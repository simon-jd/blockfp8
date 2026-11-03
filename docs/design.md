# blockfp8 design

## What is computed

DeepSeek-V3-style FP8 checkpoints (Qwen3-*-FP8 among them) store each linear
layer's weight `W [N, K]` as e4m3 values with one FP32 scale per 128x128
block, and quantize activations on the fly: one FP32 scale per token per
128-wide group of K. A linear layer is then

```
C[m, n] = sum_kb  sa[m, kb] * sb[n / 128, kb] * sum_{k in kb} A_q[m, k] * B_q[n, k]
```

with `A_q [M, K]` and `B_q [N, K]` both K-contiguous (a "TN" GEMM) and `C` in
bf16. The layouts are documented in `blockfp8/layout.py`; the scales of `A`
may have any strides (vLLM's Triton path passes them row-major, DeepGEMM and
vLLM's CUTLASS path M-major), and `quantize_activations` produces M-major.

Supported shapes: any M; N a multiple of 8; K a multiple of 16 (16-byte
loads along K); everything else is rejected with an error naming the
constraint. Tails in M, N and K are zero-filled in shared memory, so a
partial last K block contributes only its real columns.

Entry points (`blockfp8/gemm.py`, ops in `csrc/ops.cpp`):

- `gemm(a_q, a_scales, b_q, b_scales)`: pre-quantized activations.
- `quantize_cuda(x)`: bf16 activations to e4m3 and M-major scales, a
  standalone CUDA kernel (`csrc/quant_sm89.cu`).
- `gemm_bf16(x, b_q, b_scales)`: bf16 activations, quantized inside the
  GEMM (below).
- `linear(x, b_q, b_scales)`: what a linear layer calls and what the vLLM
  plugin runs: `gemm_bf16` where the plan table says fusing is faster, else
  `quantize_cuda` followed by `gemm`.

All three quantizers (the PyTorch reference `quantize_activations`, the CUDA
kernel and the fused path) use vLLM's arithmetic, `scale = max(amax, eps) /
448` with an IEEE division and `q = e4m3(clamp(x / scale))`, and the tests
check them bitwise against `per_token_group_quant_fp8`. PyTorch evaluates
`t / 448.0` as `t * (1 / 448)`, which is one ulp off for about 4% of groups,
so the reference divides tensor by tensor.

## Kernel structure (csrc/gemm_kernel.cuh)

One CTA computes a `BM x BN` tile of C over a contiguous range of 128-wide K
tiles. A K tile is exactly one scale block, which is what makes the
promotion cheap (below).

1. **Global to shared.** 16-byte `cp.async.cg` copies (L1 bypassed, the data
   is used once per CTA), predicated per chunk: chunks outside M, N or K are
   zero-filled with `src-size = 0`, so no branch is needed on the data path.
   The shared tiles form a ring of `Stages` buffers; while the MMAs consume
   buffer `i`, the copies for tile `i + Stages - 1` are in flight.
2. **Swizzled shared layout.** A tile row is 128 bytes (128 e4m3 values).
   `Swizzle<3,4,3>` XORs the 16-byte chunk index with the row index mod 8.
   Eight consecutive rows at the same logical chunk land in eight different
   physical chunks, so both the cp.async stores (8 threads per row) and the
   `ldmatrix` reads (8 rows of 16 bytes per 8x8 matrix) are free of bank
   conflicts.
3. **Shared to registers.** `ldmatrix.x4` (CuTe `SM75_U32x4_LDSM_N`). For
   8-bit data the m16n8k32 A fragment of one thread is exactly what x4
   delivers for a 16x32 tile; for B one x4 feeds two n8 MMAs, which is why
   the tiled MMA gives each warp 16 columns per step
   (`Tile<16*WarpsM, 16*WarpsN, 32>`). Fragments for k-step `j+1` are loaded
   while the MMAs of step `j` run.
4. **MMA.** `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`
   (`SM89_16x8x32_F32E4M3E4M3F32_TN`, SASS `QMMA.16832.F32.E4M3.E4M3`).
5. **Two-level accumulation.** The MMAs of one K tile accumulate into a
   separate FP32 partial; after the tile, every accumulator is promoted with
   one FMA on the CUDA cores, `acc = fma(partial, sa[m,kb] * sb[nb,kb], acc)`,
   and the partial is cleared. The scale loads for a tile are issued before
   its MMAs and consumed after them, so their latency is hidden.
6. **Epilogue.** bf16 pairs (adjacent accumulator columns) stored straight
   from registers.
7. **Split-K.** For small M a CTA handles only part of K. Each CTA writes
   its FP32 partial tile to a workspace, then increments a per-tile counter;
   the CTA that sees the count reach `splits` sums all partials in split order
   0, 1, ..., S-1 (its own from registers), writes bf16 and resets the
   counter. Arrival order never changes the arithmetic, so results are
   bitwise reproducible; the counter is only a "who is last" flag.
8. **Rasterization.** CTAs are numbered so that consecutive CTAs walk 8 tile
   rows of M before moving along N (as in Triton's `GROUP_SIZE_M`), which
   keeps the A and B panels of concurrently running CTAs in L2.
9. **Fused activation quantization** (`kFuseQuant`, `gemm_bf16`). A arrives as
   bf16 and never goes through shared memory as bf16: the `kThreads / BM`
   threads that share a tile row each hold 16-byte chunks of that row for the
   next K tile in registers, loaded one K tile ahead with `ld.global.nc`. At
   the start of each K tile, after its first MMAs are issued, they reduce the
   row's amax with shuffles, divide, convert with `cvt.rn.satfinite.e4m3x2`
   and store 8-byte pieces into the next tile's e4m3 stage; the barrier the
   K tile ends with anyway publishes that stage, so the fused kernel has no
   extra barrier and the same shared stages as the unfused one (plus one
   float scale per row and stage). Rows past M are skipped: dividing their
   zeros by an eps-sized scale would take the slow path of the IEEE division.

## Precision (results/micro/mma_precision.json)

Measured on the RTX 4090 with crafted operands (`bench/micro/mma_precision.cu`):

- The FP32-accumulating e4m3 MMA aligns its 32 products **and the
  accumulator input C** to the largest of them and keeps **13 fraction bits**
  below that leading bit. `2^16 + 2^(16-j)` is exact up to j = 13 and the
  small term vanishes from j = 14 on, whether it sits at k = 1, at k = 31 or
  comes in through C; 31 small terms vanish together, so they are dropped one
  by one, not summed first.
- In these single-sign cases the dropped bits are **truncated toward zero**
  (negative small terms are not floored). Mixed-sign sums are not fully
  characterized: a block and its negation do not always give exactly
  opposite partials (`test_cancellation_leaves_small_residual` sees residues
  above FP32 rounding, though within 2^-13 of the cancelled magnitude).
- The FP16-accumulating form keeps 10 bits, and its accumulator overflows at
  65504, which full-range e4m3 products (up to 448^2) reach in a single MMA.

What this means for the promotion interval, from random N(0, 7^2) e4m3
operands (error relative to the exact result, median over 16 384 outputs):

| K = 4096, FP32 accumulator | median rel. error | max error / sum of abs terms |
|---|---|---|
| never promote (all 128 MMAs in the MMA accumulator) | 9.8e-4 | 4.0e-4 |
| promote every 8 MMAs (K = 256) | 2.1e-4 | 2.8e-5 |
| promote every 4 MMAs (K = 128, this kernel) | 1.4e-4 | 1.5e-5 |
| promote every MMA (K = 32) | 5.6e-5 | 6.3e-6 |
| FP16 accumulator, promote every 4 MMAs | 4.6e-4 | 4.5e-5 |

Rounding the same exact results to bf16 errs by a median of 1.37e-3 (unit
roundoff 2^-8 = 3.9e-3), so without promotion the accumulation error is
already 71% of the bf16 output's own rounding error. Promoting once per
128-wide block, which the block scales require anyway, brings it to 1.37e-4,
10 times below; rounded to bf16, the promoted result errs by a median of
1.41e-3, 3% more than rounding alone. Promoting every MMA would gain another
2.4x at four times the FMAs, which a bf16 output cannot show (1.38e-3). So
the scale block size is also a sufficient promotion interval.

The tests use this model for a worst-case bound (`blockfp8.error_bound`):
one MMA errs by at most 33 * 2^-13 times its largest addend, which is at
most the block's sum of |products|; four MMAs per block, then FP32 scaling
and accumulation. On real data the kernel's error is the same as that of
vLLM's Triton block kernel to three digits (same instruction, same
promotion), which the tests also check.

## Configurations and the plan table

Twelve configurations are compiled (`csrc/gemm_sm89.cu`), named
`m<BM>n<BN>s<stages>w<warps>[b2]`; `b2` means `__launch_bounds__` allows two
CTAs per SM. Six have 128-column tiles and six 64-column tiles (16, 32, 64 and
128 rows). `bench/autotune.py` timed every configuration with every split
count on the Qwen3-8B shapes (cold L2) and `csrc/plan_table.inc` freezes the
winners (all timings in `results/gemm/autotune.json`):

- M = 1-16: `m16n64s4w4b2` (16x64 tile, 4 stages, two CTAs per SM) with
  split-K 1-4, except qkv at M = 1 (`m16n128s5w4`).
- M = 64-256: 64-column tiles (`m32n64s4w4b2`, `m16n64s4w4b2`,
  `m64n64s3w4b2`), except gate_up at M = 256 (`m64n128s2w4b2`).
- M = 1024-8192: `m128n64s2w4b2` (128x64, two 48 KB stages, two CTAs per
  SM) for o and gate_up and at M = 1024 everywhere; `m64n128s2w4b2` for qkv
  and down at 4096-8192.

The 64-column tiles are what fixed the middle range. With only 128-wide
tiles, o at M = 64 has 32 column tiles: 32 CTAs with 64-row tiles (too few
for 128 SMs), or 128 CTAs with 16-row tiles, one per SM, each streaming
512 KB of weights. 64-wide tiles double the column tiles: `m16n64s4w4b2`
runs 256 CTAs, two per SM, each streaming 256 KB. The autotuned time for o at M = 64
went from 28.7 us (`m16n128s5w4`) to 25.5 us (`m16n64s4w4b2`), and qkv at
M = 64 from 37.9 to 34.8 us (CUDA-event timings, `results/gemm/autotune.json`
before and after commit c989c75).

The second CTA per SM is what makes prefill fast. For gate_up at M = 8192
the 64x128 tile took 6576 us with four stages and one CTA per SM and 5390 us
with two stages and two CTAs; the 128x128 tile with 8 warps (one CTA per SM,
234 registers) took 6176 us (first sweep). With a single resident CTA, every
K tile ends with a barrier and the promotion FMAs, during which the tensor
pipes have nothing to do; a second, independently scheduled CTA fills those
gaps.

Fused quantization has its own table (`csrc/fused_plan_table.inc`, from
`bench/autotune.py --fused`, results in `results/gemm/autotune_fused.json`):
five small-M configurations, each timed against `quantize_cuda` + `gemm`
with the frozen plan, both with the bf16 activations L2-resident.
`prefers_fused` is true only where the fused kernel was faster: qkv at
M = 1-16, o at M = 4-16 and down at M = 16, by 0.1-1.1 us (CUDA-event
timings, about 1 us resolution in practice, so these choices are close to
coin flips). It tied on o at
M = 1, lost by 1-2 us on gate_up at M = 1-16 and down at M = 1-4, and lost by
4-365 us at M >= 64, where every CTA of a tile row re-quantizes the same rows
(3.1x slower for gate_up at M = 256).

## Measured performance (results/gemm/summary.md)

Peaks on this GPU (results/micro): FP8 MMA with FP32 accumulation 1024 FLOP
per SM per clock, 367 TFLOPS at the 2.80 GHz the microbenchmark ran at
(median SM clock during the GEMM runs: 2.78 GHz); DRAM read 957 GB/s;
L2-resident reads up to 4.4 TB/s. "Cold L2" means the weights are evicted
(a 256 MB read) and the call's activations are then read back into L2, as
they are in a model where the previous layer has just written them.

- **Prefill (M >= 1024):** 307-321 TFLOPS, 84-87% of the MMA peak. With the
  activation quantization included (`linear`), 1.67-2.03x faster than vLLM's
  actual default on this GPU (FP8 Marlin, whose bf16 MMAs peak at half the
  FP8 rate). GEMM only: 1.08-1.21x faster than vLLM's Triton kernel as
  shipped and 1.01-1.09x faster than the same kernel tuned here. cuBLASLt
  with one scale per tensor is up to 9% faster (0.91-0.93x of our time) on
  every shape but one, because it has no per-block scales to apply; on down
  at M = 8192 it measured 1.27x slower.
- **Decode (M <= 16):** a weight stream at 805-935 GB/s (84-98% of the
  measured DRAM read bandwidth). GEMM only, this is 1.02-1.14x faster than
  Marlin, which reads the same FP8 bytes. The activation quantization adds
  0.9-2.1 us on top (fused where that is faster, else blockfp8's own
  quantization kernel), against 1.8-3.0 us with vLLM's kernel, so `linear`
  is 0.99-1.06x of Marlin: ahead on qkv, o and gate_up, even on down at
  M = 1-4, and 1% behind on down at M = 16. Tuned Triton is up to 7% faster
  on o and qkv at M = 1-16.
- **Middle (M = 64-256):** 0.95-1.04x of tuned Triton (before the 64-column
  tiles: 0.79-1.01x). Still behind by 2-4% at M = 64 on all four shapes and
  by 5% on qkv at M = 256. cuBLASLt (per tensor) is up to 17% faster (0.83-1.01x).
  Against Marlin, `linear` is 1.15-2.11x except down at M = 64, where it
  ties (1.00x): Marlin (64.4 us) and the GEMM alone (61.4 us) are both
  close to the 52.6 us it takes to read the 50 MB of weights, and `linear`
  adds 2.6 us of quantization. In the previous recording (commit b3f50b9)
  the same cells were 63.0 and 63.6 us, a 5% loss; cells of unchanged
  kernels moved by up to 4.5% between the two recordings.
- **Off the tuning grid** (M = 2, 8, 32, 100, 512, 3000; second table in
  results/gemm/summary.md): the plan table takes the entry with the
  smallest tabulated M >= M, tuned Triton the nearest tabulated M. At
  M = 2, 8 and 100 the kernel loses at most 5% to any baseline but cuBLASLt
  (o at M = 8: 0.95x of tuned Triton; cuBLASLt up to 11% faster). M = 32 (planned as 64) loses 2-7% to
  Marlin and 7-14% to tuned Triton on qkv, o and down. qkv at M = 512
  (planned as 1024) takes 110.9 us, twice M = 256's 54.7 us, and loses to
  every baseline but Marlin (0.93x of default Triton, 0.88x of tuned,
  0.71x of cuBLASLt). A plan table on a finer M grid would be the fix; it
  was not built.

The tuned Triton configs are the fastest of each shape's five best warm-L2
candidates (CUDA events), re-timed with the same cold-L2 CUPTI harness as
these tables (`bench/tune_triton.py`); that changed 11 of 32 picks (the warm
pick would have been up to 12% slower, qkv at M = 64). The warm-L2 tables are in the same file.

## Why "+q" looked almost free for o at M = 64 (results/gemm/l2_probe.json)

The first benchmark (commit d9313bf) evicted everything before each cold
call, activations included. For o at M = 64 the GEMM-only time was then
27.3 us and the GEMM plus vLLM's quantizer 27.6 us, while the quantizer
alone takes 2-3 us. `bench/l2_probe.py` times the GEMM kernel alone (o, M =
16-256, split-K 1) after the flush with different predecessors. With the
plan of that time for M = 64 (`m16n128s5w4`) it takes 28.4 us with nothing
before it, 27.9 us after a spin kernel and 28.0 us after a quantizer run
whose output it does not read, but 23.8 us after the quantizer whose output
it reads and 23.6 us after a plain read of its A operand and scales. So the
gap was not a launch effect but cold activations: in the "+q" column the
GEMM read A from L2, where the quantizer had just written it, and in the
GEMM-only column from DRAM. The effect is 4.2-5.9 us with that 16x128 tile
and 0.8-2.7 us with today's 16x64 tile. The benchmark now reads every
call's activations back into L2 after the flush, which is what a real
layer sees; for o at M = 64 "+q" now adds 3.1 us like every other row.

## vLLM integration and end to end (results/e2e/summary.md)

`blockfp8_vllm.register` (entry point `vllm.general_plugins: blockfp8`)
inserts `BlockFp8LinearKernel` into vLLM's `_POSSIBLE_FP8_BLOCK_KERNELS`
for CUDA, right before `MarlinFP8ScaledMMLinearKernel`. The class derives
from vLLM's `Fp8BlockScaledMMLinearKernel`; vLLM's loader and
`process_weights_after_loading` keep the checkpoint's `[N, K]` e4m3 weights
and float32 128x128 block scales as they are, and `apply_weights` calls
`torch.ops.blockfp8.linear` on the bf16 activations. The ops have fake
(shape) implementations, so vLLM's torch.compile and CUDA graphs capture
them; split-K workspaces come from the graph pool and the counter buffers
are never freed.

`bench/e2e_vllm.py` runs Qwen3-8B-FP8 from the local snapshot (offline) in
a fresh process per run under the GPU lock, five interleaved runs per
configuration, `gpu_memory_utilization` 0.80, prefix caching off, and reads
vLLM's log to confirm which kernel it selected in every run.

| | stock (Marlin W8A16) | plugin (blockfp8 W8A8) | speedup |
|---|---|---|---|
| prefill, 64 x 256 tokens | 10228 tok/s | 17519 tok/s | 1.71x |
| prefill, 16 x 1024 tokens | 10050 tok/s | 17038 tok/s | 1.70x |
| prefill, 4 x 4096 tokens | 9460 tok/s | 15461 tok/s | 1.63x |
| decode TPOT, batch 1 | 10.12 ms | 9.95 ms | 1.017x |
| decode TPOT, batch 4 | 10.24 ms | 10.12 ms | 1.012x |
| decode TPOT, batch 16 | 10.84 ms | 10.72 ms | 1.011x |
| decode TPOT, batch 64 | 14.08 ms | 12.91 ms | 1.090x |

Prefill gains less than the GEMMs alone (1.67-2.02x in the GEMM recording
of that time) because attention, norms and sampling do not change; decode at
batch 1-16 gains 1-2%, in line with the GEMM-level 0.97-1.06x against
Marlin then (0.99-1.06x now; the linear layers are a weight stream either
way), and batch 64 gains 9%, where `linear` was 1.13-1.15x faster than
Marlin on qkv, o and gate_up (and 0.95x on down).

**Perplexity** (results/e2e/wikitext2_ppl.json, `bench/e2e_vllm.py --ppl`):
the wikitext-2-raw-v1 test split, its 4,358 lines joined with blank lines,
is 298,931 tokens; it is cut into 147 non-overlapping windows of 2048 tokens,
each scored on its own with prompt_logprobs (every token but a window's
first), in one engine process per configuration. Stock 9.7646, plugin 9.7735
(+0.09%), vLLM's Triton W8A8 path 9.7739 (+0.10%); a second stock run
reproduced the first bit for bit. The per-token logprob differences against
stock are the same for both W8A8 kernels (mean 0.062, p99 0.48, mean signed
-9e-4 for the plugin and -9.5e-4 for Triton), and between the two kernels
larger (mean 0.079, p99 0.62): they disagree token by token as much as each
does with stock, but cost the same perplexity.

**Outputs.** They are not the same, and they are not expected to be: the
plugin quantizes activations to e4m3 (W8A8, the scheme the checkpoint was
made for), Marlin keeps them in bf16 (W8A16). Over 24 chat prompts with
greedy decoding (up to 64 tokens), 7-9 sequences are identical to the first
stock run, and 57-60% of the tokens come before the first divergence.
Teacher-forced on the stock run's tokens, the logprobs of the same 1,449
tokens differ by a mean of 0.019 (p99 0.23, max 0.44); the median (1.1e-4)
said little, since half the tokens barely move while 1% move by more than
0.23. vLLM's own W8A8
path (its Triton block kernel, Marlin disabled, one run) is no closer to
stock: 9 of 24 identical, 69% before divergence, teacher-forced mean 0.019,
p99 0.21, max 0.50. The plugin and that Triton path also differ from each
other (6 of 24 identical, teacher-forced mean 0.023, p99 0.28, max 0.51).
Both quantize
activations per 1x128 group, but the Triton path does it with vLLM's
QuantFP8 op inside the compiled graph and the plugin with blockfp8's
kernels, and the GEMM-level tests establish equal accumulation error, not
bitwise-equal results; these runs do not show which difference dominates. vLLM itself is not run-to-run deterministic
here either: one of four repeated stock runs differed on one prompt
(23 of 24 identical), and one of five plugin runs differed from the other
four.
