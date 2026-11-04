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
   and the partial is cleared. A tile's activation scales travel with it:
   one thread per tile row issues a 4-byte `cp.async` of the row's scale
   into a per-stage slot, in the tile's own cp.async group, and the scales
   are read from shared memory right after the barrier that publishes the
   stage. The weight scale (one float per K tile) is a register load one
   tile ahead. Until commit bed9867 the activation scales were `__ldg`
   loads one tile ahead; in the weight-streaming regime they stalled the
   pipeline (next section but one).
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
   extra barrier and the same shared stages and scale slots as the unfused
   one; the quantizer writes the scales the unfused kernel copies. Rows
   past M are skipped: dividing their zeros by an eps-sized scale would
   take the slow path of the IEEE division.

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
CTAs per SM (the first launch on a device checks that the compiled kernel
really gets them). Six have 128-column tiles and six 64-column tiles (16,
32, 64 and 128 rows). `bench/autotune.py` times every configuration with
every split count on the Qwen3-8B shapes at every power of two M = 1-8192,
with the benchmark's own harness (`bench_gemm.time_calls`: CUPTI kernel
spans, weights cold, activations L2-resident, 5 interleaved runs), and
`csrc/plan_table.inc` freezes the winners (all timings in
`results/gemm/autotune.json`):

- M = 1-32: `m16n64s4w4b2` (16x64 tile, 4 stages, two CTAs per SM) without
  split-K; down (K = 12288) with 2 splits.
- M = 64: `m32n64s4w4b2` (down: 2 splits). M = 128-256: `m64n64s3w4b2`, or
  `m128n64s3w8` (o at 256, down at 128-256), `m64n128s2w4b2` for gate_up at
  256.
- M >= 512: `m64n128s2w4b2` (64x128, two 48 KB stages, two CTAs per SM),
  except down at M = 512-1024 (`m128n128s3w8`, one CTA of 8 warps).

Until commit 5383991 the sweep timed single calls with CUDA events
(activations cold, about 1 us resolution) on the eight benchmark batch
sizes only, and a batch size between two entries ran the plan of the next
one up. Both showed up as losses: `results/gemm/plan_probe_426750a.json`
times every plan at chosen shapes next to the tuned Triton configs, and the
automatic plan was up to 21% slower than the best plan the kernel already
had (qkv at M = 512, which ran the M = 1024 plan: 109.8 against 90.8 us
for `m64n128s2w4b2`; o and qkv at M = 32: 22.5 and 31.6 us against 21.1 and
30.2).

Between two tabulated batch sizes, `choose_plan` starts from the entry
above (the smallest tabulated M >= M). For M >= 256, where the GEMM is
compute bound, that entry's tiles can load the SMs unevenly: down at
M = 768 got the M = 1024 plan, 192 CTAs of 128x128 on 128 SMs, so half the
SMs ran two tiles and half one. `balanced_entry` therefore compares every
plan tabulated for the shape at M / 4 to 4 M (tiles of at least 64 rows)
by the work of its busiest SM, ceil(CTAs / SMs) x BM x BN x K tiles per
CTA, and takes the least (ties keep the entry above, then the nearest
entry; commit 667fc95). On the held-out batch sizes
(`results/gemm/qwen3_8b_offgrid.json`, against the recording at 35d0b77)
it took qkv, o and down at M = 384 from 85.8, 57.1 and 161.3 us to 76.1,
48.7 and 141.9 us, and qkv and down at M = 768 from 139.0 and 319.7 us to
134.7 and 248.3 us. The rule was written after those cells were probed
(`results/gemm/plan_probe_6a2a470.json`: the best plans had 3-4.5 CTAs per
SM, the plans of the entry above 0.75-2.25), so they no longer test it
blind. On six batch sizes it was not designed on (M = 320, 640, 1280, 1536,
2500, 6000; `results/gemm/qwen3_8b_check_6a2a470.json` and
`..._check_667fc95.json`) it changes four cells: o at M = 320 56.5 -> 47.1
us, down at 320 160.7 -> 139.5, down at 640 318.9 -> 244.1, and gate_up at
320 220.3 -> 223.9 (2% slower).

The 64-column tiles are what fixed the middle range in the first place.
With only 128-wide tiles, o at M = 64 has 32 column tiles: 32 CTAs with
64-row tiles (too few for 128 SMs), or 128 CTAs with 16-row tiles, one per
SM, each streaming 512 KB of weights. 64-wide tiles double the column tiles.

The second CTA per SM mattered for prefill while the activation scales were
global loads: for gate_up at M = 8192 the first sweep measured 6576 us for
the 64x128 tile with four stages and one CTA per SM, 5390 us with two stages
and two CTAs, and 6176 us for the 128x128 tile with 8 warps. With the scales
in shared memory the 8-warp 128x128 tile is within 0.6% of the two-CTA
64x128 tile (5297 against 5268 us), while the four-stage single-CTA 64x128
tile still takes 6108 us (`results/gemm/autotune.json`): with one 4-warp CTA
per SM, the barrier and the promotion FMAs that end every K tile leave the
tensor pipes idle, which a second CTA or more warps fill.

Fused quantization has its own table (`csrc/fused_plan_table.inc`, from
`bench/autotune.py --fused`, results in `results/gemm/autotune_fused.json`):
five small-M configurations at M = 1-256, each timed against
`quantize_cuda` + `gemm` with the frozen plan in the same harness. Since the
plain GEMM stages its activation scales in shared memory, the fused kernel
loses everywhere: at M = 1-16 by 0.2-1.1 us on qkv, o and down (qkv 30.9
against 30.3 us at M = 1, o 22.2 against 21.6 at M = 16) and by 1.6-5.5 us
on gate_up, by 1.0-24 us at M = 32-64 and by up to 358 us at M = 256, where
every CTA of a tile row re-quantizes the same rows. `prefers_fused` is now
false for every shape, so `linear` always runs
`quantize_cuda` and then the GEMM.

## Measured performance (results/gemm/summary.md)

Peaks on this GPU (results/micro): FP8 MMA with FP32 accumulation 1024 FLOP
per SM per clock, 367 TFLOPS at the 2.80 GHz the microbenchmark ran at
(median SM clock during the GEMM runs: 2.79 GHz); DRAM read 957 GB/s;
L2-resident reads up to 4.4 TB/s. "Cold L2" means the weights are evicted
(a 256 MB read) and the call's activations are then read back into L2, as
they are in a model where the previous layer has just written them. All
figures below are cold unless marked warm; recorded at commit 35d0b77.

- **Prefill (M >= 1024):** 305-330 TFLOPS, 83-90% of the MMA peak. With the
  activation quantization included (`linear`), 1.72-2.08x faster than vLLM's
  actual default on this GPU (FP8 Marlin, whose bf16 MMAs peak at half the
  FP8 rate). GEMM only: 1.10-1.21x faster than vLLM's Triton kernel as
  shipped and 1.03-1.09x faster than the same kernel tuned here. cuBLASLt
  with one scale per tensor is up to 9% faster (0.91-0.96x of our time),
  except on down at M = 8192 (1.23x slower).
- **Decode (M = 1-32):** a weight stream at 862-933 GB/s (90-97% of the
  measured DRAM read bandwidth). GEMM only, 1.00-1.06x of tuned Triton and
  1.00-1.07x of cuBLASLt: no loss to either. `linear` adds the separate
  quantization kernel (0.5-2.2 us; vLLM's own adds 1.4-3.2 us) and is
  1.00-1.10x of Marlin, which reads the same FP8 bytes: ties on down at
  M = 1-16 (58.1-58.7 us both; 0.998x at M = 4 and 16), ahead elsewhere.
- **Middle (M = 64-512):** 106-316 TFLOPS; 1.00-1.19x of tuned Triton,
  1.04-2.22x of default Triton, 1.03-2.16x of Marlin (`linear`). cuBLASLt
  per tensor is up to 12% faster (0.88-1.04x; o at M = 256: 28.6 against
  32.6 us).
- **Warm L2** (weights L2-resident, which a decode step of an 8 GB model
  never sees): GEMM only, tuned Triton is 12% faster on qkv at M = 32 (13.2
  against 15.0 us) and 2% on gate_up at M = 64; with the quantization kernel
  included, Marlin is faster on o at M = 1-16 (0.81-0.99x) and on down at
  M = 1 and 32 (0.99x, 0.96x), because about 2 us of quantization launch is
  a quarter of an 8 us GEMM.

Against the previous recording (commit 3660f45, the kernel and plan table
before these changes):
qkv at M = 1 28.2 us (was 29.3), o at M = 16 19.5 (21.1), down at M = 16 56.6
(57.4), qkv at M = 256 50.1 (54.7), down at M = 256 86.7 (98.8), qkv at
M = 512 88.0 (110.9 off the grid then), gate_up at M = 8192 5270 (5374).
- **Off the tuning grid** (M = 3, 12, 24, 48, 100, 384, 768, 3000, between
  the powers of two; second table of results/gemm/summary.md, recorded at
  1199990): `linear` 1.01-2.23x of Marlin; GEMM only 1.07-2.34x of default
  Triton, 0.99-1.41x of tuned Triton (with the config of the nearest tuned
  M, as vLLM picks it) and 0.89-1.09x of cuBLASLt. Still behind: tuned
  Triton by 1% on o at M = 48 and 100 and on down at M = 100, cuBLASLt by up
  to 11% (o at M = 768).

The tuned Triton configs are the fastest of each shape's five best warm-L2
candidates (CUDA events), re-timed with the same cold-L2 CUPTI harness as
these tables (`bench/tune_triton.py`), now on the same powers of two as the
plan table.

## Where the losses came from (results/gemm/plan_probe_426750a.json)

`bench/plan_probe.py` times every configuration x split count at a shape
with the benchmark's harness, next to vLLM's Triton kernel with every tuned
config of that layer, and records each plan's CTA count. At the commit
before these changes it found two causes.

1. **Plan choice.** The automatic plan was up to 21% slower than the best
   compiled plan (qkv at M = 512: 109.8 against 90.8 us; o, qkv and down at
   M = 32: 1.4-1.9 us slower), because the table was timed with CUDA events
   on cold activations and only on the eight benchmark batch sizes. Fixed by
   tuning on every power of two with the benchmark's harness (above).
2. **Activation-scale loads in the weight stream.** Even the best plan
   trailed tuned Triton by 3-7% at M = 16-64 on qkv, o and down, while it
   tied at M = 1-4. With the same CTAs and the same weight bytes,
   `m16n64s4w4b2` without split-K took 22.4 us for o at M = 16 against
   19.6 us at M = 4, and 67.0 against 56.7 us for down: the cost grew with
   the number of real rows and with K. A build that loaded no activation
   scales (wrong results, timing only) ran M = 16 at the M = 4 speed (o 19.4,
   down 56.8 us); one that loaded no activation tile instead did not (22.9,
   66.4 us). The scales were `__ldg` loads one K tile ahead whose latency
   showed up in every K tile; staging them with the tile's own cp.async
   group (commit bed9867) made the same plans 19.6 and 57.4 us
   (`results/gemm/autotune.json`). It also sped up the middle and prefill
   range (benchmark: qkv at M = 256 54.7 -> 50.1 us, gate_up at M = 8192
   5374 -> 5270 us). The two diagnostic builds were not committed; their
   numbers are in the message of commit bed9867.

What was not the cause: wave quantization at decode sizes (Triton streams
qkv at M = 16 with 96 CTAs, fewer than the 128 SMs, as fast as anything
here), split-K count (the best decode plans need none, except down), and an
L2 prefetch of the weight scales (no change).

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

Recorded at commit 6a2a470 (retuned tables, scales in shared memory):

| | stock (Marlin W8A16) | plugin (blockfp8 W8A8) | speedup |
|---|---|---|---|
| prefill, 64 x 256 tokens | 10315 tok/s | 18042 tok/s | 1.75x |
| prefill, 16 x 1024 tokens | 10160 tok/s | 17576 tok/s | 1.73x |
| prefill, 4 x 4096 tokens | 9551 tok/s | 15801 tok/s | 1.65x |
| decode TPOT, batch 1 | 10.10 ms | 9.86 ms | 1.024x |
| decode TPOT, batch 4 | 10.23 ms | 10.01 ms | 1.023x |
| decode TPOT, batch 16 | 10.84 ms | 10.51 ms | 1.032x |
| decode TPOT, batch 64 | 14.08 ms | 12.65 ms | 1.113x |

Before (commit ec74438, the plan table of that time): prefill 1.63-1.71x,
TPOT 1.011-1.017x at batch 1-16 and 1.090x at 64. Every GEMM these runs time
has a tabulated M (decode 1, 4, 16, 64; prefill chunks of 8192 tokens; TPOT
prompt batches of 128-8192 tokens), so the later balanced choice between
entries (commit 667fc95) does not change their plans. Prefill gains less
than the GEMMs alone (1.72-2.08x) because attention, norms and sampling do
not change; decode at batch 1-16 gains 2-3%, in line with the GEMM-level
1.00-1.10x against Marlin (the linear layers are a weight stream either
way), and batch 64 gains 11%, where `linear` is 1.03-1.23x faster than
Marlin.

**Perplexity** (results/e2e/wikitext2_ppl.json, `bench/e2e_vllm.py --ppl`;
stock and Triton recorded at commit 0a65e1d, the plugin at 0a65e1d and again at
2af08cb with the kernel and plan tables of this document):
the wikitext-2-raw-v1 test split, its 4,358 lines joined with blank lines,
tokenizes to 299,078 tokens (`tokens` in the JSON); it is cut into 147
non-overlapping windows of up to 2048 tokens, each scored on its own with
prompt_logprobs. A window's first token has no context and is not scored,
so 298,931 tokens are scored (`scored_tokens`), in one engine process per
configuration. Stock 9.7646, plugin 9.7735
(+0.09%; 9.77351 at 0a65e1d, 9.77347 at 2af08cb), vLLM's Triton W8A8 path
9.7739 (+0.10%); a second stock run reproduced the first bit for bit. The
per-token logprob differences against stock are the same for both W8A8
kernels (mean 0.062, p99 0.48, mean signed -9e-4 for the plugin and -9.5e-4
for Triton), and between the two kernels larger (mean 0.079, p99 0.62): they
disagree token by token as much as each does with stock, but cost the same
perplexity. The two plugin runs agree exactly on 298,862 of 298,931 token
logprobs; the other 69 differ by at most 0.79.

**Outputs** (recorded at 6a2a470; the 24-prompt prefill has an untabulated
M, so the later balanced choice may change its plan). They are not the
same, and they are not expected to be: the plugin quantizes activations to
e4m3 (W8A8, the scheme the checkpoint was made for), Marlin keeps them in
bf16 (W8A16). Over 24 chat prompts with greedy decoding (up to 64 tokens),
9 sequences are identical to the first stock run in every plugin run, and
59.5-60.0% of the tokens come before the first divergence. Teacher-forced
on the stock run's tokens, the logprobs of the same 1,449 tokens differ by
a mean of 0.018-0.019 (p99 0.22-0.24, max 0.35-0.44). vLLM's own W8A8 path
(its Triton block kernel, Marlin disabled, one run) is no closer to stock:
9 of 24 identical, 69% before divergence, teacher-forced mean 0.019, p99
0.21, max 0.50. The plugin and that Triton path also differ from each
other (7 of 24 identical, teacher-forced mean 0.023, p99 0.27-0.28, max
0.51). Both quantize activations per 1x128 group, but the Triton path does
it with vLLM's QuantFP8 op inside the compiled graph and the plugin with
blockfp8's kernels, and the GEMM-level tests establish equal accumulation
error, not bitwise-equal results; these runs do not show which difference
dominates. vLLM itself is not run-to-run deterministic here either: of the
four repeated stock runs, two matched the first on every prompt and two
did not (23 and 19 of 24 identical); all five plugin runs have the same 9
identical sequences, but one diverges from stock earlier (862 against 869
tokens before the first divergence).
