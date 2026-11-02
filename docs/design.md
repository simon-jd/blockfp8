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

Without promotion the error approaches the bf16 output's own rounding
(unit roundoff 2^-8 = 3.9e-3, typical error ~1e-3). Promoting once per
128-wide block, which the block scales require anyway, brings it to 1.4e-4,
about 28 times below bf16 rounding; promoting every MMA would gain another
2.4x at four times the FMAs, which a bf16 output cannot show. So the scale
block size is also a sufficient promotion interval.

The tests use this model for a worst-case bound (`blockfp8.error_bound`):
one MMA errs by at most 33 * 2^-13 times its largest addend, which is at
most the block's sum of |products|; four MMAs per block, then FP32 scaling
and accumulation. On real data the kernel's error is the same as that of
vLLM's Triton block kernel to three digits (same instruction, same
promotion), which the tests also check.

## Configurations and the plan table

Six configurations are compiled (`csrc/gemm_sm89.cu`), named
`m<BM>n<BN>s<stages>w<warps>[b2]`; `b2` means `__launch_bounds__` allows two
CTAs per SM. `bench/autotune.py` timed every configuration with every
split count on the Qwen3-8B shapes (cold L2) and `csrc/plan_table.inc`
freezes the winners (all timings in `results/gemm/autotune.json`):

- M <= 16: `m16n128s5w4`, split-K 1-4 (more splits for the K = 12288 down
  projection and the narrow O projection).
- M = 64-8192: mostly `m64n128s2w4b2`: a 64x128 tile, 4 warps (32x64 each),
  two 48 KB stages, 214 registers, so two CTAs share an SM.

The second CTA per SM is what makes prefill fast. For gate_up at M = 8192
the same 64x128 tile takes 6576 us with four stages and one CTA per SM and
5390 us with two stages and two CTAs; the 128x128 tile with 8 warps (one CTA
per SM, 234 registers) takes 6176 us. With a single resident CTA, every K
tile ends with a barrier and the promotion FMAs, during which the tensor
pipes have nothing to do; a second, independently scheduled CTA fills those
gaps.

## Measured performance (results/gemm/summary.md)

Peaks on this GPU (results/micro): FP8 MMA with FP32 accumulation 1024 FLOP
per SM per clock, 367 TFLOPS at the 2.80 GHz the microbenchmark ran at
(median SM clock during the GEMM runs: 2.79 GHz); DRAM read 957 GB/s;
L2-resident reads up to 4.4 TB/s.

- **Prefill (M >= 1024):** 293-320 TFLOPS, 80-87% of the MMA peak. 1.6-2.0x
  faster than vLLM's actual default on this GPU (FP8 Marlin, whose bf16 MMAs
  peak at half the FP8 rate), 1.09-1.18x faster than vLLM's Triton kernel as
  shipped and 1.01-1.06x faster than the same kernel tuned here. cuBLASLt
  with one scale per tensor is 0.88-1.07x of our time: up to 12% faster
  on most shapes, because it has no per-block scales to apply.
- **Decode (M <= 16):** a weight stream at 746-926 GB/s (78-97% of the
  measured DRAM read bandwidth). Marlin reads the same FP8 bytes, so the
  GEMMs are close (0.99-1.09x); counting the separate activation
  quantization kernel this kernel needs (about 3 us), it is 0.90-1.04x of
  Marlin: ahead only on gate_up, behind by up to 10% elsewhere (worst: O
  projection at M = 16).
- **Middle (M = 64-256): the losses.** 0.79-1.01x of tuned Triton and
  0.70-0.95x of cuBLASLt. All tiles are 128 columns wide, so for example the
  O projection at M = 64 has 32 column tiles: with 64-row tiles that is 32
  CTAs (split-K only partly helps), with 16-row tiles 128 CTAs that each
  re-read the weights. Tuned Triton picks 64-column tiles for most of these
  shapes (results/gemm/triton_tuned_configs.json). A 64-column
  configuration or stream-K is the next step.

All of the above is cold L2, the decode-realistic case (a model's weights
do not stay in a 72 MB L2 between uses); warm-L2 tables are in the same file.
