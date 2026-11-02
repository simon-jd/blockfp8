# Design trade-offs

Each entry: what was chosen, what else was possible, why, and what it costs.
Numbers come from `results/` (see the README for how they were measured).

## 1. FP32-accumulating FP8 MMA, not FP16-accumulating

- **Chosen:** `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`.
- **Alternative:** the `.f16` accumulator form of the same instruction, which
  sm_89 also has.
- **Why:** the FP16 form keeps 10 bits after alignment instead of 13 and its
  accumulator overflows at 65504, which a single MMA of full-range e4m3
  products (up to 448^2 each) can exceed; using it would mean rescaling the
  operands and promoting after every MMA (results/micro/mma_precision.json).
- **Cost:** half the peak. The FP16-accumulating MMA issues 2048 FLOP per SM
  per clock, the FP32 one 1024 (results/micro/mma_peak.json); on GeForce Ada
  the FP32-accumulating FP8 rate equals the FP16-accumulating bf16 rate and
  is twice the FP32-accumulating bf16 rate (512).

## 2. Promote once per 128-wide K block

- **Chosen:** the MMAs of one K tile accumulate into an FP32 partial; one FMA
  per accumulator folds it into the result with that block's two scales.
- **Alternatives:** (a) accumulate the whole K range inside the MMA
  accumulator and rescale it whenever the scale changes (one accumulator set,
  but every block's contribution then passes through the MMA adder together
  with the running total); (b) promote after every k32 MMA (more FMAs, no
  second accumulator set).
- **Why:** the scales change every 128 columns of K anyway, so a promotion per
  scale block is the cheapest correct scheme: 64 FMAs per thread per K tile
  next to 64 MMAs per warp. The measured MMA keeps 13 bits relative to the
  larger of its accumulator input and its products, so (a) loses accuracy
  as the running sum grows (median error 9.8e-4 at K = 4096, close to bf16
  rounding) while promotion per 128 gives 1.4e-4 (docs/design.md, Precision).
- **Cost:** a second set of accumulators in registers (64 floats per thread
  for the 128x128 tile). That is what caps the CTA tile at 128x128 with 8
  warps: a 64x64 warp tile would need 256 accumulator registers. Variant (b)
  was built to lift that cap (`StepPromote` in `csrc/gemm_kernel.cuh`), but
  it spills: instantiated for a 256x128 tile (64x64 per warp) ptxas reports
  255 registers plus 160-224 bytes of stack, and even at 128x128 it spills
  32-80 bytes, so no compiled configuration uses it.

## 3. CuTe on top of mma.sync / ldmatrix / cp.async

- **Chosen:** CuTe layouts, tiled copies and tiled MMA from CUTLASS v4.8.1,
  with the main loop written out by hand (the structure of CUTLASS's sm80
  tutorial kernel).
- **Alternatives:** CUTLASS 2.x/3.x device-level GEMM templates (no block-scale
  epilogue for sm_89; the block-scaled collectives are Hopper/Blackwell only);
  raw PTX with hand-computed fragment indices; Triton.
- **Why:** CuTe computes the swizzled shared-memory addresses and the
  ldmatrix/MMA fragment mappings, which are the error-prone part, while the
  pipeline and the promotion stay explicit and easy to change.
- **Cost:** heavy templates (about 40 s to compile four configurations) and
  register use that is hard to steer from source.

## 4. cp.async multistage pipeline

- **Chosen:** 16-byte `cp.async.cg` into a ring of 3 to 5 shared stages.
- **Alternatives:** TMA and mbarrier pipelines do not exist on sm_89; plain
  `ld.global` into registers then `st.shared` costs registers and issue slots.
- **Why:** asynchronous copies keep the loads off the register file and let
  one K tile load while another computes; zero-fill handles every tail.
- **Cost:** shared memory: a 128x128x128 FP8 stage is 32 KB, so the large
  tile fits 3 stages in the 99 KB per-block limit and runs one CTA per SM.

## 5. Tile shapes per regime

- **Large M (prefill):** 64x128 with 4 warps (32x64 per warp), 2 stages, two
  CTAs per SM. A 128x128 tile with 8 warps halves the L2 traffic per FLOP
  but fits only one CTA per SM, and measured 15% slower (docs/design.md,
  Configurations); L2 bandwidth (4.4 TB/s) was not the limit.
- **Small M (decode):** 16, 32 or 64 rows by 128 columns with 4 warps and 4-5
  stages, plus split-K. With M <= 64 the GEMM is a weight stream: FLOPs per
  weight byte are at most 2 * 64, far below the ~330 FLOP/byte the GPU needs
  to be compute-bound, so the tile only has to keep enough weight bytes in
  flight on every SM.
- **Cost of the 16-row tile at M = 1:** 15 of 16 MMA rows are padding. The
  MMA time is still hidden behind the weight stream in cold-L2 runs (746-926
  GB/s against a 957 GB/s read peak), so swapping the operands to put the
  weights in the MMA's 16-row slot was not needed.
- **Cost of 128-wide tiles:** M = 64-256 get too few CTAs or re-read weights
  and lose to tuned Triton and cuBLASLt (0.70-1.01x); see the README.

## 6. Deterministic split-K with a "last CTA reduces" counter

- **Chosen:** every split writes an FP32 partial; the last CTA to arrive
  (atomic counter) sums the partials in fixed split order and resets the
  counter.
- **Alternatives:** atomicAdd into an FP32 output (non-deterministic, needs a
  zeroed output); a second reduction kernel (deterministic, one more launch
  and a full pass over the partials); stream-K (balances waves best, much more
  bookkeeping).
- **Why:** bitwise reproducible output in one launch; only the last CTA of a
  tile pays for the reduction, and for M <= 64 a partial tile is small.
- **Cost:** a workspace of splits x M x N floats, an int counter per tile in a
  persistent buffer (so two GEMMs on different streams of one device must not
  run concurrently), and wave quantization is only as good as the split count
  chosen for each shape.

## 7. Frozen per-shape plan table

- **Chosen:** `bench/autotune.py` times every configuration and split count on
  the Qwen3-8B shapes; the winners are compiled into `csrc/plan_table.inc`.
  Other shapes use a simple M-based rule.
- **Alternatives:** runtime autotuning on first call (start-up cost, timing
  noise inside a serving process); a closed-form heuristic only.
- **Cost:** the table is specific to this GPU and these shapes.

## 8. Accepting any activation-scale strides

- **Chosen:** the kernel takes the strides of `a_scales`, so vLLM's row-major
  scales and the M-major ones both work without a copy.
- **Cost:** none measurable; each thread loads 8 scales per K tile, one K
  tile ahead of use.
