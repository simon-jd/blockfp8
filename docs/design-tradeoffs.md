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
  was built to lift that cap, but it spills: instantiated for a 256x128 tile
  (64x64 per warp) ptxas reported 255 registers plus 160-224 bytes of stack,
  and even at 128x128 it spilled 32-80 bytes. No configuration used it, so it
  was removed (the last version is `StepPromote` in `csrc/gemm_kernel.cuh` at
  commit b3f50b9).

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
- **Cost:** heavy templates (about 40 s to compile the 12 configurations and
  5 fused variants, each in a split-K and a plain form) and register use that
  is hard to steer from source.

## 4. cp.async multistage pipeline

- **Chosen:** 16-byte `cp.async.cg` into a ring of 3 to 5 shared stages.
- **Alternatives:** TMA and mbarrier pipelines do not exist on sm_89; plain
  `ld.global` into registers then `st.shared` costs registers and issue slots.
- **Why:** asynchronous copies keep the loads off the register file and let
  one K tile load while another computes; zero-fill handles every tail.
- **Cost:** shared memory: a 128x128x128 FP8 stage is 32 KB, so the large
  tile fits 3 stages in the 99 KB per-block limit and runs one CTA per SM.

## 5. Tile shapes per regime

- **Large M (prefill):** 64x128 with 4 warps, 2 stages, two CTAs per SM
  (down at M = 512-1024: 128x128 with 8 warps, one CTA per SM). With the
  activation scales in shared memory the 8-warp 128x128 tile, which halves
  the L2 traffic per FLOP, is within 0.6-4% of the two-CTA tiles at M = 8192;
  with the scales as global loads it was 15% slower (docs/design.md,
  Configurations). L2 bandwidth (4.4 TB/s) was not the limit.
- **Small and middle M:** 16-64 rows by 64 columns, 3-4 stages, two CTAs per
  SM, split-K only on down. With M <= 64 the GEMM is a weight stream: FLOPs
  per weight byte are at most 2 * 64, far below the ~330 FLOP/byte the GPU
  needs to be compute-bound, so the tile only has to keep enough weight
  bytes in flight on every SM, which 64-96 CTAs of 16x64 already do.
- **Why 64 columns:** a weight block is 128 columns, so 64-column tiles still
  read one scale per K tile, and they double the CTAs per weight block row.
  With only 128-column tiles the middle range lost up to 21% to tuned Triton
  (which picks 64-column tiles there); now it is 1.00-1.19x.
- **Cost of the 16-row tile at M = 1:** 15 of 16 MMA rows are padding. The
  MMA time is still hidden behind the weight stream (862-933 GB/s against a
  957 GB/s read peak), so swapping the operands to put the weights in the
  MMA's 16-row slot was not needed.
- **Still missing:** cuBLASLt (per-tensor scales, no promotion) is up to 12%
  faster in the middle range and 4-9% at prefill; stream-K or a persistent
  schedule was not tried.

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
  chosen for each shape. Since the activation scales moved into shared
  memory, the tuned table uses split-K only on down (K = 12288) at M <= 128;
  the decode plans of the other layers stream the weights fastest with one
  CTA per tile. Stream-K was not built: between tabulated batch sizes a
  plan with a balanced CTA count per SM recovered the losses measured
  there (docs/design.md, plan table).

## 7. Frozen per-shape plan table

- **Chosen:** `bench/autotune.py` times every configuration and split count on
  the Qwen3-8B shapes at every power of two M = 1-8192, with the benchmark's
  own harness; the winners are compiled into `csrc/plan_table.inc`. Between
  two entries, `choose_plan` takes the entry above unless another
  tabulated plan of the shape (M / 4 to 4 M) leaves less work on the busiest
  SM (M >= 256 only). Other shapes use a simple M-based rule.
- **Alternatives:** runtime autotuning on first call (start-up cost, timing
  noise inside a serving process); a closed-form heuristic only; a finer
  grid (every batch size vLLM captures in CUDA graphs).
- **Why the harness matters:** the first table was timed with CUDA events on
  cold activations at eight batch sizes; against the benchmark it picked
  plans up to 21% slower than the best one compiled
  (results/gemm/plan_probe_426750a.json).
- **Cost:** the table is specific to this GPU and these shapes, and a sweep
  takes 11 minutes of GPU time. A second table
  (`csrc/fused_plan_table.inc`) records the fused-quantization winners and
  whether each beat the separate quantization kernel + GEMM; none does now.

## 8. Accepting any activation-scale strides

- **Chosen:** the kernel takes the strides of `a_scales`, so vLLM's row-major
  scales and the M-major ones both work without a copy.
- **Cost:** none measurable. Each tile row's scale is one 4-byte `cp.async`
  per K tile into shared memory, whatever the strides; as `__ldg` loads one
  K tile ahead (until commit bed9867) the same scales stalled the weight
  stream by up to 18% at M = 16 (docs/design.md, where the losses came
  from).

## 9. Activation quantization fused into the GEMM: built, not used

- **Chosen:** `gemm_bf16` reads bf16 activations; each thread prefetches its
  16-byte chunks of the next K tile into registers one K tile ahead and
  quantizes them into the e4m3 shared stage at the start of the previous K
  tile, so the K tile's existing barrier publishes them. `linear` fuses only
  where the fused table measured it faster, which is now nowhere.
- **Alternatives:** (a) a separate quantization kernel before every GEMM
  (vLLM's way, and now `linear`'s, with blockfp8's own kernel); (b) staging
  the bf16 tile in a second cp.async ring and quantizing it in shared memory
  after it lands; (c) quantizing in the previous layer's epilogue (RMSNorm,
  attention, SiLU-and-mul), which needs control over those kernels.
- **Why it was built:** (a) costs 0.5-2.2 us per linear at decode with
  blockfp8's kernel (1.4-3.2 us with vLLM's; results/gemm/summary.md), up
  to 11% of a decode GEMM; (b) needs a second ring, which costs pipeline
  stages in the same shared memory, and a second barrier per K tile; (c) is
  outside this kernel.
- **Why it is not used:** every CTA of a tile row quantizes the same
  activation rows again. Once the plain GEMM stopped stalling on its
  activation-scale loads (commit bed9867), the fused kernel lost on every
  tuned shape: by 0.2-1.1 us at M = 1-16 on qkv, o and down, 1.6-5.5 us on
  gate_up, and by up to 358 us at M = 256 (results/gemm/autotune_fused.json).
  Before that it had won by 0.1-1.1 us on 6 of 12 decode shapes.
- **Kept as a measured negative result.** The kernel and
  `csrc/fused_plan_table.inc` ship although `linear` never runs them: the
  committed GEMM results have a `blockfp8_fused` column and
  `results/gemm/autotune_fused.json` is the evidence for the loss above, and
  both stay reproducible from the code. Every entry of the table says the
  separate kernel wins, and untuned shapes do not fuse, so `prefers_fused`
  is false everywhere. To run it:
  - `blockfp8.gemm_bf16(x, b_q, b_scales)` calls it directly (bf16
    activations; `config` and `split_k` index `blockfp8.fused_configs()`);
  - `flock /var/tmp/rtx4090.lock .venv/bin/python bench/autotune.py --fused`
    re-times the five fused configurations against `quantize_cuda` + `gemm`
    on the four layer shapes at the tuned M <= 256 and rewrites the JSON and
    the table; after
    `cmake --build build`, `linear` fuses wherever the new table says the
    fused kernel won;
  - `bench/bench_gemm.py` times it as `blockfp8_fused` (M <= 256);
  - `pytest tests/test_gemm_gpu.py -k fused` checks it bitwise against
    quantize-then-GEMM (also with split-K, inside a CUDA graph, and against
    vLLM's quantizer).

## 10. A quantization kernel of our own

- **Chosen:** `quantize_cuda` (16 threads per 128-wide group, 16-byte loads,
  shuffle reduction), used by `linear` where it does not fuse.
- **Alternative:** vLLM's `per_token_group_quant_fp8`.
- **Why:** blockfp8 then runs without vLLM; the result is bitwise the same
  (tests), and in the benchmark `linear` with it is 0.1-7.5 us faster than
  the GEMM after vLLM's kernel on every unfused shape but gate_up at
  M = 8192 (11.8 us, 0.2%, slower).
- **Cost:** one more kernel to keep bitwise equal to vLLM's (the reference
  had drifted: PyTorch's `t / 448.0` multiplies by the reciprocal; fixed in
  commit caf0bd8).

## 11. Integrating with vLLM by joining its kernel list

- **Chosen:** a `vllm.general_plugins` entry point inserts
  `BlockFp8LinearKernel` into vLLM's own list of block-FP8 linear kernels,
  right before `MarlinFP8ScaledMMLinearKernel`; its `is_supported` and
  `can_implement` claim only sm_89, bf16, 128x128 weight blocks and dynamic
  1x128 activation groups.
- **Alternatives:** patching `Fp8LinearMethod` or the model's layers;
  registering a new quantization method; editing vLLM.
- **Why:** vLLM's selector, weight loading and `process_weights_after_loading`
  stay in charge, so every other GPU and layer type keeps vLLM's choice and
  the checkpoint weights are used as loaded. Turning it off is
  `VLLM_PLUGINS` without `blockfp8`.
- **Cost:** it depends on a private list (`_POSSIBLE_FP8_BLOCK_KERNELS`) and
  the `Fp8BlockScaledMMLinearKernel` interface of vLLM 0.31, which can change
  between releases; `tests/test_vllm_plugin.py` checks both.

## 12. Benchmarking with L2-resident activations

- **Chosen:** in cold-L2 runs, the 256 MB flush is followed by a read of the
  call's activations, so weights come from DRAM and activations from L2.
- **Alternative:** evicting everything (the first benchmark).
- **Why:** in a model the activations were just written by the previous
  kernel. Evicting them made the GEMM kernel for o at M = 64 4.6 us slower
  than right after the quantizer with the 16x128 tile used then, and 2.7 us
  with the 16x64 tile used now (results/gemm/l2_probe.json), which made the
  separate quantization look nearly free there.
- **Cost:** for large M the activations do not fit in L2 anyway; the read
  then only warms part of them, which is also what a real layer sees.

