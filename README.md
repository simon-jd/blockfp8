# blockfp8

A block-scaled FP8 GEMM for NVIDIA Ada GPUs (sm_89, measured on an RTX 4090),
written with CuTe from CUTLASS v4.8.1, for FP8 LLM checkpoints such as
Qwen3-8B-FP8, with a vLLM 0.31 plugin that uses it.

**Problem.** DeepSeek-V3-style FP8 checkpoints store weights as e4m3 with one
scale per 128x128 block and quantize activations per token in 1x128 groups.
vLLM 0.31 has no FP8 tensor-core kernel for this format on Ada: its CUTLASS
block-FP8 kernel is Hopper/Blackwell only
(`cutlass_scaled_mm_supports_block_fp8(89)` is False), and what it actually
selects on this GPU is `MarlinFP8ScaledMMLinearKernel`, a weight-only kernel
that dequantizes the FP8 weights to bf16 and runs bf16 MMAs (W8A16). That was
established by asking vLLM's own selector (`init_fp8_linear_kernel`, called in
`bench/vllm_baselines.py`; the answer is recorded in
`results/gemm/qwen3_8b.json`). Its Triton W8A8 block kernel is next in line
and only runs if Marlin is disabled.

**Approach.** `mma.sync.m16n8k32` e4m3 with FP32 accumulation; cp.async
multi-stage pipeline into swizzled shared memory; ldmatrix; two-level
accumulation (MMA partials per 128-wide K block, promoted with that block's
scales on the CUDA cores); 64- and 128-column tiles, two CTAs per SM, and
deterministic split-K for small batches; a per-shape plan table from an
autotuning sweep. The per-token activation quantization runs in a kernel
of its own before the GEMM (`blockfp8.linear`); a fused variant that
quantizes inside the GEMM (`gemm_bf16`) exists but measured slower on every
shape. A `vllm.general_plugins` entry point
(`blockfp8_vllm`) puts the kernel ahead of Marlin in vLLM's own kernel list.
Details: [docs/design.md](docs/design.md),
[docs/design-tradeoffs.md](docs/design-tradeoffs.md).

**GEMM results** (Qwen3-8B layer shapes; weights cold in L2, activations
L2-resident; median of 10 interleaved runs, kernel spans from CUPTI; full
tables in [results/gemm/summary.md](results/gemm/summary.md)):

| | M = 1-32 (decode) | M = 64-512 | M = 1024-8192 (prefill) |
|---|---|---|---|
| this kernel | 862-933 GB/s of weights (DRAM read peak 957) | 106-316 TFLOPS | 305-330 TFLOPS (MMA peak 367 at 2.8 GHz) |
| vs vLLM's default here, FP8 Marlin W8A16 (ours: `linear`, quantization included) | 1.00-1.10x: **ties on down at M = 1-16** | 1.03-2.16x | **1.72-2.08x faster** |
| vs vLLM Triton block-FP8, shipped default config | 1.06-2.61x faster | 1.04-2.22x faster | 1.10-1.21x faster |
| vs the same Triton kernel tuned for this GPU | 1.00-1.06x | 1.00-1.19x | 1.03-1.09x faster |
| vs cuBLASLt FP8, one scale per tensor (coarser scaling) | 1.00-1.07x | **0.88-1.04x: loses up to 12%** | 0.91-0.96x, **loses up to 9%** (except down at M = 8192: 1.23x) |

Ratios are baseline time / our time; GEMM only except the Marlin row. The
plan table and the tuned Triton configs are both tuned on every power of
two M = 1-8192 in this harness. Against the previous recording (commit
3660f45), two changes closed the losses to tuned Triton (up to 7% at
M = 1-16 and 2-5% at M = 64-256) and to Marlin on down: the activation
scales are now copied into shared memory with each K tile instead of
being loaded from global memory in the weight stream (o at M = 16: 21.1 ->
19.5 us), and the plan table is tuned with the benchmark's own harness on a
finer grid (before, the automatic plan was up to 21% slower than the best
compiled one; [docs/design.md](docs/design.md), "Where the losses came
from"). With the weights L2-resident (warm), tuned Triton is still 12% faster
on qkv at M = 32, and `linear` is 0.81-0.99x of Marlin on o at M = 1-16.

**Batch sizes off the tuning grid** (M = 3, 12, 24, 48, 100, 384, 768,
3000, between the powers of two the tables are tuned on; cold L2,
[results/gemm/summary.md](results/gemm/summary.md)): 1.01-2.23x of Marlin,
1.07-2.34x of default Triton, 0.99-1.41x of tuned Triton (with the config
of the nearest tuned M, as vLLM picks it), **losing 1%** on o at M = 48 and
100 and down at M = 100, and 0.89-1.09x of cuBLASLt. Between two entries
the kernel takes the tabulated plan of the shape that leaves the least work
on the busiest SM: the entry above put 1.5 CTAs per SM on down at M = 768
(319.7 us); the balanced choice runs 3 per SM (248.3 us).

![speedup](docs/img/speedup.svg)

**End to end** (Qwen3-8B-FP8 in vLLM 0.31 on the RTX 4090, stock = Marlin,
plugin = blockfp8; median of 5 interleaved runs, one process each;
[results/e2e/summary.md](results/e2e/summary.md)):

| | stock | plugin | speedup |
|---|---|---|---|
| prefill, prompts of 256 / 1024 / 4096 tokens (tok/s) | 10315 / 10160 / 9551 | 18042 / 17576 / 15801 | 1.75x / 1.73x / 1.65x |
| decode TPOT, batch 1 / 4 / 16 / 64 (ms) | 10.10 / 10.23 / 10.84 / 14.08 | 9.86 / 10.01 / 10.51 / 12.65 | 1.024x / 1.023x / 1.032x / 1.113x |

Recorded at commit 6a2a470; before the scale and plan-table changes
(commit ec74438) the speedups were 1.63-1.71x for prefill and
1.011-1.017x / 1.090x for decode.

**Model quality.** wikitext-2-raw-v1 test perplexity (298,931 tokens in 147
windows of 2048, scored with prompt_logprobs; `bench/e2e_vllm.py --ppl`,
[results/e2e/wikitext2_ppl.json](results/e2e/wikitext2_ppl.json)): stock
(Marlin, W8A16) **9.7646**, plugin **9.7735** (+0.09%), vLLM's own W8A8
path (Triton, Marlin disabled) **9.7739** (+0.10%). Stock and Triton were
measured at commit 0a65e1d (recorded in 5fab662; a second stock run gave the
same logprobs bit for bit) and not repeated, since neither uses this kernel.
The plugin was measured at 0a65e1d, before the kernel and plan-table changes
(9.77351), and again at 2af08cb, which has the kernel and plan tables of this
README (9.77347, mean NLL 4e-6 lower); the figures here are from the second.
Per token, the plugin's logprobs differ from stock by a mean of 0.062 (p99
0.48, max 7.1; the same at both commits), Triton's by 0.062 (p99 0.48, max
9.6), and the plugin's from Triton's by 0.079 (p99 0.62, max 8.4): W8A8 costs
both kernels the same perplexity, and the two differ from each other by more
than either does from stock. The two plugin runs agree exactly on 298,862 of
298,931 token logprobs; the other 69 differ by at most 0.79.

**Outputs are not identical to stock**, and are not expected to be: the
plugin quantizes activations to FP8 (W8A8), Marlin does not (W8A16). Over 24
chat prompts with greedy decoding (up to 64 tokens), 9 of 24 sequences
match the stock run exactly and 60% of tokens come before the first
divergence; teacher-forced on the stock tokens, logprobs differ by a mean of
0.018-0.019 (p99 0.22-0.24, max 0.35-0.44, 1,449 tokens). vLLM's own W8A8
path is no closer to stock (9 of 24 identical, mean 0.019, p99 0.21, max
0.50), and the plugin does not match it either (7 of 24, mean 0.023, p99
0.27-0.28, max 0.51). Repeated stock runs are not fully deterministic (two
of four differed from the first, on 1 and 5 prompts).

Accuracy equals vLLM's Triton kernel (same instruction, same promotion
interval): relative Frobenius error 1.669e-3 for both on the down
projection at M = 8192 (first 256 rows), dominated by bf16 output rounding.
The FP8 MMA keeps 13 bits below the largest addend and truncates the rest
([results/micro](results/micro)). On random operands at K = 4096,
promoting every 128-wide block gives a median relative error of 1.37e-4,
10x below the 1.37e-3 median error of rounding the same exact results to
bf16; the bf16 output of the promoted sum errs by a median of 1.41e-3. The
three activation
quantizers (PyTorch reference, CUDA kernel, fused) are bitwise equal to
vLLM's `per_token_group_quant_fp8`, and the fused GEMM is bitwise equal to
quantize-then-GEMM (tests).

**Limitations.**
- cuBLASLt with per-tensor scales (no per-block promotion) is up to 12%
  faster at M = 64-512 (o at M = 256), 4-9% faster at prefill and up to 11%
  off the grid (o at M = 768); stream-K or a persistent schedule was not
  tried.
- With the weights L2-resident (warm), tuned Triton is 12% faster on qkv at
  M = 32, and `linear` is 0.81-0.99x of Marlin on o at M = 1-16: the
  separate quantization kernel (about 2 us) is a quarter of an 8 us GEMM.
  A decode step of an 8 GB model reads its weights from DRAM (cold), where
  neither loses.
- Off the tuning grid, tuned Triton is 1% faster on o at M = 48 and 100 and
  down at M = 100. The balanced plan choice between entries was written
  after the M = 384 and 768 cells were probed; on six other batch sizes it
  sped up three cells by 13-23% and slowed gate_up at M = 320 by 2%.
- The fused quantization kernel is built and tested but no longer used: it
  loses to quantize + GEMM on every tuned shape (0.2-5.5 us at M = 1-16),
  so `linear` always runs the separate kernel. Down at M = 1-16 ties with
  Marlin (0.998-1.000x).
- End to end, decode at batch 1-16 gains only 2-3% (the layers are weight
  streams for both kernels), and greedy outputs differ from stock vLLM on
  most prompts, as any W8A8 path does. Quality was checked by wikitext-2
  perplexity only (+0.09%, measured at 2af08cb); no downstream
  task was evaluated.
- One GPU (RTX 4090, clocks not lockable, median SM clock 2.79 GHz during
  the GEMM runs), shapes of one model; the plan tables are specific to both.
- N must be a multiple of 8 and K of 16; split-K counters assume calls on
  one CUDA stream per device at a time. The plugin relies on a private vLLM
  0.31 kernel list.

## Build and test

```
git submodule update --init --depth 1
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python vllm==0.31.0 pytest matplotlib
cmake -S . -B build -G Ninja -DPython3_EXECUTABLE=$PWD/.venv/bin/python && cmake --build build
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .   # registers the vLLM plugin
.venv/bin/python -m pytest tests          # GPU tests skip without a GPU or the build
```

With the package installed, every vLLM process loads the plugin; for stock
vLLM leave it out of `VLLM_PLUGINS` (for example
`VLLM_PLUGINS=lora_filesystem_resolver`). vLLM logs
`Selected BlockFp8LinearKernel for Fp8LinearMethod` when it is in use.

Benchmarks hold the machine-wide GPU lock: `flock /var/tmp/rtx4090.lock
.venv/bin/python bench/bench_gemm.py` (also `bench/micro/run_micro.py`,
`bench/autotune.py [--fused]`, `bench/tune_triton.py`, `bench/plan_probe.py`,
`bench/l2_probe.py`);
`bench/report.py` regenerates the summary and the chart.
`bench/e2e_vllm.py --runs 5` (and `--ppl`) takes the lock for each run
itself and `bench/report_e2e.py` summarizes it. `--ppl` reads
`data/wikitext-2-raw-v1/test-00000-of-00001.parquet` (download URL and
checksum in `bench/wikitext.py`).

## Layout

- `blockfp8/`: Python API (`quantize_activations`, `quantize_cuda`, `gemm`,
  `gemm_bf16`, `linear`), layouts, FP64 reference and error bound
- `blockfp8_vllm/`: the vLLM plugin (`register`, `BlockFp8LinearKernel`)
- `csrc/`: the CuTe kernel, the quantization kernel, configurations and plan
  tables, PyTorch ops
- `bench/micro/`: MMA peak, memory bandwidth, MMA precision
- `bench/`: GEMM benchmark, autotuning, Triton tuning, vLLM baselines, L2
  probe, end-to-end vLLM benchmark
- `results/`: raw results with GPU state and commit
