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
autotuning sweep. For decode batches the per-token activation quantization
runs inside the GEMM (`gemm_bf16`), elsewhere in a separate kernel of its
own; `blockfp8.linear` picks per shape. A `vllm.general_plugins` entry point
(`blockfp8_vllm`) puts the kernel ahead of Marlin in vLLM's own kernel list.
Details: [docs/design.md](docs/design.md),
[docs/design-tradeoffs.md](docs/design-tradeoffs.md).

**GEMM results** (Qwen3-8B layer shapes; weights cold in L2, activations
L2-resident; median of 10 interleaved runs, kernel spans from CUPTI; full
tables in [results/gemm/summary.md](results/gemm/summary.md)):

| | M = 1-16 (decode) | M = 64-256 | M = 1024-8192 (prefill) |
|---|---|---|---|
| this kernel | 809-934 GB/s of weights (DRAM read peak 957) | 101-299 TFLOPS | 306-320 TFLOPS (MMA peak 367 at 2.8 GHz) |
| vs vLLM's default here, FP8 Marlin W8A16 (ours: `linear`, quantization included) | 0.97-1.06x: **loses 3% on down at M = 16** | 0.95-2.12x: **loses 5% on down at M = 64** | **1.67-2.02x faster** |
| vs vLLM Triton block-FP8, shipped default config | 1.08-2.57x faster | 1.03-2.12x faster | 1.09-1.21x faster |
| vs the same Triton kernel tuned for this GPU | 0.96-1.08x: loses up to 4% on qkv and o | **0.94-1.04x: loses up to 6% at M = 64** | 1.01-1.09x faster |
| vs cuBLASLt FP8, one scale per tensor (coarser scaling) | 0.95-1.05x: loses up to 5% | **0.83-1.02x: loses up to 17%** | 0.91-0.94x, **loses up to 9%** (except down at M = 8192: 1.29x) |

Ratios are baseline time / our time; GEMM only except the Marlin row. Before
the 64-column tiles and the fused quantization (commit d5518d7), the middle
range was 0.79-1.01x of tuned Triton and decode 0.90-1.04x of Marlin.

![speedup](docs/img/speedup.svg)

**End to end** (Qwen3-8B-FP8 in vLLM 0.31 on the RTX 4090, stock = Marlin,
plugin = blockfp8; median of 5 interleaved runs, one process each;
[results/e2e/summary.md](results/e2e/summary.md)):

| | stock | plugin | speedup |
|---|---|---|---|
| prefill, prompts of 256 / 1024 / 4096 tokens (tok/s) | 10228 / 10050 / 9460 | 17519 / 17038 / 15461 | 1.71x / 1.70x / 1.63x |
| decode TPOT, batch 1 / 4 / 16 / 64 (ms) | 10.12 / 10.24 / 10.84 / 14.08 | 9.95 / 10.12 / 10.72 / 12.91 | 1.017x / 1.012x / 1.011x / 1.090x |

**Outputs are not identical to stock**, and are not expected to be: the
plugin quantizes activations to FP8 (W8A8), Marlin does not (W8A16). Over 24
chat prompts with greedy decoding (up to 64 tokens), 7-9 of 24 sequences
match the stock run exactly and 57-60% of tokens come before the first
divergence; teacher-forced on the stock tokens, logprobs differ by a median
of 1.1e-4 and at most 0.44. vLLM's own W8A8 path (Triton, Marlin disabled)
is no closer to stock (9 of 24 identical, max logprob difference 0.50), and
the plugin does not match it either (6 of 24, max 0.51). Repeated stock runs
are not fully deterministic (one of four differed on one prompt).

Accuracy equals vLLM's Triton kernel (same instruction, same promotion
interval): relative Frobenius error 1.669e-3 for both on the down
projection at M = 8192 (first 256 rows), dominated by bf16 output rounding.
The FP8 MMA keeps 13 bits below the largest addend and truncates the rest
([results/micro](results/micro)); promoting every 128-wide block keeps the
accumulation error about 28x below bf16 rounding. The three activation
quantizers (PyTorch reference, CUDA kernel, fused) are bitwise equal to
vLLM's `per_token_group_quant_fp8`, and the fused GEMM is bitwise equal to
quantize-then-GEMM (tests).

**Limitations.**
- Middle batches (M = 64) still lose 1-6% to tuned Triton, and cuBLASLt with
  per-tensor scales is up to 17% faster at M = 64-256; stream-K was not tried.
- Fusing the quantization into the GEMM only partly removes its cost: the
  fused GEMM is 1.0-5.6 us slower than the plain one, so it wins by at most
  about 1 us and only on 6 of the 12 decode shapes; the rest use a separate
  quantization kernel. Down at M = 16 and 64 stays 3-5% behind Marlin.
- End to end, decode at batch 1-16 gains only 1-2% (the layers are weight
  streams for both kernels), and greedy outputs differ from stock vLLM on
  most prompts, as any W8A8 path does; no task-accuracy evaluation was run.
- One GPU (RTX 4090, clocks not lockable, median SM clock 2.78 GHz during
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
`bench/autotune.py [--fused]`, `bench/tune_triton.py`, `bench/l2_probe.py`);
`bench/report.py` regenerates the summary and the chart.
`bench/e2e_vllm.py --runs 5` takes the lock for each run itself and
`bench/report_e2e.py` summarizes it.

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
