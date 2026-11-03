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
| this kernel | 805-935 GB/s of weights (DRAM read peak 957) | 102-298 TFLOPS | 307-321 TFLOPS (MMA peak 367 at 2.8 GHz) |
| vs vLLM's default here, FP8 Marlin W8A16 (ours: `linear`, quantization included) | 0.99-1.06x: **loses 1% on down at M = 16** | 1.00-2.11x | **1.67-2.03x faster** |
| vs vLLM Triton block-FP8, shipped default config | 1.07-2.58x faster | 1.02-2.14x faster | 1.08-1.21x faster |
| vs the same Triton kernel tuned for this GPU | **0.93-1.07x: loses up to 7% on o and qkv** | **0.95-1.04x: loses up to 5% on qkv** | 1.01-1.09x faster |
| vs cuBLASLt FP8, one scale per tensor (coarser scaling) | 0.95-1.07x: loses up to 5% | **0.83-1.01x: loses up to 17%** | 0.91-0.93x, **loses up to 9%** (except down at M = 8192: 1.27x) |

Ratios are baseline time / our time; GEMM only except the Marlin row. The
tuned Triton config of each shape is the fastest of its five best warm-L2
candidates re-timed in this cold-L2 harness (that changed 11 of 32 picks;
the warm pick would have been up to 12% slower, on qkv at M = 64). Against
the previous recording (commit b3f50b9), cells of unchanged kernels moved by
up to 4.5% (cuBLASLt on qkv at M = 64; down at M = 64: blockfp8 63.6 -> 61.4
us, Marlin 63.0 -> 64.4 us, which turned a 5% loss to Marlin into a tie), so
margins of a few percent are within session-to-session variation. Before
the 64-column tiles and the fused quantization (commit d5518d7), the middle
range was 0.79-1.01x of tuned Triton and decode 0.90-1.04x of Marlin.

**Batch sizes off the tuning grid** (M = 2, 8, 32, 100, 512, 3000, none of
which the plan tables or the tuned Triton configs have; cold L2,
[results/gemm/summary.md](results/gemm/summary.md)): 0.93-2.10x of Marlin,
**losing 2-7% at M = 32** (down 7%, o 4%, qkv 2%); 0.93-2.36x of default
Triton, **losing 7% on qkv at M = 512**; 0.86-1.18x of tuned Triton (with the
config of the nearest tuned M, as vLLM picks it), **losing up to 14%** (o at
M = 32) and 12% (qkv at M = 512); 0.71-1.04x of cuBLASLt. qkv at M = 512 runs
the M = 1024 plan entry and takes 110.9 us, twice the 54.7 us of M = 256.

![speedup](docs/img/speedup.svg)

**End to end** (Qwen3-8B-FP8 in vLLM 0.31 on the RTX 4090, stock = Marlin,
plugin = blockfp8; median of 5 interleaved runs, one process each;
[results/e2e/summary.md](results/e2e/summary.md)):

| | stock | plugin | speedup |
|---|---|---|---|
| prefill, prompts of 256 / 1024 / 4096 tokens (tok/s) | 10228 / 10050 / 9460 | 17519 / 17038 / 15461 | 1.71x / 1.70x / 1.63x |
| decode TPOT, batch 1 / 4 / 16 / 64 (ms) | 10.12 / 10.24 / 10.84 / 14.08 | 9.95 / 10.12 / 10.72 / 12.91 | 1.017x / 1.012x / 1.011x / 1.090x |

**Model quality.** wikitext-2-raw-v1 test perplexity (298,931 tokens in 147
windows of 2048, scored with prompt_logprobs; `bench/e2e_vllm.py --ppl`,
[results/e2e/wikitext2_ppl.json](results/e2e/wikitext2_ppl.json)): stock
(Marlin, W8A16) **9.7646**, plugin **9.7735** (+0.09%), vLLM's own W8A8
path (Triton, Marlin disabled) **9.7739** (+0.10%); a second stock run gave
the same logprobs bit for bit. Per token, the plugin's logprobs differ from
stock by a mean of 0.062 (p99 0.48, max 7.1), Triton's by 0.062 (p99 0.48,
max 9.6), and the plugin's from Triton's by 0.079 (p99 0.62): W8A8 costs both
kernels the same perplexity, and the two differ from each other by more than
either does from stock.

**Outputs are not identical to stock**, and are not expected to be: the
plugin quantizes activations to FP8 (W8A8), Marlin does not (W8A16). Over 24
chat prompts with greedy decoding (up to 64 tokens), 7-9 of 24 sequences
match the stock run exactly and 57-60% of tokens come before the first
divergence; teacher-forced on the stock tokens, logprobs differ by a mean of
0.019 (p99 0.23, max 0.44, 1,449 tokens). vLLM's own W8A8 path is no closer
to stock (9 of 24 identical, mean 0.019, p99 0.21, max 0.50), and the plugin
does not match it either (6 of 24, mean 0.023, p99 0.28, max 0.51). Repeated
stock runs are not fully deterministic (one of four differed on one prompt).

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
- Tuned Triton is up to 7% faster at M = 1-16 (o, qkv), 2-4% faster at
  M = 64 and 5% on qkv at M = 256; cuBLASLt with per-tensor scales is up to 17%
  faster at M = 64-256; stream-K was not tried.
- The plan tables only know the tuning grid. Off it, M = 32 loses 2-7% to
  Marlin and up to 14% to tuned Triton, and qkv at M = 512 is 7% slower
  than even the default Triton config.
- Fusing the quantization into the GEMM only partly removes its cost: the
  fused GEMM is 1.0-5.6 us slower than the plain one, so it wins by at most
  about 1 us (at the timer's resolution) and only on 6 of the 12 decode
  shapes; the rest use a separate quantization kernel. Down at M = 16 stays
  1% behind Marlin.
- End to end, decode at batch 1-16 gains only 1-2% (the layers are weight
  streams for both kernels), and greedy outputs differ from stock vLLM on
  most prompts, as any W8A8 path does. Quality was checked by wikitext-2
  perplexity only (+0.09%); no downstream task was evaluated.
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
