"""Tune vLLM's Triton block-FP8 kernel for this GPU, so the baseline is not
only the untuned default vLLM falls back to on an RTX 4090.

The search space follows vLLM's own tuning script
(benchmarks/kernels/benchmark_w8a8_block_fp8.py), trimmed where an option
cannot matter (BLOCK_K is the 128-wide scale block; tiles taller than 2*M only
add masked rows). Timing: CUDA events, warm L2, median of 3 x 10 calls.
Output: results/gemm/triton_tuned_configs.json, one entry per layer in the
file format vLLM reads from its configs directory ({M: config}).

    flock /var/tmp/rtx4090.lock .venv/bin/python bench/tune_triton.py
"""

import itertools
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_state  # noqa: E402

import blockfp8  # noqa: E402
from bench_gemm import LAYERS, M_VALUES  # noqa: E402


def candidates(m):
    bms = [bm for bm in (16, 32, 64, 128, 256) if bm <= max(16, 2 * m)]
    groups = (8, 32) if m >= 1024 else (32,)
    for bm, bn, warps, stages, gm in itertools.product(bms, (64, 128, 256), (4, 8), (2, 3, 4), groups):
        if bm * bn >= 128 * 256 and warps == 4:
            continue  # too few warps for the largest tiles
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": gm,
               "num_warps": warps, "num_stages": stages}


def time_us(fn, calls=10, reps=3):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(reps):
        s.record()
        for _ in range(calls):
            fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1e3 / calls)
    return statistics.median(out)


def main():
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    import triton
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    kernel = fp8_utils._w8a8_triton_block_scaled_mm
    out = {"header": gpu_state.header(benchmark="tune_triton", triton=triton.__version__),
           "gpu_before": gpu_state.nvidia_smi(), "layers": {}}
    t0 = time.time()
    for name, (n, k) in LAYERS.items():
        g = torch.Generator(device="cuda").manual_seed(0)
        b, sb = blockfp8.quantize_weights(torch.randn(n, k, generator=g, device="cuda") * 0.02)
        best = {}
        for m in M_VALUES:
            a, sa = blockfp8.quantize_activations(torch.randn(m, k, generator=g, device="cuda"),
                                                  column_major_scales=False)
            c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

            def run(cfg):
                grid = (triton.cdiv(m, cfg["BLOCK_SIZE_M"]) * triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)
                kernel[grid](a, b, c, sa, sb, m, n, k, 128, 128, a.stride(-2), a.stride(-1),
                             b.stride(1), b.stride(0), c.stride(-2), c.stride(-1), sa.stride(-2),
                             sa.stride(-1), sb.stride(1), sb.stride(0), **cfg)

            timed = []
            for cfg in candidates(m):
                try:
                    run(cfg)
                    timed.append((time_us(lambda: run(cfg)), cfg))
                except Exception as ex:  # out of shared memory / registers
                    timed.append((float("inf"), {**cfg, "error": type(ex).__name__}))
            timed.sort(key=lambda t: t[0])
            best[str(m)] = timed[0][1]
            print(f"{name} M={m}: best {timed[0][0]:.1f} us {timed[0][1]} ({len(timed)} tried)",
                  flush=True)
        out["layers"][name] = {"N": n, "K": k, "configs": best}
    out["elapsed_s"] = time.time() - t0
    out["gpu_after"] = gpu_state.nvidia_smi()
    path = os.path.join(gpu_state.REPO, "results", "gemm", "triton_tuned_configs.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.relpath(path, gpu_state.REPO)}")


if __name__ == "__main__":
    main()
