"""Sweep kernel configuration x split count per shape and freeze the winners.

For every Qwen3-8B layer shape and every M of the tuning grid (TUNE_M, the
powers of two 1-8192, so that a batch size between two entries is less than
2x below the entry it uses), every compiled configuration is timed with every
split count that leaves each split at least one K tile. Timing: the benchmark's own harness
(bench_gemm.time_calls: CUPTI kernel spans, a 256 MB read evicting the
weights and the call's activations read back into L2, interleaved runs,
median of the per-run medians), so a plan is chosen by the measure it is
reported with. Writes results/gemm/autotune.json (all timings) and
csrc/plan_table.inc, the table choose_plan() consults (rebuild the extension
afterwards).

With --fused, sweeps the fused-quantization configurations instead (bf16
activations, M <= 256) and times the alternative they replace: the separate
quantization kernel (blockfp8.quantize_cuda) followed by the GEMM with the
frozen plan; vLLM's per_token_group_quant_fp8 + GEMM is recorded too. Writes
results/gemm/autotune_fused.json and csrc/fused_plan_table.inc, which
choose_fused_plan() and prefers_fused() consult.

    flock /var/tmp/rtx4090.lock .venv/bin/python bench/autotune.py [--fused]
"""

import argparse
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
from bench_gemm import LAYERS, time_calls  # noqa: E402

TUNE_M = [1 << i for i in range(14)]  # 1, 2, 4, ..., 8192 (bench/tune_triton.py too)
SPLITS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
FUSED_M = [m for m in TUNE_M if m <= 256]
RUNS = 5


def calls_per_run(m):
    return 10 if m <= 256 else 5 if m <= 1024 else 3


def candidates(names, m, k_tiles):
    """(config, name, splits) worth timing at batch size m."""
    for cfg, cname in enumerate(names):
        bm = int(cname.split("n")[0][1:])
        if bm > 2 * max(m, 16) and bm > 16:
            continue  # a tile mostly of padding rows never wins
        if m >= 1024 and bm < 64:
            continue  # 16- and 32-row tiles re-read the weights M / BM times
        for sp in SPLITS:
            if sp > k_tiles or (m >= 512 and sp > 4):
                continue
            if -(-k_tiles // -(-k_tiles // sp)) != sp:
                continue  # the launcher would run fewer splits (no empty ones): a duplicate
            yield cfg, cname, sp


def sweep(names, m, k_tiles, make_fn, acts, extra=None):
    """Times every candidate plan (and the callables in extra) in one interleaved
    measurement; returns (plan rows fastest first, {extra name: us})."""
    fns, plans = {}, {}
    for cfg, cname, sp in candidates(names, m, k_tiles):
        label = f"{cname}/{sp}"
        fns[label] = make_fn(cfg, sp)
        plans[label] = {"config": cfg, "name": cname, "splits": sp}
    fns.update(extra or {})
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    timed = time_calls(fns, {k: acts for k in fns}, RUNS, calls_per_run(m), modes=("cold",))
    us = {k: statistics.median(timed[(k, "cold")]) for k in fns}
    rows = sorted(({**plans[k], "us": us[k], "runs_us": timed[(k, "cold")]} for k in plans),
                  key=lambda r: r["us"])
    return rows, {k: us[k] for k in (extra or {})}


def main_fused():
    import vllm_baselines

    names = blockfp8.fused_configs()
    out = {"header": gpu_state.header(benchmark="autotune_fused", configs=names,
                                      unfused_configs=blockfp8.configs(), runs=RUNS,
                                      timing="bench_gemm.time_calls, cold L2"),
           "gpu_before": gpu_state.nvidia_smi(), "shapes": []}
    t0 = time.time()
    for name, (n, k) in LAYERS.items():
        g = torch.Generator(device="cuda").manual_seed(0)
        b, sb = blockfp8.quantize_weights(torch.randn(n, k, generator=g, device="cuda") * 0.02)
        for m in FUSED_M:
            x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
            extra = {
                "separate": lambda: blockfp8.gemm(*blockfp8.quantize_cuda(x), b, sb),
                "separate_vllm": lambda: blockfp8.gemm(
                    *vllm_baselines.vllm_group_quant(x, column_major=True), b, sb),
            }
            rows, ex = sweep(names, m, -(-k // 128),
                             lambda cfg, sp: lambda: blockfp8.gemm_bf16(x, b, sb, config=cfg,
                                                                        split_k=sp),
                             [x], extra)
            best = rows[0]
            wins = best["us"] < ex["separate"]
            out["shapes"].append({"layer": name, "N": n, "K": k, "M": m, "best": best,
                                  "separate_us": ex["separate"],
                                  "separate_vllm_quant_us": ex["separate_vllm"],
                                  "wins": wins, "all": rows})
            print(f"{name} M={m}: fused {best['name']} splits={best['splits']} {best['us']:.1f} us,"
                  f" quant + gemm {ex['separate']:.1f} us (vLLM quant {ex['separate_vllm']:.1f})"
                  f" -> {'fused' if wins else 'separate'}", flush=True)
    out["elapsed_s"] = time.time() - t0
    out["gpu_after"] = gpu_state.nvidia_smi()
    with open(os.path.join(gpu_state.REPO, "results", "gemm", "autotune_fused.json"), "w") as f:
        json.dump(out, f, indent=1)
    lines = [
        "// Generated by bench/autotune.py --fused from results/gemm/autotune_fused.json; do not edit.",
        f"// Fused configurations: {', '.join(names)}",
        "// {N, K, M, config, splits, wins}",
    ]
    for s in out["shapes"]:
        lines.append(f"{{{s['N']}, {s['K']}, {s['M']}, {s['best']['config']}, "
                     f"{s['best']['splits']}, {int(s['wins'])}}},")
    with open(os.path.join(gpu_state.REPO, "csrc", "fused_plan_table.inc"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/gemm/autotune_fused.json and csrc/fused_plan_table.inc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", action="store_true", help="sweep the fused-quantization kernels")
    args = ap.parse_args()
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    if args.fused:
        return main_fused()
    names = blockfp8.configs()
    out = {"header": gpu_state.header(benchmark="autotune", configs=names, runs=RUNS,
                                      timing="bench_gemm.time_calls, cold L2"),
           "gpu_before": gpu_state.nvidia_smi(), "shapes": []}
    t0 = time.time()
    for name, (n, k) in LAYERS.items():
        g = torch.Generator(device="cuda").manual_seed(0)
        b, sb = blockfp8.quantize_weights(torch.randn(n, k, generator=g, device="cuda") * 0.02)
        k_tiles = -(-k // 128)
        for m in TUNE_M:
            a, sa = blockfp8.quantize_activations(torch.randn(m, k, generator=g, device="cuda"))
            rows, _ = sweep(names, m, k_tiles,
                            lambda cfg, sp: lambda: blockfp8.gemm(a, sa, b, sb, config=cfg,
                                                                  split_k=sp),
                            [a, sa])
            best = rows[0]
            out["shapes"].append({"layer": name, "N": n, "K": k, "M": m, "best": best, "all": rows})
            print(f"{name} M={m}: {best['name']} splits={best['splits']} {best['us']:.1f} us",
                  flush=True)
        del b, sb
        torch.cuda.empty_cache()
    out["elapsed_s"] = time.time() - t0
    out["gpu_after"] = gpu_state.nvidia_smi()
    res = os.path.join(gpu_state.REPO, "results", "gemm", "autotune.json")
    os.makedirs(os.path.dirname(res), exist_ok=True)
    with open(res, "w") as f:
        json.dump(out, f, indent=1)
    lines = [
        "// Generated by bench/autotune.py from results/gemm/autotune.json; do not edit.",
        f"// Configurations: {', '.join(names)}",
        "// {N, K, M, config, splits}",
    ]
    for s in out["shapes"]:
        lines.append(f"{{{s['N']}, {s['K']}, {s['M']}, {s['best']['config']}, {s['best']['splits']}}},")
    with open(os.path.join(gpu_state.REPO, "csrc", "plan_table.inc"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/gemm/autotune.json and csrc/plan_table.inc")


if __name__ == "__main__":
    main()
