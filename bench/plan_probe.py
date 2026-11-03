"""Time every compiled configuration x split count at chosen shapes, with the
benchmark's harness, next to vLLM's Triton kernel with each config of
results/gemm/triton_tuned_configs.json for the same layer.

The plan table (bench/autotune.py) and the lookup decide which of these the
kernel runs; this probe shows what the best plan would have been and how the
CTA count of each plan fills the 128 SMs, to explain where blockfp8 loses.
Timing: bench_gemm.time_calls (CUPTI spans, weights cold and activations
L2-resident in cold mode; everything L2-resident in warm mode).

    flock /var/tmp/rtx4090.lock .venv/bin/python bench/plan_probe.py \
        --shapes qkv:32 o:16 --out results/gemm/plan_probe.json
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_state  # noqa: E402

import blockfp8  # noqa: E402
from bench_gemm import LAYERS, TUNED, TritonConfigRecorder, make_operands, time_calls  # noqa: E402

SPLITS = (1, 2, 3, 4, 6, 8, 12, 16)
SMEM_PER_SM = 100 * 1024  # sm_89; the driver reserves 1 KB per CTA


def ctas_per_sm(name):
    """Resident CTAs per SM as shared memory allows (the b2 configs are built for 2)."""
    bm, rest = name[1:].split("n", 1)
    bn = int(rest.split("s")[0])
    stages = int(rest.split("s")[1].split("w")[0])
    smem = (int(bm) + bn) * 128 * stages
    return max(1, SMEM_PER_SM // (smem + 1024)), int(bm), bn


def triton_fns(ops, m, triton_mod):
    import triton as tl_triton

    with open(TUNED) as f:
        layer = next(v for v in json.load(f)["layers"].values()
                     if v["N"] == ops["b_q"].shape[0] and v["K"] == ops["b_q"].shape[1])
    nearest = min(layer["configs"], key=lambda x: abs(int(x) - m))
    a_q, a_s = blockfp8.quantize_activations(ops["x"][:m], column_major_scales=False)
    b, b_s = ops["b_q"], ops["b_s"]
    n, k = b.shape
    c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    fns, info, seen = {}, {}, set()
    for tm, cfg in layer["configs"].items():
        cfg = {kk: v for kk, v in cfg.items() if kk != "error"}
        key = json.dumps(cfg, sort_keys=True)
        if key in seen and tm != nearest:
            continue
        seen.add(key)
        grid = (tl_triton.cdiv(m, cfg["BLOCK_SIZE_M"]) * tl_triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)

        def fn(cfg=cfg, grid=grid):
            triton_mod.kernel[grid](a_q, b, c, a_s, b_s, m, n, k, 128, 128, a_q.stride(-2),
                                    a_q.stride(-1), b.stride(1), b.stride(0), c.stride(-2),
                                    c.stride(-1), a_s.stride(-2), a_s.stride(-1), b_s.stride(1),
                                    b_s.stride(0), **cfg)
        label = f"triton@{tm}"
        fns[label] = fn
        info[label] = {"impl": "vllm_triton", "config": cfg, "tuned_for_m": int(tm),
                       "nearest": tm == nearest, "ctas": grid[0]}
    return fns, info, [a_q, a_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", required=True, help="layer:M ...")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--only", help="regex: time only the blockfp8 plans whose label matches")
    ap.add_argument("--out", default=os.path.join(gpu_state.REPO, "results", "gemm", "plan_probe.json"))
    args = ap.parse_args()
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    rec = TritonConfigRecorder()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    names = blockfp8.configs()
    out = {"header": gpu_state.header(benchmark="plan_probe", runs=args.runs, calls=args.calls,
                                      configs=names, num_sms=sms),
           "gpu_before": gpu_state.nvidia_smi(), "shapes": []}
    t0 = time.time()
    ops_cache = {}
    for spec in args.shapes:
        layer, m = spec.split(":")
        m = int(m)
        n, k = LAYERS[layer]
        if layer not in ops_cache:
            ops_cache.clear()
            torch.cuda.empty_cache()
            ops_cache[layer] = make_operands(n, k, 8192)
        ops = ops_cache[layer]
        a_q, a_s = blockfp8.quantize_activations(ops["x"][:m])
        k_tiles = -(-k // 128)
        fns, info, acts = {}, {}, {}
        for cfg, cname in enumerate(names):
            per_sm, bm, bn = ctas_per_sm(cname)
            if bm > 2 * max(m, 16) and bm > 16:
                continue
            tiles = -(-m // bm) * -(-n // bn)
            for sp in SPLITS:
                kps = -(-k_tiles // sp)
                eff = -(-k_tiles // kps)  # splits the launcher really uses
                if sp > k_tiles or eff != sp:
                    continue
                label = f"{cname}/{sp}"
                if args.only and not re.search(args.only, label):
                    continue
                fns[label] = (lambda cfg=cfg, sp=sp:
                              blockfp8.gemm(a_q, a_s, ops["b_q"], ops["b_s"], config=cfg, split_k=sp))
                ctas = tiles * sp
                info[label] = {"impl": "blockfp8", "config": cfg, "name": cname, "splits": sp,
                               "tiles": tiles, "ctas": ctas, "ctas_per_sm": per_sm,
                               "waves": ctas / (sms * per_sm)}
                acts[label] = [a_q, a_s]
        fns["auto"] = lambda: blockfp8.gemm(a_q, a_s, ops["b_q"], ops["b_s"])
        info["auto"] = {"impl": "blockfp8_auto"}
        acts["auto"] = [a_q, a_s]
        tf, ti, tacts = triton_fns(ops, m, rec)
        fns.update(tf)
        info.update(ti)
        for lab in tf:
            acts[lab] = tacts
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()
        timed = time_calls(fns, acts, args.runs, args.calls)
        rows = []
        for lab, inf in info.items():
            row = {"label": lab, **inf}
            for mode in ("cold", "warm"):
                row[f"{mode}_us"] = statistics.median(timed[(lab, mode)])
            rows.append(row)
        rows.sort(key=lambda r: r["cold_us"])
        out["shapes"].append({"layer": layer, "N": n, "K": k, "M": m, "rows": rows})
        bf = [r for r in rows if r["impl"] == "blockfp8"]
        tr = [r for r in rows if r["impl"] == "vllm_triton"]
        tn = next(r for r in tr if r["nearest"])
        print(f"{layer} M={m}: best blockfp8 {bf[0]['label']} {bf[0]['cold_us']:.1f} "
              f"(warm best {min(r['warm_us'] for r in bf):.1f}), auto "
              f"{next(r for r in rows if r['label'] == 'auto')['cold_us']:.1f}; triton nearest {tn['label']} "
              f"{tn['cold_us']:.1f}, best {tr[0]['label']} {tr[0]['cold_us']:.1f}", flush=True)
    out["elapsed_s"] = time.time() - t0
    out["gpu_after"] = gpu_state.nvidia_smi()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.relpath(args.out, gpu_state.REPO)}")


if __name__ == "__main__":
    main()
