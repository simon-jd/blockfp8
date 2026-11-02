"""Does the GEMM run faster when its activations are L2-resident?

The first GEMM benchmark (commit d9313bf) evicted everything before each
cold-L2 call, so the GEMM-only column read A and its scales from DRAM while
the "+q" column read them from L2, where the quantizer had just written them.
For o at M = 64 "+q" then added only 0.3 us. This probe times the GEMM
kernel alone (CUPTI duration) after a 256 MB flush, preceded by:
  A  nothing
  B  vLLM's quantizer, whose output the GEMM then reads
  C  vLLM's quantizer on the same input, output discarded (GEMM reads old A)
  D  a read of A and its scales (A L2-resident, no quantizer)
  E  a 5000-cycle spin kernel (a gap, nothing L2-resident)
Writes results/gemm/l2_probe.json.

    flock /var/tmp/rtx4090.lock .venv/bin/python bench/l2_probe.py
"""

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_state  # noqa: E402
import vllm_baselines  # noqa: E402

import blockfp8  # noqa: E402

N, K = 4096, 4096  # the o projection
REPS = 30
CONFIGS = ["m16n128s5w4", "m16n64s4w4b2"]  # the M = 64 plans before / after 64-column tiles


def main():
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    names = blockfp8.configs()
    g = torch.Generator(device="cuda").manual_seed(0)
    b, sb = blockfp8.quantize_weights(torch.randn(N, K, generator=g, device="cuda") * 0.02)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    sink = torch.empty((), dtype=torch.int64, device="cuda")
    out = {"header": gpu_state.header(benchmark="l2_probe", N=N, K=K, reps=REPS),
           "gpu_before": gpu_state.nvidia_smi(), "rows": []}
    for m in (16, 64, 256):
        x = torch.randn(m, K, generator=g, device="cuda").to(torch.bfloat16)
        a, sa = blockfp8.quantize_activations(x)
        for cname in CONFIGS:
            cfg = names.index(cname)

            def gemm(a, sa):
                return blockfp8.gemm(a, sa, b, sb, config=cfg, split_k=1)

            variants = {
                "A nothing": lambda: gemm(a, sa),
                "B quantizer, GEMM reads its output": lambda: gemm(
                    *vllm_baselines.vllm_group_quant(x, True)),
                "C quantizer, GEMM reads the old A": lambda: (
                    vllm_baselines.vllm_group_quant(x, True), gemm(a, sa)),
                "D read of A and scales": lambda: (
                    a.view(torch.uint8).sum(dtype=torch.int64), sa.sum(), gemm(a, sa)),
                "E 5000-cycle spin": lambda: (torch.cuda._sleep(5000), gemm(a, sa)),
            }
            for f in variants.values():
                f()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(REPS):
                    for f in variants.values():
                        torch.sum(flush, dim=0, dtype=torch.int64, out=sink)
                        f()
                torch.cuda.synchronize()
            path = os.path.join(gpu_state.REPO, "build", f"l2probe-{os.getpid()}.json")
            prof.export_chrome_trace(path)
            with open(path) as f:
                events = json.load(f)["traceEvents"]
            os.remove(path)
            gemms = sorted((e for e in events if e.get("cat") == "kernel"
                            and "gemm_kernel" in e["name"]), key=lambda e: e["ts"])
            assert len(gemms) == REPS * len(variants), len(gemms)
            keys = list(variants)
            durs = {k: [e["dur"] for i, e in enumerate(gemms) if keys[i % len(keys)] == k]
                    for k in keys}
            row = {"M": m, "config": cname, "splits": 1,
                   "gemm_kernel_us_median": {k: statistics.median(v) for k, v in durs.items()},
                   "gemm_kernel_us": durs}
            out["rows"].append(row)
            print(m, cname, {k[0]: round(v, 1) for k, v in row["gemm_kernel_us_median"].items()},
                  flush=True)
    out["gpu_after"] = gpu_state.nvidia_smi()
    with open(os.path.join(gpu_state.REPO, "results", "gemm", "l2_probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/gemm/l2_probe.json")


if __name__ == "__main__":
    main()
