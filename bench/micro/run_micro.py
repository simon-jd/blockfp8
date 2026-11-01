"""Run the microbenchmarks and write results/micro/*.json.

Run it holding the machine-wide GPU lock:
    flock /var/tmp/rtx4090.lock .venv/bin/python bench/micro/run_micro.py
"""

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_state  # noqa: E402

REPO = gpu_state.REPO
OUT = os.path.join(REPO, "results", "micro")


def run_binary(name: str, args: list[str]) -> dict:
    before = gpu_state.nvidia_smi()
    proc = subprocess.run(
        [os.path.join(REPO, "build", name), *args], capture_output=True, text=True, check=True
    )
    after = gpu_state.nvidia_smi()
    return {"data": json.loads(proc.stdout), "gpu_before": before, "gpu_after": after}


def summarize_precision(data: dict) -> dict:
    """Retained bits and rounding of the in-MMA sum, from the crafted cases."""
    out = {}
    for acc in ("f32acc", "f16acc"):
        al = data[f"alignment_{acc}"]
        js = al["j"]
        res = {}
        for name in ("single_k1", "single_k31", "c_big", "c_small", "many"):
            obs, ex = al[name]["observed_minus_big"], al[name]["exact_minus_big"]
            kept = 0
            for j, o, e in zip(js, obs, ex):
                if o != e:
                    break
                kept = j
            res[name] = {"exact_up_to_j": kept, "first_inexact": [
                {"j": j, "observed": o, "exact": e} for j, o, e in zip(js, obs, ex) if o != e
            ][:3]}
        # Rounding: compare the small term against truncation toward zero,
        # toward -inf, and round-to-nearest on a grid of 2^(big - F).
        big = 16 if acc == "f32acc" else 8
        f = res["single_k1"]["exact_up_to_j"]
        q = 2.0 ** (big - f)
        modes = {"toward_zero": math.trunc, "toward_minus_inf": math.floor, "nearest_even": round}
        rounding = {}
        for name in ("s=+1.5", "s=+1.75", "s=-1", "s=-1.5", "s=-1.75"):
            obs, ex = al[name]["observed_minus_big"], al[name]["exact_minus_big"]
            match = {m: all(o == fn(e / q) * q for o, e in zip(obs, ex)) for m, fn in modes.items()}
            rounding[name] = [m for m, ok in match.items() if ok]
        res["fraction_bits_kept_below_largest_term"] = f
        res["rounding_of_dropped_bits"] = rounding
        out[acc] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["mma_peak", "dram_bw", "mma_precision"])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    os.makedirs(OUT, exist_ok=True)
    for name in args.only:
        extra = [str(args.reps)] if name == "mma_peak" else [str(2 * args.reps)] if name == "dram_bw" else []
        res = {**gpu_state.header(benchmark=name, argv=extra), **run_binary(name, extra)}
        if name == "mma_precision":
            res["summary"] = summarize_precision(res["data"])
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote results/micro/{name}.json")
        if name == "mma_precision":
            print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
