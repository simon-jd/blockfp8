"""Turn results/gemm/qwen3_8b.json (+ results/micro, + the off-grid batch sizes
in results/gemm/qwen3_8b_offgrid.json) into tables and a chart.

Writes results/gemm/summary.md and docs/img/speedup.svg. No GPU needed.
"""

import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMM = os.path.join(REPO, "results", "gemm", "qwen3_8b.json")
OFFGRID = os.path.join(REPO, "results", "gemm", "qwen3_8b_offgrid.json")
MICRO = os.path.join(REPO, "results", "micro")
SHORT = {
    "blockfp8": "blockfp8",
    "blockfp8+quant": "blockfp8 +q",
    "blockfp8_fused": "blockfp8 fused",
    "blockfp8_linear": "blockfp8 linear",
    "vllm_group_quant": "vLLM quant alone",
    "vllm_triton": "Triton",
    "vllm_triton+quant": "Triton +q",
    "vllm_triton_tuned": "Triton tuned",
    "vllm_marlin_fp8": "Marlin W8A16",
    "cublaslt_fp8": "cuBLASLt FP8/tensor",
    "cublas_bf16": "cuBLAS BF16",
}


def load():
    with open(GEMM) as f:
        g = json.load(f)
    peaks = {}
    try:
        with open(os.path.join(MICRO, "mma_peak.json")) as f:
            mp = json.load(f)["data"]["results"]
        peaks["fp8_tflops"] = mp[0]["tflops"]["median"]
        peaks["fp8_flop_per_clk_sm"] = mp[0]["flop_per_sm_per_clk"]["median"]
        peaks["bf16_tflops"] = mp[2]["tflops"]["median"]
        with open(os.path.join(MICRO, "dram_bw.json")) as f:
            bw = json.load(f)["data"]["results"]
        peaks["dram_read_gbps"] = max(
            r["gb_per_s"]["median"] for r in bw if r["kind"] == "read" and r["l2"] == "cold"
            and r["working_set_mb"] >= 256
        )
    except (OSError, KeyError, IndexError):
        pass
    return g, peaks


def linear_us(cell):
    """blockfp8.linear's time and path (fused, or its own quantization + GEMM)."""
    r = cell["blockfp8_linear"]
    return r["median_us"], "fused" if r.get("linear_prefers_fused") else "quant+gemm"


def table(g, peaks, l2):
    rows = [r for r in g["results"] if r["l2"] == l2]
    impls = [i for i in SHORT if any(r["impl"] == i for r in rows)]
    out = []
    for layer in dict.fromkeys(r["layer"] for r in rows):
        lr = [r for r in rows if r["layer"] == layer]
        n, k = lr[0]["N"], lr[0]["K"]
        out.append(f"\n### {layer} (N={n}, K={k}), {l2} L2, median us [IQR]\n")
        out.append("| M | " + " | ".join(SHORT[i] for i in impls)
                   + " | vs Triton | vs Triton tuned | linear runs | vs Marlin | blockfp8 TFLOPS |"
                   " blockfp8 GB/s |")
        out.append("|" + "---|" * (len(impls) + 7))
        for m in sorted({r["M"] for r in lr}):
            cell = {r["impl"]: r for r in lr if r["M"] == m}
            vals = [f"{cell[i]['median_us']:.1f} [{cell[i]['p25_us']:.1f}-{cell[i]['p75_us']:.1f}]"
                    if i in cell else "-" for i in impls]
            ours = cell["blockfp8"]["median_us"]
            vs_t = cell["vllm_triton"]["median_us"] / ours
            vs_tt = cell["vllm_triton_tuned"]["median_us"] / ours if "vllm_triton_tuned" in cell \
                else float("nan")
            lin, path = linear_us(cell)
            vs_m = cell["vllm_marlin_fp8"]["median_us"] / lin if "vllm_marlin_fp8" in cell \
                else float("nan")
            out.append(f"| {m} | " + " | ".join(vals) + f" | {vs_t:.2f}x | {vs_tt:.2f}x | {path} |"
                       f" {vs_m:.2f}x | "
                       f"{cell['blockfp8']['tflops']:.0f} | {cell['blockfp8']['gbps_fp8_operands']:.0f} |")
    return "\n".join(out)


def offgrid(path):
    """Cold-L2 ratios at batch sizes none of the tables was tuned on."""
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        g = json.load(f)
    rows = [r for r in g["results"] if r["l2"] == "cold"]
    out = ["\n## Batch sizes outside the tuning grid (cold L2)\n",
           f"Source: results/gemm/qwen3_8b_offgrid.json, commit {g['git']['commit'][:10]}"
           f"{' (dirty)' if g['git']['dirty'] else ''}, {g['time_utc']}, {g['runs']} interleaved runs."
           " The plan tables and the tuned Triton configs are tuned on the powers of two "
           "1-8192 only; these M lie between them: blockfp8 takes the entry with the smallest "
           "tabulated M >= M, tuned Triton the nearest tabulated M (vLLM's rule; Triton M column).",
           "Ratios are baseline time / blockfp8 time (Marlin: / blockfp8.linear time).\n",
           "| layer | M | blockfp8 us | linear us (runs) | vs Triton | vs Triton tuned (M) |"
           " vs Marlin | vs cuBLASLt FP8/tensor |",
           "|---|---|---|---|---|---|---|---|"]
    for layer in dict.fromkeys(r["layer"] for r in rows):
        lr = [r for r in rows if r["layer"] == layer]
        for m in sorted({r["M"] for r in lr}):
            c = {r["impl"]: r for r in lr if r["M"] == m}
            ours = c["blockfp8"]["median_us"]
            lin, path_ = linear_us(c)
            tt = c.get("vllm_triton_tuned")
            out.append(
                f"| {layer} | {m} | {ours:.1f} | {lin:.1f} ({path_}) | "
                f"{c['vllm_triton']['median_us'] / ours:.2f}x | "
                + (f"{tt['median_us'] / ours:.2f}x ({tt.get('triton_config_m', '?')})" if tt else "-")
                + f" | {c['vllm_marlin_fp8']['median_us'] / lin:.2f}x | "
                f"{c['cublaslt_fp8']['median_us'] / ours:.2f}x |")
    return "\n".join(out) + "\n"


def chart(g, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in g["results"] if r["l2"] == "cold"]
    layers = list(dict.fromkeys(r["layer"] for r in rows))
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.6), sharey=True)
    series = [("vllm_marlin_fp8", "vs vLLM default: FP8 Marlin W8A16 (ours: linear, incl. quant)",
               "#d62728"),
              ("vllm_triton", "vs vLLM Triton, default config (GEMM only)", "#1f77b4"),
              ("vllm_triton_tuned", "vs vLLM Triton, tuned here (GEMM only)", "#2ca02c"),
              ("cublaslt_fp8", "vs cuBLASLt FP8 per-tensor (GEMM only)", "#7f7f7f")]
    for ax, layer in zip(axes, layers):
        lr = [r for r in rows if r["layer"] == layer]
        ms = sorted({r["M"] for r in lr})
        t = {(r["impl"], r["M"]): r["median_us"] for r in lr}
        lin = {m: linear_us({r["impl"]: r for r in lr if r["M"] == m})[0] for m in ms}
        for impl, label, color in series:
            ys = [t[(impl, m)] / (lin[m] if impl == "vllm_marlin_fp8" else t[("blockfp8", m)])
                  for m in ms]
            ax.plot(ms, ys, marker="o", label=label, color=color)
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_title(f"{layer} N={lr[0]['N']} K={lr[0]['K']}")
        ax.set_xlabel("M (tokens)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("speedup of blockfp8 (>1 = faster)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("RTX 4090, cold L2, median of interleaved runs", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)


def main():
    g, peaks = load()
    head = [
        "# GEMM results (generated by bench/report.py)",
        "",
        f"Source: results/gemm/qwen3_8b.json, commit {g['git']['commit'][:10]}"
        f"{' (dirty)' if g['git']['dirty'] else ''}, {g['time_utc']}, {g['runs']} interleaved runs,"
        f" vLLM {g['vllm']}, torch {g['torch']}, triton {g['triton']}.",
        f"vLLM's block-FP8 kernel choice on this GPU: {g.get('vllm_selected_block_fp8_kernel')}.",
        f"Measured peaks: {json.dumps(peaks)}",
        "",
        "+q = including the per-token-group activation quantization kernel (vLLM's CUDA one);"
        " fused = bf16 activations quantized inside the GEMM (M <= 256 only).",
        "linear = blockfp8.linear, what the vLLM plugin runs: fused, or blockfp8's own quantization"
        " kernel + GEMM, per the plan table (linear runs = which).",
        "vs Triton (tuned) = Triton time / blockfp8 time (GEMM only); "
        "vs Marlin = Marlin time / blockfp8.linear time (quantization included).",
    ]
    clocks = []
    for st in g["gpu_state"].values():
        clocks += [float(s["clocks.sm"].split()[0]) for s in st["samples"]
                   if s.get("clocks.sm") and s["clocks.sm"].split()[0].isdigit()]
    if clocks:
        head.append(f"SM clock during the runs (nvidia-smi, 1 s samples): median "
                    f"{statistics.median(clocks):.0f} MHz, min {min(clocks):.0f}, max {max(clocks):.0f}.")
    body = ("\n".join(head) + "\n" + table(g, peaks, "cold") + "\n" + table(g, peaks, "warm") + "\n"
            + offgrid(OFFGRID))
    with open(os.path.join(REPO, "results", "gemm", "summary.md"), "w") as f:
        f.write(body)
    chart(g, os.path.join(REPO, "docs", "img", "speedup.svg"))
    print(body)


if __name__ == "__main__":
    main()
