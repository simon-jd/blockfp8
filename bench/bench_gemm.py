"""GEMM benchmark on the Qwen3-8B linear-layer shapes.

Implementations, each with the operands it would get in practice:
  blockfp8           this kernel on pre-quantized activations (M-major scales)
  blockfp8+quant     the same, plus vLLM's CUDA per-token-group quantization of
                     the bf16 activations (what a linear layer has to run)
  vllm_triton        vLLM 0.31 w8a8_triton_block_scaled_mm on pre-quantized
                     activations (row-major scales, as vLLM passes them)
  vllm_triton+quant  the same, plus the quantization (vLLM's W8A8 Triton path)
  vllm_triton_tuned  the same Triton kernel with the best config found for this
                     GPU and shape by bench/tune_triton.py (vLLM ships none)
  vllm_marlin_fp8    vLLM's FP8 Marlin kernel: weight-only FP8, bf16 activations
                     and bf16 MMAs (W8A16); no activation quantization needed
  cublaslt_fp8       torch._scaled_mm with one scale per tensor (cuBLASLt):
                     coarser scaling, an upper bound on plain FP8 GEMM speed
  cublas_bf16        torch.matmul in bf16 (cuBLAS), unquantized

Timing: CUPTI kernel timestamps through torch.profiler. Every timed call is
preceded by a marker kernel (torch.cuda._sleep), and in cold-L2 mode by a read
of a 256 MB buffer that evicts the operands. A call's time is the span from
its first kernel's start to its last kernel's end. One run times `calls`
calls of every implementation in both L2 modes, implementations in a rotating
order; the result is the median over >= 10 such interleaved runs, with the IQR.

Run it holding the GPU lock:
    flock /var/tmp/rtx4090.lock .venv/bin/python bench/bench_gemm.py
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_state  # noqa: E402

import blockfp8  # noqa: E402
import vllm_baselines  # noqa: E402

LAYERS = {
    "qkv": (6144, 4096),
    "o": (4096, 4096),
    "gate_up": (24576, 4096),
    "down": (4096, 12288),
}
M_VALUES = [1, 4, 16, 64, 256, 1024, 4096, 8192]
IMPLS = ["blockfp8", "blockfp8+quant", "vllm_triton", "vllm_triton+quant", "vllm_triton_tuned",
         "vllm_marlin_fp8", "cublaslt_fp8", "cublas_bf16"]
TUNED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gemm",
                     "triton_tuned_configs.json")
MARKER = "spin_kernel"  # the kernel torch.cuda._sleep launches


class TritonConfigRecorder:
    """Wraps vLLM's Triton kernel to record the launch config it really uses."""

    def __init__(self):
        from vllm.model_executor.layers.quantization.utils import fp8_utils

        self.mod = fp8_utils
        self.kernel = fp8_utils._w8a8_triton_block_scaled_mm
        self.last = None
        rec = self

        class Proxy:
            def __getitem__(self, grid):
                launch = rec.kernel[grid]

                def run(*args, **kwargs):
                    rec.last = {k: v for k, v in kwargs.items() if k.isupper() or k.startswith("num_")}
                    return launch(*args, **kwargs)

                return run

        fp8_utils._w8a8_triton_block_scaled_mm = Proxy()
        self.mm = fp8_utils.w8a8_triton_block_scaled_mm


class Sampler(threading.Thread):
    """Samples nvidia-smi once a second while a layer is measured."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            s = gpu_state.nvidia_smi()
            self.samples.append({k: s.get(k) for k in ("timestamp", "clocks.sm", "power.draw",
                                                       "temperature.gpu", "utilization.gpu",
                                                       "clocks_event_reasons.active")})
            self.stop.wait(1.0)


def make_operands(n, k, m_max, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(m_max, k, generator=g, device="cuda", dtype=torch.bfloat16)
    b_q, b_s = blockfp8.quantize_weights(w)
    w_amax = w.abs().amax().float()
    w_pt = (w.float() / (w_amax / 448)).to(blockfp8.FP8_DTYPE)
    return {"w": w, "x": x, "b_q": b_q, "b_s": b_s, "w_pt": w_pt, "w_pt_s": (w_amax / 448).reshape(())}


def make_calls(impl, ops, m, triton):
    """Returns (callable, info) for one implementation at batch size m."""
    x = ops["x"][:m]
    if impl == "blockfp8":
        a_q, a_s = blockfp8.quantize_activations(x)
        return (lambda: blockfp8.gemm(a_q, a_s, ops["b_q"], ops["b_s"])), {}
    if impl == "blockfp8+quant":
        def fn():
            a_q, a_s = vllm_baselines.vllm_group_quant(x, column_major=True)
            return blockfp8.gemm(a_q, a_s, ops["b_q"], ops["b_s"])
        return fn, {}
    if impl == "vllm_triton":
        a_q, a_s = blockfp8.quantize_activations(x, column_major_scales=False)
        fn = lambda: triton.mm(a_q, ops["b_q"], a_s, ops["b_s"], [128, 128], torch.bfloat16)  # noqa
        fn()
        return fn, {"triton_config": dict(triton.last)}
    if impl == "vllm_triton+quant":
        def fn():
            a_q, a_s = vllm_baselines.vllm_group_quant(x, column_major=False)
            return triton.mm(a_q, ops["b_q"], a_s, ops["b_s"], [128, 128], torch.bfloat16)
        fn()
        return fn, {"triton_config": dict(triton.last)}
    if impl == "vllm_triton_tuned":
        import triton as tl_triton

        with open(TUNED) as f:
            layer = next(v for v in json.load(f)["layers"].values()
                         if v["N"] == ops["b_q"].shape[0] and v["K"] == ops["b_q"].shape[1])
        cfg = {k: v for k, v in layer["configs"][str(m)].items() if k != "error"}
        a_q, a_s = blockfp8.quantize_activations(x, column_major_scales=False)
        b, b_s = ops["b_q"], ops["b_s"]
        n, k = b.shape
        c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        grid = (tl_triton.cdiv(m, cfg["BLOCK_SIZE_M"]) * tl_triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)

        def fn():
            triton.kernel[grid](a_q, b, c, a_s, b_s, m, n, k, 128, 128, a_q.stride(-2),
                                a_q.stride(-1), b.stride(1), b.stride(0), c.stride(-2),
                                c.stride(-1), a_s.stride(-2), a_s.stride(-1), b_s.stride(1),
                                b_s.stride(0), **cfg)
            return c
        return fn, {"triton_config": cfg}
    if impl == "vllm_marlin_fp8":
        if "marlin" not in ops:
            ops["marlin"] = vllm_baselines.MarlinFp8Linear(ops["b_q"], ops["b_s"])
        mar = ops["marlin"]
        return (lambda: mar(x)), {}
    if impl == "cublaslt_fp8":
        amax = x.abs().amax().float()
        s = (amax / 448).reshape(())
        info = {}
        mp = m
        try:
            a_pt = (x.float() / s).to(blockfp8.FP8_DTYPE)
            torch._scaled_mm(a_pt, ops["w_pt"].t(), s, ops["w_pt_s"], out_dtype=torch.bfloat16)
        except RuntimeError as e:  # cuBLASLt FP8 needs M % 16 == 0 on some versions
            mp = -(-m // 16) * 16
            info = {"padded_m": mp, "reason": str(e).splitlines()[0][:200]}
        xp = torch.nn.functional.pad(x, (0, 0, 0, mp - m)) if mp != m else x
        a_pt = (xp.float() / s).to(blockfp8.FP8_DTYPE)
        wt = ops["w_pt"].t()
        return (lambda: torch._scaled_mm(a_pt, wt, s, ops["w_pt_s"], out_dtype=torch.bfloat16)), info
    if impl == "cublas_bf16":
        wt = ops["w"].t()
        return (lambda: torch.matmul(x, wt)), {}
    raise ValueError(impl)


def _errors(c, ref):
    err = (c.double() - ref).abs()
    rel = err / ref.abs().clamp_min(1e-30)
    return {"max_abs_err": err.max().item(), "max_rel_err": rel.max().item(),
            "median_rel_err": rel.median().item(), "rel_fro_err": (err.norm() / ref.norm()).item()}


def check_accuracy(ops, m, triton):
    """Errors vs FP64 references on (up to) the first 256 rows.

    The W8A8 kernels are compared with the exact block-scaled product of the
    quantized operands. Marlin never quantizes activations, so it is compared
    with bf16 activations times the dequantized FP8 weights.
    """
    rows = min(m, 256)
    x = ops["x"][:rows]
    a_q, a_s = blockfp8.quantize_activations(x)
    ref = blockfp8.reference_gemm(a_q, a_s, ops["b_q"], ops["b_s"])
    out = {
        "blockfp8": _errors(blockfp8.gemm(a_q, a_s, ops["b_q"], ops["b_s"]), ref),
        "vllm_triton": _errors(triton.mm(a_q, ops["b_q"], a_s.contiguous(), ops["b_s"],
                                         [128, 128], torch.bfloat16), ref),
    }
    if "marlin" in ops:
        ref_w = x.double() @ blockfp8.dequantize_weights(ops["b_q"], ops["b_s"]).t()
        out["vllm_marlin_fp8"] = _errors(ops["marlin"](x), ref_w)
    return out


def measure_layer(name, n, k, m_values, runs, triton):
    ops = make_operands(n, k, max(m_values))
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    flush_out = torch.empty((), dtype=torch.int64, device="cuda")
    results = []
    sampler = Sampler()
    sampler.start()
    state_before = gpu_state.nvidia_smi()
    for m in m_values:
        calls_per = 20 if m <= 256 else 10 if m <= 1024 else 5
        fns, infos = {}, {}
        for impl in IMPLS:
            fns[impl], infos[impl] = make_calls(impl, ops, m, triton)
            for _ in range(3):
                fns[impl]()  # warm-up / compile
        torch.cuda.synchronize()
        labels = []
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for r in range(runs):
                order = IMPLS[r % len(IMPLS):] + IMPLS[: r % len(IMPLS)]
                for impl in order:
                    for mode in ("warm", "cold"):
                        for _ in range(calls_per):
                            if mode == "cold":
                                torch.sum(flush, dtype=torch.int64, out=flush_out)
                            torch.cuda._sleep(1)
                            fns[impl]()
                            labels.append((r, impl, mode))
            torch.cuda._sleep(1)
            torch.cuda.synchronize()
        spans = split_spans(prof, labels)
        for impl in IMPLS:
            for mode in ("warm", "cold"):
                per_run = []
                for r in range(runs):
                    v = [s for (rr, ii, mm), s in spans if rr == r and ii == impl and mm == mode]
                    per_run.append(statistics.median(v))
                q = statistics.quantiles(per_run, n=4, method="inclusive")
                med = statistics.median(per_run)
                flops = 2.0 * m * n * k
                bytes_fp8 = m * k + n * k + 2 * m * n
                results.append({
                    "layer": name, "N": n, "K": k, "M": m, "impl": impl, "l2": mode,
                    "median_us": med, "p25_us": q[0], "p75_us": q[2], "runs_us": per_run,
                    "calls_per_run": calls_per, "tflops": flops / med / 1e6,
                    "gbps_fp8_operands": bytes_fp8 / med / 1e3, **infos[impl],
                })
        acc = check_accuracy(ops, m, triton)
        for row in results:
            if row["M"] == m and row["impl"] in acc:
                row["accuracy_first_rows"] = acc[row["impl"]]
        line = " ".join(
            f"{r['impl']}={r['median_us']:.1f}" for r in results if r["M"] == m and r["l2"] == "cold"
        )
        print(f"{name} M={m}: cold us {line}", flush=True)
    sampler.stop.set()
    sampler.join()
    return results, {"before": state_before, "after": gpu_state.nvidia_smi(),
                     "samples": sampler.samples}


def split_spans(prof, labels):
    """Group kernels by the marker launched before each call; returns [(label, span_us)].

    Launch order per call is [flush if cold], marker, the call's kernels; so the
    kernels between two markers are one call plus, if the next call is cold,
    the next call's flush, which is dropped.
    """
    path = os.path.join(gpu_state.REPO, "build", f"trace-{os.getpid()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    os.remove(path)
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    groups, cur = [], None
    for e in kernels:
        if MARKER in e["name"]:
            cur = []
            groups.append(cur)
        elif cur is not None:
            cur.append(e)
    groups = groups[:-1]  # the final marker closes the last call
    if len(groups) != len(labels):
        raise RuntimeError(f"{len(groups)} kernel groups for {len(labels)} calls")
    out = []
    for i, (lab, g) in enumerate(zip(labels, groups)):
        if i + 1 < len(labels) and labels[i + 1][2] == "cold":
            g = g[:-1]
        if not g:
            raise RuntimeError(f"no kernels recorded for {lab}")
        span = max(e["ts"] + e["dur"] for e in g) - min(e["ts"] for e in g)
        out.append((lab, span))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="*", default=list(LAYERS))
    ap.add_argument("--m", nargs="*", type=int, default=M_VALUES)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--impls", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(gpu_state.REPO, "results", "gemm", "qwen3_8b.json"))
    args = ap.parse_args()
    if args.runs < 10:
        print("note: fewer than 10 runs; not a reportable measurement", flush=True)
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    import triton
    import vllm
    from vllm.model_executor.layers.quantization.utils.fp8_utils import get_w8a8_block_fp8_configs

    global IMPLS
    if args.impls:
        IMPLS = args.impls
    elif not os.path.exists(TUNED):
        IMPLS = [i for i in IMPLS if i != "vllm_triton_tuned"]
    rec = TritonConfigRecorder()
    meta = gpu_state.header(
        benchmark="gemm", runs=args.runs, impls=IMPLS, torch=torch.__version__, cuda=torch.version.cuda,
        vllm=vllm.__version__, triton=triton.__version__, device=torch.cuda.get_device_name(),
        configs=blockfp8.configs(),
        vllm_selected_block_fp8_kernel=vllm_baselines.selected_block_fp8_kernel(),
        vllm_tuned_config_files={
            name: get_w8a8_block_fp8_configs(n, k, 128, 128) is not None
            for name, (n, k) in LAYERS.items()
        },
    )
    all_rows, states = [], {}
    t0 = time.time()
    for name in args.layers:
        n, k = LAYERS[name]
        rows, st = measure_layer(name, n, k, args.m, args.runs, rec)
        all_rows += rows
        states[name] = st
        torch.cuda.empty_cache()
    meta["elapsed_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({**meta, "gpu_state": states, "results": all_rows}, f, indent=1)
    print(f"wrote {os.path.relpath(args.out, gpu_state.REPO)}")


if __name__ == "__main__":
    main()
