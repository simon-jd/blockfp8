"""End to end: Qwen3-8B-FP8 in vLLM 0.31, stock (FP8 Marlin, W8A16) vs the
blockfp8 plugin (W8A8, blockfp8.linear), on the local HF cache only.

One run = one fresh process holding the GPU lock, with one configuration:
  * prefill throughput: batches of random-token prompts of each length,
    max_tokens=1, prefix caching off; prompt tokens / wall time;
  * decode TPOT: B prompts of 128 tokens, generated twice with ignore_eos to
    8 and to 136 tokens; TPOT = time difference / 128 (prefill cancels);
  * outputs: greedy generation (up to 64 tokens) over a fixed set of chat prompts,
    with the logprob of each generated token, and teacher-forced logprobs
    (prompt_logprobs) of a reference continuation: the first stock run's own
    greedy output, so every run scores the same token sequences.
Each figure is the median of 3 repetitions inside the run. The driver runs
--runs rounds of (stock, plugin), alternating which goes first, and writes
results/e2e/qwen3_8b_fp8.json; --extra triton appends one run of vLLM's own
W8A8 path (Marlin disabled) as a reference for the output check.

--ppl: model quality instead of speed. wikitext-2-raw-v1 test (bench/wikitext.py;
lines joined with blank lines, as in the usual perplexity scripts), tokenized
once and cut into non-overlapping windows of PPL_WINDOW tokens; each window is
scored on its own with prompt_logprobs (every token but its first). One process
per configuration: stock, plugin, triton, then stock again (its run-to-run
noise). Writes results/e2e/wikitext2_ppl.json: perplexity per run, per-window
NLL sums, and statistics of the per-token logprob differences between runs (the
per-token logprobs stay in build/e2e).
bench/report_e2e.py summarizes both files.

    .venv/bin/python bench/e2e_vllm.py --runs 5      # takes the lock per run itself
    .venv/bin/python bench/e2e_vllm.py --extra triton
    .venv/bin/python bench/e2e_vllm.py --ppl
"""

import argparse
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_state  # noqa: E402
import wikitext  # noqa: E402

MODEL = "Qwen/Qwen3-8B-FP8"
# The local HF cache entry has no refs/ (so offline lookup by name fails); load
# the snapshot directory itself. Nothing is downloaded.
SNAPSHOTS = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots")
PREFILL_LENS = [256, 1024, 4096]
PREFILL_TOKENS = 16384  # prompt tokens per prefill measurement
DECODE_BATCHES = [1, 4, 16, 64]
DECODE_PROMPT = 128
DECODE_STEPS = (8, 136)
GEN_TOKENS = 64
REPS = 3
ENGINE = dict(gpu_memory_utilization=0.80, max_model_len=8192, max_num_batched_tokens=8192,
              max_num_seqs=128, enable_prefix_caching=False, seed=0)
OTHER_PLUGINS = "lora_filesystem_resolver,lora_hf_hub_resolver"
PPL_WINDOW = 2048
# One window per scheduler step: prompt logprobs materialize [tokens, vocab] FP32 logits.
PPL_ENGINE = dict(ENGINE, max_num_batched_tokens=PPL_WINDOW)
PPL_ORDER = [("ppl1", "stock"), ("ppl1", "plugin"), ("ppl1", "triton"), ("ppl2", "stock")]
OUT_DIR = os.path.join(gpu_state.REPO, "results", "e2e")
RAW_DIR = os.path.join(gpu_state.REPO, "build", "e2e")  # per-run files and logs
PROMPTS = [
    "Explain why the sky is blue in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is the difference between a process and a thread?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give me a haiku about autumn rain.",
    "How does a hash table handle collisions? Name two strategies.",
    "Translate into French: The library opens at nine and closes at six.",
    "List five prime numbers greater than 100.",
    "What causes the seasons on Earth?",
    "Write a SQL query that returns the ten most recent orders per customer.",
    "Explain gradient descent to a high-school student.",
    "What are the main differences between TCP and UDP?",
    "Describe the water cycle step by step.",
    "Write a short email asking a colleague to review a pull request.",
    "Why is the derivative of sin(x) equal to cos(x)? Sketch an argument.",
    "Name three sorting algorithms and their average time complexity.",
    "What is a GPU warp, and why does divergence slow it down?",
    "Give a recipe for a simple tomato pasta sauce.",
    "Explain the CAP theorem with an example.",
    "What is 17 times 23? Show the steps.",
    "Write a bash one-liner that counts lines in all .py files under a directory.",
    "What does a compiler's register allocator do?",
    "Compare renewable and non-renewable energy sources briefly.",
    "Explain what a mutex is and when you would use one.",
]


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            s = gpu_state.nvidia_smi()
            self.samples.append({k: s.get(k) for k in ("timestamp", "clocks.sm", "power.draw",
                                                       "temperature.gpu", "memory.used")})
            self.stop.wait(1.0)


def rand_prompts(n, length, seed):
    from vllm.inputs import TokensPrompt

    rng = random.Random(seed)
    return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000) for _ in range(length)])
            for _ in range(n)]


def timed(fn):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def run_one(config, out, reference):
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    snapshot = sorted(os.listdir(SNAPSHOTS))[-1]
    model_dir = os.path.join(SNAPSHOTS, snapshot)
    meta = gpu_state.header(benchmark="e2e", config=config, model=MODEL, snapshot=snapshot,
                            engine=ENGINE,
                            vllm=vllm.__version__, torch=torch.__version__,
                            vllm_plugins=os.environ.get("VLLM_PLUGINS"))
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    llm = LLM(model=model_dir, **ENGINE)
    meta["load_s"] = time.time() - t0
    res = {"prefill": [], "decode": []}

    for length in PREFILL_LENS:
        prompts = rand_prompts(max(1, PREFILL_TOKENS // length), length, seed=length)
        sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        llm.generate(prompts, sp, use_tqdm=False)  # warm-up
        med, ts = timed(lambda: llm.generate(prompts, sp, use_tqdm=False))
        tokens = len(prompts) * length
        res["prefill"].append({"prompt_len": length, "prompts": len(prompts), "tokens": tokens,
                               "s": med, "s_reps": ts, "tokens_per_s": tokens / med})
        print(f"[{config}] prefill L={length}: {tokens / med:.0f} tok/s", flush=True)

    for batch in DECODE_BATCHES:
        prompts = rand_prompts(batch, DECODE_PROMPT, seed=10_000 + batch)
        short, long_ = (SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
                        for n in DECODE_STEPS)
        llm.generate(prompts, long_, use_tqdm=False)  # warm-up
        t_short, ts_short = timed(lambda: llm.generate(prompts, short, use_tqdm=False))
        t_long, ts_long = timed(lambda: llm.generate(prompts, long_, use_tqdm=False))
        tpot = (t_long - t_short) / (DECODE_STEPS[1] - DECODE_STEPS[0])
        res["decode"].append({"batch": batch, "prompt_len": DECODE_PROMPT, "steps": DECODE_STEPS,
                              "s_short": t_short, "s_long": t_long, "s_short_reps": ts_short,
                              "s_long_reps": ts_long, "tpot_ms": tpot * 1e3})
        print(f"[{config}] decode B={batch}: TPOT {tpot * 1e3:.2f} ms", flush=True)

    tok = llm.get_tokenizer()
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
             for p in PROMPTS]
    prompt_ids = [tok(c, add_special_tokens=False).input_ids for c in chats]
    greedy = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
                          SamplingParams(max_tokens=GEN_TOKENS, temperature=0.0, logprobs=0),
                          use_tqdm=False)
    gen = []
    for o in greedy:
        c = o.outputs[0]
        gen.append({"tokens": list(c.token_ids),
                    "logprobs": [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)]})
    ref = gen
    if reference:
        with open(reference) as f:
            ref = json.load(f)["outputs"]["greedy"]
    forced_in = [TokensPrompt(prompt_token_ids=ids + r["tokens"]) for ids, r in zip(prompt_ids, ref)]
    forced = llm.generate(forced_in, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
                          use_tqdm=False)
    teacher = []
    for ids, r, o in zip(prompt_ids, ref, forced):
        lps = o.prompt_logprobs[len(ids):]
        teacher.append([lp[t].logprob for t, lp in zip(r["tokens"], lps)])
    res["outputs"] = {"prompt_ids": prompt_ids, "greedy": gen, "teacher_forced": teacher,
                      "reference": "run 1 stock" if reference else "self"}

    sampler.stop.set()
    sampler.join()
    meta["gpu_samples"] = sampler.samples
    meta["gpu_after"] = gpu_state.nvidia_smi()
    with open(out, "w") as f:
        json.dump({**meta, **res}, f)
    print(f"[{config}] wrote {os.path.relpath(out, gpu_state.REPO)}", flush=True)
    shut_down(llm)


def shut_down(llm):
    """Stop the engine and every child process, then exit. An EngineCore left
    running inherits the GPU lock's file descriptor and holds the lock."""
    import psutil

    llm.llm_engine.engine_core.shutdown(timeout=60)
    children = psutil.Process().children(recursive=True)
    _, alive = psutil.wait_procs(children, timeout=30)
    for p in alive:
        print(f"killing leftover child {p.pid} ({p.name()})", flush=True)
        p.kill()
    psutil.wait_procs(alive, timeout=10)
    # Interpreter teardown after vLLM aborts in a C++ destructor; nothing is left to do.
    sys.stdout.flush()
    os._exit(0)


def run_one_ppl(config, out):
    """wikitext-2 test perplexity of one configuration (one process)."""
    if not gpu_state.holds_gpu_lock():
        sys.exit(f"refusing to measure without holding {gpu_state.LOCK} (run under flock)")
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows, digest = wikitext.load_test()
    snapshot = sorted(os.listdir(SNAPSHOTS))[-1]
    meta = gpu_state.header(benchmark="e2e_ppl", config=config, model=MODEL, snapshot=snapshot,
                            engine=PPL_ENGINE, vllm=vllm.__version__, torch=torch.__version__,
                            vllm_plugins=os.environ.get("VLLM_PLUGINS"),
                            dataset={"name": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
                                     "split": "test", "file": wikitext.URL, "sha256": digest,
                                     "rows": len(rows)})
    meta["gpu_before"] = gpu_state.nvidia_smi()
    llm = LLM(model=os.path.join(SNAPSHOTS, snapshot), **PPL_ENGINE)
    ids = llm.get_tokenizer()("\n\n".join(rows), add_special_tokens=False).input_ids
    windows = [ids[i:i + PPL_WINDOW] for i in range(0, len(ids), PPL_WINDOW)]
    windows = [w for w in windows if len(w) > 1]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in windows],
                        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
                        use_tqdm=False)
    elapsed = time.time() - t0
    lps = [[o.prompt_logprobs[j][w[j]].logprob for j in range(1, len(w))]
           for w, o in zip(windows, outs)]
    scored = sum(map(len, lps))
    nll = -sum(map(sum, lps)) / scored
    res = {"tokens": len(ids), "window": PPL_WINDOW, "windows": len(windows),
           "scored_tokens": scored, "nll": nll, "ppl": math.exp(nll), "score_s": elapsed,
           "window_nll_sums": [-sum(x) for x in lps], "logprobs": lps,
           "gpu_after": gpu_state.nvidia_smi()}
    with open(out, "w") as f:
        json.dump({**meta, **res}, f)
    print(f"[{config}] wikitext-2 ppl {res['ppl']:.4f} over {scored} tokens", flush=True)
    shut_down(llm)


CONFIG_ENV = {
    "stock": {"VLLM_PLUGINS": OTHER_PLUGINS},
    "plugin": {"VLLM_PLUGINS": OTHER_PLUGINS + ",blockfp8"},
    # vLLM's own W8A8 path: its Triton block-FP8 kernel, the next candidate
    # once Marlin and Humming are disabled (the log records what vLLM chose).
    "triton": {"VLLM_PLUGINS": OTHER_PLUGINS,
               "VLLM_DISABLED_KERNELS": "MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel"},
}


def run_config(tag, config, reference, ppl=False):
    """One measurement process under the GPU lock; returns its result file."""
    base_env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    VLLM_CACHE_ROOT=os.path.join(RAW_DIR, "vllm-cache"),
                    TRITON_CACHE_DIR=os.path.join(RAW_DIR, "triton-cache"),
                    TORCHINDUCTOR_CACHE_DIR=os.path.join(RAW_DIR, "inductor-cache"))
    out = os.path.join(RAW_DIR, f"{tag}-{config}.json")
    log = os.path.join(RAW_DIR, f"{tag}-{config}.log")
    cmd = ["flock", gpu_state.LOCK, sys.executable, os.path.abspath(__file__),
           "--one", config, "--out", out]
    if reference:
        cmd += ["--reference", reference]
    if ppl:
        cmd += ["--ppl"]
    t0 = time.time()
    with open(log, "w") as f:
        # Own process group, so nothing of this run can outlive it holding the lock.
        p = subprocess.Popen(cmd, env=dict(base_env, **CONFIG_ENV[config]), stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        p.wait()
    try:
        os.killpg(p.pid, signal.SIGKILL)
        print(f"{tag} {config}: killed processes left in its group", flush=True)
    except ProcessLookupError:
        pass
    if p.returncode != 0 or not os.path.exists(out):
        sys.exit(f"{tag} {config} failed (exit {p.returncode}); see {log}")
    with open(log, errors="replace") as f:
        selected = sorted({line.split("Selected ", 1)[1].split()[0] for line in f
                           if "Selected " in line and "for Fp8LinearMethod" in line})
    with open(out) as f:
        d = json.load(f)
    d["selected_from_log"] = selected
    d["tag"] = tag
    with open(out, "w") as f:
        json.dump(d, f)
    print(f"{tag} {config}: {time.time() - t0:.0f} s, selected {selected}", flush=True)
    return out


def write_combined(files):
    runs_data = []
    for path in files:
        with open(path) as f:
            runs_data.append(json.load(f))
    with open(os.path.join(OUT_DIR, "qwen3_8b_fp8.json"), "w") as f:
        json.dump({"runs": runs_data, "files": [os.path.relpath(p, gpu_state.REPO) for p in files]},
                  f)
    print("wrote results/e2e/qwen3_8b_fp8.json")


def drive(runs):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    reference = os.path.join(RAW_DIR, "run1-stock.json")
    files = []
    for r in range(1, runs + 1):
        # The first stock run produces the reference outputs; then alternate.
        order = ["stock", "plugin"] if r % 2 else ["plugin", "stock"]
        for config in order:
            ref = None if (r == 1 and config == "stock") else reference
            files.append(run_config(f"run{r}", config, ref))
    write_combined(files)


def extra(config):
    """One more run of another configuration, appended to the combined results."""
    with open(os.path.join(OUT_DIR, "qwen3_8b_fp8.json")) as f:
        files = [os.path.join(gpu_state.REPO, p) for p in json.load(f)["files"]]
    files.append(run_config("extra", config, os.path.join(RAW_DIR, "run1-stock.json")))
    write_combined(files)


def logprob_diffs(a, b):
    """Per-token logprob differences a - b over the same scored tokens."""
    import numpy as np

    d = np.concatenate([np.subtract(wa, wb) for wa, wb in zip(a, b)])
    ad = np.abs(d)
    return {"tokens": int(d.size), "mean_abs": float(ad.mean()),
            "p99_abs": float(np.percentile(ad, 99)), "p999_abs": float(np.percentile(ad, 99.9)),
            "max_abs": float(ad.max()), "mean_signed": float(d.mean()),
            "identical": int((d == 0).sum())}


def drive_ppl():
    os.makedirs(RAW_DIR, exist_ok=True)
    runs = []
    for tag, config in PPL_ORDER:
        path = run_config(f"{tag}", config, None, ppl=True)
        with open(path) as f:
            runs.append(json.load(f))
    if len({r["scored_tokens"] for r in runs}) != 1:
        sys.exit("runs scored different token counts")
    names = [f"{r['config']} ({r['tag']})" for r in runs]
    pairs = [(1, 0), (2, 0), (1, 2), (3, 0)]  # plugin, triton, plugin vs triton, stock vs stock
    summary = {
        "runs": [{k: v for k, v in r.items() if k != "logprobs"} for r in runs],
        "logprob_diffs": [{"run": names[i], "vs": names[j],
                           **logprob_diffs(runs[i]["logprobs"], runs[j]["logprobs"])}
                          for i, j in pairs],
        "raw_files": [os.path.relpath(os.path.join(RAW_DIR, f"{r['tag']}-{r['config']}.json"),
                                      gpu_state.REPO) for r in runs],
    }
    path = os.path.join(OUT_DIR, "wikitext2_ppl.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    for r, n in zip(runs, names):
        print(f"{n}: ppl {r['ppl']:.4f}")
    print(f"wrote {os.path.relpath(path, gpu_state.REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--extra", choices=list(CONFIG_ENV),
                    help="append one run of this configuration to the existing results")
    ap.add_argument("--one", choices=list(CONFIG_ENV), help="one measurement process (internal)")
    ap.add_argument("--out")
    ap.add_argument("--reference")
    ap.add_argument("--ppl", action="store_true",
                    help="wikitext-2 perplexity of stock, plugin, triton and stock again")
    args = ap.parse_args()
    if args.one and args.ppl:
        run_one_ppl(args.one, args.out)
    elif args.one:
        run_one(args.one, args.out, args.reference)
    elif args.ppl:
        drive_ppl()
    elif args.extra:
        extra(args.extra)
    else:
        drive(args.runs)


if __name__ == "__main__":
    main()
