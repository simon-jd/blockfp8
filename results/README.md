# Results and run history

Every result file records the commit it ran at, whether the tree was dirty,
the GPU lock and nvidia-smi state (`git` and `time_utc` in its header or in
each run). The summaries are generated from them: `bench/report.py` writes
`gemm/summary.md` and the chart, `bench/report_e2e.py` writes
`e2e/summary.md`.

## Current files

"Ran at" is the commit in the file's header; "recorded in" is the commit
that added the file in its current form. Dates are UTC, from the header.

| file | ran at | date | recorded in | what it measured |
|---|---|---|---|---|
| `micro/mma_peak.json` | ba22c42 | 2026-11-01 | 84e417e | `mma.sync` peak throughput (FP8, bf16) |
| `micro/dram_bw.json` | ba22c42 | 2026-11-01 | 84e417e | DRAM read bandwidth |
| `micro/mma_precision.json` | 0a65e1d | 2026-11-03 | c388eb6 | bits the FP8 MMA keeps; promotion error vs bf16 output rounding |
| `gemm/l2_probe.json` | 907d1f6 | 2026-11-02 | 8ac4972 | what cold activations cost the GEMM after an L2 flush |
| `gemm/autotune.json` | bed9867, dirty (1) | 2026-11-03 | 5383991 | plan-table sweep on M = 1-8192 (powers of two); source of `csrc/plan_table.inc` |
| `gemm/autotune_fused.json` | bed9867, dirty (1) | 2026-11-03 | d7fc008 | fused-quantization sweep at M <= 256 against quantize + GEMM; source of `csrc/fused_plan_table.inc` |
| `gemm/triton_tuned_configs.json` | bed9867, dirty (2) | 2026-11-03 | 604eb9f | vLLM Triton configs tuned on the same powers of two |
| `gemm/plan_probe_426750a.json` | 426750a | 2026-11-03 | 86294e6 | every plan at M = 1-512 next to tuned Triton, with the plan table of that time: the automatic plan was up to 21% slower than the best compiled one (qkv at M = 512: 109.8 vs 90.8 us) |
| `gemm/plan_probe_6a2a470.json` | 6a2a470 | 2026-11-03 | 1199990 | every plan of qkv, o and down at M = 384 and down at 768, before the balanced plan choice (667fc95) |
| `gemm/qwen3_8b.json` | 35d0b77 | 2026-11-03 | 6a2a470 | the GEMM benchmark on the tuned M, cold and warm L2, 10 interleaved runs (3) |
| `gemm/qwen3_8b_offgrid.json` | 1199990 | 2026-11-03 | 2ba27bd | M = 3, 12, 24, 48, 100, 384, 768, 3000, with the balanced plan choice |
| `gemm/qwen3_8b_check_6a2a470.json` | 6a2a470 | 2026-11-03 | da328c8 | M = 320, 640, 1280, 1536, 2500, 6000 before the balanced plan choice |
| `gemm/qwen3_8b_check_667fc95.json` | 1199990 | 2026-11-03 | da328c8 | the same batch sizes with it (the plan code of 667fc95) |
| `e2e/qwen3_8b_fp8.json` | 607348f; Triton run 91d7c06 | 2026-11-04 | 91d7c06; f256fd0 | vLLM end to end: 5 interleaved stock and plugin runs, prefill, decode TPOT, outputs; one run of vLLM's W8A8 Triton path (4) |
| `e2e/wikitext2_ppl.json` | 0a65e1d; 2af08cb | 2026-11-03 | 5fab662; 36c0285 | wikitext-2 perplexity: stock, plugin, Triton and stock again at 0a65e1d (recorded in 5fab662); the plugin again at 2af08cb (recorded in 36c0285) |

(1) Recorded with the new tuning harness uncommitted; it was committed in
5383991. (2) Recorded with the tuning change uncommitted; committed in
604eb9f. (3) Recorded before the balanced plan choice (667fc95), which does
not change the plan of any tabulated M. (4) The stock and plugin runs ran at 607348f and were recorded in 91d7c06; the Triton run ran at 91d7c06, which only adds those results, so all eleven runs used the same code.

## Earlier recordings

Superseded files stay in git history (`git show <commit>:results/<file>`).
The docs compare against these:

| file | recorded in | ran at | date | what changed since |
|---|---|---|---|---|
| `gemm/qwen3_8b.json` | 3660f45 | e49ac24 | 2026-11-02 | Before the activation scales were staged in shared memory (bed9867) and the plan table was retuned in the benchmark's harness (5383991). Tuned Triton was then up to 7% faster at M = 1-16 and 2-5% faster at M = 64-256, and Marlin was faster on down; o at M = 16 took 21.1 us (19.5 now). |
| `gemm/qwen3_8b_offgrid.json` | d049515 | f74798d | 2026-11-02 | Off-grid batch sizes with the plan tables of that time. |
| `gemm/qwen3_8b_offgrid.json` | 6a2a470 | 35d0b77, dirty (the grid run had just rewritten its own file) | 2026-11-03 | Before the balanced plan choice: qkv, o and down at M = 384 took 85.8, 57.1 and 161.3 us (now 76.1, 48.7, 141.9); qkv and down at M = 768 took 139.0 and 319.7 us (now 134.7, 248.3). |
| `e2e/qwen3_8b_fp8.json` | 84a2f29 | ec74438 (Triton run: 907d1f6) | 2026-11-02 | Before the scale staging and the retuned plan tables: prefill 1.63-1.71x, decode TPOT 1.011-1.017x at batch 1-16 and 1.090x at 64. |
| `e2e/qwen3_8b_fp8.json` | acf1170 | 6a2a470 | 2026-11-03 | Ran before the balanced plan choice (667fc95): the tree stayed at 6a2a470 until after the last run, and `blockfp8/_C.so` was rebuilt only afterwards. The Triton run says dirty only because the combined file had just been rewritten. Prefill 1.65-1.75x, TPOT 1.023-1.032x at batch 1-16 and 1.113x at 64; 9 of 24 greedy plugin sequences matched stock (10 now). |
