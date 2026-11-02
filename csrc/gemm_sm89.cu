// Compiled kernel configurations, the split-K plan and the launcher.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "gemm_kernel.cuh"

namespace blockfp8 {

namespace {

struct ConfigEntry {
  const char* name;
  int bm, bn, threads, smem;
  void (*launch)(const KernelParams&, dim3, cudaStream_t);
};

template <class Cfg, bool kSplitK, bool kFuse>
void launch_cfg(const KernelParams& kp, dim3 grid, cudaStream_t stream) {
  auto kernel = gemm_kernel<Cfg, kSplitK, kFuse>;
  constexpr int smem = kFuse ? Cfg::kSmemBytesFused : Cfg::kSmemBytes;
  static bool attr_set = [&] {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    return true;
  }();
  (void)attr_set;
  kernel<<<grid, Cfg::kThreads, smem, stream>>>(kp);
}

template <class Cfg, bool kFuse>
void launch_any(const KernelParams& kp, dim3 grid, cudaStream_t stream) {
  if (kp.splits > 1)
    launch_cfg<Cfg, true, kFuse>(kp, grid, stream);
  else
    launch_cfg<Cfg, false, kFuse>(kp, grid, stream);
}

template <class Cfg, bool kFuse = false>
constexpr ConfigEntry entry(const char* name) {
  static_assert((kFuse ? Cfg::kSmemBytesFused : Cfg::kSmemBytes) <= 99 * 1024,
                "more shared memory than one sm_89 CTA can have");
  return {name, Cfg::BM, Cfg::BN, Cfg::kThreads,
          kFuse ? Cfg::kSmemBytesFused : Cfg::kSmemBytes, &launch_any<Cfg, kFuse>};
}

//                     BM   BN  stages warps(M x N)
using CfgL128 = Config<128, 128, 3, 2, 4>;
using CfgM64 = Config<64, 128, 4, 2, 2>;
using CfgM32 = Config<32, 128, 4, 1, 4>;
using CfgM16 = Config<16, 128, 5, 1, 4>;
// Two CTAs per SM (48 KB of shared memory, <= 255 registers for 2 x 128 threads).
using CfgM64x2 = Config<64, 128, 2, 2, 2, 2>;
using CfgL128S2 = Config<128, 128, 2, 2, 4>;
// 64-column tiles: twice as many CTAs per weight block row, so mid-size M
// fills the GPU without split-K and without every CTA re-reading wide tiles.
using CfgM16N64x2 = Config<16, 64, 4, 1, 4, 2>;
using CfgM32N64x2 = Config<32, 64, 4, 2, 2, 2>;
using CfgM64N64x2 = Config<64, 64, 3, 2, 2, 2>;
using CfgL128N64 = Config<128, 64, 3, 2, 2>;
using CfgL128N64x2 = Config<128, 64, 2, 2, 2, 2>;
using CfgL128N64W8 = Config<128, 64, 3, 4, 2>;

const ConfigEntry kConfigs[] = {
    entry<CfgL128>("m128n128s3w8"),
    entry<CfgM64>("m64n128s4w4"),
    entry<CfgM32>("m32n128s4w4"),
    entry<CfgM16>("m16n128s5w4"),
    entry<CfgM64x2>("m64n128s2w4b2"),
    entry<CfgL128S2>("m128n128s2w8"),
    entry<CfgM16N64x2>("m16n64s4w4b2"),
    entry<CfgM32N64x2>("m32n64s4w4b2"),
    entry<CfgM64N64x2>("m64n64s3w4b2"),
    entry<CfgL128N64>("m128n64s3w4"),
    entry<CfgL128N64x2>("m128n64s2w4b2"),
    entry<CfgL128N64W8>("m128n64s3w8"),
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

// Fused-quantization configurations: the small-M tiles above (only a float
// scale per row and stage is added to their shared memory), plus a deeper
// 16x64 pipeline with one CTA per SM.
using FCfgM16N64S8 = Config<16, 64, 8, 1, 4>;

const ConfigEntry kFusedConfigs[] = {
    entry<CfgM16, true>("m16n128s5w4"),      entry<CfgM16N64x2, true>("m16n64s4w4b2"),
    entry<FCfgM16N64S8, true>("m16n64s8w4"), entry<CfgM32, true>("m32n128s4w4"),
    entry<CfgM32N64x2, true>("m32n64s4w4b2"),
};
constexpr int kNumFusedConfigs = sizeof(kFusedConfigs) / sizeof(kFusedConfigs[0]);

int cdiv(int a, int b) { return (a + b - 1) / b; }

const ConfigEntry& entry_of(Plan plan) {
  const int n = plan.fused ? kNumFusedConfigs : kNumConfigs;
  if (plan.config < 0 || plan.config >= n) throw std::out_of_range("bad config id");
  return plan.fused ? kFusedConfigs[plan.config] : kConfigs[plan.config];
}

}  // namespace

int num_configs() { return kNumConfigs; }
int num_fused_configs() { return kNumFusedConfigs; }

const char* config_name(int config) { return entry_of({config, 1, false}).name; }
const char* fused_config_name(int config) { return entry_of({config, 1, true}).name; }

void plan_shape(const GemmProblem& p, Plan plan, int* tiles, int* k_tiles) {
  const ConfigEntry& e = entry_of(plan);
  *tiles = cdiv(p.M, e.bm) * cdiv(p.N, e.bn);
  *k_tiles = cdiv(p.K, 128);
}

namespace {

struct TableEntry {
  int n, k, m, config, splits;
};

// Winners of the autotuning sweep (bench/autotune.py) on the RTX 4090.
const TableEntry kPlanTable[] = {
#include "plan_table.inc"
    {0, 0, 0, 0, 0}};

// Fused-quantization winners, and whether each beat the separate
// quantization kernel + the GEMM above (bench/autotune.py --fused).
struct FusedEntry {
  int n, k, m, config, splits, wins;
};
const FusedEntry kFusedTable[] = {
#include "fused_plan_table.inc"
    {0, 0, 0, 0, 0, 0}};

// Tuned shape: the entry with the smallest tabulated M >= M (else the largest).
template <class E>
const E* lookup(const E* table, int n_entries, int num_cfgs, int M, int N, int K) {
  const E* best = nullptr;
  for (int i = 0; i < n_entries; ++i) {
    const E& t = table[i];
    if (t.n != N || t.k != K || t.config >= num_cfgs) continue;
    bool better = !best || (t.m >= M ? (best->m < M || t.m < best->m) : (best->m < M && t.m > best->m));
    if (better) best = &t;
  }
  return best;
}

template <class E, size_t n>
constexpr int entries(const E (&)[n]) {
  return int(n) - 1;  // without the terminator
}

Plan heuristic_plan(int M, int N, int K, int num_sms, bool fused) {
  // Fused: m16n128 (0) or m32n128 (3); otherwise by M.
  int config = fused ? (M > 16 ? 3 : 0) : M > 64 ? 0 : M > 32 ? 1 : M > 16 ? 2 : 3;
  const ConfigEntry& e = entry_of({config, 1, fused});
  int tiles = cdiv(M, e.bm) * cdiv(N, e.bn);
  int k_tiles = cdiv(K, 128);
  int splits = 1;
  if (fused || config != 0) {
    // Enough CTAs that every SM streams weights; at least 4 K tiles each.
    splits = std::max(1, std::min(cdiv(num_sms, tiles), k_tiles / 4));
  }
  return {config, splits, fused};
}

}  // namespace

Plan choose_plan(int M, int N, int K, int num_sms) {
  if (const TableEntry* t = lookup(kPlanTable, entries(kPlanTable), kNumConfigs, M, N, K))
    return {t->config, t->splits, false};
  return heuristic_plan(M, N, K, num_sms, false);
}

Plan choose_fused_plan(int M, int N, int K, int num_sms) {
  if (const FusedEntry* t = lookup(kFusedTable, entries(kFusedTable), kNumFusedConfigs, M, N, K))
    if (M <= t->m) return {t->config, t->splits, true};
  return heuristic_plan(M, N, K, num_sms, true);
}

bool prefers_fused(int M, int N, int K) {
  const FusedEntry* t = lookup(kFusedTable, entries(kFusedTable), kNumFusedConfigs, M, N, K);
  if (!t) return M <= 16;
  return M <= t->m && t->wins;
}

namespace {

KernelParams make_params(const GemmProblem& p, Plan plan, float* workspace, int* counters) {
  const ConfigEntry& e = entry_of(plan);
  KernelParams kp{};
  kp.p = p;
  kp.tiles_m = cdiv(p.M, e.bm);
  kp.tiles_n = cdiv(p.N, e.bn);
  kp.k_tiles = cdiv(p.K, 128);
  int splits = std::max(1, std::min(plan.splits, kp.k_tiles));
  kp.k_tiles_per_split = cdiv(kp.k_tiles, splits);
  kp.splits = cdiv(kp.k_tiles, kp.k_tiles_per_split);  // no empty splits
  kp.group_m = 8;
  kp.workspace = workspace;
  kp.counters = counters;
  kp.eps = p.eps;
  return kp;
}

}  // namespace

size_t workspace_bytes(const GemmProblem& p, Plan plan) {
  KernelParams kp = make_params(p, plan, nullptr, nullptr);
  if (kp.splits == 1) return 0;
  const ConfigEntry& e = entry_of(plan);
  return size_t(kp.splits) * kp.tiles_m * kp.tiles_n * e.bm * e.bn * sizeof(float);
}

int counters_needed(const GemmProblem& p, Plan plan) {
  KernelParams kp = make_params(p, plan, nullptr, nullptr);
  return kp.splits == 1 ? 0 : kp.tiles_m * kp.tiles_n;
}

void launch_gemm(const GemmProblem& p, Plan plan, float* workspace, int* counters,
                 cudaStream_t stream) {
  KernelParams kp = make_params(p, plan, workspace, counters);
  dim3 grid(kp.tiles_m * kp.tiles_n, kp.splits);
  entry_of(plan).launch(kp, grid, stream);
}

}  // namespace blockfp8
