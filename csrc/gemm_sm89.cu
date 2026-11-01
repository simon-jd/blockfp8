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

template <class Cfg, bool kSplitK>
void launch_cfg(const KernelParams& kp, dim3 grid, cudaStream_t stream) {
  auto kernel = gemm_kernel<Cfg, kSplitK>;
  static bool attr_set = [&] {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmemBytes);
    return true;
  }();
  (void)attr_set;
  kernel<<<grid, Cfg::kThreads, Cfg::kSmemBytes, stream>>>(kp);
}

template <class Cfg>
void launch_any(const KernelParams& kp, dim3 grid, cudaStream_t stream) {
  if (kp.splits > 1)
    launch_cfg<Cfg, true>(kp, grid, stream);
  else
    launch_cfg<Cfg, false>(kp, grid, stream);
}

template <class Cfg>
constexpr ConfigEntry entry(const char* name) {
  return {name, Cfg::BM, Cfg::BN, Cfg::kThreads, Cfg::kSmemBytes, &launch_any<Cfg>};
}

//                     BM   BN  stages warps(M x N)
using CfgL128 = Config<128, 128, 3, 2, 4>;
using CfgM64 = Config<64, 128, 4, 2, 2>;
using CfgM32 = Config<32, 128, 4, 1, 4>;
using CfgM16 = Config<16, 128, 5, 1, 4>;
// Two CTAs per SM (48 KB of shared memory, <= 255 registers for 2 x 128 threads).
using CfgM64x2 = Config<64, 128, 2, 2, 2, 2>;
using CfgL128S2 = Config<128, 128, 2, 2, 4>;

const ConfigEntry kConfigs[] = {
    entry<CfgL128>("m128n128s3w8"),
    entry<CfgM64>("m64n128s4w4"),
    entry<CfgM32>("m32n128s4w4"),
    entry<CfgM16>("m16n128s5w4"),
    entry<CfgM64x2>("m64n128s2w4b2"),
    entry<CfgL128S2>("m128n128s2w8"),
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

int num_configs() { return kNumConfigs; }

const char* config_name(int config) {
  if (config < 0 || config >= kNumConfigs) throw std::out_of_range("bad config id");
  return kConfigs[config].name;
}

void plan_shape(const GemmProblem& p, int config, int* tiles, int* k_tiles) {
  const ConfigEntry& e = kConfigs[config];
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

Plan heuristic_plan(int M, int N, int K, int num_sms) {
  int config = M > 64 ? 0 : M > 32 ? 1 : M > 16 ? 2 : 3;
  const ConfigEntry& e = kConfigs[config];
  int tiles = cdiv(M, e.bm) * cdiv(N, e.bn);
  int k_tiles = cdiv(K, 128);
  int splits = 1;
  if (config != 0) {
    // Enough CTAs that every SM streams weights; at least 4 K tiles each.
    splits = std::max(1, std::min(cdiv(num_sms, tiles), k_tiles / 4));
  }
  return {config, splits};
}

}  // namespace

Plan choose_plan(int M, int N, int K, int num_sms) {
  // Tuned shape: the entry with the smallest tabulated M >= M (else the largest).
  const TableEntry* best = nullptr;
  for (const TableEntry& t : kPlanTable) {
    if (t.n != N || t.k != K || t.config >= kNumConfigs) continue;
    bool better = !best || (t.m >= M ? (best->m < M || t.m < best->m) : (best->m < M && t.m > best->m));
    if (better) best = &t;
  }
  if (best) return {best->config, best->splits};
  return heuristic_plan(M, N, K, num_sms);
}

namespace {

KernelParams make_params(const GemmProblem& p, Plan plan, float* workspace, int* counters) {
  const ConfigEntry& e = kConfigs[plan.config];
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
  return kp;
}

}  // namespace

size_t workspace_bytes(const GemmProblem& p, Plan plan) {
  KernelParams kp = make_params(p, plan, nullptr, nullptr);
  if (kp.splits == 1) return 0;
  const ConfigEntry& e = kConfigs[plan.config];
  return size_t(kp.splits) * kp.tiles_m * kp.tiles_n * e.bm * e.bn * sizeof(float);
}

int counters_needed(const GemmProblem& p, Plan plan) {
  KernelParams kp = make_params(p, plan, nullptr, nullptr);
  return kp.splits == 1 ? 0 : kp.tiles_m * kp.tiles_n;
}

void launch_gemm(const GemmProblem& p, Plan plan, float* workspace, int* counters,
                 cudaStream_t stream) {
  if (plan.config < 0 || plan.config >= kNumConfigs) throw std::out_of_range("bad config id");
  KernelParams kp = make_params(p, plan, workspace, counters);
  dim3 grid(kp.tiles_m * kp.tiles_n, kp.splits);
  kConfigs[plan.config].launch(kp, grid, stream);
}

}  // namespace blockfp8
