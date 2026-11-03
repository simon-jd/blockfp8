// Host-side interface of the block-scaled FP8 GEMM (no PyTorch types here).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace blockfp8 {

// C[M, N] (bf16, row-major) = sum_kb sa[m, kb] * sb[n / 128, kb] *
//                              sum_{k in kb} A[m, k] * B[n, k]
// A [M, K] and B [N, K] are e4m3, K-contiguous. Strides are in elements.
// With a fused plan, a points at bf16 activations [M, K] (row stride lda),
// which the kernel quantizes per row per 128-wide K group (amax floored at
// eps); sa is then unused.
struct GemmProblem {
  const void* a;
  int64_t lda;
  const void* b;
  int64_t ldb;
  const float* sa;
  int64_t sa_m, sa_k;
  const float* sb;
  int64_t sb_n, sb_k;
  void* c;
  int64_t ldc;
  int M, N, K;
  float eps = 1e-10f;
};

struct Plan {
  int config;          // index into the compiled (fused, if fused) configurations
  int splits;          // K splits (1 = no split-K)
  bool fused = false;  // A is bf16 and is quantized inside the GEMM
};

int num_configs();
const char* config_name(int config);
int num_fused_configs();
const char* fused_config_name(int config);

// The built-in choice of kernel configuration and split count.
Plan choose_plan(int M, int N, int K, int num_sms);
Plan choose_fused_plan(int M, int N, int K, int num_sms);

// True where the autotuning sweep found the fused kernel faster than the
// separate quantization kernel followed by the GEMM (untuned shapes: false;
// it lost on every tuned shape, results/gemm/autotune_fused.json).
bool prefers_fused(int M, int N, int K);

// Number of CTA tiles and K tiles for a plan's configuration.
void plan_shape(const GemmProblem& p, Plan plan, int* tiles, int* k_tiles);

// Split-K scratch: fp32 partial tiles (bytes) and one int arrival counter per tile.
size_t workspace_bytes(const GemmProblem& p, Plan plan);
int counters_needed(const GemmProblem& p, Plan plan);

// counters must hold counters_needed() zeros; the kernel leaves them zero.
void launch_gemm(const GemmProblem& p, Plan plan, float* workspace, int* counters,
                 cudaStream_t stream);

// Separate activation quantization: x [M, K] bf16 (row stride ldx) to
// q [M, K] e4m3 (row stride ldq) and one float scale per row per 128-wide K
// group at s[m * s_m + kb * s_k]; K must be a multiple of 8.
struct QuantProblem {
  const void* x;
  int64_t ldx;
  void* q;
  int64_t ldq;
  float* s;
  int64_t s_m, s_k;
  int M, K;
  float eps;
};
void launch_quantize(const QuantProblem& p, cudaStream_t stream);

}  // namespace blockfp8
