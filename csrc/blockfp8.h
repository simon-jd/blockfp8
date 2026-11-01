// Host-side interface of the block-scaled FP8 GEMM (no PyTorch types here).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace blockfp8 {

// C[M, N] (bf16, row-major) = sum_kb sa[m, kb] * sb[n / 128, kb] *
//                              sum_{k in kb} A[m, k] * B[n, k]
// A [M, K] and B [N, K] are e4m3, K-contiguous. Strides are in elements.
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
};

struct Plan {
  int config;  // index into the compiled configurations
  int splits;  // K splits (1 = no split-K)
};

int num_configs();
const char* config_name(int config);

// The built-in choice of kernel configuration and split count.
Plan choose_plan(int M, int N, int K, int num_sms);

// Number of CTA tiles and K tiles for a configuration; splits must not exceed k_tiles.
void plan_shape(const GemmProblem& p, int config, int* tiles, int* k_tiles);

// Split-K scratch: fp32 partial tiles (bytes) and one int arrival counter per tile.
size_t workspace_bytes(const GemmProblem& p, Plan plan);
int counters_needed(const GemmProblem& p, Plan plan);

// counters must hold counters_needed() zeros; the kernel leaves them zero.
void launch_gemm(const GemmProblem& p, Plan plan, float* workspace, int* counters,
                 cudaStream_t stream);

}  // namespace blockfp8
