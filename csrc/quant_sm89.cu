// Per-token-group (1 x 128) activation quantization to e4m3: the separate
// kernel blockfp8.linear runs where the plan table does not fuse it into the
// GEMM. Same arithmetic as the fused quantizer in gemm_kernel.cuh and as
// vLLM's per_token_group_quant_fp8: amax over the group floored at eps,
// scale = amax / 448, q = e4m3(clamp(x / scale, -448, 448)), IEEE division.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "blockfp8.h"

namespace blockfp8 {

namespace {

constexpr int kThreads = 256;
constexpr int kLanesPerGroup = 16;  // 8 values (16 bytes) per thread

__global__ void __launch_bounds__(kThreads) quantize_kernel(QuantProblem p) {
  const int kb_count = (p.K + 127) / 128;
  const int64_t gid = (int64_t(blockIdx.x) * kThreads + threadIdx.x) / kLanesPerGroup;
  const int lane = threadIdx.x % kLanesPerGroup;
  const int64_t row = gid / kb_count;
  const int kb = int(gid % kb_count);
  const int col = kb * 128 + lane * 8;
  const bool valid = row < p.M, in = valid && col < p.K;  // K % 8 == 0

  uint4 v = make_uint4(0u, 0u, 0u, 0u);  // outside K: zeros, which leave amax alone
  if (in) v = __ldg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.x) +
                                                   row * p.ldx + col));
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
  float f[8];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 t = __bfloat1622float2(h[e]);
    f[2 * e] = t.x;
    f[2 * e + 1] = t.y;
  }
  float amax = p.eps;
#pragma unroll
  for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
  // Every thread takes part, so the 16-lane groups of a warp reduce together.
#pragma unroll
  for (int o = kLanesPerGroup / 2; o > 0; o /= 2)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float scale = amax / 448.f;

  if (in) {
    uint32_t packed[2] = {0u, 0u};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 q = make_float2(fminf(fmaxf(f[2 * e] / scale, -448.f), 448.f),
                                   fminf(fmaxf(f[2 * e + 1] / scale, -448.f), 448.f));
      const uint32_t pair = __nv_cvt_float2_to_fp8x2(q, __NV_SATFINITE, __NV_E4M3);
      packed[e / 2] |= pair << (16 * (e % 2));
    }
    *reinterpret_cast<uint2*>(static_cast<uint8_t*>(p.q) + row * p.ldq + col) =
        make_uint2(packed[0], packed[1]);
  }
  if (valid && lane == 0) p.s[row * p.s_m + int64_t(kb) * p.s_k] = scale;
}

}  // namespace

void launch_quantize(const QuantProblem& p, cudaStream_t stream) {
  const int64_t threads = int64_t(p.M) * ((p.K + 127) / 128) * kLanesPerGroup;
  const int64_t blocks = (threads + kThreads - 1) / kThreads;
  if (blocks == 0) return;
  quantize_kernel<<<dim3(unsigned(blocks)), kThreads, 0, stream>>>(p);
}

}  // namespace blockfp8
