// PyTorch operator registration: torch.ops.blockfp8.gemm / config_names.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <mutex>
#include <string>
#include <vector>

#include "blockfp8.h"

namespace {

// Split-K arrival counters, one zeroed buffer per device. The kernel resets
// every counter it uses, so the buffer stays zero between calls. Calls that
// run concurrently on different streams of one device must not share it;
// vLLM issues linear layers on one stream.
at::Tensor counters_for(const at::Device& device, int needed) {
  static std::mutex mu;
  static std::vector<at::Tensor> per_device;
  std::lock_guard<std::mutex> lock(mu);
  size_t idx = device.index();
  if (per_device.size() <= idx) per_device.resize(idx + 1);
  at::Tensor& t = per_device[idx];
  if (!t.defined() || t.numel() < needed) {
    int64_t size = std::max<int64_t>(needed, 4096);
    t = at::zeros({size}, at::TensorOptions().dtype(at::kInt).device(device));
  }
  return t;
}

at::Tensor gemm(const at::Tensor& a, const at::Tensor& a_scales, const at::Tensor& b,
                const at::Tensor& b_scales, int64_t config, int64_t split_k) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a_scales.is_cuda() && b_scales.is_cuda(),
              "blockfp8.gemm: all operands must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && b.scalar_type() == at::kFloat8_e4m3fn,
              "blockfp8.gemm: a and b must be float8_e4m3fn");
  TORCH_CHECK(a_scales.scalar_type() == at::kFloat && b_scales.scalar_type() == at::kFloat,
              "blockfp8.gemm: scales must be float32");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1), "blockfp8.gemm: bad shapes");
  const int64_t M = a.size(0), N = b.size(0), K = a.size(1);
  const int64_t kb = (K + 127) / 128, nb = (N + 127) / 128;
  TORCH_CHECK(K > 0 && K % 16 == 0, "blockfp8.gemm: K must be a positive multiple of 16");
  TORCH_CHECK(N > 0 && N % 8 == 0, "blockfp8.gemm: N must be a positive multiple of 8");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1 && a.stride(0) % 16 == 0 &&
                  b.stride(0) % 16 == 0,
              "blockfp8.gemm: a and b must be row-major with 16-element aligned rows");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0,
              "blockfp8.gemm: a and b must be 16-byte aligned");
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(0) == M && a_scales.size(1) == kb,
              "blockfp8.gemm: a_scales must be [M, ceil(K/128)]");
  TORCH_CHECK(b_scales.dim() == 2 && b_scales.size(0) == nb && b_scales.size(1) == kb,
              "blockfp8.gemm: b_scales must be [ceil(N/128), ceil(K/128)]");
  TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX, "blockfp8.gemm: too large");
  TORCH_CHECK(M * a_scales.stride(0) <= INT32_MAX,
              "blockfp8.gemm: a_scales row offsets must fit in 32 bits");

  const at::cuda::CUDAGuard guard(a.device());
  at::Tensor c = at::empty({M, N}, a.options().dtype(at::kBFloat16));
  if (M == 0) return c;

  blockfp8::GemmProblem p{};
  p.a = a.data_ptr();
  p.lda = a.stride(0);
  p.b = b.data_ptr();
  p.ldb = b.stride(0);
  p.sa = a_scales.data_ptr<float>();
  p.sa_m = a_scales.stride(0);
  p.sa_k = a_scales.stride(1);
  p.sb = b_scales.data_ptr<float>();
  p.sb_n = b_scales.stride(0);
  p.sb_k = b_scales.stride(1);
  p.c = c.data_ptr();
  p.ldc = c.stride(0);
  p.M = int(M);
  p.N = int(N);
  p.K = int(K);

  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  blockfp8::Plan plan = blockfp8::choose_plan(p.M, p.N, p.K, sms);
  if (config >= 0) {
    TORCH_CHECK(config < blockfp8::num_configs(), "blockfp8.gemm: config out of range");
    plan.config = int(config);
    if (split_k <= 0) split_k = 1;
  }
  if (split_k > 0) plan.splits = int(split_k);

  at::Tensor workspace;
  float* ws = nullptr;
  int* counters = nullptr;
  if (size_t bytes = blockfp8::workspace_bytes(p, plan)) {
    workspace = at::empty({int64_t(bytes / sizeof(float))}, a.options().dtype(at::kFloat));
    ws = workspace.data_ptr<float>();
    counters = counters_for(a.device(), blockfp8::counters_needed(p, plan)).data_ptr<int>();
  }
  blockfp8::launch_gemm(p, plan, ws, counters, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

std::string config_names() {
  std::string out;
  for (int i = 0; i < blockfp8::num_configs(); ++i) {
    if (i) out += ";";
    out += blockfp8::config_name(i);
  }
  return out;
}

}  // namespace

TORCH_LIBRARY(blockfp8, m) {
  m.def(
      "gemm(Tensor a, Tensor a_scales, Tensor b, Tensor b_scales, int config=-1, "
      "int split_k=0) -> Tensor");
  m.def("config_names() -> str", &config_names);
}

TORCH_LIBRARY_IMPL(blockfp8, CUDA, m) { m.impl("gemm", &gemm); }
