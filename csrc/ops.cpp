// PyTorch operator registration: torch.ops.blockfp8.gemm, gemm_bf16 (fused
// activation quantization), quantize, linear, prefers_fused, config_names,
// fused_config_names.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "blockfp8.h"

namespace {

// Split-K arrival counters, one zeroed buffer per device. The kernel resets
// every counter it uses, so the buffer stays zero between calls. Calls that
// run concurrently on different streams of one device must not share it;
// vLLM issues linear layers on one stream. A buffer that turns out too small
// is replaced but kept alive, since captured CUDA graphs may still use it;
// the first one is large enough for every plan of the tuned shapes.
at::Tensor counters_for(const at::Device& device, int needed) {
  static std::mutex mu;
  static std::vector<at::Tensor> per_device, retired;
  std::lock_guard<std::mutex> lock(mu);
  size_t idx = device.index();
  if (per_device.size() <= idx) per_device.resize(idx + 1);
  at::Tensor& t = per_device[idx];
  if (!t.defined() || t.numel() < needed) {
    if (t.defined()) retired.push_back(t);
    int64_t size = std::max<int64_t>(needed, 1 << 16);
    t = at::zeros({size}, at::TensorOptions().dtype(at::kInt).device(device));
  }
  return t;
}

void check_weights(const at::Tensor& b, const at::Tensor& b_scales, int64_t K) {
  TORCH_CHECK(b.is_cuda() && b_scales.is_cuda(), "blockfp8: all operands must be CUDA tensors");
  TORCH_CHECK(b.scalar_type() == at::kFloat8_e4m3fn, "blockfp8: b must be float8_e4m3fn");
  TORCH_CHECK(b_scales.scalar_type() == at::kFloat, "blockfp8: scales must be float32");
  TORCH_CHECK(b.dim() == 2 && b.size(1) == K, "blockfp8: bad shapes");
  const int64_t N = b.size(0), kb = (K + 127) / 128, nb = (N + 127) / 128;
  TORCH_CHECK(K > 0 && K % 16 == 0, "blockfp8: K must be a positive multiple of 16");
  TORCH_CHECK(N > 0 && N % 8 == 0, "blockfp8: N must be a positive multiple of 8");
  TORCH_CHECK(b.stride(1) == 1 && b.stride(0) % 16 == 0,
              "blockfp8: b must be row-major with 16-element aligned rows");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0,
              "blockfp8: b must be 16-byte aligned");
  TORCH_CHECK(b_scales.dim() == 2 && b_scales.size(0) == nb && b_scales.size(1) == kb,
              "blockfp8: b_scales must be [ceil(N/128), ceil(K/128)]");
  TORCH_CHECK(N <= INT32_MAX && K <= INT32_MAX, "blockfp8: too large");
}

// Fills in B, C and the shape; the caller sets A (and its scales).
blockfp8::GemmProblem problem(const at::Tensor& b, const at::Tensor& b_scales, at::Tensor& c,
                              int64_t M) {
  blockfp8::GemmProblem p{};
  p.b = b.data_ptr();
  p.ldb = b.stride(0);
  p.sb = b_scales.data_ptr<float>();
  p.sb_n = b_scales.stride(0);
  p.sb_k = b_scales.stride(1);
  p.c = c.data_ptr();
  p.ldc = c.stride(0);
  p.M = int(M);
  p.N = int(b.size(0));
  p.K = int(b.size(1));
  return p;
}

void run(const blockfp8::GemmProblem& p, blockfp8::Plan plan, int64_t config, int64_t split_k,
         int num_configs, const at::Tensor& like) {
  if (config >= 0) {
    TORCH_CHECK(config < num_configs, "blockfp8: config out of range");
    plan.config = int(config);
    if (split_k <= 0) split_k = 1;
  }
  if (split_k > 0) plan.splits = int(split_k);

  at::Tensor workspace;
  float* ws = nullptr;
  int* counters = nullptr;
  if (size_t bytes = blockfp8::workspace_bytes(p, plan)) {
    workspace = at::empty({int64_t(bytes / sizeof(float))}, like.options().dtype(at::kFloat));
    ws = workspace.data_ptr<float>();
    counters = counters_for(like.device(), blockfp8::counters_needed(p, plan)).data_ptr<int>();
  }
  blockfp8::launch_gemm(p, plan, ws, counters, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

at::Tensor gemm(const at::Tensor& a, const at::Tensor& a_scales, const at::Tensor& b,
                const at::Tensor& b_scales, int64_t config, int64_t split_k) {
  TORCH_CHECK(a.is_cuda() && a_scales.is_cuda(), "blockfp8.gemm: all operands must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn, "blockfp8.gemm: a must be float8_e4m3fn");
  TORCH_CHECK(a_scales.scalar_type() == at::kFloat, "blockfp8.gemm: scales must be float32");
  TORCH_CHECK(a.dim() == 2, "blockfp8.gemm: bad shapes");
  const int64_t M = a.size(0), K = a.size(1), kb = (K + 127) / 128;
  check_weights(b, b_scales, K);
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) % 16 == 0,
              "blockfp8.gemm: a must be row-major with 16-element aligned rows");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "blockfp8.gemm: a must be 16-byte aligned");
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(0) == M && a_scales.size(1) == kb,
              "blockfp8.gemm: a_scales must be [M, ceil(K/128)]");
  TORCH_CHECK(M <= INT32_MAX && M * a_scales.stride(0) <= INT32_MAX,
              "blockfp8.gemm: a_scales row offsets must fit in 32 bits");

  const at::cuda::CUDAGuard guard(a.device());
  at::Tensor c = at::empty({M, b.size(0)}, a.options().dtype(at::kBFloat16));
  if (M == 0) return c;
  blockfp8::GemmProblem p = problem(b, b_scales, c, M);
  p.a = a.data_ptr();
  p.lda = a.stride(0);
  p.sa = a_scales.data_ptr<float>();
  p.sa_m = a_scales.stride(0);
  p.sa_k = a_scales.stride(1);
  run(p, blockfp8::choose_plan(p.M, p.N, p.K, num_sms()), config, split_k,
      blockfp8::num_configs(), a);
  return c;
}

// x [M, K] bf16 -> C [M, N] bf16, with x quantized per row per 128-wide K
// group inside the GEMM, bitwise as vLLM's per_token_group_quant_fp8 (eps
// = its eps) followed by gemm() would compute it.
at::Tensor gemm_bf16(const at::Tensor& x, const at::Tensor& b, const at::Tensor& b_scales,
                     int64_t config, int64_t split_k, double eps) {
  TORCH_CHECK(x.is_cuda(), "blockfp8.gemm_bf16: all operands must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "blockfp8.gemm_bf16: x must be bfloat16");
  TORCH_CHECK(x.dim() == 2, "blockfp8.gemm_bf16: bad shapes");
  const int64_t M = x.size(0), K = x.size(1);
  check_weights(b, b_scales, K);
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0,
              "blockfp8.gemm_bf16: x must be row-major with 8-element aligned rows");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "blockfp8.gemm_bf16: x must be 16-byte aligned");
  TORCH_CHECK(M <= INT32_MAX, "blockfp8.gemm_bf16: too large");
  TORCH_CHECK(eps > 0, "blockfp8.gemm_bf16: eps must be positive");

  const at::cuda::CUDAGuard guard(x.device());
  at::Tensor c = at::empty({M, b.size(0)}, x.options().dtype(at::kBFloat16));
  if (M == 0) return c;
  blockfp8::GemmProblem p = problem(b, b_scales, c, M);
  p.a = x.data_ptr();
  p.lda = x.stride(0);
  p.eps = float(eps);
  run(p, blockfp8::choose_fused_plan(p.M, p.N, p.K, num_sms()), config, split_k,
      blockfp8::num_fused_configs(), x);
  return c;
}

bool prefers_fused(int64_t M, int64_t N, int64_t K) {
  return blockfp8::prefers_fused(int(M), int(N), int(K));
}

void check_activations(const at::Tensor& x, const char* op) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2,
              op, ": x must be a 2-D bfloat16 CUDA tensor");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              op, ": x must be row-major with 16-byte aligned rows");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) % 16 == 0, op, ": K must be a positive multiple of 16");
  TORCH_CHECK(x.size(0) <= INT32_MAX && x.size(1) <= INT32_MAX, op, ": too large");
}

// x [M, K] bf16 -> (q [M, K] e4m3, scales [M, ceil(K/128)] float32 with
// M-contiguous strides (1, M)), bitwise as vLLM's per_token_group_quant_fp8.
std::tuple<at::Tensor, at::Tensor> quantize(const at::Tensor& x, double eps) {
  check_activations(x, "blockfp8.quantize");
  TORCH_CHECK(eps > 0, "blockfp8.quantize: eps must be positive");
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), K = x.size(1), kb = (K + 127) / 128;
  at::Tensor q = at::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
  at::Tensor s = at::empty({kb, M}, x.options().dtype(at::kFloat)).t();
  blockfp8::QuantProblem p{x.data_ptr(), x.stride(0), q.data_ptr(), q.stride(0),
                           s.data_ptr<float>(), s.stride(0), s.stride(1), int(M), int(K),
                           float(eps)};
  blockfp8::launch_quantize(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, s};
}

// A linear layer: fused quantization where the plan table prefers it
// (decode batches), else quantize() followed by gemm().
at::Tensor linear(const at::Tensor& x, const at::Tensor& b, const at::Tensor& b_scales) {
  check_activations(x, "blockfp8.linear");
  if (blockfp8::prefers_fused(int(x.size(0)), int(b.size(0)), int(x.size(1))))
    return gemm_bf16(x, b, b_scales, -1, 0, 1e-10);
  auto [q, s] = quantize(x, 1e-10);
  return gemm(q, s, b, b_scales, -1, 0);
}

std::string names(int n, const char* (*name)(int)) {
  std::string out;
  for (int i = 0; i < n; ++i) {
    if (i) out += ";";
    out += name(i);
  }
  return out;
}

std::string config_names() { return names(blockfp8::num_configs(), blockfp8::config_name); }
std::string fused_config_names() {
  return names(blockfp8::num_fused_configs(), blockfp8::fused_config_name);
}

}  // namespace

TORCH_LIBRARY(blockfp8, m) {
  m.def(
      "gemm(Tensor a, Tensor a_scales, Tensor b, Tensor b_scales, int config=-1, "
      "int split_k=0) -> Tensor");
  m.def(
      "gemm_bf16(Tensor x, Tensor b, Tensor b_scales, int config=-1, int split_k=0, "
      "float eps=1e-10) -> Tensor");
  m.def("quantize(Tensor x, float eps=1e-10) -> (Tensor, Tensor)");
  m.def("linear(Tensor x, Tensor b, Tensor b_scales) -> Tensor");
  m.def("prefers_fused(int m, int n, int k) -> bool", &prefers_fused);
  m.def("config_names() -> str", &config_names);
  m.def("fused_config_names() -> str", &fused_config_names);
}

TORCH_LIBRARY_IMPL(blockfp8, CUDA, m) {
  m.impl("gemm", &gemm);
  m.impl("gemm_bf16", &gemm_bf16);
  m.impl("quantize", &quantize);
  m.impl("linear", &linear);
}
