// Effective accumulation precision of Ada's FP8 mma.sync (m16n8k32, e4m3).
//
// Each test case is one warp computing D = A[16 x K] * B[8 x K]^T + C with
// K = 32 * L, as a chain of L mma instructions. With promote_every = P > 0 the
// chain runs in a separate accumulator that is added into an FP32 register
// accumulator on the CUDA cores every P instructions (the GEMM's two-level
// accumulation); P = 0 keeps the whole sum inside the MMA accumulator.
//
// The program prints raw observations as JSON; bench/micro/run_micro.py turns
// them into the summary (retained bits, rounding behaviour, error vs K).
// Every e4m3 x e4m3 product is exact, so the exact result is a short sum that
// the host evaluates in long double.
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

#include "common.cuh"

// ---------------------------------------------------------------- e4m3 codes
static double e4m3_value(uint8_t c) {
  int s = c >> 7, e = (c >> 3) & 0xf, m = c & 7;
  double v = e == 0 ? std::ldexp(m, -9) : std::ldexp(8 + m, e - 10);
  return s ? -v : v;
}
static bool e4m3_finite(uint8_t c) { return (c & 0x7f) != 0x7f; }

static uint8_t e4m3_exact(double v) {
  for (int c = 0; c < 256; ++c)
    if (e4m3_finite(uint8_t(c)) && e4m3_value(uint8_t(c)) == v) return uint8_t(c);
  std::fprintf(stderr, "%g is not an e4m3 value\n", v);
  std::exit(1);
}

struct E4m3Table {
  std::vector<std::pair<double, uint8_t>> sorted;  // all finite values, ascending
  E4m3Table() {
    for (int c = 0; c < 256; ++c)
      if (e4m3_finite(uint8_t(c))) sorted.push_back({e4m3_value(uint8_t(c)), uint8_t(c)});
    std::sort(sorted.begin(), sorted.end());
  }
  uint8_t nearest(double v) const {
    auto it = std::lower_bound(sorted.begin(), sorted.end(), std::make_pair(v, uint8_t(0)));
    if (it == sorted.end()) return sorted.back().second;
    if (it == sorted.begin()) return it->second;
    auto lo = it - 1;
    return (v - lo->first) <= (it->first - v) ? lo->second : it->second;
  }
  // Two positive e4m3 codes whose product is exactly target (> 0).
  void product_pair(double target, uint8_t& ca, uint8_t& cb) const {
    for (auto& [va, a] : sorted) {
      if (va <= 0) continue;
      double vb = target / va;
      auto it = std::lower_bound(sorted.begin(), sorted.end(), std::make_pair(vb, uint8_t(0)));
      if (it != sorted.end() && it->first == vb && va * vb == target) {
        ca = a, cb = it->second;
        return;
      }
    }
    std::fprintf(stderr, "%g is not a product of two e4m3 values\n", target);
    std::exit(1);
  }
};
static const E4m3Table kE4m3;

// ---------------------------------------------------------------- device side
__device__ __forceinline__ void mma_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
      "{%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float2 unpack_half2(uint32_t v) {
  return __half22float2(*reinterpret_cast<__half2*>(&v));
}

// One warp per case. A: [16][K] bytes, B: [8][K] bytes (n-major), C, D: [16][8].
// Fragment layouts follow the PTX ISA for mma.m16n8k32 with 8-bit operands.
template <bool kF16Acc>
__global__ void mma_cases(const uint8_t* A, const uint8_t* B, const float* C, float* D, int L,
                          int P) {
  int lane = threadIdx.x, g = lane >> 2, t = lane & 3, K = 32 * L;
  const uint8_t* a_ = A + size_t(blockIdx.x) * 16 * K;
  const uint8_t* b_ = B + size_t(blockIdx.x) * 8 * K;
  const float* c_ = C + blockIdx.x * 128;
  float acc[4] = {c_[g * 8 + 2 * t], c_[g * 8 + 2 * t + 1], c_[(g + 8) * 8 + 2 * t],
                  c_[(g + 8) * 8 + 2 * t + 1]};
  float part[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t h[2] = {0u, 0u};
  if (kF16Acc && P == 0) h[0] = pack_half2(acc[0], acc[1]), h[1] = pack_half2(acc[2], acc[3]);

  for (int i = 0; i < L; ++i) {
    int k0 = 32 * i + 4 * t;
    uint32_t a[4] = {ld4(a_ + g * K + k0), ld4(a_ + (g + 8) * K + k0), ld4(a_ + g * K + k0 + 16),
                     ld4(a_ + (g + 8) * K + k0 + 16)};
    uint32_t b[2] = {ld4(b_ + g * K + k0), ld4(b_ + g * K + k0 + 16)};
    bool promote = P > 0 && ((i + 1) % P == 0 || i == L - 1);
    if (!kF16Acc) {
      if (P == 0) {
        mma_f32(acc, a, b);
      } else {
        mma_f32(part, a, b);
        if (promote)
          for (int j = 0; j < 4; ++j) acc[j] += part[j], part[j] = 0.f;
      }
    } else {
      mma_f16(h, a, b);
      if (promote) {
        float2 lo = unpack_half2(h[0]), hi = unpack_half2(h[1]);
        acc[0] += lo.x, acc[1] += lo.y, acc[2] += hi.x, acc[3] += hi.y;
        h[0] = h[1] = 0u;
      }
    }
  }
  if (kF16Acc && P == 0) {
    float2 lo = unpack_half2(h[0]), hi = unpack_half2(h[1]);
    acc[0] = lo.x, acc[1] = lo.y, acc[2] = hi.x, acc[3] = hi.y;
  }
  float* d_ = D + blockIdx.x * 128;
  d_[g * 8 + 2 * t] = acc[0];
  d_[g * 8 + 2 * t + 1] = acc[1];
  d_[(g + 8) * 8 + 2 * t] = acc[2];
  d_[(g + 8) * 8 + 2 * t + 1] = acc[3];
}

// ---------------------------------------------------------------- host side
struct Batch {
  int L, cases;
  std::vector<uint8_t> A, B;
  std::vector<float> C;
  Batch(int L_, int cases_)
      : L(L_), cases(cases_), A(size_t(cases_) * 16 * 32 * L_, 0), B(size_t(cases_) * 8 * 32 * L_, 0),
        C(size_t(cases_) * 128, 0.f) {}
  int K() const { return 32 * L; }
  uint8_t& a(int c, int m, int k) { return A[(size_t(c) * 16 + m) * K() + k]; }
  uint8_t& b(int c, int n, int k) { return B[(size_t(c) * 8 + n) * K() + k]; }
  float& cin(int c, int m, int n) { return C[size_t(c) * 128 + m * 8 + n]; }

  // Exact D and the sum of |products| + |C| per output, in long double.
  void exact(std::vector<long double>& d, std::vector<long double>& mag) {
    d.assign(size_t(cases) * 128, 0.0L);
    mag.assign(size_t(cases) * 128, 0.0L);
    for (int c = 0; c < cases; ++c)
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
          long double s = cin(c, m, n), sa = std::fabs(cin(c, m, n));
          for (int k = 0; k < K(); ++k) {
            long double p = (long double)e4m3_value(a(c, m, k)) * e4m3_value(b(c, n, k));
            s += p;
            sa += std::fabs(p);
          }
          d[size_t(c) * 128 + m * 8 + n] = s;
          mag[size_t(c) * 128 + m * 8 + n] = sa;
        }
  }
  // Sequential FP32 sum on the host, the CUDA-core alternative.
  std::vector<float> host_fp32() {
    std::vector<float> d(size_t(cases) * 128);
    for (int c = 0; c < cases; ++c)
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
          float s = cin(c, m, n);
          for (int k = 0; k < K(); ++k) s += float(e4m3_value(a(c, m, k)) * e4m3_value(b(c, n, k)));
          d[size_t(c) * 128 + m * 8 + n] = s;
        }
    return d;
  }

  std::vector<float> run(bool f16acc, int P) {
    uint8_t *dA, *dB;
    float *dC, *dD;
    CUDA_CHECK(cudaMalloc(&dA, A.size()));
    CUDA_CHECK(cudaMalloc(&dB, B.size()));
    CUDA_CHECK(cudaMalloc(&dC, C.size() * 4));
    CUDA_CHECK(cudaMalloc(&dD, C.size() * 4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice));
    if (f16acc)
      mma_cases<true><<<cases, 32>>>(dA, dB, dC, dD, L, P);
    else
      mma_cases<false><<<cases, 32>>>(dA, dB, dC, dD, L, P);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> D(C.size());
    CUDA_CHECK(cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    return D;
  }
};

static void print_vec(const char* key, const std::vector<double>& v) {
  std::printf("\"%s\": [", key);
  for (size_t i = 0; i < v.size(); ++i) {
    if (std::isfinite(v[i]))
      std::printf("%s%.17g", i ? ", " : "", v[i]);
    else
      std::printf("%snull", i ? ", " : "");
  }
  std::printf("]");
}

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("{\"device\": \"%s\",\n", prop.name);
  std::mt19937_64 rng(1234);

  // 1. Fragment-layout sanity: small integers sum exactly at any precision.
  {
    Batch bt(1, 64);
    std::uniform_int_distribution<int> small(-4, 4);
    for (int c = 0; c < bt.cases; ++c) {
      for (int m = 0; m < 16; ++m)
        for (int k = 0; k < 32; ++k) bt.a(c, m, k) = e4m3_exact(small(rng));
      for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 32; ++k) bt.b(c, n, k) = e4m3_exact(small(rng));
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) bt.cin(c, m, n) = float(small(rng));
    }
    std::vector<long double> ex, mag;
    bt.exact(ex, mag);
    for (int f16 = 0; f16 < 2; ++f16) {
      auto d = bt.run(f16, 0);
      int bad = 0;
      for (size_t i = 0; i < d.size(); ++i) bad += (long double)d[i] != ex[i];
      std::printf("\"layout_sanity_%s_mismatches\": %d,\n", f16 ? "f16acc" : "f32acc", bad);
    }
  }

  // 2-5. One large product 2^16 plus small terms 2^(16-j), row 0 / column 0.
  //  single_k1:   small term in the same mma, at k = 1
  //  single_k31:  small term at k = 31
  //  c_big:       the 2^16 comes in through the accumulator C, the small term is a product
  //  c_small:     the small term comes in through C, the 2^16 is a product
  //  many:        31 small products next to the large one
  //  s=<x>:       small product is x * 2^(16-j), for rounding behaviour
  const int J = 35;
  struct Exp {
    const char* name;
    int mode;
    double s;
  };
  const Exp exps[] = {{"single_k1", 0, 1.0},  {"single_k31", 1, 1.0}, {"c_big", 2, 1.0},
                      {"c_small", 3, 1.0},    {"many", 4, 1.0},      {"s=+1.5", 0, 1.5},
                      {"s=+1.75", 0, 1.75},   {"s=-1", 0, -1.0},     {"s=-1.5", 0, -1.5},
                      {"s=-1.75", 0, -1.75}};
  for (int f16 = 0; f16 < 2; ++f16) {
    std::printf("\"alignment_%s\": {\"j\": [", f16 ? "f16acc" : "f32acc");
    for (int j = 0; j < J; ++j) std::printf("%s%d", j ? ", " : "", j);
    std::printf("]");
    // FP16 accumulators overflow at 2^16; use 2^8 as the large term there.
    int big = f16 ? 8 : 16;
    for (const Exp& e : exps) {
      Batch bt(1, J);
      for (int j = 0; j < J; ++j) {
        uint8_t x, y, u, v;
        kE4m3.product_pair(std::ldexp(1.0, big), x, y);
        // Smallest product of two e4m3 values is 2^-18; s * 2^-18 needs a bit more.
        int es = std::max(big - j, std::fabs(e.s) == 1.0 ? -18 : -16);
        double small = std::ldexp(std::fabs(e.s), es);
        kE4m3.product_pair(small, u, v);
        uint8_t su = u | (e.s < 0 ? 0x80 : 0);
        switch (e.mode) {
          case 0:
          case 1: {
            int ks = e.mode == 0 ? 1 : 31;
            bt.a(j, 0, 0) = x, bt.b(j, 0, 0) = y;
            bt.a(j, 0, ks) = su, bt.b(j, 0, ks) = v;
            break;
          }
          case 2:
            bt.cin(j, 0, 0) = float(std::ldexp(1.0, big));
            bt.a(j, 0, 1) = su, bt.b(j, 0, 1) = v;
            break;
          case 3:
            bt.cin(j, 0, 0) = float(small);
            bt.a(j, 0, 0) = x, bt.b(j, 0, 0) = y;
            break;
          case 4:
            bt.a(j, 0, 0) = x, bt.b(j, 0, 0) = y;
            for (int k = 1; k < 32; ++k) bt.a(j, 0, k) = su, bt.b(j, 0, k) = v;
            break;
        }
      }
      std::vector<long double> ex, mag;
      bt.exact(ex, mag);
      auto d = bt.run(f16, 0);
      std::vector<double> got, want;
      for (int j = 0; j < J; ++j) {
        got.push_back(double(d[size_t(j) * 128] - std::ldexp(1.0, big)));
        want.push_back(double(ex[size_t(j) * 128] - std::ldexp(1.0L, big)));
      }
      std::printf(", \"%s\": {", e.name);
      print_vec("observed_minus_big", got);
      std::printf(", ");
      print_vec("exact_minus_big", want);
      std::printf("}");
    }
    std::printf("},\n");
  }

  // 6-7. Random operands: one mma (K = 32) and chains up to K = 4096 with
  // different promotion intervals. Values are N(0, 7^2) samples rounded to the
  // nearest e4m3: small enough that FP16 accumulators do not overflow over
  // K = 4096 (with the full e4m3 range they would), so both kinds compare.
  std::normal_distribution<double> gauss(0.0, 1.0);
  auto fill_gauss = [&](Batch& bt) {
    for (auto& x : bt.A) x = kE4m3.nearest(gauss(rng) * 7.0);
    for (auto& x : bt.B) x = kE4m3.nearest(gauss(rng) * 7.0);
  };
  struct Run {
    int L, P;
    bool f16;
  };
  const Run runs[] = {{1, 0, false},   {1, 0, true},     {4, 0, false},   {4, 1, false},
                      {32, 0, false},  {32, 1, false},   {32, 4, false},  {128, 0, false},
                      {128, 1, false}, {128, 2, false},  {128, 4, false}, {128, 8, false},
                      {128, 32, false}, {128, 1, true},  {128, 4, true},  {128, 0, true}};
  std::printf("\"random\": [\n");
  bool first = true;
  int last_L = -1;
  Batch* bt = nullptr;
  std::vector<long double> ex, mag;
  std::vector<float> host;
  for (const Run& r : runs) {
    if (r.L != last_L) {
      delete bt;
      bt = new Batch(r.L, r.L == 1 ? 1024 : 128);
      fill_gauss(*bt);
      bt->exact(ex, mag);
      host = bt->host_fp32();
      last_L = r.L;
    }
    auto d = bt->run(r.f16, r.P);
    std::vector<double> rel_mag, rel_exact, host_rel_mag;
    for (size_t i = 0; i < d.size(); ++i) {
      rel_mag.push_back(double(std::fabs(d[i] - ex[i]) / mag[i]));
      if (std::fabs(ex[i]) > 0) rel_exact.push_back(double(std::fabs(d[i] - ex[i]) / std::fabs(ex[i])));
      host_rel_mag.push_back(double(std::fabs(host[i] - ex[i]) / mag[i]));
    }
    std::printf("%s{\"K\": %d, \"promote_every_mma\": %d, \"accumulator\": \"%s\", \"outputs\": %zu, ",
                first ? "" : ",\n", 32 * r.L, r.P, r.f16 ? "f16" : "f32", d.size());
    print_stats("err_over_abs_sum", stats_of(rel_mag));
    std::printf(", ");
    print_stats("err_over_abs_exact", stats_of(rel_exact));
    std::printf(", ");
    print_stats("host_fp32_sequential_err_over_abs_sum", stats_of(host_rel_mag));
    std::printf("}");
    first = false;
  }
  delete bt;
  std::printf("\n]}\n");
  return 0;
}
