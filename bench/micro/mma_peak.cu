// Sustained mma.sync throughput on register operands.
//
// Every warp runs independent accumulator chains of one MMA instruction for a
// fixed number of iterations; no memory traffic is involved. The B operand
// alternates between b and -b so accumulators stay bounded (FP16 ones too).
// Reported: TFLOPS from event timing, FLOP per SM per clock from clock64()
// (independent of the boost clock), and the SM clock the run achieved.
#include <cuda_fp16.h>

#include <cstdint>

#include "common.cuh"

enum Variant { E4M3_F32 = 0, E4M3_F16 = 1, BF16_F32 = 2, F16_F16 = 3 };

template <int V>
struct Mma;

template <>
struct Mma<E4M3_F32> {
  static constexpr int kFlops = 2 * 16 * 8 * 32;
  static constexpr const char* kName = "mma.m16n8k32.f32.e4m3.e4m3.f32";
  using Acc = float[4];
  __device__ static void run(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

template <>
struct Mma<E4M3_F16> {
  static constexpr int kFlops = 2 * 16 * 8 * 32;
  static constexpr const char* kName = "mma.m16n8k32.f16.e4m3.e4m3.f16";
  __device__ static void run(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
        "{%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

template <>
struct Mma<BF16_F32> {
  static constexpr int kFlops = 2 * 16 * 8 * 16;
  static constexpr const char* kName = "mma.m16n8k16.f32.bf16.bf16.f32";
  __device__ static void run(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

template <>
struct Mma<F16_F16> {
  static constexpr int kFlops = 2 * 16 * 8 * 16;
  static constexpr const char* kName = "mma.m16n8k16.f16.f16.f16.f16";
  __device__ static void run(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
        "{%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

template <int V>
struct AccOf {
  using T = float[4];
};
template <>
struct AccOf<E4M3_F16> {
  using T = uint32_t[2];
};
template <>
struct AccOf<F16_F16> {
  using T = uint32_t[2];
};

__device__ uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Operand bits: random mantissas and signs, exponents kept near 1.0.
__device__ uint32_t operand_bits(int v, uint32_t seed) {
  uint32_t r = hash32(seed);
  if (v == E4M3_F32 || v == E4M3_F16) {
    // e4m3: s eeee mmm, exponent field 6..8 -> values in [0.5, 4)
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
      uint32_t byte = (r >> (8 * i)) & 0x87u;  // sign + 3 mantissa bits
      byte |= (6u + ((r >> (8 * i + 3)) % 3u)) << 3;
      out |= byte << (8 * i);
    }
    return out;
  }
  // bf16 / fp16 pairs: random mantissa, exponent near 1.0
  uint32_t lo = r & 0x807fu, hi = (r >> 16) & 0x807fu;
  uint32_t e = (v == BF16_F32) ? 0x3f80u : 0x3c00u;
  if (v == F16_F16) {
    lo = r & 0x83ffu;
    hi = (r >> 16) & 0x83ffu;
  }
  return (lo | e) | ((hi | e) << 16);
}

constexpr int kChains = 4;

template <int V>
__global__ void mma_loop(int iters, float* sink, unsigned long long* cycles) {
  uint32_t a[4];
  for (int i = 0; i < 4; ++i) a[i] = operand_bits(V, threadIdx.x * 131u + blockIdx.x * 7u + i);
  uint32_t b0 = operand_bits(V, threadIdx.x * 17u + 1000u);
  uint32_t b1 = operand_bits(V, threadIdx.x * 19u + 2000u);
  // Negating every element flips each element's sign bit.
  uint32_t flip = (V == E4M3_F32 || V == E4M3_F16) ? 0x80808080u : 0x80008000u;
  uint32_t nb0 = b0 ^ flip, nb1 = b1 ^ flip;

  typename AccOf<V>::T acc[kChains];
  for (int c = 0; c < kChains; ++c)
    for (int j = 0; j < int(sizeof(acc[c]) / 4); ++j) reinterpret_cast<uint32_t*>(acc[c])[j] = 0;

  __syncthreads();
  unsigned long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) Mma<V>::run(acc[c], a, b0, b1);
#pragma unroll
    for (int c = 0; c < kChains; ++c) Mma<V>::run(acc[c], a, nb0, nb1);
  }
  __syncthreads();
  unsigned long long t1 = clock64();

  float s = 0.f;
  for (int c = 0; c < kChains; ++c)
    for (int j = 0; j < int(sizeof(acc[c]) / 4); ++j) s += float(reinterpret_cast<uint32_t*>(acc[c])[j] & 1u);
  if (s == -1.f) sink[0] = s;  // never true; keeps the chains alive
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

template <int V>
void bench(int sms, int blocks_per_sm, int threads, int iters, int reps, bool first) {
  int blocks = sms * blocks_per_sm;
  float* sink;
  unsigned long long* cycles;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&cycles, blocks * sizeof(unsigned long long)));
  mma_loop<V><<<blocks, threads>>>(iters / 10, sink, cycles);  // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> tflops, flop_per_clk, mhz;
  std::vector<unsigned long long> host(blocks);
  double mmas = double(blocks) * (threads / 32) * iters * 2 * kChains;
  for (int r = 0; r < reps; ++r) {
    double ms = time_ms([&] { mma_loop<V><<<blocks, threads>>>(iters, sink, cycles); });
    CUDA_CHECK(cudaMemcpy(host.data(), cycles, blocks * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    double cyc = 0;
    for (auto c : host) cyc = std::max(cyc, double(c));
    double flops = mmas * Mma<V>::kFlops;
    tflops.push_back(flops / (ms * 1e-3) / 1e12);
    flop_per_clk.push_back(flops / sms / cyc);
    mhz.push_back(cyc / (ms * 1e-3) / 1e6);
  }
  std::printf("%s{\"instruction\": \"%s\", \"blocks\": %d, \"threads\": %d, \"iters\": %d, "
              "\"chains_per_warp\": %d, \"reps\": %d, ",
              first ? "" : ",\n", Mma<V>::kName, blocks, threads, iters, 2 * kChains, reps);
  print_stats("tflops", stats_of(tflops));
  std::printf(", ");
  print_stats("flop_per_sm_per_clk", stats_of(flop_per_clk));
  std::printf(", ");
  print_stats("sm_mhz_from_clock64", stats_of(mhz));
  std::printf("}");
  CUDA_CHECK(cudaFree(sink));
  CUDA_CHECK(cudaFree(cycles));
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 10;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int sms = prop.multiProcessorCount;
  std::printf("{\"device\": \"%s\", \"sm\": \"%d%d\", \"sms\": %d, \"results\": [\n", prop.name,
              prop.major, prop.minor, sms);
  // 4 blocks x 256 threads per SM = 8 warps per SM sub-partition, each with
  // 8 MMAs in flight: enough to cover the MMA pipeline latency.
  bench<E4M3_F32>(sms, 4, 256, 20000, reps, true);
  bench<E4M3_F16>(sms, 4, 256, 20000, reps, false);
  bench<BF16_F32>(sms, 4, 256, 20000, reps, false);
  bench<F16_F16>(sms, 4, 256, 20000, reps, false);
  std::printf("\n]}\n");
  return 0;
}
