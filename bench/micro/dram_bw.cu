// Memory bandwidth: read-only, write-only and copy kernels (16-byte accesses,
// grid-stride loops) over working sets from 8 MB to 1 GB.
//
// "warm" repeats the kernel back to back, so working sets that fit in the
// 72 MB L2 are served from L2. "cold" reads a separate 256 MB buffer before
// every timed launch, which evicts the working set without leaving dirty
// lines behind; only the measured kernel is timed. Decode GEMMs read their weights once per call, so the cold read
// figure is the roofline for them; the warm figure applies when a layer's
// weights are still in L2 from the previous call.
#include <cstdint>

#include "common.cuh"

__global__ void read_kernel(const int4* __restrict__ src, size_t n, int* sink) {
  int acc = 0;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    int4 v0 = src[i], v1 = src[i + stride], v2 = src[i + 2 * stride], v3 = src[i + 3 * stride];
    acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
    acc ^= v2.x ^ v2.y ^ v2.z ^ v2.w ^ v3.x ^ v3.y ^ v3.z ^ v3.w;
  }
  for (; i < n; i += stride) {
    int4 v = src[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x7fffffff) sink[0] = acc;  // keeps the loads; practically never stores
}

__global__ void write_kernel(int4* __restrict__ dst, size_t n) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  int4 v = make_int4(1, 2, 3, 4);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = v;
}

__global__ void copy_kernel(const int4* __restrict__ src, int4* __restrict__ dst, size_t n) {
  size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + stride < n; i += 2 * stride) {
    int4 v0 = src[i], v1 = src[i + stride];
    dst[i] = v0;
    dst[i + stride] = v1;
  }
  for (; i < n; i += stride) dst[i] = src[i];
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int sms = prop.multiProcessorCount;
  const int threads = 256;
  const int blocks = sms * (prop.maxThreadsPerMultiProcessor / threads);
  const size_t max_bytes = size_t(1) << 30;
  const size_t flush_bytes = size_t(256) << 20;

  int4 *a, *b, *flush;
  int* sink;
  CUDA_CHECK(cudaMalloc(&a, max_bytes));
  CUDA_CHECK(cudaMalloc(&b, max_bytes));
  CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
  CUDA_CHECK(cudaMemset(a, 1, max_bytes));
  CUDA_CHECK(cudaMemset(b, 0, max_bytes));

  std::printf("{\"device\": \"%s\", \"sms\": %d, \"l2_bytes\": %d, \"blocks\": %d, "
              "\"threads\": %d, \"reps\": %d, \"results\": [\n",
              prop.name, sms, prop.l2CacheSize, blocks, threads, reps);
  bool first = true;
  for (size_t mb : {8, 16, 32, 64, 128, 256, 512, 1024}) {
    size_t bytes = mb << 20, n = bytes / sizeof(int4);
    for (int kind = 0; kind < 3; ++kind) {
      const char* kname = kind == 0 ? "read" : kind == 1 ? "write" : "copy";
      double moved = kind == 2 ? 2.0 * bytes : double(bytes);
      for (int cold = 0; cold < 2; ++cold) {
        auto launch = [&] {
          if (kind == 0) read_kernel<<<blocks, threads>>>(a, n, sink);
          if (kind == 1) write_kernel<<<blocks, threads>>>(b, n);
          if (kind == 2) copy_kernel<<<blocks, threads>>>(a, b, n);
        };
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> gbs;
        for (int r = 0; r < reps; ++r) {
          if (cold) read_kernel<<<blocks, threads>>>(flush, flush_bytes / sizeof(int4), sink);
          gbs.push_back(moved / (time_ms(launch) * 1e-3) / 1e9);
        }
        std::printf("%s{\"kind\": \"%s\", \"l2\": \"%s\", \"working_set_mb\": %zu, ",
                    first ? "" : ",\n", kname, cold ? "cold" : "warm", mb);
        print_stats("gb_per_s", stats_of(gbs));
        std::printf("}");
        first = false;
      }
    }
  }
  std::printf("\n]}\n");
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(flush));
  return 0;
}
