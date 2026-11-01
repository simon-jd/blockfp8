// Shared helpers for the microbenchmarks: error checking, event timing, stats.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err_ = (expr);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

struct Stats {
  double median, p25, p75, min, max;
};

inline Stats stats_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto q = [&](double f) {
    double pos = f * (v.size() - 1);
    size_t lo = static_cast<size_t>(pos);
    size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
  };
  return {q(0.5), q(0.25), q(0.75), v.front(), v.back()};
}

// Times one launch of f() on the default stream with CUDA events (ms).
template <class F>
double time_ms(F&& f) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  f();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

inline void print_num(double v) {
  if (v == v && v - v == 0)  // finite
    std::printf("%.6g", v);
  else
    std::printf("null");
}

inline void print_stats(const char* key, const Stats& s) {
  std::printf("\"%s\": {\"median\": ", key);
  print_num(s.median);
  std::printf(", \"p25\": ");
  print_num(s.p25);
  std::printf(", \"p75\": ");
  print_num(s.p75);
  std::printf(", \"min\": ");
  print_num(s.min);
  std::printf(", \"max\": ");
  print_num(s.max);
  std::printf("}");
}
