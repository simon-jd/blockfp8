cmake_minimum_required(VERSION 3.26)
project(blockfp8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# FP8 mma.sync exists from sm_89 (Ada) on; this project targets exactly sm_89.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(BLOCKFP8_CUDA_FLAGS -lineinfo --expt-relaxed-constexpr)

# Microbenchmarks: plain CUDA programs, no PyTorch.
foreach(name mma_peak dram_bw mma_precision)
  add_executable(${name} bench/micro/${name}.cu)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${BLOCKFP8_CUDA_FLAGS}>)
endforeach()

