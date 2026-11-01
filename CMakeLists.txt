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

# PyTorch extension: blockfp8/_C.so, loaded with torch.ops.load_library.
option(BLOCKFP8_BUILD_EXTENSION "Build the PyTorch extension" ON)
set(BLOCKFP8_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/blockfp8 CACHE PATH "Where _C.so goes")
if(BLOCKFP8_BUILD_EXTENSION)
  set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
  if(NOT EXISTS ${CUTLASS_DIR}/include/cute/tensor.hpp)
    message(FATAL_ERROR "CUTLASS missing: run git submodule update --init --depth 1")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
  set(TORCH_CUDA_ARCH_LIST "8.9")
  find_package(Torch REQUIRED)

  add_library(blockfp8_C SHARED csrc/ops.cpp csrc/gemm_sm89.cu)
  target_include_directories(blockfp8_C PRIVATE ${CUTLASS_DIR}/include csrc)
  target_link_libraries(blockfp8_C PRIVATE ${TORCH_LIBRARIES})
  target_compile_options(blockfp8_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${BLOCKFP8_CUDA_FLAGS}>)
  set_target_properties(blockfp8_C PROPERTIES
    PREFIX "" OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY ${BLOCKFP8_OUTPUT_DIR})
endif()
