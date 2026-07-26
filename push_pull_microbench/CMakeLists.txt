cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(_preferred_nvcc "$ENV{HOME}/.conda/envs/torch2.8/bin/nvcc")
  if(EXISTS "${_preferred_nvcc}")
    set(CMAKE_CUDA_COMPILER "${_preferred_nvcc}" CACHE FILEPATH "CUDA compiler")
  endif()
endif()

project(push_pull_microbench
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(push_pull_bench
  src/main.cu
)

target_include_directories(push_pull_bench PRIVATE include)
target_compile_features(push_pull_bench PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(push_pull_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(push_pull_bench PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

