cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

get_property(_puercgp_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _puercgp_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL
    "Export compile_commands.json for clangd and other language servers.")

if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(_puercgp_nvcc
    NAMES nvcc
    HINTS
      "$ENV{CONDA_PREFIX}/bin"
      "$ENV{HOME}/.conda/envs/torch2.8/bin"
      "$ENV{HOME}/.conda/envs/gum/bin"
      "$ENV{HOME}/.conda/envs/groute/bin"
    NO_DEFAULT_PATH)
  if(_puercgp_nvcc)
    get_filename_component(_puercgp_cuda_bin "${_puercgp_nvcc}" DIRECTORY)
    get_filename_component(_puercgp_cuda_root "${_puercgp_cuda_bin}/.." ABSOLUTE)
    set(CMAKE_CUDA_COMPILER "${_puercgp_nvcc}" CACHE FILEPATH "CUDA compiler")
    set(CUDAToolkit_ROOT "${_puercgp_cuda_root}" CACHE PATH "CUDA toolkit root")
  endif()
endif()

project(puercgp
  VERSION 0.2.0
  LANGUAGES CXX CUDA
  DESCRIPTION "Standalone policy-oriented concurrent GPU graph processing framework"
)

option(PUERCGP_BUILD_EXAMPLES "Build puercgp smoke and validation examples." ON)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
endif()

add_library(puercgp INTERFACE)
add_library(puercgp::puercgp ALIAS puercgp)

target_compile_features(puercgp INTERFACE cxx_std_17 cuda_std_17)

target_compile_options(puercgp
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_include_directories(puercgp
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(puercgp INTERFACE CUDA::cudart)

if(PUERCGP_BUILD_EXAMPLES)
  foreach(_puercgp_example smoke_frontier validate_bfs validate_sssp smoke_reduce_ops smoke_hybrid_init smoke_hybrid_push validate_hybrid validate_wcc validate_hybrid_real bench_hybrid)
    add_executable(${_puercgp_example} examples/${_puercgp_example}.cu)
    target_link_libraries(${_puercgp_example} PRIVATE puercgp::puercgp)
    set_target_properties(${_puercgp_example}
      PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  endforeach()
endif()
