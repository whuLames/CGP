#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace puercgp {
namespace detail {

__host__ __device__ __forceinline__ std::size_t value_index(
    std::size_t vertex,
    std::size_t query_id,
    std::size_t query_stride) {
  return vertex * query_stride + query_id;
}

}  // namespace detail
}  // namespace puercgp
