#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

template <typename vertex_t>
__global__ void compact_shared_pull_frontier_kernel(
    const query_mask_t* next_frontier_mask,
    std::size_t vertex_count,
    const unsigned long long* unique_offsets,
    vertex_t* frontier_vertices) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t vertex = tid; vertex < vertex_count; vertex += stride) {
    if (next_frontier_mask[vertex] == 0) {
      continue;
    }
    unsigned long long position = unique_offsets[vertex] - 1ULL;
    frontier_vertices[position] = static_cast<vertex_t>(vertex);
  }
}

}  // namespace detail
}  // namespace puercgp
