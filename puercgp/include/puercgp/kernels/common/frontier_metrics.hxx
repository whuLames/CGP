#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <puercgp/core/cuda_utils.hxx>
#include <puercgp/core/mask.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

template <typename graph_t, typename vertex_t>
__global__ void compute_shared_frontier_metrics_kernel(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t active_slots,
    unsigned long long* actual_edge_count,
    unsigned long long* virtual_edge_count,
    unsigned long long* active_pair_count) {
  extern __shared__ unsigned long long shared[];
  unsigned long long* actual_sums = shared;
  unsigned long long* virtual_sums = shared + blockDim.x;
  unsigned long long* pair_sums = shared + 2 * blockDim.x;

  unsigned long long local_actual = 0;
  unsigned long long local_virtual = 0;
  unsigned long long local_pairs = 0;

  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t i = tid; i < unique_count; i += stride) {
    vertex_t vertex = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[vertex] & active_slots;
    int active_queries = mask_popcount(active_mask);
    if (active_queries == 0) {
      continue;
    }

    auto begin = graph.get_starting_edge(vertex);
    auto end = graph.get_starting_edge(vertex + 1);
    unsigned long long degree =
        static_cast<unsigned long long>(end - begin);
    local_actual += degree;
    local_virtual += degree * static_cast<unsigned long long>(active_queries);
    local_pairs += static_cast<unsigned long long>(active_queries);
  }

  actual_sums[threadIdx.x] = local_actual;
  virtual_sums[threadIdx.x] = local_virtual;
  pair_sums[threadIdx.x] = local_pairs;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      actual_sums[threadIdx.x] += actual_sums[threadIdx.x + offset];
      virtual_sums[threadIdx.x] += virtual_sums[threadIdx.x + offset];
      pair_sums[threadIdx.x] += pair_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(actual_edge_count, actual_sums[0]);
    atomicAdd(virtual_edge_count, virtual_sums[0]);
    atomicAdd(active_pair_count, pair_sums[0]);
  }
}

template <typename graph_t, typename vertex_t>
void launch_compute_shared_frontier_metrics(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t active_slots,
    unsigned long long* actual_edge_count,
    unsigned long long* virtual_edge_count,
    unsigned long long* active_pair_count,
    int threads,
    cudaStream_t stream) {
  const int blocks = grid_for(unique_count, threads);
  const std::size_t shared_bytes =
      3ULL * static_cast<std::size_t>(threads) * sizeof(unsigned long long);
  compute_shared_frontier_metrics_kernel<<<blocks, threads, shared_bytes,
                                           stream>>>(
      graph, frontier_vertices, frontier_mask, unique_count, active_slots,
      actual_edge_count, virtual_edge_count, active_pair_count);
}

}  // namespace detail
}  // namespace puercgp
