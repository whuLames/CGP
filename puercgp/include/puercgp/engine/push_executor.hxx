#pragma once

#include <algorithm>
#include <cstddef>

#include <cuda_runtime.h>

#include <puercgp/core/cuda_utils.hxx>
#include <puercgp/core/types.hxx>
#include <puercgp/kernels/push/shared_push_kernels.hxx>

namespace puercgp {
namespace detail {

template <typename Policy, typename graph_t, typename vertex_t>
void launch_shared_push_simple(graph_t graph,
                               const vertex_t* frontier_vertices,
                               const query_mask_t* frontier_mask,
                               std::size_t unique_count,
                               query_mask_t* visited_mask,
                               query_mask_t* next_frontier_mask,
                               vertex_t* next_frontier_vertices,
                               unsigned long long* next_unique_count,
                               unsigned long long* next_pair_count,
                               typename Policy::value_type* values,
                               int query_count,
                               vertex_t level,
                               int threads,
                               cudaStream_t stream) {
  int blocks = static_cast<int>(std::min<std::size_t>(
      std::max<std::size_t>(unique_count, 1), 65535));
  expand_shared_node_kernel<Policy, graph_t, vertex_t>
      <<<blocks, threads, 0, stream>>>(
          graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
          next_frontier_mask, next_frontier_vertices, next_unique_count,
          next_pair_count, values, query_count, level);
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_shared_push_query_parallel(graph_t graph,
                                       const vertex_t* frontier_vertices,
                                       const query_mask_t* frontier_mask,
                                       std::size_t unique_count,
                                       query_mask_t* visited_mask,
                                       query_mask_t* next_frontier_mask,
                                       vertex_t* next_frontier_vertices,
                                       unsigned long long* next_unique_count,
                                       unsigned long long* next_pair_count,
                                       typename Policy::value_type* values,
                                       int query_count,
                                       vertex_t level,
                                       int threads,
                                       cudaStream_t stream) {
  int blocks = static_cast<int>(std::min<std::size_t>(
      std::max<std::size_t>(unique_count, 1), 65535));
  expand_shared_node_query_parallel_kernel<Policy, graph_t, vertex_t>
      <<<blocks, threads, 0, stream>>>(
          graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
          next_frontier_mask, next_frontier_vertices, next_unique_count,
          next_pair_count, values, query_count, level);
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_shared_push_warp(graph_t graph,
                             const vertex_t* frontier_vertices,
                             const query_mask_t* frontier_mask,
                             std::size_t unique_count,
                             query_mask_t* visited_mask,
                             query_mask_t* next_frontier_mask,
                             vertex_t* next_frontier_vertices,
                             unsigned long long* next_unique_count,
                             unsigned long long* next_pair_count,
                             typename Policy::value_type* values,
                             int query_count,
                             vertex_t level,
                             int threads,
                             cudaStream_t stream) {
  int blocks = grid_for(unique_count * 32ULL, threads);
  expand_shared_node_warp_kernel<Policy, graph_t, vertex_t>
      <<<blocks, threads, 0, stream>>>(
          graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
          next_frontier_mask, next_frontier_vertices, next_unique_count,
          next_pair_count, values, query_count, level);
}

}  // namespace detail
}  // namespace puercgp
