#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <puercgp/core/cuda_utils.hxx>
#include <puercgp/core/types.hxx>
#include <puercgp/kernels/common/pull_postprocess.hxx>
#include <puercgp/kernels/pull/fused_pull_kernels.hxx>

namespace puercgp {
namespace detail {

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_compute(graph_t graph,
                               int query_count,
                               typename Policy::value_type* values,
                               query_mask_t* visited_mask,
                               query_mask_t* next_frontier_mask,
                               unsigned long long* unique_flags,
                               unsigned long long* pair_counts,
                               query_mask_t active_slots,
                               cudaStream_t stream) {
  launch_fused_pull<Policy, graph_t, vertex_t>(
      graph, query_count, values, visited_mask, next_frontier_mask,
      unique_flags, pair_counts, active_slots, stream);
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_compute_scheduled(
    graph_t graph, int query_count, typename Policy::value_type* values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, query_mask_t* active_union,
    cudaStream_t stream) {
  launch_fused_pull_scheduled<Policy, graph_t, vertex_t>(
      graph, query_count, values, visited_mask, next_frontier_mask,
      unique_flags, pair_counts, active_slots, active_union, stream);
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_sum_pull_compute(
    graph_t graph, int query_count, const float* old_values,
    float* next_values, const float* damping_factors, const float* epsilons,
    const vertex_t* personalization_sources, const float* dangling_mass,
    query_mask_t* next_frontier_mask, unsigned long long* unique_flags,
    unsigned long long* pair_counts, query_mask_t* changed_queries,
    query_mask_t active_slots, cudaStream_t stream) {
  launch_fused_pull_sum<Policy, graph_t, vertex_t>(
      graph, query_count, old_values, next_values, damping_factors, epsilons,
      personalization_sources, dangling_mass, next_frontier_mask, unique_flags,
      pair_counts, changed_queries, active_slots, stream);
}

template <typename vertex_t>
void launch_pull_frontier_compact(const query_mask_t* next_frontier_mask,
                                  std::size_t vertex_count,
                                  const unsigned long long* unique_offsets,
                                  vertex_t* frontier_vertices,
                                  int threads,
                                  cudaStream_t stream) {
  compact_shared_pull_frontier_kernel<vertex_t>
      <<<grid_for(vertex_count, threads), threads, 0, stream>>>(
          next_frontier_mask, vertex_count, unique_offsets, frontier_vertices);
}

}  // namespace detail
}  // namespace puercgp
