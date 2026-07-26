#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include <puercgp/core/atomics.hxx>
#include <puercgp/core/layout.hxx>
#include <puercgp/core/mask.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

template <typename Policy, typename vertex_t>
__global__ void init_rank_values_kernel(
    std::size_t vertex_count, int query_count, query_mask_t valid_slots,
    const vertex_t *personalization_sources, float *ranks,
    query_mask_t *frontier_mask, vertex_t *frontier_vertices) {
  const std::size_t total =
      vertex_count * static_cast<std::size_t>(query_count);
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t position = tid; position < total; position += stride) {
    const int query_id =
        static_cast<int>(position % static_cast<std::size_t>(query_count));
    const vertex_t vertex =
        static_cast<vertex_t>(position / static_cast<std::size_t>(query_count));
    ranks[position] = Policy::personalization_value(
        vertex, query_id, vertex_count, personalization_sources);
    if (query_id == 0) {
      frontier_mask[vertex] = valid_slots;
      frontier_vertices[vertex] = vertex;
    }
  }
}

template <typename graph_t, typename vertex_t>
__global__ void compute_rank_dangling_mass_kernel(
    graph_t graph, std::size_t vertex_count, int query_count,
    const float *ranks, query_mask_t active_slots, float *dangling_mass) {
  const std::size_t total =
      vertex_count * static_cast<std::size_t>(query_count);
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t position = tid; position < total; position += stride) {
    const int query_id =
        static_cast<int>(position % static_cast<std::size_t>(query_count));
    if ((active_slots & query_bit(query_id)) == 0) {
      continue;
    }
    const vertex_t vertex =
        static_cast<vertex_t>(position / static_cast<std::size_t>(query_count));
    if (graph.get_starting_edge(vertex) ==
        graph.get_starting_edge(vertex + 1)) {
      atomicAdd(dangling_mass + query_id, ranks[position]);
    }
  }
}

template <typename Policy, typename vertex_t>
__global__ void initialize_rank_output_kernel(
    std::size_t vertex_count, int query_count, const float *old_values,
    float *next_values, const float *damping_factors,
    const vertex_t *personalization_sources, const float *dangling_mass,
    query_mask_t active_slots) {
  const std::size_t total =
      vertex_count * static_cast<std::size_t>(query_count);
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t position = tid; position < total; position += stride) {
    const int query_id =
        static_cast<int>(position % static_cast<std::size_t>(query_count));
    if ((active_slots & query_bit(query_id)) == 0) {
      next_values[position] = old_values[position];
      continue;
    }
    const vertex_t vertex =
        static_cast<vertex_t>(position / static_cast<std::size_t>(query_count));
    const float personalization = Policy::personalization_value(
        vertex, query_id, vertex_count, personalization_sources);
    const float damping = damping_factors[query_id];
    next_values[position] =
        ((1.0f - damping) + damping * dangling_mass[query_id]) *
        personalization;
  }
}

template <typename graph_t, typename vertex_t>
__global__ void fused_rank_push_kernel(graph_t graph, std::size_t vertex_count,
                                       int query_count, const float *old_values,
                                       float *next_values,
                                       const float *damping_factors,
                                       query_mask_t active_slots) {
  constexpr int warp_size = 32;
  constexpr int warps_per_block = 8;
  constexpr int max_queries = 64;
  __shared__ float contributions[warps_per_block][max_queries];

  const int lane = threadIdx.x & (warp_size - 1);
  const int local_warp = threadIdx.x / warp_size;
  const std::size_t warp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      warp_size;
  const std::size_t warp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) / warp_size;

  for (std::size_t source_index = warp_id; source_index < vertex_count;
       source_index += warp_stride) {
    const vertex_t source = static_cast<vertex_t>(source_index);
    const auto begin = graph.get_starting_edge(source);
    const auto end = graph.get_starting_edge(source + 1);
    const auto degree = end - begin;

    for (int query_id = lane; query_id < query_count; query_id += warp_size) {
      float contribution = 0.0f;
      if (degree != 0 && (active_slots & query_bit(query_id)) != 0) {
        contribution = damping_factors[query_id] *
                       old_values[value_index(
                           source_index, static_cast<std::size_t>(query_id),
                           static_cast<std::size_t>(query_count))] /
                       static_cast<float>(degree);
      }
      contributions[local_warp][query_id] = contribution;
    }
    __syncwarp();

    for (auto edge = begin + lane; edge < end; edge += warp_size) {
      const vertex_t destination = graph.get_destination_vertex(edge);
      query_mask_t bits = active_slots;
      while (bits != 0) {
        const int query_id = mask_ffs(bits) - 1;
        atomicAdd(next_values +
                      value_index(static_cast<std::size_t>(destination),
                                  static_cast<std::size_t>(query_id),
                                  static_cast<std::size_t>(query_count)),
                  contributions[local_warp][query_id]);
        bits &= bits - 1;
      }
    }
    __syncwarp();
  }
}

__global__ void mark_rank_changes_kernel(
    std::size_t vertex_count, int query_count, const float *old_values,
    const float *next_values, const float *epsilons,
    query_mask_t *next_frontier_mask, unsigned long long *unique_flags,
    unsigned long long *pair_counts, query_mask_t *changed_queries,
    query_mask_t active_slots) {
  __shared__ query_mask_t thread_masks[128];

  const std::size_t vertex = blockIdx.x * blockDim.y + threadIdx.y;
  const int query_id = threadIdx.x;
  const int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  query_mask_t local_mask = 0;

  if (vertex < vertex_count && query_id < query_count &&
      (active_slots & query_bit(query_id)) != 0) {
    const std::size_t position =
        value_index(vertex, static_cast<std::size_t>(query_id),
                    static_cast<std::size_t>(query_count));
    if (fabsf(next_values[position] - old_values[position]) >
        epsilons[query_id]) {
      local_mask = query_bit(query_id);
    }
  }

  thread_masks[shared_index] = local_mask;
  __syncthreads();

  if (vertex < vertex_count && threadIdx.x == 0) {
    query_mask_t changed_mask = 0;
    const int row_base = threadIdx.y * blockDim.x;
    for (int q = 0; q < query_count; ++q) {
      changed_mask |= thread_masks[row_base + q];
    }
    next_frontier_mask[vertex] = changed_mask;
    unique_flags[vertex] = changed_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(changed_mask));
    if (changed_mask != 0) {
      atomic_or_query_mask(changed_queries, changed_mask);
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_rank_push(
    graph_t graph, std::size_t vertex_count, int query_count,
    const float *old_values, float *next_values, const float *damping_factors,
    const float *epsilons, const vertex_t *personalization_sources,
    const float *dangling_mass, query_mask_t *next_frontier_mask,
    unsigned long long *unique_flags, unsigned long long *pair_counts,
    query_mask_t *changed_queries, query_mask_t active_slots,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int warps_per_block = threads / 32;
  const std::size_t value_count =
      vertex_count * static_cast<std::size_t>(query_count);
  const int value_blocks = std::max(
      1,
      std::min(static_cast<int>((value_count + threads - 1) / threads), 65535));
  initialize_rank_output_kernel<Policy, vertex_t>
      <<<value_blocks, threads, 0, stream>>>(
          vertex_count, query_count, old_values, next_values, damping_factors,
          personalization_sources, dangling_mass, active_slots);

  const int push_blocks = std::max(
      1, std::min(static_cast<int>((vertex_count + warps_per_block - 1) /
                                   warps_per_block),
                  65535));
  fused_rank_push_kernel<graph_t, vertex_t>
      <<<push_blocks, threads, 0, stream>>>(graph, vertex_count, query_count,
                                            old_values, next_values,
                                            damping_factors, active_slots);

  const int tile_rows = std::max(1, 128 / query_count);
  const int change_blocks =
      static_cast<int>((vertex_count + tile_rows - 1) / tile_rows);
  mark_rank_changes_kernel<<<change_blocks, dim3(query_count, tile_rows), 0,
                             stream>>>(
      vertex_count, query_count, old_values, next_values, epsilons,
      next_frontier_mask, unique_flags, pair_counts, changed_queries,
      active_slots);
}

} // namespace detail
} // namespace puercgp
