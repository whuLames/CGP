#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include <puercgp/backend/pull_graph_access.hxx>
#include <puercgp/core/layout.hxx>
#include <puercgp/core/mask.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void fused_pull_simple_kernel(
    graph_t graph,
    int query_count,
    typename Policy::value_type* values,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts) {
  using value_t = typename Policy::value_type;
  __shared__ query_mask_t thread_masks[128];

  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);
  std::size_t vertex = blockIdx.x * blockDim.y + threadIdx.y;
  int query_id = threadIdx.x;
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  query_mask_t local_mask = 0;

  if (vertex < vertex_count && query_id < query_count) {
    value_t acc = Policy::infinity();
    auto begin =
        get_pull_starting_edge(graph, static_cast<vertex_t>(vertex));
    auto end =
        get_pull_starting_edge(graph, static_cast<vertex_t>(vertex + 1));
    for (auto edge = begin; edge < end; ++edge) {
      vertex_t neighbor = get_pull_neighbor_vertex(graph, edge);
      value_t nb_val =
          values[value_index(static_cast<std::size_t>(neighbor),
                             static_cast<std::size_t>(query_id),
                             query_stride)];
      if (nb_val != Policy::infinity()) {
        value_t candidate =
            Policy::relax(nb_val, get_pull_edge_weight(graph, edge));
        if (candidate < acc) {
          acc = candidate;
        }
      }
    }
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query_id), query_stride);
    if (Policy::should_update(acc, values[value_pos])) {
      values[value_pos] = acc;
      local_mask = query_bit(query_id);
    }
  }

  thread_masks[shared_index] = local_mask;
  __syncthreads();

  if (vertex < vertex_count && threadIdx.x == 0) {
    query_mask_t improved_mask = 0;
    int row_base = threadIdx.y * blockDim.x;
    for (int q = 0; q < query_count; ++q) {
      improved_mask |= thread_masks[row_base + q];
    }
    next_frontier_mask[vertex] = improved_mask;
    visited_mask[vertex] |= improved_mask;
    unique_flags[vertex] = improved_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved_mask));
  }
}

template <int TILE_ROW, typename Policy, typename graph_t, typename vertex_t>
__global__ void fused_pull_smem_kernel(
    graph_t graph,
    int query_count,
    typename Policy::value_type* values,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts) {
  using value_t = typename Policy::value_type;
  constexpr int warp_size = 32;
  __shared__ vertex_t neighbor_tile[TILE_ROW][warp_size];
  __shared__ query_mask_t lane_masks[TILE_ROW][warp_size];

  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);
  std::size_t vertex =
      blockIdx.x * static_cast<std::size_t>(TILE_ROW) + threadIdx.y;
  int lane = threadIdx.x & (warp_size - 1);
  int query0 = lane;
  int query1 = lane + warp_size;
  bool row_valid = vertex < vertex_count;
  value_t acc0 = Policy::infinity();
  value_t acc1 = Policy::infinity();

  decltype(get_pull_starting_edge(graph, static_cast<vertex_t>(0))) begin = 0;
  decltype(begin) end = 0;
  if (row_valid) {
    begin = get_pull_starting_edge(graph, static_cast<vertex_t>(vertex));
    end = get_pull_starting_edge(graph, static_cast<vertex_t>(vertex + 1));
  }
  for (auto tile = begin; tile < end; tile += warp_size) {
    auto remaining = end - tile;
    int tile_count =
        remaining < warp_size ? static_cast<int>(remaining) : warp_size;
    if (lane < tile_count) {
      neighbor_tile[threadIdx.y][lane] =
          get_pull_neighbor_vertex(graph, tile + lane);
    }
    __syncthreads();

    for (int i = 0; i < tile_count; ++i) {
      vertex_t neighbor = neighbor_tile[threadIdx.y][i];
      auto weight = get_pull_edge_weight(graph, tile + i);
      if (query0 < query_count) {
        value_t nb_val =
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query0),
                               query_stride)];
        if (nb_val != Policy::infinity()) {
          value_t candidate = Policy::relax(nb_val, weight);
          if (candidate < acc0) {
            acc0 = candidate;
          }
        }
      }
      if (query1 < query_count) {
        value_t nb_val =
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query1),
                               query_stride)];
        if (nb_val != Policy::infinity()) {
          value_t candidate = Policy::relax(nb_val, weight);
          if (candidate < acc1) {
            acc1 = candidate;
          }
        }
      }
    }
    __syncthreads();
  }

  query_mask_t local_mask = 0;
  if (row_valid && query0 < query_count) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query0), query_stride);
    if (Policy::should_update(acc0, values[value_pos])) {
      values[value_pos] = acc0;
      local_mask |= query_bit(query0);
    }
  }
  if (row_valid && query1 < query_count) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query1), query_stride);
    if (Policy::should_update(acc1, values[value_pos])) {
      values[value_pos] = acc1;
      local_mask |= query_bit(query1);
    }
  }
  lane_masks[threadIdx.y][lane] = local_mask;
  __syncthreads();

  if (row_valid && lane == 0) {
    query_mask_t improved_mask = 0;
    for (int i = 0; i < warp_size; ++i) {
      improved_mask |= lane_masks[threadIdx.y][i];
    }
    next_frontier_mask[vertex] = improved_mask;
    visited_mask[vertex] |= improved_mask;
    unique_flags[vertex] = improved_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved_mask));
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull(graph_t graph,
                       int query_count,
                       typename Policy::value_type* values,
                       query_mask_t* visited_mask,
                       query_mask_t* next_frontier_mask,
                       unsigned long long* unique_flags,
                       unsigned long long* pair_counts,
                       cudaStream_t stream) {
  int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  if (query_count <= 32) {
    int tile_row = std::max(1, 128 / std::max(1, query_count));
    int grid_x = (vertex_count + tile_row - 1) / tile_row;
    fused_pull_simple_kernel<Policy, graph_t, vertex_t>
        <<<grid_x, dim3(query_count, tile_row), 0, stream>>>(
            graph, query_count, values, visited_mask, next_frontier_mask,
            unique_flags, pair_counts);
  } else if (query_count <= 64) {
    constexpr int tile_row = 4;
    int grid_x = (vertex_count + tile_row - 1) / tile_row;
    fused_pull_smem_kernel<tile_row, Policy, graph_t, vertex_t>
        <<<grid_x, dim3(32, tile_row), 0, stream>>>(
            graph, query_count, values, visited_mask, next_frontier_mask,
            unique_flags, pair_counts);
  } else {
    throw std::invalid_argument("fused pull supports at most 64 queries");
  }
}

}  // namespace detail
}  // namespace puercgp
