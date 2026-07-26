#pragma once

#include <cstddef>
#include <type_traits>

#include <cuda_runtime.h>

#include <puercgp/algorithms/bfs.hxx>
#include <puercgp/core/atomics.hxx>
#include <puercgp/core/layout.hxx>
#include <puercgp/core/mask.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

template <typename Policy, typename graph_t, typename vertex_t,
          bool use_start_levels = false, bool track_active = false>
__global__ void expand_shared_node_kernel(
    graph_t graph,
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
    query_mask_t active_slots,
    vertex_t level,
    const int* start_levels,
    query_mask_t* active_union) {
  using value_t = typename Policy::value_type;
  std::size_t vertex_count = graph.get_number_of_vertices();

  for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
    vertex_t source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source] & active_slots;
    if (active_mask == 0) {
      continue;
    }
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);

    for (auto edge = begin + threadIdx.x; edge < end; edge += blockDim.x) {
      vertex_t neighbor = graph.get_destination_vertex(edge);
      query_mask_t improved = 0;

      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        query_mask_t old_visited =
            atomic_or_query_mask(visited_mask + neighbor, active_mask);

        improved = active_mask & ~old_visited;
        query_mask_t bits = improved;

        while (bits != 0) {
          int query_id = mask_ffs(bits) - 1;
          if (query_id < query_count) {
            int next_level = static_cast<int>(level) + 1;
            if constexpr (use_start_levels) {
              next_level = static_cast<int>(values[value_index(
                               static_cast<std::size_t>(source),
                               static_cast<std::size_t>(query_id),
                               static_cast<std::size_t>(query_count))]) +
                  1;
            }
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query_id),
                               static_cast<std::size_t>(query_count))] =
                static_cast<value_t>(next_level);
          }
          bits &= (bits - 1);
        }
      } else {
        query_mask_t bits = active_mask;
        while (bits != 0) {
          int query_id = mask_ffs(bits) - 1;
          if (query_id < query_count) {
            value_t source_distance =
                values[value_index(static_cast<std::size_t>(source),
                                   static_cast<std::size_t>(query_id),
                                   static_cast<std::size_t>(query_count))];
            if (source_distance != Policy::infinity()) {
              value_t candidate =
                  Policy::relax(source_distance, graph.get_edge_weight(edge));
              value_t old = atomic_reduce_value<Policy>(
                  values + value_index(static_cast<std::size_t>(neighbor),
                                       static_cast<std::size_t>(query_id),
                                       static_cast<std::size_t>(query_count)),
                  candidate);
              if (Policy::should_update(candidate, old)) {
                improved |= query_bit(query_id);
              }
            }
          }
          bits &= (bits - 1);
        }
      }

      if (improved == 0) {
        continue;
      }

      if constexpr (track_active) {
        mark_next_shared_frontier_with_signal(
            neighbor, improved, next_frontier_mask, next_frontier_vertices,
            next_unique_count, next_pair_count, active_union);
      } else {
        mark_next_shared_frontier(neighbor, improved, next_frontier_mask,
                                  next_frontier_vertices, next_unique_count,
                                  next_pair_count);
      }
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t,
          bool use_start_levels = false, bool track_active = false>
__global__ void expand_shared_node_query_parallel_kernel(
    graph_t graph,
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
    query_mask_t active_slots,
    vertex_t level,
    const int* start_levels,
    query_mask_t* active_union) {
  using value_t = typename Policy::value_type;
  constexpr int warp_size = 32;
  std::size_t vertex_count = graph.get_number_of_vertices();
  int lane = threadIdx.x & (warp_size - 1);
  int warp_in_block = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;

  for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
    vertex_t source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source] & active_slots;
    if (active_mask == 0) {
      continue;
    }
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);

    for (auto edge = begin + warp_in_block; edge < end;
         edge += warps_per_block) {
      vertex_t neighbor = graph.get_destination_vertex(edge);
      query_mask_t improved = 0;

      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        if (lane == 0) {
          query_mask_t old_visited =
              atomic_or_query_mask(visited_mask + neighbor, active_mask);
          improved = active_mask & ~old_visited;
        }
        improved = static_cast<query_mask_t>(__shfl_sync(
            0xffffffffU, static_cast<unsigned long long>(improved), 0));
        for (int query_id = lane; query_id < query_count; query_id += 32) {
          if ((improved & query_bit(query_id)) != 0) {
            int next_level = static_cast<int>(level) + 1;
            if constexpr (use_start_levels) {
              next_level = static_cast<int>(values[value_index(
                               static_cast<std::size_t>(source),
                               static_cast<std::size_t>(query_id),
                               static_cast<std::size_t>(query_count))]) +
                  1;
            }
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query_id),
                               static_cast<std::size_t>(query_count))] =
                static_cast<value_t>(next_level);
          }
        }
      } else {
        for (int query_id = lane; query_id < query_count; query_id += 32) {
          if ((active_mask & query_bit(query_id)) != 0) {
            value_t source_distance =
                values[value_index(static_cast<std::size_t>(source),
                                   static_cast<std::size_t>(query_id),
                                   static_cast<std::size_t>(query_count))];
            if (source_distance != Policy::infinity()) {
              value_t candidate =
                  Policy::relax(source_distance, graph.get_edge_weight(edge));
              value_t old = atomic_reduce_value<Policy>(
                  values + value_index(static_cast<std::size_t>(neighbor),
                                       static_cast<std::size_t>(query_id),
                                       static_cast<std::size_t>(query_count)),
                  candidate);
              if (Policy::should_update(candidate, old)) {
                improved |= query_bit(query_id);
              }
            }
          }
        }
        unsigned long long reduced =
            static_cast<unsigned long long>(improved);
        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
          reduced |= __shfl_down_sync(0xffffffffU, reduced, offset);
        }
        improved =
            static_cast<query_mask_t>(__shfl_sync(0xffffffffU, reduced, 0));
      }

      if (lane == 0 && improved != 0) {
        if constexpr (track_active) {
          mark_next_shared_frontier_with_signal(
              neighbor, improved, next_frontier_mask, next_frontier_vertices,
              next_unique_count, next_pair_count, active_union);
        } else {
          mark_next_shared_frontier(neighbor, improved, next_frontier_mask,
                                    next_frontier_vertices, next_unique_count,
                                    next_pair_count);
        }
      }
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t,
          bool use_start_levels = false, bool track_active = false>
__global__ void expand_shared_node_warp_kernel(
    graph_t graph,
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
    query_mask_t active_slots,
    vertex_t level,
    const int* start_levels,
    query_mask_t* active_union) {
  using value_t = typename Policy::value_type;
  constexpr int warp_size = 32;
  std::size_t vertex_count = graph.get_number_of_vertices();
  int lane = threadIdx.x & (warp_size - 1);
  std::size_t warp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  std::size_t warp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) >> 5;

  for (std::size_t i = warp_id; i < unique_count; i += warp_stride) {
    vertex_t source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source] & active_slots;
    if (active_mask == 0) {
      continue;
    }
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);

    for (auto edge = begin + lane; edge < end; edge += warp_size) {
      vertex_t neighbor = graph.get_destination_vertex(edge);
      query_mask_t improved = 0;

      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        query_mask_t old_visited =
            atomic_or_query_mask(visited_mask + neighbor, active_mask);
        improved = active_mask & ~old_visited;
        query_mask_t bits = improved;

        while (bits != 0) {
          int query_id = mask_ffs(bits) - 1;
          if (query_id < query_count) {
            int next_level = static_cast<int>(level) + 1;
            if constexpr (use_start_levels) {
              next_level = static_cast<int>(values[value_index(
                               static_cast<std::size_t>(source),
                               static_cast<std::size_t>(query_id),
                               static_cast<std::size_t>(query_count))]) +
                  1;
            }
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query_id),
                               static_cast<std::size_t>(query_count))] =
                static_cast<value_t>(next_level);
          }
          bits &= (bits - 1);
        }
      } else {
        query_mask_t bits = active_mask;
        while (bits != 0) {
          int query_id = mask_ffs(bits) - 1;
          if (query_id < query_count) {
            value_t source_distance =
                values[value_index(static_cast<std::size_t>(source),
                                   static_cast<std::size_t>(query_id),
                                   static_cast<std::size_t>(query_count))];
            if (source_distance != Policy::infinity()) {
              value_t candidate =
                  Policy::relax(source_distance, graph.get_edge_weight(edge));
              value_t old = atomic_reduce_value<Policy>(
                  values + value_index(static_cast<std::size_t>(neighbor),
                                       static_cast<std::size_t>(query_id),
                                       static_cast<std::size_t>(query_count)),
                  candidate);
              if (Policy::should_update(candidate, old)) {
                improved |= query_bit(query_id);
              }
            }
          }
          bits &= (bits - 1);
        }
      }

      if (improved == 0) {
        continue;
      }

      if constexpr (track_active) {
        mark_next_shared_frontier_with_signal(
            neighbor, improved, next_frontier_mask, next_frontier_vertices,
            next_unique_count, next_pair_count, active_union);
      } else {
        mark_next_shared_frontier(neighbor, improved, next_frontier_mask,
                                  next_frontier_vertices, next_unique_count,
                                  next_pair_count);
      }
    }
  }
}

}  // namespace detail
}  // namespace puercgp
