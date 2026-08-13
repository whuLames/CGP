#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include <puercgp/backend/pull_graph_access.hxx>
#include <puercgp/core/layout.hxx>
#include <puercgp/core/mask.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace detail {

template <typename graph_t, typename vertex_t>
__device__ int pull_degree_bucket(graph_t graph, vertex_t vertex, int t2,
                                  int t4, int t8) {
  const auto begin = get_pull_starting_edge(graph, vertex);
  const auto end = get_pull_starting_edge(graph, vertex + 1);
  const auto degree = end - begin;
  if (degree >= t8) {
    return 3;
  }
  if (degree >= t4) {
    return 2;
  }
  return degree >= t2 ? 1 : 0;
}

template <typename graph_t, typename vertex_t>
__global__ void count_pull_degree_buckets_kernel(
    graph_t graph, int t2, int t4, int t8,
    unsigned long long* bucket_counts) {
  const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;
  const std::size_t vertex_count = graph.get_number_of_vertices();
  for (std::size_t vertex = first; vertex < vertex_count; vertex += stride) {
    const int bucket = pull_degree_bucket(
        graph, static_cast<vertex_t>(vertex), t2, t4, t8);
    atomicAdd(bucket_counts + bucket, 1ULL);
  }
}

template <typename graph_t, typename vertex_t>
__global__ void fill_pull_degree_buckets_kernel(
    graph_t graph, int t2, int t4, int t8,
    const unsigned long long* bucket_offsets,
    unsigned long long* bucket_positions, vertex_t* vertices) {
  const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;
  const std::size_t vertex_count = graph.get_number_of_vertices();
  for (std::size_t vertex = first; vertex < vertex_count; vertex += stride) {
    const int bucket = pull_degree_bucket(
        graph, static_cast<vertex_t>(vertex), t2, t4, t8);
    const auto position = atomicAdd(bucket_positions + bucket, 1ULL);
    vertices[bucket_offsets[bucket] + position] =
        static_cast<vertex_t>(vertex);
  }
}

template <typename graph_t, typename vertex_t>
__global__ void count_pull_high_degree_segments_kernel(
    graph_t graph, const vertex_t* row_ids, std::size_t row_count,
    int edges_per_segment, unsigned int* segment_counts) {
  const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;
  for (std::size_t row = first; row < row_count; row += stride) {
    const auto vertex = row_ids[row];
    const auto begin = get_pull_starting_edge(graph, vertex);
    const auto end = get_pull_starting_edge(graph, vertex + 1);
    const auto degree = static_cast<std::size_t>(end - begin);
    segment_counts[row] = static_cast<unsigned int>(
        (degree + static_cast<std::size_t>(edges_per_segment) - 1) /
        static_cast<std::size_t>(edges_per_segment));
  }
}

template <typename graph_t, typename vertex_t>
__global__ void fill_pull_high_degree_segments_kernel(
    graph_t graph, const vertex_t* row_ids, std::size_t row_count,
    int edges_per_segment, const unsigned int* segment_offsets,
    vertex_t* segment_vertices, typename graph_t::edge_type* segment_begins,
    typename graph_t::edge_type* segment_ends) {
  const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;
  for (std::size_t row = first; row < row_count; row += stride) {
    const auto vertex = row_ids[row];
    const auto row_begin = get_pull_starting_edge(graph, vertex);
    const auto row_end = get_pull_starting_edge(graph, vertex + 1);
    const unsigned int output_begin = segment_offsets[row];
    const unsigned int output_end = segment_offsets[row + 1];
    for (unsigned int output = output_begin; output < output_end; ++output) {
      const auto local_segment = output - output_begin;
      const auto begin = row_begin +
          static_cast<typename graph_t::edge_type>(local_segment) *
              edges_per_segment;
      const auto end = begin + edges_per_segment < row_end
                           ? begin + edges_per_segment
                           : row_end;
      segment_vertices[output] = vertex;
      segment_begins[output] = begin;
      segment_ends[output] = end;
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void fused_pull_high_degree_segments_kernel(
    graph_t graph, int query_count, const vertex_t* segment_vertices,
    const typename graph_t::edge_type* segment_begins,
    const typename graph_t::edge_type* segment_ends,
    std::size_t segment_count,
    const typename Policy::value_type* input_values,
    typename Policy::value_type* output_values,
    query_mask_t* next_frontier_mask, query_mask_t active_slots) {
  using value_t = typename Policy::value_type;
  constexpr int warp_size = 32;
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  const int query_warps = query_count > warp_size ? 2 : 1;
  const int segments_per_block = (blockDim.x / warp_size) / query_warps;
  const int segment_slot = warp / query_warps;
  const int query_group = warp % query_warps;
  const std::size_t segment =
      static_cast<std::size_t>(blockIdx.x) * segments_per_block + segment_slot;
  const bool valid_segment = segment < segment_count;
  const int query = query_group * warp_size + lane;
  const bool query_active = valid_segment && query < query_count &&
      (active_slots & query_bit(query)) != 0;

  value_t aggregate = Policy::infinity();
  if (query_active) {
    const auto begin = segment_begins[segment];
    const auto end = segment_ends[segment];
    for (auto edge = begin; edge < end; ++edge) {
      const auto neighbor = get_pull_neighbor_vertex(graph, edge);
      const auto neighbor_value = input_values[value_index(
          static_cast<std::size_t>(neighbor),
          static_cast<std::size_t>(query),
          static_cast<std::size_t>(query_count))];
      if (neighbor_value != Policy::infinity()) {
        const auto candidate =
            Policy::relax(neighbor_value, get_pull_edge_weight(graph, edge));
        if (Policy::should_update(candidate, aggregate)) {
          aggregate = candidate;
        }
      }
    }
  }

  bool updated = false;
  vertex_t vertex{};
  if (query_active && aggregate != Policy::infinity()) {
    vertex = segment_vertices[segment];
    const auto position = value_index(
        static_cast<std::size_t>(vertex), static_cast<std::size_t>(query),
        static_cast<std::size_t>(query_count));
    const auto previous = atomic_reduce_value<Policy>(output_values + position,
                                                       aggregate);
    updated = Policy::should_update(aggregate, previous);
  }
  const unsigned int improved = __ballot_sync(0xffffffffU, updated);
  if (valid_segment && lane == 0 && improved != 0) {
    vertex = segment_vertices[segment];
    const query_mask_t shifted =
        static_cast<query_mask_t>(improved) << (query_group * warp_size);
    atomic_or_query_mask(next_frontier_mask + vertex, shifted);
  }
}

template <typename vertex_t>
__global__ void finalize_pull_high_degree_segments_kernel(
    const vertex_t* row_ids, std::size_t row_count,
    query_mask_t* visited_mask, const query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots) {
  const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;
  for (std::size_t row = first; row < row_count; row += stride) {
    const auto vertex = row_ids[row];
    const query_mask_t improved = next_frontier_mask[vertex] & active_slots;
    atomic_or_query_mask(visited_mask + vertex, improved);
    unique_flags[vertex] = improved != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved));
  }
}

__global__ void trace_pull_frontier_kernel(
    const query_mask_t* next_frontier_mask, std::size_t vertex_count,
    unsigned int iteration, unsigned long long* iteration_sum,
    unsigned int* update_count, unsigned int* first_iteration,
    unsigned int* last_iteration) {
  const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x;
  for (std::size_t vertex = first; vertex < vertex_count; vertex += stride) {
    const auto updates = static_cast<unsigned int>(
        mask_popcount(next_frontier_mask[vertex]));
    if (updates == 0) {
      continue;
    }
    if (update_count[vertex] == 0) {
      first_iteration[vertex] = iteration;
    }
    iteration_sum[vertex] +=
        static_cast<unsigned long long>(iteration) * updates;
    update_count[vertex] += updates;
    last_iteration[vertex] = iteration;
  }
}

inline void launch_trace_pull_frontier(
    const query_mask_t* next_frontier_mask, std::size_t vertex_count,
    unsigned int iteration, unsigned long long* iteration_sum,
    unsigned int* update_count, unsigned int* first_iteration,
    unsigned int* last_iteration, cudaStream_t stream) {
  if (vertex_count == 0) {
    return;
  }
  constexpr int threads = 256;
  const auto raw_blocks =
      (vertex_count + static_cast<std::size_t>(threads) - 1) / threads;
  const int blocks = static_cast<int>(
      std::min<std::size_t>(raw_blocks, static_cast<std::size_t>(65535)));
  trace_pull_frontier_kernel<<<blocks, threads, 0, stream>>>(
      next_frontier_mask, vertex_count, iteration, iteration_sum,
      update_count, first_iteration, last_iteration);
}

template <int WARPS_PER_ROW, int WARPS_PER_BLOCK, typename Policy,
          typename graph_t, typename vertex_t>
__global__ void fused_pull_degree_aware_kernel(
    graph_t graph, int query_count, const vertex_t* row_ids,
    std::size_t row_count, const typename Policy::value_type* input_values,
    typename Policy::value_type* output_values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots) {
  static_assert(WARPS_PER_ROW == 1 || WARPS_PER_ROW == 2 ||
                    WARPS_PER_ROW == 4 || WARPS_PER_ROW == 8,
                "warps per row must be 1, 2, 4, or 8");
  static_assert(WARPS_PER_BLOCK % WARPS_PER_ROW == 0,
                "row groups must fit in a block");
  using value_t = typename Policy::value_type;
  constexpr int warp_size = 32;
  constexpr int rows_per_block = WARPS_PER_BLOCK / WARPS_PER_ROW;
  __shared__ vertex_t neighbor_cache[WARPS_PER_BLOCK * warp_size];
  __shared__ value_t weight_cache[WARPS_PER_BLOCK * warp_size];
  __shared__ value_t partial[WARPS_PER_BLOCK * 64];

  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  const int row_slot = warp / WARPS_PER_ROW;
  const int local_warp = warp % WARPS_PER_ROW;
  const std::size_t row_index =
      static_cast<std::size_t>(blockIdx.x) * rows_per_block + row_slot;
  const bool valid_row = row_index < row_count;
  const vertex_t vertex = valid_row ? row_ids[row_index] : vertex_t{0};
  const int query0 = lane;
  const int query1 = lane + warp_size;
  const bool query0_active = query0 < query_count &&
      (active_slots & query_bit(query0)) != 0;
  const bool query1_active = query1 < query_count &&
      (active_slots & query_bit(query1)) != 0;

  decltype(get_pull_starting_edge(graph, vertex_t{0})) row_begin = 0;
  decltype(row_begin) row_end = 0;
  if (valid_row) {
    row_begin = get_pull_starting_edge(graph, vertex);
    row_end = get_pull_starting_edge(graph, vertex + 1);
  }
  const auto degree = row_end - row_begin;
  const auto part_begin =
      row_begin + degree * local_warp / WARPS_PER_ROW;
  const auto part_end =
      row_begin + degree * (local_warp + 1) / WARPS_PER_ROW;
  const int cache_base = warp * warp_size;
  value_t acc0 = Policy::infinity();
  value_t acc1 = Policy::infinity();

  for (auto tile = part_begin; tile < part_end; tile += warp_size) {
    const auto edge = tile + lane;
    if (edge < part_end) {
      neighbor_cache[cache_base + lane] =
          get_pull_neighbor_vertex(graph, edge);
      weight_cache[cache_base + lane] =
          static_cast<value_t>(get_pull_edge_weight(graph, edge));
    }
    __syncwarp();
    const int tile_size = static_cast<int>(
        part_end - tile < warp_size ? part_end - tile : warp_size);
    for (int i = 0; i < tile_size; ++i) {
      const auto neighbor = neighbor_cache[cache_base + i];
      const auto weight = weight_cache[cache_base + i];
      if (query0_active) {
        const auto neighbor_value = input_values[value_index(
            static_cast<std::size_t>(neighbor),
            static_cast<std::size_t>(query0),
            static_cast<std::size_t>(query_count))];
        if (neighbor_value != Policy::infinity()) {
          const auto candidate = Policy::relax(neighbor_value, weight);
          if (Policy::should_update(candidate, acc0)) {
            acc0 = candidate;
          }
        }
      }
      if (query1_active) {
        const auto neighbor_value = input_values[value_index(
            static_cast<std::size_t>(neighbor),
            static_cast<std::size_t>(query1),
            static_cast<std::size_t>(query_count))];
        if (neighbor_value != Policy::infinity()) {
          const auto candidate = Policy::relax(neighbor_value, weight);
          if (Policy::should_update(candidate, acc1)) {
            acc1 = candidate;
          }
        }
      }
    }
    __syncwarp();
  }

  partial[warp * 64 + query0] = acc0;
  partial[warp * 64 + query1] = acc1;
  __syncthreads();

  bool updated0 = false;
  bool updated1 = false;
  if (valid_row && local_warp == 0) {
    value_t total0 = Policy::infinity();
    value_t total1 = Policy::infinity();
#pragma unroll
    for (int part = 0; part < WARPS_PER_ROW; ++part) {
      const int source_warp = row_slot * WARPS_PER_ROW + part;
      const auto value0 = partial[source_warp * 64 + query0];
      const auto value1 = partial[source_warp * 64 + query1];
      if (Policy::should_update(value0, total0)) {
        total0 = value0;
      }
      if (Policy::should_update(value1, total1)) {
        total1 = value1;
      }
    }
    if (query0_active) {
      const auto position = value_index(
          static_cast<std::size_t>(vertex),
          static_cast<std::size_t>(query0),
          static_cast<std::size_t>(query_count));
      updated0 = Policy::should_update(total0, input_values[position]);
      if (updated0) {
        output_values[position] = total0;
      }
    }
    if (query1_active) {
      const auto position = value_index(
          static_cast<std::size_t>(vertex),
          static_cast<std::size_t>(query1),
          static_cast<std::size_t>(query_count));
      updated1 = Policy::should_update(total1, input_values[position]);
      if (updated1) {
        output_values[position] = total1;
      }
    }
  }

  const unsigned int low_mask = __ballot_sync(0xffffffffU, updated0);
  const unsigned int high_mask = __ballot_sync(0xffffffffU, updated1);
  if (valid_row && local_warp == 0 && lane == 0) {
    const query_mask_t improved_mask =
        static_cast<query_mask_t>(low_mask) |
        (static_cast<query_mask_t>(high_mask) << 32);
    next_frontier_mask[vertex] = improved_mask;
    atomic_or_query_mask(visited_mask + vertex, improved_mask);
    unique_flags[vertex] = improved_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved_mask));
  }
}

template <int WARPS_PER_ROW, int WARPS_PER_BLOCK, typename Policy,
          typename graph_t, typename vertex_t>
void launch_fused_pull_degree_bucket(
    graph_t graph, int query_count, const vertex_t* row_ids,
    std::size_t row_count, const typename Policy::value_type* input_values,
    typename Policy::value_type* output_values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, cudaStream_t stream) {
  if (row_count == 0) {
    return;
  }
  constexpr int rows_per_block = WARPS_PER_BLOCK / WARPS_PER_ROW;
  const int blocks = static_cast<int>(
      (row_count + static_cast<std::size_t>(rows_per_block) - 1) /
      static_cast<std::size_t>(rows_per_block));
  fused_pull_degree_aware_kernel<WARPS_PER_ROW, WARPS_PER_BLOCK, Policy,
                                 graph_t, vertex_t>
      <<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
          graph, query_count, row_ids, row_count, input_values, output_values,
          visited_mask,
          next_frontier_mask, unique_flags, pair_counts, active_slots);
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_indices(
    graph_t graph, int query_count, typename Policy::value_type* values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, const vertex_t* vertex_indices,
    std::size_t vertex_index_count, bool reverse_vertices,
    cudaStream_t stream, bool accumulate_frontier = false,
    const typename Policy::value_type* input_values = nullptr);

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_degree_aware(
    graph_t graph, int query_count, const vertex_t* bucket_vertices,
    const std::size_t* bucket_offsets, const int* bucket_order,
    typename Policy::value_type* values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, cudaStream_t stream,
    const typename Policy::value_type* input_values = nullptr) {
  const auto* read_values = input_values == nullptr ? values : input_values;
  for (int position = 0; position < 4; ++position) {
    const int bucket = bucket_order[position];
    const auto* rows = bucket_vertices + bucket_offsets[bucket];
    const auto count = bucket_offsets[bucket + 1] - bucket_offsets[bucket];
    switch (bucket) {
      case 0:
        launch_fused_pull_indices<Policy, graph_t, vertex_t>(
            graph, query_count, values, visited_mask, next_frontier_mask,
            unique_flags, pair_counts, active_slots, rows, count, false,
            stream, false, read_values);
        break;
      case 1:
        launch_fused_pull_degree_bucket<2, 2, Policy>(
            graph, query_count, rows, count, read_values, values, visited_mask,
            next_frontier_mask, unique_flags, pair_counts, active_slots,
            stream);
        break;
      case 2:
        launch_fused_pull_degree_bucket<4, 4, Policy>(
            graph, query_count, rows, count, read_values, values, visited_mask,
            next_frontier_mask, unique_flags, pair_counts, active_slots,
            stream);
        break;
      case 3:
        launch_fused_pull_degree_bucket<8, 8, Policy>(
            graph, query_count, rows, count, read_values, values, visited_mask,
            next_frontier_mask, unique_flags, pair_counts, active_slots,
            stream);
        break;
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t,
          bool track_active = false>
__global__ void fused_pull_simple_kernel(
    graph_t graph,
    int query_count,
    const typename Policy::value_type* input_values,
    typename Policy::value_type* output_values,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts,
    query_mask_t active_slots,
    query_mask_t* active_union,
    const vertex_t* vertex_indices,
    std::size_t vertex_index_count,
    std::size_t vertex_begin,
    std::size_t vertex_end,
    bool reverse_vertices,
    unsigned int trace_iteration,
    unsigned long long* trace_iteration_sum,
    unsigned int* trace_update_count,
    unsigned int* trace_first_iteration,
    unsigned int* trace_last_iteration,
    bool accumulate_frontier) {
  using value_t = typename Policy::value_type;
  __shared__ query_mask_t thread_masks[128];

  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);
  const std::size_t local_vertex = blockIdx.x * blockDim.y + threadIdx.y;
  const std::size_t range_size =
      vertex_indices == nullptr ? vertex_end - vertex_begin
                                : vertex_index_count;
  std::size_t vertex = vertex_end;
  if (local_vertex < range_size) {
    if (vertex_indices != nullptr) {
      const std::size_t index =
          reverse_vertices ? range_size - 1 - local_vertex : local_vertex;
      vertex = static_cast<std::size_t>(vertex_indices[index]);
    } else {
      vertex = reverse_vertices ? vertex_end - 1 - local_vertex
                                : vertex_begin + local_vertex;
    }
  }
  int query_id = threadIdx.x;
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  query_mask_t local_mask = 0;

  if (vertex < vertex_count && query_id < query_count &&
      (active_slots & query_bit(query_id)) != 0) {
    value_t acc = Policy::infinity();
    auto begin =
        get_pull_starting_edge(graph, static_cast<vertex_t>(vertex));
    auto end =
        get_pull_starting_edge(graph, static_cast<vertex_t>(vertex + 1));
    for (auto edge = begin; edge < end; ++edge) {
      vertex_t neighbor = get_pull_neighbor_vertex(graph, edge);
      value_t nb_val =
          input_values[value_index(static_cast<std::size_t>(neighbor),
                                   static_cast<std::size_t>(query_id),
                                   query_stride)];
      if (nb_val != Policy::infinity()) {
        value_t candidate =
            Policy::relax(nb_val, get_pull_edge_weight(graph, edge));
        if (Policy::should_update(candidate, acc)) {
          acc = candidate;
        }
      }
    }
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query_id), query_stride);
    if (Policy::should_update(acc, input_values[value_pos])) {
      output_values[value_pos] = acc;
      local_mask = query_bit(query_id);
    }
  }

  thread_masks[shared_index] = local_mask;
  __syncthreads();

  if (vertex < vertex_count && threadIdx.x == 0) {
    query_mask_t sweep_improved_mask = 0;
    int row_base = threadIdx.y * blockDim.x;
    for (int q = 0; q < query_count; ++q) {
      sweep_improved_mask |= thread_masks[row_base + q];
    }
    query_mask_t improved_mask = sweep_improved_mask;
    if (accumulate_frontier) {
      improved_mask |= next_frontier_mask[vertex];
    }
    next_frontier_mask[vertex] = improved_mask;
    atomic_or_query_mask(visited_mask + vertex, improved_mask);
    unique_flags[vertex] = improved_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved_mask));
    if (sweep_improved_mask != 0 && trace_update_count != nullptr) {
      const auto improved_count =
          static_cast<unsigned int>(mask_popcount(sweep_improved_mask));
      if (trace_update_count[vertex] == 0) {
        trace_first_iteration[vertex] = trace_iteration;
      }
      trace_iteration_sum[vertex] +=
          static_cast<unsigned long long>(trace_iteration) * improved_count;
      trace_update_count[vertex] += improved_count;
      trace_last_iteration[vertex] = trace_iteration;
    }
    if constexpr (track_active) {
      if (improved_mask != 0) {
        atomic_or_query_mask(active_union, improved_mask);
      }
    }
  }
}

template <int TILE_ROW, typename Policy, typename graph_t, typename vertex_t,
          bool track_active = false>
__global__ void fused_pull_smem_kernel(
    graph_t graph,
    int query_count,
    typename Policy::value_type* values,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts,
    query_mask_t active_slots,
    query_mask_t* active_union) {
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
  bool query0_active = query0 < query_count &&
      (active_slots & query_bit(query0)) != 0;
  bool query1_active = query1 < query_count &&
      (active_slots & query_bit(query1)) != 0;
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
      if (query0_active) {
        value_t nb_val =
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query0),
                               query_stride)];
        if (nb_val != Policy::infinity()) {
          value_t candidate = Policy::relax(nb_val, weight);
          if (Policy::should_update(candidate, acc0)) {
            acc0 = candidate;
          }
        }
      }
      if (query1_active) {
        value_t nb_val =
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query1),
                               query_stride)];
        if (nb_val != Policy::infinity()) {
          value_t candidate = Policy::relax(nb_val, weight);
          if (Policy::should_update(candidate, acc1)) {
            acc1 = candidate;
          }
        }
      }
    }
    __syncthreads();
  }

  query_mask_t local_mask = 0;
  if (row_valid && query0_active) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query0), query_stride);
    if (Policy::should_update(acc0, values[value_pos])) {
      values[value_pos] = acc0;
      local_mask |= query_bit(query0);
    }
  }
  if (row_valid && query1_active) {
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
    atomic_or_query_mask(visited_mask + vertex, improved_mask);
    unique_flags[vertex] = improved_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved_mask));
    if constexpr (track_active) {
      if (improved_mask != 0) {
        atomic_or_query_mask(active_union, improved_mask);
      }
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_range(
    graph_t graph, int query_count, typename Policy::value_type* values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, std::size_t vertex_begin,
    std::size_t vertex_end, bool reverse_vertices, cudaStream_t stream,
    unsigned int trace_iteration = 0,
    unsigned long long* trace_iteration_sum = nullptr,
    unsigned int* trace_update_count = nullptr,
    unsigned int* trace_first_iteration = nullptr,
    unsigned int* trace_last_iteration = nullptr,
    bool accumulate_frontier = false,
    const typename Policy::value_type* input_values = nullptr) {
  if (vertex_begin >= vertex_end)
    return;
  const auto* read_values = input_values == nullptr ? values : input_values;
  if (query_count <= 64) {
    // The row-query layout is faster than the smem-tiled variant on the
    // current Q=16/32/64 BFS hybrid matrix. Keep the smem kernel above for
    // focused experiments, but use simple as the default fused pull path.
    int tile_row = std::max(1, 128 / std::max(1, query_count));
    const auto range_size = vertex_end - vertex_begin;
    int grid_x = static_cast<int>((range_size + tile_row - 1) / tile_row);
    fused_pull_simple_kernel<Policy, graph_t, vertex_t>
        <<<grid_x, dim3(query_count, tile_row), 0, stream>>>(
            graph, query_count, read_values, values, visited_mask,
            next_frontier_mask, unique_flags, pair_counts, active_slots,
            nullptr, nullptr, 0, vertex_begin, vertex_end, reverse_vertices,
            trace_iteration, trace_iteration_sum, trace_update_count,
            trace_first_iteration, trace_last_iteration, accumulate_frontier);
  } else {
    throw std::invalid_argument("fused pull supports at most 64 queries");
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_indices(
    graph_t graph, int query_count, typename Policy::value_type* values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, const vertex_t* vertex_indices,
    std::size_t vertex_index_count, bool reverse_vertices,
    cudaStream_t stream, bool accumulate_frontier,
    const typename Policy::value_type* input_values) {
  if (vertex_index_count == 0) {
    return;
  }
  if (query_count <= 0 || query_count > 64) {
    throw std::invalid_argument("indexed fused pull supports 1-64 queries");
  }
  const int tile_row = std::max(1, 128 / query_count);
  const int grid_x = static_cast<int>(
      (vertex_index_count + static_cast<std::size_t>(tile_row) - 1) /
      static_cast<std::size_t>(tile_row));
  const auto* read_values = input_values == nullptr ? values : input_values;
  fused_pull_simple_kernel<Policy, graph_t, vertex_t>
      <<<grid_x, dim3(query_count, tile_row), 0, stream>>>(
          graph, query_count, read_values, values, visited_mask,
          next_frontier_mask, unique_flags, pair_counts, active_slots, nullptr,
          vertex_indices, vertex_index_count, 0,
          graph.get_number_of_vertices(), reverse_vertices, 0, nullptr,
          nullptr, nullptr, nullptr, accumulate_frontier);
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_degree_segmented(
    graph_t graph, int query_count, const vertex_t* bucket_vertices,
    const std::size_t* bucket_offsets, const int* bucket_order,
    const vertex_t* segment_vertices,
    const typename graph_t::edge_type* segment_begins,
    const typename graph_t::edge_type* segment_ends,
    std::size_t segment_count, typename Policy::value_type* values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, int segment_threads, cudaStream_t stream,
    const typename Policy::value_type* input_values = nullptr) {
  const auto* read_values = input_values == nullptr ? values : input_values;
  for (int position = 0; position < 4; ++position) {
    const int bucket = bucket_order[position];
    const auto* rows = bucket_vertices + bucket_offsets[bucket];
    const auto count = bucket_offsets[bucket + 1] - bucket_offsets[bucket];
    if (bucket != 3) {
      launch_fused_pull_indices<Policy, graph_t, vertex_t>(
          graph, query_count, values, visited_mask, next_frontier_mask,
          unique_flags, pair_counts, active_slots, rows, count, false, stream,
          false, read_values);
      continue;
    }
    if (segment_count != 0) {
      const int query_warps = query_count > 32 ? 2 : 1;
      const int segments_per_block =
          (segment_threads / 32) / query_warps;
      const int blocks = static_cast<int>(
          (segment_count + static_cast<std::size_t>(segments_per_block) - 1) /
          static_cast<std::size_t>(segments_per_block));
      fused_pull_high_degree_segments_kernel<Policy, graph_t, vertex_t>
          <<<blocks, segment_threads, 0, stream>>>(
              graph, query_count, segment_vertices, segment_begins,
              segment_ends, segment_count, read_values, values,
              next_frontier_mask, active_slots);
    }
  }

  const auto* high_degree_rows = bucket_vertices + bucket_offsets[3];
  const auto high_degree_count = bucket_offsets[4] - bucket_offsets[3];
  if (high_degree_count != 0) {
    constexpr int threads = 256;
    const auto raw_blocks =
        (high_degree_count + static_cast<std::size_t>(threads) - 1) /
        static_cast<std::size_t>(threads);
    const int blocks = static_cast<int>(
        raw_blocks < 65535 ? raw_blocks : static_cast<std::size_t>(65535));
    finalize_pull_high_degree_segments_kernel<<<blocks, threads, 0, stream>>>(
        high_degree_rows, high_degree_count, visited_mask,
        next_frontier_mask, unique_flags, pair_counts, active_slots);
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
                       query_mask_t active_slots,
                       cudaStream_t stream,
                       const typename Policy::value_type* input_values =
                           nullptr) {
  launch_fused_pull_range<Policy, graph_t, vertex_t>(
      graph, query_count, values, visited_mask, next_frontier_mask,
      unique_flags, pair_counts, active_slots, 0,
      graph.get_number_of_vertices(), false, stream, 0, nullptr, nullptr,
      nullptr, nullptr, false, input_values);
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_scheduled(
    graph_t graph, int query_count, typename Policy::value_type* values,
    query_mask_t* visited_mask, query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags, unsigned long long* pair_counts,
    query_mask_t active_slots, query_mask_t* active_union,
    cudaStream_t stream,
    const typename Policy::value_type* input_values = nullptr) {
  if (query_count <= 0 || query_count > 64) {
    throw std::invalid_argument(
        "scheduled fused pull supports between 1 and 64 queries");
  }
  const int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  const int tile_row = std::max(1, 128 / query_count);
  const int grid_x = (vertex_count + tile_row - 1) / tile_row;
  const auto* read_values = input_values == nullptr ? values : input_values;
  fused_pull_simple_kernel<Policy, graph_t, vertex_t, true>
      <<<grid_x, dim3(query_count, tile_row), 0, stream>>>(
          graph, query_count, read_values, values, visited_mask,
          next_frontier_mask, unique_flags, pair_counts, active_slots,
          active_union, nullptr, 0, 0, graph.get_number_of_vertices(), false,
          0, nullptr, nullptr, nullptr, nullptr, false);
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void fused_pull_sum_kernel(
    graph_t graph, int query_count, const float* old_values,
    float* next_values, const float* damping_factors, const float* epsilons,
    const vertex_t* personalization_sources, const float* dangling_mass,
    query_mask_t* next_frontier_mask, unsigned long long* unique_flags,
    unsigned long long* pair_counts, query_mask_t* changed_queries,
    query_mask_t active_slots) {
  static_assert(Policy::reduction == reduction_kind_t::sum,
                "sum pull requires a sum-reduction policy");
  __shared__ query_mask_t thread_masks[128];

  const std::size_t vertex_count = graph.get_number_of_vertices();
  const std::size_t vertex = blockIdx.x * blockDim.y + threadIdx.y;
  const int query_id = threadIdx.x;
  const int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  query_mask_t local_mask = 0;

  if (vertex < vertex_count && query_id < query_count) {
    const std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query_id),
                    static_cast<std::size_t>(query_count));
    const float old_value = old_values[value_pos];
    const query_mask_t query_mask = query_bit(query_id);

    if ((active_slots & query_mask) == 0) {
      next_values[value_pos] = old_value;
    } else {
      float incoming_sum = 0.0f;
      auto begin = get_pull_starting_edge(graph, static_cast<vertex_t>(vertex));
      auto end =
          get_pull_starting_edge(graph, static_cast<vertex_t>(vertex + 1));
      for (auto edge = begin; edge < end; ++edge) {
        const vertex_t source = get_pull_neighbor_vertex(graph, edge);
        const auto source_begin = graph.get_starting_edge(source);
        const auto source_end = graph.get_starting_edge(source + 1);
        const auto source_degree = source_end - source_begin;
        if (source_degree == 0) {
          continue;
        }
        incoming_sum +=
            old_values[value_index(static_cast<std::size_t>(source),
                                   static_cast<std::size_t>(query_id),
                                   static_cast<std::size_t>(query_count))] /
            static_cast<float>(source_degree);
      }

      const float personalization = Policy::personalization_value(
          static_cast<vertex_t>(vertex), query_id, vertex_count,
          personalization_sources);
      const float damping = damping_factors[query_id];
      const float next_value =
          (1.0f - damping) * personalization +
          damping *
              (incoming_sum + dangling_mass[query_id] * personalization);
      next_values[value_pos] = next_value;
      if (fabsf(next_value - old_value) > epsilons[query_id]) {
        local_mask = query_mask;
      }
    }
  }

  thread_masks[shared_index] = local_mask;
  __syncthreads();

  if (vertex < vertex_count && threadIdx.x == 0) {
    query_mask_t improved_mask = 0;
    const int row_base = threadIdx.y * blockDim.x;
    for (int q = 0; q < query_count; ++q) {
      improved_mask |= thread_masks[row_base + q];
    }
    next_frontier_mask[vertex] = improved_mask;
    unique_flags[vertex] = improved_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved_mask));
    if (improved_mask != 0) {
      atomic_or_query_mask(changed_queries, improved_mask);
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
void launch_fused_pull_sum(
    graph_t graph, int query_count, const float* old_values,
    float* next_values, const float* damping_factors, const float* epsilons,
    const vertex_t* personalization_sources, const float* dangling_mass,
    query_mask_t* next_frontier_mask, unsigned long long* unique_flags,
    unsigned long long* pair_counts, query_mask_t* changed_queries,
    query_mask_t active_slots, cudaStream_t stream) {
  if (query_count <= 0 || query_count > 64) {
    throw std::invalid_argument("sum pull supports between 1 and 64 queries");
  }
  const int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  const int tile_rows = std::max(1, 128 / query_count);
  const int grid_x = (vertex_count + tile_rows - 1) / tile_rows;
  fused_pull_sum_kernel<Policy, graph_t, vertex_t>
      <<<grid_x, dim3(query_count, tile_rows), 0, stream>>>(
          graph, query_count, old_values, next_values, damping_factors,
          epsilons, personalization_sources, dangling_mass,
          next_frontier_mask, unique_flags, pair_counts, changed_queries,
          active_slots);
}

}  // namespace detail
}  // namespace puercgp
