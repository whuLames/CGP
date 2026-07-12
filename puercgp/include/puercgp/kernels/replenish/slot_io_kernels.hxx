#pragma once

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include <puercgp/algorithms/hybrid.hxx>
#include <puercgp/core/atomics.hxx>
#include <puercgp/core/cuda_utils.hxx>
#include <puercgp/core/layout.hxx>
#include <puercgp/core/query_descriptor.hxx>

namespace puercgp {
namespace hybrid_detail {

using puercgp::detail::atomic_or_query_mask;
using puercgp::detail::grid_for;
using puercgp::detail::query_bit;
using puercgp::detail::throw_if_cuda_error;
using puercgp::detail::value_index;

__global__ void clear_single_slot_kernel(
    int slot,
    int query_count,
    std::size_t vertex_count,
    algorithms::unified_value_t* values,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    query_mask_t* next_frontier_mask) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  unsigned long long clear_mask_ll =
      static_cast<unsigned long long>(~query_bit(slot));
  std::size_t q_stride = static_cast<std::size_t>(query_count);
  for (std::size_t v = tid; v < vertex_count; v += stride) {
    values[value_index(v, static_cast<std::size_t>(slot), q_stride)] =
        algorithms::unified_infinity();
    atomicAnd(reinterpret_cast<unsigned long long*>(visited_mask + v),
              clear_mask_ll);
    atomicAnd(reinterpret_cast<unsigned long long*>(frontier_mask + v),
              clear_mask_ll);
    atomicAnd(reinterpret_cast<unsigned long long*>(next_frontier_mask + v),
              clear_mask_ll);
  }
}

__global__ void set_slot_source_kernel(
    int slot,
    algorithms::algo_kind_t kind,
    int source,
    algorithms::unified_value_t source_value,
    int query_count,
    algorithms::unified_value_t* values,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    int* frontier_vertices,
    unsigned long long* unique_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  query_mask_t bit = query_bit(slot);
  std::size_t q_stride = static_cast<std::size_t>(query_count);
  values[value_index(static_cast<std::size_t>(source),
                     static_cast<std::size_t>(slot), q_stride)] = source_value;
  if (kind == algorithms::algo_kind_t::bfs) {
    atomic_or_query_mask(visited_mask + source, bit);
  }
  query_mask_t old = atomic_or_query_mask(frontier_mask + source, bit);
  if (old == 0) {
    unsigned long long position = atomicAdd(unique_count, 1ULL);
    frontier_vertices[position] = source;
  }
}

template <typename graph_t>
void launch_reinit_single_slot(graph_t graph,
                               int slot,
                               query_descriptor_t desc,
                               int query_count,
                               algorithms::unified_value_t* values,
                               query_mask_t* visited_mask,
                               query_mask_t* frontier_mask,
                               query_mask_t* next_frontier_mask,
                               int* frontier_vertices,
                               unsigned long long* unique_count,
                               cudaStream_t stream) {
  if (desc.kind == algorithms::algo_kind_t::wcc) {
    throw std::invalid_argument(
        "reinit_single_slot does not support WCC (label pollution)");
  }
  constexpr int threads = 256;
  std::size_t V = graph.get_number_of_vertices();
  clear_single_slot_kernel<<<grid_for(V, threads), threads, 0, stream>>>(
      slot, query_count, V, values, visited_mask, frontier_mask,
      next_frontier_mask);
  set_slot_source_kernel<<<1, 1, 0, stream>>>(
      slot, desc.kind, desc.source, desc.source_value, query_count, values,
      visited_mask, frontier_mask, frontier_vertices, unique_count);
  throw_if_cuda_error(cudaGetLastError(), "reinit_single_slot");
}

__global__ void snapshot_slot_values_kernel(
    const algorithms::unified_value_t* values,
    const query_mask_t* visited_mask,
    int slot,
    algorithms::algo_kind_t kind,
    int query_count,
    std::size_t vertex_count,
    algorithms::unified_value_t* final_row) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  std::size_t q_stride = static_cast<std::size_t>(query_count);
  for (std::size_t v = tid; v < vertex_count; v += stride) {
    const bool bfs_unvisited =
        kind == algorithms::algo_kind_t::bfs &&
        (visited_mask[v] & query_bit(slot)) == 0;
    final_row[v] = bfs_unvisited
                       ? algorithms::unified_infinity()
                       : values[value_index(v, static_cast<std::size_t>(slot),
                                            q_stride)];
  }
}

inline void launch_snapshot_slot_values(
    int slot,
    algorithms::algo_kind_t kind,
    int query_count,
    std::size_t vertex_count,
    const algorithms::unified_value_t* values,
    const query_mask_t* visited_mask,
    algorithms::unified_value_t* final_row,
    cudaStream_t stream) {
  constexpr int threads = 256;
  snapshot_slot_values_kernel<<<grid_for(vertex_count, threads), threads, 0,
                                stream>>>(
      values, visited_mask, slot, kind, query_count, vertex_count, final_row);
  throw_if_cuda_error(cudaGetLastError(), "snapshot_slot_values");
}

__global__ void snapshot_multi_slot_kernel(
    const int* converged_slots,
    const int* final_query_ids,
    const algorithms::algo_kind_t* converged_kinds,
    int num_converged,
    int query_count,
    std::size_t vertex_count,
    const algorithms::unified_value_t* values,
    const query_mask_t* visited_mask,
    algorithms::unified_value_t* final_buffer,
    std::size_t final_row_stride) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  std::size_t q_stride = static_cast<std::size_t>(query_count);
  for (std::size_t v = tid; v < vertex_count; v += stride) {
    for (int i = 0; i < num_converged; ++i) {
      int s = converged_slots[i];
      std::size_t pos = value_index(v, static_cast<std::size_t>(s), q_stride);
      std::size_t final_pos =
          static_cast<std::size_t>(final_query_ids[i]) * final_row_stride + v;
      const bool bfs_unvisited =
          converged_kinds[i] == algorithms::algo_kind_t::bfs &&
          (visited_mask[v] & query_bit(s)) == 0;
      final_buffer[final_pos] =
          bfs_unvisited ? algorithms::unified_infinity() : values[pos];
    }
  }
}

__global__ void reset_reused_slots_kernel(
    const int* value_reset_slots,
    int num_value_reset_slots,
    int query_count,
    std::size_t vertex_count,
    algorithms::unified_value_t* values,
    query_mask_t* visited_mask,
    query_mask_t visited_keep_mask) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  std::size_t q_stride = static_cast<std::size_t>(query_count);
  const unsigned long long keep_ll =
      static_cast<unsigned long long>(visited_keep_mask);
  for (std::size_t v = tid; v < vertex_count; v += stride) {
    for (int i = 0; i < num_value_reset_slots; ++i) {
      int s = value_reset_slots[i];
      values[value_index(v, static_cast<std::size_t>(s), q_stride)] =
          algorithms::unified_infinity();
    }
    if (visited_keep_mask != ~query_mask_t{0}) {
      atomicAnd(reinterpret_cast<unsigned long long*>(visited_mask + v),
                keep_ll);
    }
  }
}

__global__ void set_sources_multi_kernel(
    const int* slots,
    const algorithms::algo_kind_t* kinds,
    const int* sources,
    const algorithms::unified_value_t* source_values,
    int num_slots,
    int query_count,
    algorithms::unified_value_t* values,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    int* frontier_vertices,
    unsigned long long* unique_count,
    int* slot_start_levels,
    int start_level) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(num_slots)) return;
  int s = slots[i];
  int src = sources[i];
  query_mask_t bit = query_bit(s);
  slot_start_levels[s] = start_level;
  std::size_t q_stride = static_cast<std::size_t>(query_count);
  values[value_index(static_cast<std::size_t>(src),
                     static_cast<std::size_t>(s), q_stride)] = source_values[i];
  if (kinds[i] == algorithms::algo_kind_t::bfs) {
    atomic_or_query_mask(visited_mask + src, bit);
  }
  query_mask_t old = atomic_or_query_mask(frontier_mask + src, bit);
  if (old == 0) {
    unsigned long long position = atomicAdd(unique_count, 1ULL);
    frontier_vertices[position] = src;
  }
}

inline void launch_snapshot_multi_slot(
    const int* converged_slots, const int* final_query_ids, int num_converged,
    const algorithms::algo_kind_t* converged_kinds,
    int query_count, std::size_t vertex_count,
    const algorithms::unified_value_t* values,
    const query_mask_t* visited_mask, algorithms::unified_value_t* final_buffer,
    std::size_t final_row_stride, cudaStream_t stream) {
  if (num_converged <= 0 || final_buffer == nullptr) return;
  constexpr int threads = 256;
  snapshot_multi_slot_kernel<<<grid_for(vertex_count, threads), threads, 0,
                               stream>>>(
      converged_slots, final_query_ids, converged_kinds, num_converged, query_count,
      vertex_count, values, visited_mask, final_buffer, final_row_stride);
  throw_if_cuda_error(cudaGetLastError(), "snapshot_multi_slot");
}

inline void launch_reset_reused_slots(
    const int* value_reset_slots, int num_value_reset_slots,
    query_mask_t visited_clear_bits, int query_count,
    std::size_t vertex_count, algorithms::unified_value_t* values,
    query_mask_t* visited_mask, cudaStream_t stream) {
  if (num_value_reset_slots <= 0 && visited_clear_bits == 0) return;
  constexpr int threads = 256;
  reset_reused_slots_kernel<<<grid_for(vertex_count, threads), threads, 0,
                              stream>>>(
      value_reset_slots, num_value_reset_slots, query_count, vertex_count,
      values, visited_mask, ~visited_clear_bits);
  throw_if_cuda_error(cudaGetLastError(), "reset_reused_slots");
}

inline void launch_set_sources_multi(
    const int* slots, const algorithms::algo_kind_t* kinds,
    const int* sources, const algorithms::unified_value_t* source_values,
    int num_slots, int query_count,
    algorithms::unified_value_t* values,
    query_mask_t* visited_mask, query_mask_t* frontier_mask,
    int* frontier_vertices, unsigned long long* unique_count,
    int* slot_start_levels, int start_level,
    cudaStream_t stream) {
  if (num_slots <= 0) return;
  constexpr int threads = 64;
  set_sources_multi_kernel<<<grid_for(num_slots, threads), threads, 0,
                             stream>>>(
      slots, kinds, sources, source_values, num_slots, query_count, values,
      visited_mask, frontier_mask, frontier_vertices, unique_count,
      slot_start_levels, start_level);
  throw_if_cuda_error(cudaGetLastError(), "set_sources_multi");
}

}  // namespace hybrid_detail
}  // namespace puercgp
