#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include <puercgp/algorithms/bfs.hxx>
#include <puercgp/algorithms/sssp.hxx>
#include <puercgp/algorithms/wcc.hxx>
#include <puercgp/backend/graph_adapter.hxx>
#include <puercgp/core/atomics.hxx>
#include <puercgp/core/cuda_utils.hxx>
#include <puercgp/core/layout.hxx>
#include <puercgp/core/mask.hxx>
#include <puercgp/core/query_batch.hxx>
#include <puercgp/core/result.hxx>
#include <puercgp/engine/pull_executor.hxx>
#include <puercgp/kernels/common/pull_postprocess.hxx>
#include <puercgp/kernels/pull/fused_pull_kernels.hxx>
#include <puercgp/kernels/push/shared_push_kernels.hxx>
#include <puercgp/state/frontier_storage.hxx>

namespace puercgp {
namespace detail {

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void init_list_kernel(graph_t graph,
                                 const vertex_t* sources,
                                 int query_count,
                                 typename Policy::value_type* values,
                                 frontier_item_t<vertex_t>* frontier) {
  using value_t = typename Policy::value_type;
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t total = static_cast<std::size_t>(query_count) * vertex_count;
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < total; i += stride) {
    values[i] = Policy::infinity();
  }
  for (std::size_t q = tid; q < static_cast<std::size_t>(query_count);
       q += stride) {
    vertex_t source = sources[q];
    values[value_index(static_cast<std::size_t>(source), q,
                       static_cast<std::size_t>(query_count))] =
        Policy::source_value();
    frontier[q] = {static_cast<vertex_t>(q), source};
  }
  (void)static_cast<value_t>(0);
}

__global__ void reset_counter_kernel(unsigned long long* counter) {
  *counter = 0ULL;
}

template <typename Policy, typename value_t>
__global__ void fill_values_kernel(value_t* values, std::size_t total) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < total; i += stride) {
    values[i] = Policy::infinity();
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void init_shared_sources_kernel(
    graph_t graph,
    const vertex_t* sources,
    int query_count,
    typename Policy::value_type* values,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    vertex_t* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t q = tid; q < static_cast<std::size_t>(query_count);
       q += stride) {
    vertex_t source = sources[q];
    query_mask_t bit = query_bit(static_cast<int>(q));
    values[value_index(static_cast<std::size_t>(source), q,
                       static_cast<std::size_t>(query_count))] =
        Policy::source_value();
    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      atomic_or_query_mask(visited_mask + source, bit);
    }
    query_mask_t old = atomic_or_query_mask(frontier_mask + source, bit);
    if (old == 0) {
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = source;
    }
  }
}

// WCC per-vertex init：每个顶点的 label = vertex_id，所有顶点入 frontier
// 仅在 Policy == wcc_policy 时由 init 段调用
template <typename graph_t, typename vertex_t>
__global__ void init_wcc_labels_kernel(
    graph_t graph,
    int query_count,
    float* values,
    query_mask_t* frontier_mask,
    vertex_t* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t v = tid; v < vertex_count; v += stride) {
    for (int q = 0; q < query_count; ++q) {
      values[value_index(v, static_cast<std::size_t>(q),
                         static_cast<std::size_t>(query_count))] =
          static_cast<float>(static_cast<int>(v));  // label = vertex id
    }
    for (int q = 0; q < query_count; ++q) {
      query_mask_t bit = query_bit(q);
      query_mask_t old = atomic_or_query_mask(frontier_mask + v, bit);
      if (old == 0) {
        unsigned long long position = atomicAdd(unique_count, 1ULL);
        frontier_vertices[position] = static_cast<vertex_t>(v);
      }
    }
  }
}

template <typename vertex_t>
__global__ void list_to_bitmap_kernel(const frontier_item_t<vertex_t>* frontier,
                                      std::size_t count,
                                      int query_count,
                                      unsigned char* bitmap) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < count; i += stride) {
    auto item = frontier[i];
    bitmap[static_cast<std::size_t>(item.vertex) *
               static_cast<std::size_t>(query_count) +
           static_cast<std::size_t>(item.query_id)] = 1;
  }
}

template <typename vertex_t>
__global__ void shared_to_bitmap_kernel(const vertex_t* frontier_vertices,
                                        const query_mask_t* frontier_mask,
                                        std::size_t unique_count,
                                        int query_count,
                                        unsigned char* bitmap) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < unique_count; i += stride) {
    vertex_t vertex = frontier_vertices[i];
    query_mask_t mask = frontier_mask[vertex];
    while (mask != 0) {
      int query_id = mask_ffs(mask) - 1;
      if (query_id < query_count) {
        bitmap[static_cast<std::size_t>(vertex) *
                   static_cast<std::size_t>(query_count) +
               static_cast<std::size_t>(query_id)] = 1;
      }
      mask &= (mask - 1);
    }
  }
}

template <typename Policy, typename vertex_t>
__global__ void list_to_dense_kernel(const frontier_item_t<vertex_t>* frontier,
                                     std::size_t count,
                                     int m_eff,
                                     const typename Policy::value_type* values,
                                     std::size_t vertex_count,
                                     float* dense) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < count; i += stride) {
    auto item = frontier[i];
    std::size_t dense_index =
        static_cast<std::size_t>(item.vertex) * static_cast<std::size_t>(m_eff) +
        static_cast<std::size_t>(item.query_id);
    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      dense[dense_index] = 1.0f;
    } else {
      dense[dense_index] =
          values[value_index(static_cast<std::size_t>(item.vertex),
                             static_cast<std::size_t>(item.query_id),
                             static_cast<std::size_t>(m_eff))];
    }
  }
}

template <typename Policy, typename vertex_t>
__global__ void shared_to_dense_kernel(const vertex_t* frontier_vertices,
                                       const query_mask_t* frontier_mask,
                                       std::size_t unique_count,
                                       int query_count,
                                       int m_eff,
                                       const typename Policy::value_type* values,
                                       std::size_t vertex_count,
                                       float* dense) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < unique_count; i += stride) {
    vertex_t vertex = frontier_vertices[i];
    query_mask_t mask = frontier_mask[vertex];
    while (mask != 0) {
      int query_id = mask_ffs(mask) - 1;
      if (query_id < query_count) {
        std::size_t dense_index =
            static_cast<std::size_t>(vertex) * static_cast<std::size_t>(m_eff) +
            static_cast<std::size_t>(query_id);
        if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
          dense[dense_index] = 1.0f;
        } else {
          dense[dense_index] =
              values[value_index(static_cast<std::size_t>(vertex),
                                 static_cast<std::size_t>(query_id),
                                 static_cast<std::size_t>(m_eff))];
        }
      }
      mask &= (mask - 1);
    }
  }
}

template <typename vertex_t>
__global__ void bitmap_to_list_kernel(const unsigned char* bitmap,
                                      std::size_t total_pairs,
                                      int query_count,
                                      frontier_item_t<vertex_t>* frontier,
                                      unsigned long long* out_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < total_pairs; i += stride) {
    if (bitmap[i] == 0) {
      continue;
    }
    unsigned long long position = atomicAdd(out_count, 1ULL);
    frontier[position] = {
        static_cast<vertex_t>(i % static_cast<std::size_t>(query_count)),
        static_cast<vertex_t>(i / static_cast<std::size_t>(query_count))};
  }
}

template <typename Policy, typename vertex_t>
__global__ void dense_to_list_kernel(const float* dense,
                                     std::size_t vertex_count,
                                     int query_count,
                                     int m_eff,
                                     frontier_item_t<vertex_t>* frontier,
                                     unsigned long long* out_count) {
  std::size_t total_pairs =
      vertex_count * static_cast<std::size_t>(query_count);
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < total_pairs; i += stride) {
    std::size_t vertex = i / static_cast<std::size_t>(query_count);
    std::size_t query_id = i % static_cast<std::size_t>(query_count);
    float value = dense[vertex * static_cast<std::size_t>(m_eff) + query_id];
    bool active = false;
    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      active = value != 0.0f;
    } else {
      active = value != Policy::infinity();
    }
    if (!active) {
      continue;
    }
    unsigned long long position = atomicAdd(out_count, 1ULL);
    frontier[position] = {static_cast<vertex_t>(query_id),
                          static_cast<vertex_t>(vertex)};
  }
}

template <typename vertex_t>
__global__ void bitmap_to_shared_kernel(const unsigned char* bitmap,
                                        std::size_t vertex_count,
                                        int query_count,
                                        query_mask_t* frontier_mask,
                                        vertex_t* frontier_vertices,
                                        unsigned long long* unique_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t vertex = tid; vertex < vertex_count; vertex += stride) {
    query_mask_t mask = 0;
    for (int q = 0; q < query_count; ++q) {
      if (bitmap[vertex * static_cast<std::size_t>(query_count) +
                 static_cast<std::size_t>(q)] != 0) {
        mask |= query_bit(q);
      }
    }
    if (mask != 0) {
      frontier_mask[vertex] = mask;
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = static_cast<vertex_t>(vertex);
    }
  }
}

template <typename Policy, typename vertex_t>
__global__ void dense_to_shared_kernel(const float* dense,
                                       std::size_t vertex_count,
                                       int query_count,
                                       int m_eff,
                                       query_mask_t* frontier_mask,
                                       vertex_t* frontier_vertices,
                                       unsigned long long* unique_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t vertex = tid; vertex < vertex_count; vertex += stride) {
    query_mask_t mask = 0;
    for (int q = 0; q < query_count; ++q) {
      float value = dense[vertex * static_cast<std::size_t>(m_eff) +
                          static_cast<std::size_t>(q)];
      bool active = false;
      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        active = value != 0.0f;
      } else {
        active = value != Policy::infinity();
      }
      if (active) {
        mask |= query_bit(q);
      }
    }
    if (mask != 0) {
      frontier_mask[vertex] = mask;
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = static_cast<vertex_t>(vertex);
    }
  }
}

template <typename graph_t, typename vertex_t>
__global__ void compute_degrees_kernel(graph_t graph,
                                       const frontier_item_t<vertex_t>* frontier,
                                       std::size_t count,
                                       unsigned long long* degrees) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < count; i += stride) {
    vertex_t vertex = frontier[i].vertex;
    degrees[i] = static_cast<unsigned long long>(
        graph.get_starting_edge(vertex + 1) - graph.get_starting_edge(vertex));
  }
}

template <typename graph_t>
__global__ void compute_bitmap_degrees_kernel(graph_t graph,
                                              const unsigned char* bitmap,
                                              std::size_t total_pairs,
                                              int query_count,
                                              unsigned long long* edge_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < total_pairs; i += stride) {
    if (bitmap[i] == 0) {
      continue;
    }
    auto vertex = static_cast<typename graph_t::vertex_type>(
        i / static_cast<std::size_t>(query_count));
    auto degree = static_cast<unsigned long long>(
        graph.get_starting_edge(vertex + 1) - graph.get_starting_edge(vertex));
    atomicAdd(edge_count, degree);
  }
}

// 计算节点共享信息
template <typename graph_t, typename vertex_t>
__global__ void compute_shared_degrees_kernel(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    unsigned long long* actual_degrees,
    unsigned long long* virtual_degrees) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x; // The number of launched thread
  for (std::size_t i = tid; i < unique_count; i += stride) {
    vertex_t vertex = frontier_vertices[i];
    query_mask_t mask = frontier_mask[vertex];

    auto degree = static_cast<unsigned long long>(
        graph.get_starting_edge(vertex + 1) - graph.get_starting_edge(vertex));

    actual_degrees[i] = degree;
    virtual_degrees[i] =
        degree * static_cast<unsigned long long>(mask_popcount(mask));
  }
}

__device__ __forceinline__ std::size_t find_frontier_index_by_end(
    const unsigned long long* edge_ends,
    std::size_t count,
    unsigned long long edge_ordinal) {
  std::size_t left = 0;
  std::size_t right = count;
  while (left < right) {
    std::size_t mid = left + ((right - left) >> 1);
    if (edge_ends[mid] <= edge_ordinal) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void expand_edge_balanced_kernel(
    graph_t graph,
    const frontier_item_t<vertex_t>* in_frontier,
    std::size_t in_count,
    const unsigned long long* edge_offsets,
    const unsigned long long* edge_ends,
    unsigned long long total_edges,
    frontier_item_t<vertex_t>* out_frontier,
    unsigned long long* out_count,
    typename Policy::value_type* values,
    int query_count,
    vertex_t level) {
  using value_t = typename Policy::value_type;
  std::size_t vertex_count = graph.get_number_of_vertices();
  unsigned long long global_stride =
      static_cast<unsigned long long>(blockDim.x) * gridDim.x;

  __shared__ unsigned int block_count;
  __shared__ unsigned long long block_base;

  for (unsigned long long block_edge_base =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x;
       block_edge_base < total_edges; block_edge_base += global_stride) {
    if (threadIdx.x == 0) {
      block_count = 0;
    }
    __syncthreads();

    bool discovered = false;
    unsigned int local_position = 0;
    frontier_item_t<vertex_t> output_item{};

    unsigned long long edge_ordinal = block_edge_base + threadIdx.x;
    if (edge_ordinal < total_edges) {
      std::size_t frontier_index =
          find_frontier_index_by_end(edge_ends, in_count, edge_ordinal);
      auto item = in_frontier[frontier_index];
      auto source = item.vertex;
      auto query_id = item.query_id;
      auto query_stride = static_cast<std::size_t>(query_count);
      auto local_edge = edge_ordinal - edge_offsets[frontier_index];
      auto edge = graph.get_starting_edge(source) + local_edge;
      vertex_t neighbor = graph.get_destination_vertex(edge);

      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        value_t* distance =
            values + value_index(static_cast<std::size_t>(neighbor),
                                 static_cast<std::size_t>(query_id),
                                 query_stride);
        value_t old =
            atomicCAS(distance, Policy::infinity(), static_cast<value_t>(level + 1));
        if (old == Policy::infinity()) {
          local_position = atomicAdd(&block_count, 1U);
          output_item = {query_id, neighbor};
          discovered = true;
        }
      } else {
        value_t source_distance =
            values[value_index(static_cast<std::size_t>(source),
                               static_cast<std::size_t>(query_id),
                               query_stride)];
        if (source_distance != Policy::infinity()) {
          value_t candidate = Policy::relax(source_distance,
                                            graph.get_edge_weight(edge));
          value_t* distance =
              values + value_index(static_cast<std::size_t>(neighbor),
                                   static_cast<std::size_t>(query_id),
                                   query_stride);
          value_t old = atomic_min_value(distance, candidate);
          if (candidate < old) {
            local_position = atomicAdd(&block_count, 1U);
            output_item = {query_id, neighbor};
            discovered = true;
          }
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0 && block_count > 0) {
      block_base =
          atomicAdd(out_count, static_cast<unsigned long long>(block_count));
    }
    __syncthreads();
    if (discovered) {
      out_frontier[block_base + local_position] = output_item;
    }
    __syncthreads();
  }
}

template <typename vertex_t>
__global__ void clear_shared_frontier_mask_kernel(
    const vertex_t* frontier_vertices,
    std::size_t unique_count,
    query_mask_t* frontier_mask) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < unique_count; i += stride) {
    frontier_mask[frontier_vertices[i]] = 0;
  }
}

constexpr int shared_node_degree_low_threshold = 8;
constexpr int shared_node_degree_high_threshold = 512;

template <typename graph_t, typename vertex_t>
__global__ void expand_shared_node_degree_low_bfs_kernel(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    algorithms::bfs_policy::value_type* values,
    int query_count,
    vertex_t level) {
  using value_t = algorithms::bfs_policy::value_type;
  constexpr int subwarp_size = 8;
  int lane = threadIdx.x & (subwarp_size - 1);
  std::size_t subwarp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      subwarp_size;
  std::size_t subwarp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) / subwarp_size;

  for (std::size_t i = subwarp_id; i < unique_count; i += subwarp_stride) {
    vertex_t source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source];
    if (active_mask == 0) {
      continue;
    }
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);
    auto degree = end - begin;
    if (degree > shared_node_degree_low_threshold) {
      continue;
    }

    if (lane < degree) {
      auto edge = begin + lane;
      vertex_t neighbor = graph.get_destination_vertex(edge);
      query_mask_t old_visited =
          atomic_or_query_mask(visited_mask + neighbor, active_mask);
      query_mask_t improved = active_mask & ~old_visited;
      query_mask_t bits = improved;

      while (bits != 0) {
        int query_id = mask_ffs(bits) - 1;
        if (query_id < query_count) {
          values[value_index(static_cast<std::size_t>(neighbor),
                             static_cast<std::size_t>(query_id),
                             static_cast<std::size_t>(query_count))] =
              static_cast<value_t>(level + 1);
        }
        bits &= (bits - 1);
      }

      mark_next_shared_frontier(neighbor, improved, next_frontier_mask,
                                next_frontier_vertices, next_unique_count,
                                next_pair_count);
    }
  }
}

template <int query_tile, typename graph_t, typename vertex_t>
__global__ void expand_shared_node_degree_medium_bfs_kernel(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    algorithms::bfs_policy::value_type* values,
    int query_count,
    vertex_t level) {
  using value_t = algorithms::bfs_policy::value_type;
  constexpr int warp_size = 32;
  constexpr int edge_lanes = warp_size / query_tile;
  static_assert(query_tile == 2 || query_tile == 4 || query_tile == 8 ||
                    query_tile == 16 || query_tile == 32,
                "unsupported query tile");

  int lane = threadIdx.x & (warp_size - 1);
  int edge_slot = lane / query_tile;
  int query_slot = lane - edge_slot * query_tile;
  int leader_lane = edge_slot * query_tile;
  unsigned int tile_mask = 0xffffffffU;
  if constexpr (query_tile < warp_size) {
    tile_mask =
        ((1U << query_tile) - 1U) << static_cast<unsigned int>(leader_lane);
  }
  std::size_t warp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  std::size_t warp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) >> 5;

  for (std::size_t i = warp_id; i < unique_count; i += warp_stride) {
    vertex_t source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source];
    if (active_mask == 0) {
      continue;
    }
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);
    auto degree = end - begin;
    if (degree <= shared_node_degree_low_threshold ||
        degree > shared_node_degree_high_threshold) {
      continue;
    }

    for (auto local_edge = edge_slot; local_edge < degree;
         local_edge += edge_lanes) {
      auto edge = begin + local_edge;
      vertex_t neighbor = graph.get_destination_vertex(edge);
      query_mask_t improved = 0;
      if (query_slot == 0) {
        query_mask_t old_visited =
            atomic_or_query_mask(visited_mask + neighbor, active_mask);
        improved = active_mask & ~old_visited;
      }
      unsigned long long improved_bits = static_cast<unsigned long long>(improved);
      improved_bits = __shfl_sync(tile_mask, improved_bits, leader_lane);
      improved = static_cast<query_mask_t>(improved_bits);

      for (int query_id = query_slot; query_id < query_count;
           query_id += query_tile) {
        if ((improved & query_bit(query_id)) != 0) {
          values[value_index(static_cast<std::size_t>(neighbor),
                             static_cast<std::size_t>(query_id),
                             static_cast<std::size_t>(query_count))] =
              static_cast<value_t>(level + 1);
        }
      }
      __syncwarp(tile_mask);

      if (query_slot == 0) {
        mark_next_shared_frontier(neighbor, improved, next_frontier_mask,
                                  next_frontier_vertices, next_unique_count,
                                  next_pair_count);
      }
    }
  }
}

template <typename graph_t, typename vertex_t>
__global__ void expand_shared_node_degree_high_bfs_kernel(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    algorithms::bfs_policy::value_type* values,
    int query_count,
    vertex_t level) {
  using value_t = algorithms::bfs_policy::value_type;

  for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
    vertex_t source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source];
    if (active_mask == 0) {
      continue;
    }
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);
    auto degree = end - begin;
    if (degree <= shared_node_degree_high_threshold) {
      continue;
    }

    for (auto edge = begin + threadIdx.x; edge < end; edge += blockDim.x) {
      vertex_t neighbor = graph.get_destination_vertex(edge);
      query_mask_t old_visited =
          atomic_or_query_mask(visited_mask + neighbor, active_mask);
      query_mask_t improved = active_mask & ~old_visited;
      query_mask_t bits = improved;

      while (bits != 0) {
        int query_id = mask_ffs(bits) - 1;
        if (query_id < query_count) {
          values[value_index(static_cast<std::size_t>(neighbor),
                             static_cast<std::size_t>(query_id),
                             static_cast<std::size_t>(query_count))] =
              static_cast<value_t>(level + 1);
        }
        bits &= (bits - 1);
      }

      mark_next_shared_frontier(neighbor, improved, next_frontier_mask,
                                next_frontier_vertices, next_unique_count,
                                next_pair_count);
    }
  }
}

template <int query_tile, typename graph_t, typename vertex_t>
void launch_expand_shared_node_degree_medium_bfs(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    algorithms::bfs_policy::value_type* values,
    int query_count,
    vertex_t level,
    int blocks,
    int threads,
    cudaStream_t stream) {
  expand_shared_node_degree_medium_bfs_kernel<query_tile, graph_t, vertex_t>
      <<<blocks, threads, 0, stream>>>(
          graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
          next_frontier_mask, next_frontier_vertices, next_unique_count,
          next_pair_count, values, query_count, level);
}

template <typename graph_t, typename vertex_t>
void launch_expand_shared_node_degree_bfs(
    graph_t graph,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    algorithms::bfs_policy::value_type* values,
    int query_count,
    vertex_t level,
    int threads,
    cudaStream_t stream) {
  int low_blocks = grid_for(unique_count * 8ULL, threads);
  int warp_blocks = grid_for(unique_count * 32ULL, threads);
  int high_blocks = static_cast<int>(
      std::min<std::size_t>(std::max<std::size_t>(unique_count, 1), 65535));

  expand_shared_node_degree_low_bfs_kernel<graph_t, vertex_t>
      <<<low_blocks, threads, 0, stream>>>(
          graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
          next_frontier_mask, next_frontier_vertices, next_unique_count,
          next_pair_count, values, query_count, level);

  if (query_count <= 2) {
    launch_expand_shared_node_degree_medium_bfs<2>(
        graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
        next_frontier_mask, next_frontier_vertices, next_unique_count,
        next_pair_count, values, query_count, level, warp_blocks, threads,
        stream);
  } else if (query_count <= 4) {
    launch_expand_shared_node_degree_medium_bfs<4>(
        graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
        next_frontier_mask, next_frontier_vertices, next_unique_count,
        next_pair_count, values, query_count, level, warp_blocks, threads,
        stream);
  } else if (query_count <= 8) {
    launch_expand_shared_node_degree_medium_bfs<8>(
        graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
        next_frontier_mask, next_frontier_vertices, next_unique_count,
        next_pair_count, values, query_count, level, warp_blocks, threads,
        stream);
  } else if (query_count <= 16) {
    launch_expand_shared_node_degree_medium_bfs<16>(
        graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
        next_frontier_mask, next_frontier_vertices, next_unique_count,
        next_pair_count, values, query_count, level, warp_blocks, threads,
        stream);
  } else {
    launch_expand_shared_node_degree_medium_bfs<32>(
        graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
        next_frontier_mask, next_frontier_vertices, next_unique_count,
        next_pair_count, values, query_count, level, warp_blocks, threads,
        stream);
  }

  expand_shared_node_degree_high_bfs_kernel<graph_t, vertex_t>
      <<<high_blocks, threads, 0, stream>>>(
          graph, frontier_vertices, frontier_mask, unique_count, visited_mask,
          next_frontier_mask, next_frontier_vertices, next_unique_count,
          next_pair_count, values, query_count, level);
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void pull_expand_kernel(graph_t graph,
                                   int query_count,
                                   const unsigned char* frontier_bitmap,
                                   unsigned char* next_frontier_bitmap,
                                   unsigned long long* out_count,
                                   query_mask_t* visited_mask,
                                   query_mask_t* next_frontier_mask,
                                   vertex_t* next_frontier_vertices,
                                   unsigned long long* next_unique_count,
                                   bool update_shared,
                                   typename Policy::value_type* values,
                                   vertex_t level) {
  using value_t = typename Policy::value_type;
  constexpr unsigned int warp_size = 32;
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t total_pairs =
      vertex_count * static_cast<std::size_t>(query_count);
  unsigned int lane = threadIdx.x & (warp_size - 1);
  std::size_t warp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  std::size_t warp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) >> 5;

  for (std::size_t pair = warp_id; pair < total_pairs; pair += warp_stride) {
    vertex_t query_id =
        static_cast<vertex_t>(pair % static_cast<std::size_t>(query_count));
    vertex_t vertex =
        static_cast<vertex_t>(pair / static_cast<std::size_t>(query_count));
    std::size_t query_stride = static_cast<std::size_t>(query_count);

    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      if (frontier_bitmap[static_cast<std::size_t>(vertex) *
                              static_cast<std::size_t>(query_count) +
                          static_cast<std::size_t>(query_id)] == 0) {
        continue;
      }
      auto begin = graph.get_starting_edge(vertex);
      auto end = graph.get_starting_edge(vertex + 1);
      for (auto edge = begin + lane; edge < end; edge += warp_size) {
        vertex_t neighbor = graph.get_destination_vertex(edge);
        value_t* distance =
            values + value_index(static_cast<std::size_t>(neighbor),
                                 static_cast<std::size_t>(query_id),
                                 query_stride);
        value_t old = atomicCAS(distance, Policy::infinity(),
                                static_cast<value_t>(level + 1));
        if (old != Policy::infinity()) {
          continue;
        }
        next_frontier_bitmap[static_cast<std::size_t>(neighbor) *
                                 static_cast<std::size_t>(query_count) +
                             static_cast<std::size_t>(query_id)] = 1;
        if (update_shared) {
          query_mask_t bit = query_bit(query_id);
          atomic_or_query_mask(visited_mask + neighbor, bit);
          query_mask_t old_next =
              atomic_or_query_mask(next_frontier_mask + neighbor, bit);
          if (old_next == 0) {
            unsigned long long position = atomicAdd(next_unique_count, 1ULL);
            next_frontier_vertices[position] = neighbor;
          }
        }
        atomicAdd(out_count, 1ULL);
      }
    } else {
      if (frontier_bitmap[static_cast<std::size_t>(vertex) *
                              static_cast<std::size_t>(query_count) +
                          static_cast<std::size_t>(query_id)] == 0) {
        continue;
      }
      value_t source_distance =
          values[value_index(static_cast<std::size_t>(vertex),
                             static_cast<std::size_t>(query_id),
                             query_stride)];
      if (source_distance == Policy::infinity()) {
        continue;
      }
      auto begin = graph.get_starting_edge(vertex);
      auto end = graph.get_starting_edge(vertex + 1);
      for (auto edge = begin + lane; edge < end; edge += warp_size) {
        vertex_t neighbor = graph.get_destination_vertex(edge);
        value_t candidate =
            Policy::relax(source_distance, graph.get_edge_weight(edge));
        value_t old = atomic_min_value(
            values + value_index(static_cast<std::size_t>(neighbor),
                                 static_cast<std::size_t>(query_id),
                                 query_stride),
            candidate);
        if (candidate >= old) {
          continue;
        }
        next_frontier_bitmap[static_cast<std::size_t>(neighbor) *
                                 static_cast<std::size_t>(query_count) +
                             static_cast<std::size_t>(query_id)] = 1;
        if (update_shared) {
          query_mask_t bit = query_bit(query_id);
          query_mask_t old_next =
              atomic_or_query_mask(next_frontier_mask + neighbor, bit);
          if (old_next == 0) {
            unsigned long long position = atomicAdd(next_unique_count, 1ULL);
            next_frontier_vertices[position] = neighbor;
          }
        }
        atomicAdd(out_count, 1ULL);
      }
    }
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void pull_values_simple_kernel(
    graph_t graph,
    int query_count,
    typename Policy::value_type* values,
    vertex_t level,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count) {
  using value_t = typename Policy::value_type;
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);
  std::size_t vertex = blockIdx.x * blockDim.y + threadIdx.y;
  int query_id = threadIdx.x;
  if (vertex >= vertex_count || query_id >= query_count) {
    return;
  }

  auto source_index = value_index(vertex, static_cast<std::size_t>(query_id),
                                  query_stride);
  auto begin = graph.get_starting_edge(static_cast<vertex_t>(vertex));
  auto end = graph.get_starting_edge(static_cast<vertex_t>(vertex + 1));
  if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
    if (values[source_index] != static_cast<value_t>(level)) {
      return;
    }
    for (auto edge = begin; edge < end; ++edge) {
      vertex_t neighbor = graph.get_destination_vertex(edge);
      auto neighbor_index =
          value_index(static_cast<std::size_t>(neighbor),
                      static_cast<std::size_t>(query_id), query_stride);
      value_t old = atomicCAS(values + neighbor_index, Policy::infinity(),
                              static_cast<value_t>(level + 1));
      if (old == Policy::infinity()) {
        query_mask_t bit = query_bit(query_id);
        if (visited_mask != nullptr) {
          atomic_or_query_mask(visited_mask + neighbor, bit);
        }
        mark_next_shared_frontier(neighbor, bit, next_frontier_mask,
                                  next_frontier_vertices, next_unique_count,
                                  next_pair_count);
      }
    }
  } else {
    value_t source_distance = values[source_index];
    if (source_distance == Policy::infinity()) {
      return;
    }
    for (auto edge = begin; edge < end; ++edge) {
      vertex_t neighbor = graph.get_destination_vertex(edge);
      value_t candidate =
          Policy::relax(source_distance, graph.get_edge_weight(edge));
      value_t old = atomic_min_value(
          values + value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query_id),
                               query_stride),
          candidate);
      if (candidate < old) {
        mark_next_shared_frontier(neighbor, query_bit(query_id),
                                  next_frontier_mask, next_frontier_vertices,
                                  next_unique_count, next_pair_count);
      }
    }
  }
}

template <typename graph_t, typename vertex_t>
__global__ void pull_bfs_source_mask_kernel(
    graph_t graph,
    int query_count,
    typename algorithms::bfs_policy::value_type* values,
    vertex_t level,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count) {
  using value_t = typename algorithms::bfs_policy::value_type;
  __shared__ query_mask_t query_masks[256];
  __shared__ query_mask_t active_mask;

  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);

  for (std::size_t vertex = blockIdx.x; vertex < vertex_count;
       vertex += gridDim.x) {
    query_mask_t local_mask = 0;
    if (threadIdx.x < static_cast<unsigned int>(query_count)) {
      int query_id = static_cast<int>(threadIdx.x);
      value_t distance =
          values[value_index(vertex, static_cast<std::size_t>(query_id),
                             query_stride)];
      if (distance == static_cast<value_t>(level)) {
        local_mask = query_bit(query_id);
      }
    }
    query_masks[threadIdx.x] = local_mask;
    __syncthreads();

    if (threadIdx.x == 0) {
      query_mask_t mask = 0;
      for (int q = 0; q < query_count; ++q) {
        mask |= query_masks[q];
      }
      active_mask = mask;
    }
    __syncthreads();

    query_mask_t mask = active_mask;
    if (mask != 0) {
      auto begin =
          graph.get_starting_edge(static_cast<vertex_t>(vertex));
      auto end =
          graph.get_starting_edge(static_cast<vertex_t>(vertex + 1));
      for (auto edge = begin + threadIdx.x; edge < end; edge += blockDim.x) {
        vertex_t neighbor = graph.get_destination_vertex(edge);
        query_mask_t old_visited =
            atomic_or_query_mask(visited_mask + neighbor, mask);
        query_mask_t improved = mask & ~old_visited;
        query_mask_t bits = improved;
        while (bits != 0) {
          int query_id = mask_ffs(bits) - 1;
          if (query_id < query_count) {
            values[value_index(static_cast<std::size_t>(neighbor),
                               static_cast<std::size_t>(query_id),
                               query_stride)] =
                static_cast<value_t>(level + 1);
          }
          bits &= (bits - 1);
        }
        mark_next_shared_frontier(neighbor, improved, next_frontier_mask,
                                  next_frontier_vertices, next_unique_count,
                                  next_pair_count);
      }
    }
    __syncthreads();
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void pull_values_warp_coarsened_kernel(
    graph_t graph,
    int query_count,
    typename Policy::value_type* values,
    vertex_t level,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    int tile_row) {
  using value_t = typename Policy::value_type;
  constexpr int warp_size = 32;
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);
  std::size_t vertex = blockIdx.x * static_cast<std::size_t>(tile_row) +
                       static_cast<std::size_t>(threadIdx.y);
  int lane = threadIdx.x & (warp_size - 1);
  if (vertex >= vertex_count) {
    return;
  }

  auto begin = graph.get_starting_edge(static_cast<vertex_t>(vertex));
  auto end = graph.get_starting_edge(static_cast<vertex_t>(vertex + 1));
  for (int query_id = lane; query_id < query_count; query_id += warp_size) {
    auto source_index = value_index(vertex, static_cast<std::size_t>(query_id),
                                    query_stride);
    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      if (values[source_index] != static_cast<value_t>(level)) {
        continue;
      }
      for (auto edge = begin; edge < end; ++edge) {
        vertex_t neighbor = graph.get_destination_vertex(edge);
        auto neighbor_index =
            value_index(static_cast<std::size_t>(neighbor),
                        static_cast<std::size_t>(query_id), query_stride);
        value_t old = atomicCAS(values + neighbor_index, Policy::infinity(),
                                static_cast<value_t>(level + 1));
        if (old == Policy::infinity()) {
          query_mask_t bit = query_bit(query_id);
          if (visited_mask != nullptr) {
            atomic_or_query_mask(visited_mask + neighbor, bit);
          }
          mark_next_shared_frontier(neighbor, bit, next_frontier_mask,
                                    next_frontier_vertices, next_unique_count,
                                    next_pair_count);
        }
      }
    } else {
      value_t source_distance = values[source_index];
      if (source_distance == Policy::infinity()) {
        continue;
      }
      for (auto edge = begin; edge < end; ++edge) {
        vertex_t neighbor = graph.get_destination_vertex(edge);
        value_t candidate =
            Policy::relax(source_distance, graph.get_edge_weight(edge));
        value_t old = atomic_min_value(
            values + value_index(static_cast<std::size_t>(neighbor),
                                 static_cast<std::size_t>(query_id),
                                 query_stride),
            candidate);
        if (candidate < old) {
          mark_next_shared_frontier(neighbor, query_bit(query_id),
                                    next_frontier_mask, next_frontier_vertices,
                                    next_unique_count, next_pair_count);
        }
      }
    }
  }
}

template <typename graph_t, typename vertex_t>
__global__ void fused_ge_bfs_pull_simple_kernel(
    graph_t graph,
    int query_count,
    typename algorithms::bfs_policy::value_type* values,
    vertex_t level,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts) {
  using value_t = typename algorithms::bfs_policy::value_type;
  __shared__ query_mask_t thread_masks[128];

  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);
  std::size_t vertex = blockIdx.x * blockDim.y + threadIdx.y;
  int query_id = threadIdx.x;
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  query_mask_t local_mask = 0;

  if (vertex < vertex_count && query_id < query_count) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query_id), query_stride);
    bool acc = false;
    auto begin = graph.get_starting_edge(static_cast<vertex_t>(vertex));
    auto end = graph.get_starting_edge(static_cast<vertex_t>(vertex + 1));
    for (auto edge = begin; edge < end; ++edge) {
      vertex_t neighbor = graph.get_destination_vertex(edge);
      if (values[value_index(static_cast<std::size_t>(neighbor),
                             static_cast<std::size_t>(query_id),
                             query_stride)] == static_cast<value_t>(level)) {
        acc = true;
      }
    }
    if (acc && values[value_pos] == algorithms::bfs_policy::infinity()) {
      values[value_pos] = static_cast<value_t>(level + 1);
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

template <int TILE_ROW, typename graph_t, typename vertex_t>
__global__ void fused_ge_bfs_pull_smem_kernel(
    graph_t graph,
    int query_count,
    typename algorithms::bfs_policy::value_type* values,
    vertex_t level,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts) {
  using value_t = typename algorithms::bfs_policy::value_type;
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
  bool acc0 = false;
  bool acc1 = false;

  decltype(graph.get_starting_edge(static_cast<vertex_t>(0))) begin = 0;
  decltype(begin) end = 0;
  if (row_valid) {
    begin = graph.get_starting_edge(static_cast<vertex_t>(vertex));
    end = graph.get_starting_edge(static_cast<vertex_t>(vertex + 1));
  }
  for (auto tile = begin; tile < end; tile += warp_size) {
    auto remaining = end - tile;
    int tile_count =
        remaining < warp_size ? static_cast<int>(remaining) : warp_size;
    if (lane < tile_count) {
      neighbor_tile[threadIdx.y][lane] =
          graph.get_destination_vertex(tile + lane);
    }
    __syncthreads();

    for (int i = 0; i < tile_count; ++i) {
      vertex_t neighbor = neighbor_tile[threadIdx.y][i];
      if (query0 < query_count &&
          values[value_index(static_cast<std::size_t>(neighbor),
                             static_cast<std::size_t>(query0),
                             query_stride)] == static_cast<value_t>(level)) {
        acc0 = true;
      }
      if (query1 < query_count &&
          values[value_index(static_cast<std::size_t>(neighbor),
                             static_cast<std::size_t>(query1),
                             query_stride)] == static_cast<value_t>(level)) {
        acc1 = true;
      }
    }
    __syncthreads();
  }

  query_mask_t local_mask = 0;
  if (row_valid && query0 < query_count && acc0) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query0), query_stride);
    if (values[value_pos] == algorithms::bfs_policy::infinity()) {
      values[value_pos] = static_cast<value_t>(level + 1);
      local_mask |= query_bit(query0);
    }
  }
  if (row_valid && query1 < query_count && acc1) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query1), query_stride);
    if (values[value_pos] == algorithms::bfs_policy::infinity()) {
      values[value_pos] = static_cast<value_t>(level + 1);
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

template <typename graph_t, typename vertex_t>
void launch_fused_ge_bfs_pull(graph_t graph,
                              int query_count,
                              typename algorithms::bfs_policy::value_type* values,
                              vertex_t level,
                              query_mask_t* visited_mask,
                              query_mask_t* next_frontier_mask,
                              unsigned long long* unique_flags,
                              unsigned long long* pair_counts,
                              cudaStream_t stream) {
  int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  if (query_count <= 32) {
    int tile_row = std::max(1, 128 / std::max(1, query_count));
    int grid_x = (vertex_count + tile_row - 1) / tile_row;
    fused_ge_bfs_pull_simple_kernel<graph_t, vertex_t>
        <<<grid_x, dim3(query_count, tile_row), 0, stream>>>(
            graph, query_count, values, level, visited_mask,
            next_frontier_mask, unique_flags, pair_counts);
  } else if (query_count <= 64) {
    constexpr int tile_row = 4;
    int grid_x = (vertex_count + tile_row - 1) / tile_row;
    fused_ge_bfs_pull_smem_kernel<tile_row, graph_t, vertex_t>
        <<<grid_x, dim3(32, tile_row), 0, stream>>>(
            graph, query_count, values, level, visited_mask,
            next_frontier_mask, unique_flags, pair_counts);
  } else {
    throw std::invalid_argument(
        "fused GE BFS shared pull supports at most 64 queries");
  }
}

/* ================================================================
 * Algorithm-independent fused pull kernels.
 *
 * Both BFS and SSSP share the min-plus semiring:
 *   acc = +inf
 *   for each neighbor u:
 *     candidate = Policy::relax(values[u,q], weight)
 *     acc = min(acc, candidate)
 *   if Policy::should_update(acc, values[v,q]):
 *     values[v,q] = acc; mark frontier
 * ================================================================ */

template <typename Policy, typename graph_t, typename vertex_t>
void launch_shared_pull_values(graph_t graph,
                               int query_count,
                               typename Policy::value_type* values,
                               vertex_t level,
                               query_mask_t* visited_mask,
                               query_mask_t* next_frontier_mask,
                               vertex_t* next_frontier_vertices,
                               unsigned long long* next_unique_count,
                               unsigned long long* next_pair_count,
                               cudaStream_t stream) {
  int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
    int blocks = std::max(1, std::min(vertex_count, 65535));
    pull_bfs_source_mask_kernel<graph_t, vertex_t>
        <<<blocks, 256, 0, stream>>>(
            graph, query_count, values, level, visited_mask, next_frontier_mask,
            next_frontier_vertices, next_unique_count, next_pair_count);
    return;
  }
  if (query_count <= 32) {
    int eff_tile = std::max(1, 128 / std::max(1, query_count));
    int grid_x = (vertex_count + eff_tile - 1) / eff_tile;
    pull_values_simple_kernel<Policy, graph_t, vertex_t>
        <<<grid_x, dim3(query_count, eff_tile), 0, stream>>>(
            graph, query_count, values, level, visited_mask, next_frontier_mask,
            next_frontier_vertices, next_unique_count, next_pair_count);
  } else {
    int tile_row = 4;
    int grid_x = (vertex_count + tile_row - 1) / tile_row;
    pull_values_warp_coarsened_kernel<Policy, graph_t, vertex_t>
        <<<grid_x, dim3(32, tile_row), 0, stream>>>(
            graph, query_count, values, level, visited_mask, next_frontier_mask,
            next_frontier_vertices, next_unique_count, next_pair_count,
            tile_row);
  }
}

/*
@todo:

*/
template <int M, typename graph_t>
__global__ void ge_spmm_bfs_simple_kernel(graph_t graph,
                                          const float* input,
                                          float* output,
                                          int tile_row) {
  using vertex_t = typename graph_t::vertex_type;
  int row = tile_row * blockIdx.x + threadIdx.y;
  if (row >= graph.get_number_of_vertices()) {
    return;
  }
  int col = threadIdx.x; 
  float source_active =
      input[static_cast<std::size_t>(row) * M + static_cast<std::size_t>(col)];
  if (source_active <= 0.0f) {
    return;
  }
  auto begin = graph.get_starting_edge(static_cast<vertex_t>(row));
  auto end = graph.get_starting_edge(static_cast<vertex_t>(row + 1));

  for (auto edge = begin; edge < end; ++edge) {
    vertex_t neighbor = graph.get_destination_vertex(edge);
    atomicAdd(output + static_cast<std::size_t>(neighbor) * M +
                         static_cast<std::size_t>(col),
              1.0f);
  }
}

template <int M, typename graph_t>
__global__ void ge_spmm_sssp_simple_kernel(graph_t graph,
                                           const float* input,
                                           float* output,
                                           int tile_row) {
  using vertex_t = typename graph_t::vertex_type;
  int row = tile_row * blockIdx.x + threadIdx.y;
  if (row >= graph.get_number_of_vertices()) {
    return;
  }
  int col = threadIdx.x;
  float source_distance =
      input[static_cast<std::size_t>(row) * M + static_cast<std::size_t>(col)];
  if (source_distance == algorithms::sssp_policy::infinity()) {
    return;
  }
  auto begin = graph.get_starting_edge(static_cast<vertex_t>(row));
  auto end = graph.get_starting_edge(static_cast<vertex_t>(row + 1));
  for (auto edge = begin; edge < end; ++edge) {
    vertex_t neighbor = graph.get_destination_vertex(edge);
    float candidate = source_distance + static_cast<float>(graph.get_edge_weight(edge));
    atomic_min_value(output + static_cast<std::size_t>(neighbor) * M +
                                  static_cast<std::size_t>(col),
                     candidate);
  }
}

template <typename graph_t>
void launch_ge_spmm_bfs(graph_t graph,
                        const float* input,
                        float* output,
                        int m_eff,
                        cudaStream_t stream) {
  int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  int tile_row = 8;
#define PUERCGP_LAUNCH_BFS_GE(M_VALUE)                                       \
  do {                                                                       \
    int eff_tile = std::max(tile_row, std::max(1, 128 / (M_VALUE)));         \
    int grid_x = (vertex_count + eff_tile - 1) / eff_tile;                   \
    ge_spmm_bfs_simple_kernel<M_VALUE, graph_t>                              \
        <<<grid_x, dim3(M_VALUE, eff_tile), 0, stream>>>(graph, input,       \
                                                         output, eff_tile);  \
  } while (0)
  switch (m_eff) {
    case 1: PUERCGP_LAUNCH_BFS_GE(1); break;
    case 2: PUERCGP_LAUNCH_BFS_GE(2); break;
    case 4: PUERCGP_LAUNCH_BFS_GE(4); break;
    case 8: PUERCGP_LAUNCH_BFS_GE(8); break;
    case 16: PUERCGP_LAUNCH_BFS_GE(16); break;
    case 32: PUERCGP_LAUNCH_BFS_GE(32); break;
    case 64: PUERCGP_LAUNCH_BFS_GE(64); break;
    case 80: PUERCGP_LAUNCH_BFS_GE(80); break;
    case 96: PUERCGP_LAUNCH_BFS_GE(96); break;
    case 128: PUERCGP_LAUNCH_BFS_GE(128); break;
    default: throw std::invalid_argument("unsupported GE-SpMM query dimension");
  }
#undef PUERCGP_LAUNCH_BFS_GE
}

template <typename graph_t>
void launch_ge_spmm_sssp(graph_t graph,
                         const float* input,
                         float* output,
                         int m_eff,
                         cudaStream_t stream) {
  int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  int tile_row = 8;
#define PUERCGP_LAUNCH_SSSP_GE(M_VALUE)                                      \
  do {                                                                       \
    int eff_tile = std::max(tile_row, std::max(1, 128 / (M_VALUE)));         \
    int grid_x = (vertex_count + eff_tile - 1) / eff_tile;                   \
    ge_spmm_sssp_simple_kernel<M_VALUE, graph_t>                             \
        <<<grid_x, dim3(M_VALUE, eff_tile), 0, stream>>>(graph, input,       \
                                                         output, eff_tile);  \
  } while (0)
  switch (m_eff) {
    case 1: PUERCGP_LAUNCH_SSSP_GE(1); break;
    case 2: PUERCGP_LAUNCH_SSSP_GE(2); break;
    case 4: PUERCGP_LAUNCH_SSSP_GE(4); break;
    case 8: PUERCGP_LAUNCH_SSSP_GE(8); break;
    case 16: PUERCGP_LAUNCH_SSSP_GE(16); break;
    case 32: PUERCGP_LAUNCH_SSSP_GE(32); break;
    case 64: PUERCGP_LAUNCH_SSSP_GE(64); break;
    default: throw std::invalid_argument("unsupported GE-SpMM query dimension");
  }
#undef PUERCGP_LAUNCH_SSSP_GE
}

template <typename Policy, typename vertex_t>
__global__ void ge_spmm_postprocess_kernel(
    const float* spmm_out,
    std::size_t vertex_count,
    int query_count,
    int m_eff,
    typename Policy::value_type* values,
    vertex_t level,
    float* next_dense,
    unsigned long long* out_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    bool update_shared) {
  using value_t = typename Policy::value_type;
  std::size_t total_pairs =
      vertex_count * static_cast<std::size_t>(query_count);
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < total_pairs; i += stride) {
    std::size_t vertex = i / static_cast<std::size_t>(query_count);
    std::size_t query_id = i % static_cast<std::size_t>(query_count);
    std::size_t dense_index =
        vertex * static_cast<std::size_t>(m_eff) + query_id;
    std::size_t value_pos =
        value_index(vertex, query_id, static_cast<std::size_t>(query_count));

    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      if (spmm_out[dense_index] <= 0.0f ||
          values[value_pos] != Policy::infinity()) {
        continue;
      }
      values[value_pos] = static_cast<value_t>(level + 1);
      next_dense[dense_index] = 1.0f;
    } else {
      float candidate = spmm_out[dense_index];
      if (candidate == Policy::infinity() || candidate >= values[value_pos]) {
        continue;
      }
      values[value_pos] = candidate;
      next_dense[dense_index] = candidate;
    }

    atomicAdd(out_count, 1ULL);
    if (update_shared) {
      query_mask_t bit = query_bit(static_cast<int>(query_id));
      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        atomic_or_query_mask(visited_mask + vertex, bit);
      }
      query_mask_t old_next =
          atomic_or_query_mask(next_frontier_mask + vertex, bit);
      if (old_next == 0) {
        unsigned long long position = atomicAdd(next_unique_count, 1ULL);
        next_frontier_vertices[position] = static_cast<vertex_t>(vertex);
      }
    }
  }
}

template <typename Policy>
inline int effective_query_dim(int query_count, pull_strategy_t strategy) {
  (void)strategy;
  return query_count;
}

}  // namespace detail

template <typename Policy>
class frontier_engine {
 public:
  using vertex_type = typename Policy::vertex_type;
  using value_type = typename Policy::value_type;
  using result_type = run_result_t<vertex_type, value_type>;

  template <typename graph_t>
  result_type run(graph_t& graph,
                  const query_batch<vertex_type>& queries,
                  execution_context& context,
                  const run_options& options) const {
    static_assert(std::is_same<vertex_type, int>::value,
                  "puercgp frontier policies currently require int vertices");

    queries.validate(options.max_queries);
    const int query_count = static_cast<int>(queries.size());
    const auto vertex_count =
        static_cast<std::size_t>(graph.get_number_of_vertices());
    const auto edge_count = static_cast<std::size_t>(graph.get_number_of_edges());

    const std::size_t pair_capacity =
        static_cast<std::size_t>(query_count) * vertex_count;
    std::size_t list_capacity = pair_capacity;

    if constexpr (std::is_same<Policy, algorithms::sssp_policy>::value) {
      list_capacity =
          static_cast<std::size_t>(query_count) * std::max(vertex_count, edge_count);
    }

    bool shared_push = options.push_strategy == push_strategy_t::shared_node ||
                       options.push_strategy ==
                           push_strategy_t::shared_node_query_parallel ||
                       options.push_strategy == push_strategy_t::shared_node_warp ||
                       options.push_strategy == push_strategy_t::shared_node_degree;
    if (shared_push && query_count > 64) {
      throw std::invalid_argument(
          "shared_node push strategies support at most 64 queries");
    }

    int m_eff =
        detail::effective_query_dim<Policy>(query_count, options.pull_strategy);
    std::size_t dense_capacity = pair_capacity;
    double pull_frontier_threshold =
        options.pull_frontier_ratio * static_cast<double>(pair_capacity);
    double pull_edge_threshold = options.pull_edge_ratio *
                                 static_cast<double>(query_count) *
                                 static_cast<double>(edge_count);

    cudaStream_t stream = context.stream();
    auto policy = thrust::cuda::par.on(stream);
    constexpr int threads = 256;

    thrust::device_vector<vertex_type> device_sources(queries.sources());
    thrust::device_vector<value_type> values(pair_capacity);

    // frontier a & b can be removed
    thrust::device_vector<frontier_item_t<vertex_type>> frontier_a(
        shared_push ? 0 : list_capacity);
    thrust::device_vector<frontier_item_t<vertex_type>> frontier_b(
        shared_push ? 0 : list_capacity);

    thrust::device_vector<unsigned char> frontier_bitmap(shared_push ? 0 : pair_capacity);
    thrust::device_vector<unsigned char> next_frontier_bitmap(shared_push ? 0 : pair_capacity);

    thrust::device_vector<float> frontier_dense;
    thrust::device_vector<float> next_frontier_dense;
    thrust::device_vector<float> ge_spmm_out;

    if (!shared_push && options.pull_strategy == pull_strategy_t::ge_spmm) {
      frontier_dense.resize(dense_capacity);
      next_frontier_dense.resize(dense_capacity);
      ge_spmm_out.resize(dense_capacity);
    }

    thrust::device_vector<unsigned long long> out_count(1);
    thrust::device_vector<unsigned long long> shared_pair_count(1);
    thrust::device_vector<unsigned long long> frontier_degrees(
        shared_push ? 0 : list_capacity);
    thrust::device_vector<unsigned long long> edge_offsets(
        shared_push ? 0 : list_capacity);
    thrust::device_vector<unsigned long long> edge_ends(
        shared_push ? 0 : list_capacity);

    thrust::device_vector<query_mask_t> visited_mask;
    thrust::device_vector<query_mask_t> frontier_mask;
    thrust::device_vector<query_mask_t> next_frontier_mask;
    thrust::device_vector<vertex_type> shared_frontier_a;
    thrust::device_vector<vertex_type> shared_frontier_b;
    thrust::device_vector<unsigned long long> shared_actual_degrees;
    thrust::device_vector<unsigned long long> shared_virtual_degrees;
    
    if (shared_push) {
      visited_mask.resize(vertex_count);
      frontier_mask.resize(vertex_count);
      next_frontier_mask.resize(vertex_count);
      shared_frontier_a.resize(vertex_count);
      shared_frontier_b.resize(vertex_count);
      shared_actual_degrees.resize(vertex_count);
      shared_virtual_degrees.resize(vertex_count);
    }

    auto wall_start = std::chrono::high_resolution_clock::now();
    detail::cuda_event_timer total_timer;
    total_timer.begin(stream);

    int init_blocks = detail::grid_for(pair_capacity, threads);

    std::size_t current_count = queries.size();
    std::size_t current_unique_count = queries.size();
    if (shared_push) {
      detail::fill_values_kernel<Policy><<<init_blocks, threads, 0, stream>>>(
          thrust::raw_pointer_cast(values.data()), pair_capacity);
      detail::throw_if_cuda_error(cudaGetLastError(), "fill_values_kernel");
      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0,
                            vertex_count * sizeof(query_mask_t), stream),
            "cudaMemsetAsync(visited_mask)");
      }
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                          vertex_count * sizeof(query_mask_t), stream),
          "cudaMemsetAsync(frontier_mask)");
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                          vertex_count * sizeof(query_mask_t), stream),
          "cudaMemsetAsync(next_frontier_mask)");
      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(out_count.data()));
      if constexpr (std::is_same<Policy, algorithms::wcc_policy>::value) {
        detail::init_wcc_labels_kernel<graph_t, vertex_type><<<detail::grid_for(
            vertex_count * static_cast<std::size_t>(query_count), threads),
            threads, 0, stream>>>(
            graph, query_count,
            thrust::raw_pointer_cast(values.data()),
            thrust::raw_pointer_cast(frontier_mask.data()),
            thrust::raw_pointer_cast(shared_frontier_a.data()),
            thrust::raw_pointer_cast(out_count.data()));
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "init_wcc_labels_kernel");
      } else {
        detail::init_shared_sources_kernel<Policy>
            <<<detail::grid_for(queries.size(), threads), threads, 0, stream>>>(
                graph, thrust::raw_pointer_cast(device_sources.data()),
                query_count, thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(visited_mask.data()),
                thrust::raw_pointer_cast(frontier_mask.data()),
                thrust::raw_pointer_cast(shared_frontier_a.data()),
                thrust::raw_pointer_cast(out_count.data()));
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "init_shared_sources_kernel");
      }
      unsigned long long unique_count = 0;
      detail::throw_if_cuda_error(
          cudaMemcpyAsync(&unique_count, thrust::raw_pointer_cast(out_count.data()),
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                          stream),
          "cudaMemcpyAsync(unique_count)");
      context.synchronize();
      current_unique_count = static_cast<std::size_t>(unique_count);
    } else {
      detail::init_list_kernel<Policy><<<init_blocks, threads, 0, stream>>>(
          graph, thrust::raw_pointer_cast(device_sources.data()), query_count,
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(frontier_a.data()));
      detail::throw_if_cuda_error(cudaGetLastError(), "init_list_kernel");
      context.synchronize();
    }

    result_type result;
    result.options = options;
    result.effective_query_dim = m_eff;
    result.frontier_sizes.push_back(current_count);
    result.unique_frontier_sizes.push_back(shared_push ? current_unique_count
                                                       : current_count);

    frontier_repr_t current_repr =
        shared_push ? frontier_repr_t::shared : frontier_repr_t::list;
    vertex_type level = 0;

    while (current_count > 0 &&
           (options.max_iterations <= 0 ||
            result.iterations < options.max_iterations)) {
      unsigned long long total_edges = 0;
      unsigned long long actual_edges = 0;
      unsigned long long virtual_edges = 0;
      bool edge_count_ready = false;
      iteration_profile_t profile;
      profile.iteration = static_cast<int>(level);
      profile.frontier_size = current_count;
      profile.unique_frontier_size =
          shared_push ? current_unique_count : current_count;
      profile.pull_frontier_threshold = pull_frontier_threshold;
      profile.pull_edge_threshold = pull_edge_threshold;
      auto iteration_start = std::chrono::high_resolution_clock::now();

      auto compact_bitmap_to_list = [&]() {
        profile.compact_ms += detail::timed_gpu(stream, options.profile_iterations,
                                                [&]() {
          detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
              thrust::raw_pointer_cast(out_count.data()));
          detail::bitmap_to_list_kernel<vertex_type>
              <<<detail::grid_for(pair_capacity, threads), threads, 0, stream>>>(
                  thrust::raw_pointer_cast(frontier_bitmap.data()), pair_capacity,
                  query_count, thrust::raw_pointer_cast(frontier_a.data()),
                  thrust::raw_pointer_cast(out_count.data()));
        });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "bitmap_to_list_kernel");
        unsigned long long compact_count = 0;
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&compact_count,
                            thrust::raw_pointer_cast(out_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(compact_count)");
        context.synchronize();
        current_count = static_cast<std::size_t>(compact_count);
        current_repr = frontier_repr_t::list;
      };

      auto compact_bitmap_to_shared = [&]() {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                            vertex_count * sizeof(query_mask_t), stream),
            "cudaMemsetAsync(frontier_mask)");
        profile.compact_ms += detail::timed_gpu(stream, options.profile_iterations,
                                                [&]() {
          detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
              thrust::raw_pointer_cast(out_count.data()));
          detail::bitmap_to_shared_kernel<vertex_type>
              <<<detail::grid_for(vertex_count, threads), threads, 0, stream>>>(
                  thrust::raw_pointer_cast(frontier_bitmap.data()), vertex_count,
                  query_count, thrust::raw_pointer_cast(frontier_mask.data()),
                  thrust::raw_pointer_cast(shared_frontier_a.data()),
                  thrust::raw_pointer_cast(out_count.data()));
        });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "bitmap_to_shared_kernel");
        unsigned long long compact_unique_count = 0;
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&compact_unique_count,
                            thrust::raw_pointer_cast(out_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(compact_unique_count)");
        context.synchronize();
        current_unique_count = static_cast<std::size_t>(compact_unique_count);
        current_repr = frontier_repr_t::shared;
      };

      auto compact_dense_to_list = [&]() {
        profile.dense_compact_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
                  thrust::raw_pointer_cast(out_count.data()));
              detail::dense_to_list_kernel<Policy, vertex_type>
                  <<<detail::grid_for(pair_capacity, threads), threads, 0,
                     stream>>>(thrust::raw_pointer_cast(frontier_dense.data()),
                                vertex_count, query_count, m_eff,
                                thrust::raw_pointer_cast(frontier_a.data()),
                                thrust::raw_pointer_cast(out_count.data()));
            });
        detail::throw_if_cuda_error(cudaGetLastError(), "dense_to_list_kernel");
        unsigned long long compact_count = 0;
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&compact_count,
                            thrust::raw_pointer_cast(out_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(compact_count)");
        context.synchronize();
        current_count = static_cast<std::size_t>(compact_count);
        current_repr = frontier_repr_t::list;
      };

      auto compact_dense_to_shared = [&]() {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                            vertex_count * sizeof(query_mask_t), stream),
            "cudaMemsetAsync(frontier_mask)");
        profile.dense_compact_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
                  thrust::raw_pointer_cast(out_count.data()));
              detail::dense_to_shared_kernel<Policy, vertex_type>
                  <<<detail::grid_for(vertex_count, threads), threads, 0,
                     stream>>>(thrust::raw_pointer_cast(frontier_dense.data()),
                                vertex_count, query_count, m_eff,
                                thrust::raw_pointer_cast(frontier_mask.data()),
                                thrust::raw_pointer_cast(shared_frontier_a.data()),
                                thrust::raw_pointer_cast(out_count.data()));
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "dense_to_shared_kernel");
        unsigned long long compact_unique_count = 0;
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&compact_unique_count,
                            thrust::raw_pointer_cast(out_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(compact_unique_count)");
        context.synchronize();
        current_unique_count = static_cast<std::size_t>(compact_unique_count);
        current_repr = frontier_repr_t::shared;
      };

      auto build_dense_frontier = [&]() {
        if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
          detail::throw_if_cuda_error(
              cudaMemsetAsync(thrust::raw_pointer_cast(frontier_dense.data()), 0,
                              dense_capacity * sizeof(float), stream),
              "cudaMemsetAsync(frontier_dense)");
        } else {
          thrust::fill(policy, frontier_dense.begin(), frontier_dense.end(),
                       Policy::infinity());
        }
        std::size_t build_count =
            shared_push ? current_unique_count : current_count;
        profile.dense_build_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              if (current_repr == frontier_repr_t::shared) {
                detail::shared_to_dense_kernel<Policy, vertex_type>
                    <<<detail::grid_for(build_count, threads), threads, 0,
                       stream>>>(
                        thrust::raw_pointer_cast(shared_frontier_a.data()),
                        thrust::raw_pointer_cast(frontier_mask.data()),
                        current_unique_count, query_count, m_eff,
                        thrust::raw_pointer_cast(values.data()), vertex_count,
                        thrust::raw_pointer_cast(frontier_dense.data()));
              } else {
                detail::list_to_dense_kernel<Policy, vertex_type>
                    <<<detail::grid_for(build_count, threads), threads, 0,
                       stream>>>(
                        thrust::raw_pointer_cast(frontier_a.data()),
                        current_count, m_eff,
                        thrust::raw_pointer_cast(values.data()), vertex_count,
                        thrust::raw_pointer_cast(frontier_dense.data()));
              }
            });
        detail::throw_if_cuda_error(cudaGetLastError(), "build_dense_frontier");
        current_repr = frontier_repr_t::dense;
      };

      auto compute_push_edge_count = [&]() {
        profile.degree_scan_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::compute_degrees_kernel<graph_t, vertex_type>
                  <<<detail::grid_for(current_count, threads), threads, 0,
                     stream>>>(graph, thrust::raw_pointer_cast(frontier_a.data()),
                                current_count,
                                thrust::raw_pointer_cast(frontier_degrees.data()));
              thrust::exclusive_scan(policy, frontier_degrees.begin(),
                                     frontier_degrees.begin() + current_count,
                                     edge_offsets.begin(), 0ULL);
              thrust::inclusive_scan(policy, frontier_degrees.begin(),
                                     frontier_degrees.begin() + current_count,
                                     edge_ends.begin());
            });
        detail::throw_if_cuda_error(cudaGetLastError(), "compute_degrees_kernel");
        auto sync_start = std::chrono::high_resolution_clock::now();
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&total_edges,
                            thrust::raw_pointer_cast(edge_ends.data()) +
                                current_count - 1,
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(total_edges)");
        context.synchronize();
        profile.count_sync_ms += detail::elapsed_ms(sync_start);
        edge_count_ready = true;
      };

      auto compute_shared_push_edge_count = [&]() {
        profile.degree_scan_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::compute_shared_degrees_kernel<graph_t, vertex_type>
                  <<<detail::grid_for(current_unique_count, threads), threads, 0,
                     stream>>>(
                      graph, thrust::raw_pointer_cast(shared_frontier_a.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(shared_actual_degrees.data()),
                      thrust::raw_pointer_cast(shared_virtual_degrees.data()));
              thrust::inclusive_scan(
                  policy, shared_actual_degrees.begin(),
                  shared_actual_degrees.begin() + current_unique_count,
                  shared_actual_degrees.begin());
              thrust::inclusive_scan(
                  policy, shared_virtual_degrees.begin(),
                  shared_virtual_degrees.begin() + current_unique_count,
                  shared_virtual_degrees.begin());
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "compute_shared_degrees_kernel");
        auto sync_start = std::chrono::high_resolution_clock::now();
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&actual_edges,
                            thrust::raw_pointer_cast(shared_actual_degrees.data()) +
                                current_unique_count - 1,
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(actual_edges)");
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&virtual_edges,
                            thrust::raw_pointer_cast(shared_virtual_degrees.data()) +
                                current_unique_count - 1,
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(virtual_edges)");
        context.synchronize();
        profile.count_sync_ms += detail::elapsed_ms(sync_start);
        total_edges = virtual_edges;
        edge_count_ready = true;
      };

      if (options.traversal_mode == traversal_mode_t::push &&
          current_repr == frontier_repr_t::bitmap) {
        shared_push ? compact_bitmap_to_shared() : compact_bitmap_to_list();
      }
      if (options.traversal_mode == traversal_mode_t::push &&
          current_repr == frontier_repr_t::dense) {
        shared_push ? compact_dense_to_shared() : compact_dense_to_list();
      }

      // profile 模式下强制计算 edge_count（即使 shared_node_warp），
      // 以便 hybrid 策略评估时能拿到完整的 degree_sum / virtual_edge_count。
      // 非.profile 模式保持原 skip 行为避免性能损失。
      bool skip_shared_direct_edge_count =
          options.traversal_mode == traversal_mode_t::push &&
          current_repr == frontier_repr_t::shared &&
          !options.profile_iterations &&
          (options.push_strategy == push_strategy_t::shared_node_warp ||
           options.push_strategy == push_strategy_t::shared_node_degree);

      bool forced_pull = options.traversal_mode == traversal_mode_t::pull;
      bool pull_by_frontier =
          options.traversal_mode == traversal_mode_t::hybrid &&
          static_cast<double>(current_count) >= pull_frontier_threshold;
      bool need_edge_count_for_decision = !forced_pull && !pull_by_frontier;

      if (current_repr == frontier_repr_t::shared &&
          need_edge_count_for_decision &&
          !skip_shared_direct_edge_count) {
        compute_shared_push_edge_count();
      } else if (current_repr == frontier_repr_t::list &&
                 need_edge_count_for_decision) {
        compute_push_edge_count();
      } else if (options.traversal_mode == traversal_mode_t::hybrid &&
                 need_edge_count_for_decision &&
                 static_cast<double>(current_count) < pull_frontier_threshold) {
        if (shared_push) {
          current_repr == frontier_repr_t::bitmap ? compact_bitmap_to_shared()
                                                  : compact_dense_to_shared();
          compute_shared_push_edge_count();
        } else {
          current_repr == frontier_repr_t::bitmap ? compact_bitmap_to_list()
                                                  : compact_dense_to_list();
          compute_push_edge_count();
        }
      }

      bool use_pull = false;
      if (options.traversal_mode == traversal_mode_t::pull) {
        use_pull = true;
      } else if (options.traversal_mode == traversal_mode_t::hybrid) {
        use_pull = static_cast<double>(current_count) >= pull_frontier_threshold;
        if (!use_pull && edge_count_ready) {
          use_pull = static_cast<double>(total_edges) >= pull_edge_threshold;
        }
      }

      if (use_pull && !shared_push &&
          options.pull_strategy == pull_strategy_t::bitmap &&
          current_repr != frontier_repr_t::bitmap) {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(frontier_bitmap.data()), 0,
                            pair_capacity * sizeof(unsigned char), stream),
            "cudaMemsetAsync(frontier_bitmap)");
        std::size_t bitmap_count =
            shared_push ? current_unique_count : current_count;
        profile.bitmap_build_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              if (current_repr == frontier_repr_t::shared) {
                detail::shared_to_bitmap_kernel<vertex_type>
                    <<<detail::grid_for(bitmap_count, threads), threads, 0,
                       stream>>>(
                        thrust::raw_pointer_cast(shared_frontier_a.data()),
                        thrust::raw_pointer_cast(frontier_mask.data()),
                        current_unique_count, query_count,
                        thrust::raw_pointer_cast(frontier_bitmap.data()));
              } else {
                detail::list_to_bitmap_kernel<vertex_type>
                    <<<detail::grid_for(bitmap_count, threads), threads, 0,
                       stream>>>(
                        thrust::raw_pointer_cast(frontier_a.data()),
                        current_count, query_count,
                        thrust::raw_pointer_cast(frontier_bitmap.data()));
              }
            });
        detail::throw_if_cuda_error(cudaGetLastError(), "bitmap build");
        current_repr = frontier_repr_t::bitmap;
      }

      if (use_pull && !shared_push &&
          options.pull_strategy == pull_strategy_t::ge_spmm &&
          current_repr != frontier_repr_t::dense) {
        build_dense_frontier();
      }

      if (use_pull && current_repr == frontier_repr_t::bitmap &&
          !edge_count_ready && options.profile_iterations) {
        profile.degree_scan_ms += detail::timed_gpu(stream, true, [&]() {
          detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
              thrust::raw_pointer_cast(out_count.data()));
          detail::compute_bitmap_degrees_kernel<graph_t>
              <<<detail::grid_for(pair_capacity, threads), threads, 0, stream>>>(
                  graph, thrust::raw_pointer_cast(frontier_bitmap.data()),
                  pair_capacity, query_count,
                  thrust::raw_pointer_cast(out_count.data()));
        });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "compute_bitmap_degrees_kernel");
        auto sync_start = std::chrono::high_resolution_clock::now();
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&total_edges, thrust::raw_pointer_cast(out_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(total_edges)");
        context.synchronize();
        profile.count_sync_ms += detail::elapsed_ms(sync_start);
        edge_count_ready = true;
      }

      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(out_count.data()));
      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(shared_pair_count.data()));

      if (use_pull && shared_push) {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()),
                            0, vertex_count * sizeof(query_mask_t), stream),
            "cudaMemsetAsync(next_frontier_mask)");
        auto launch_legacy_shared_pull = [&]() {
          detail::launch_shared_pull_values<Policy, graph_t, vertex_type>(
              graph, query_count, thrust::raw_pointer_cast(values.data()), level,
              thrust::raw_pointer_cast(visited_mask.data()),
              thrust::raw_pointer_cast(next_frontier_mask.data()),
              thrust::raw_pointer_cast(shared_frontier_b.data()),
              thrust::raw_pointer_cast(shared_pair_count.data()),
              thrust::raw_pointer_cast(out_count.data()), stream);
        };
        if (options.pull_strategy == pull_strategy_t::ge_spmm) {
          auto launch_fused_pull = [&]() {
            detail::launch_fused_pull_compute<Policy, graph_t, vertex_type>(
                graph, query_count,
                thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(visited_mask.data()),
                thrust::raw_pointer_cast(next_frontier_mask.data()),
                thrust::raw_pointer_cast(shared_actual_degrees.data()),
                thrust::raw_pointer_cast(shared_virtual_degrees.data()),
                stream);
          };
          float pull_ms = detail::timed_gpu(
              stream, options.profile_iterations, launch_fused_pull);
          profile.pull_kernel_ms += pull_ms;
          profile.ge_spmm_pull_kernel_ms += pull_ms;
          detail::throw_if_cuda_error(cudaGetLastError(),
                                      "launch_fused_pull");
          profile.compact_ms += detail::timed_gpu(
              stream, options.profile_iterations, [&]() {
                thrust::inclusive_scan(
                    policy, shared_actual_degrees.begin(),
                    shared_actual_degrees.begin() + vertex_count,
                    shared_actual_degrees.begin());
                thrust::inclusive_scan(
                    policy, shared_virtual_degrees.begin(),
                    shared_virtual_degrees.begin() + vertex_count,
                    shared_virtual_degrees.begin());
                detail::launch_pull_frontier_compact<vertex_type>(
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    vertex_count,
                    thrust::raw_pointer_cast(shared_actual_degrees.data()),
                    thrust::raw_pointer_cast(shared_frontier_b.data()), threads,
                    stream);
                detail::throw_if_cuda_error(
                    cudaMemcpyAsync(
                        thrust::raw_pointer_cast(shared_pair_count.data()),
                        thrust::raw_pointer_cast(shared_actual_degrees.data()) +
                            vertex_count - 1,
                        sizeof(unsigned long long), cudaMemcpyDeviceToDevice,
                        stream),
                    "cudaMemcpyAsync(shared_pull_unique_count)");
                detail::throw_if_cuda_error(
                    cudaMemcpyAsync(
                        thrust::raw_pointer_cast(out_count.data()),
                        thrust::raw_pointer_cast(shared_virtual_degrees.data()) +
                            vertex_count - 1,
                        sizeof(unsigned long long), cudaMemcpyDeviceToDevice,
                        stream),
                    "cudaMemcpyAsync(shared_pull_pair_count)");
              });
          detail::throw_if_cuda_error(
              cudaGetLastError(), "compact_shared_pull_frontier_kernel");
        } else {
          float pull_ms = detail::timed_gpu(
              stream, options.profile_iterations, launch_legacy_shared_pull);
          profile.pull_kernel_ms += pull_ms;
          detail::throw_if_cuda_error(cudaGetLastError(),
                                      "launch_shared_pull_values");
        }
      } else if (use_pull && options.pull_strategy == pull_strategy_t::ge_spmm) {
        if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
          detail::throw_if_cuda_error(
              cudaMemsetAsync(thrust::raw_pointer_cast(ge_spmm_out.data()), 0,
                              dense_capacity * sizeof(float), stream),
              "cudaMemsetAsync(ge_spmm_out)");
          detail::throw_if_cuda_error(
              cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_dense.data()),
                              0, dense_capacity * sizeof(float), stream),
              "cudaMemsetAsync(next_frontier_dense)");
        } else {
          thrust::fill(policy, ge_spmm_out.begin(), ge_spmm_out.end(),
                       Policy::infinity());
          thrust::fill(policy, next_frontier_dense.begin(),
                       next_frontier_dense.end(), Policy::infinity());
        }
        if (shared_push) {
          detail::throw_if_cuda_error(
              cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()),
                              0, vertex_count * sizeof(query_mask_t), stream),
              "cudaMemsetAsync(next_frontier_mask)");
        }
        profile.ge_spmm_pull_kernel_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
                detail::launch_ge_spmm_bfs(
                    graph, thrust::raw_pointer_cast(frontier_dense.data()),
                    thrust::raw_pointer_cast(ge_spmm_out.data()), m_eff, stream);
              } else {
                detail::launch_ge_spmm_sssp(
                    graph, thrust::raw_pointer_cast(frontier_dense.data()),
                    thrust::raw_pointer_cast(ge_spmm_out.data()), m_eff, stream);
              }
            });
        detail::throw_if_cuda_error(cudaGetLastError(), "GE-SpMM kernel");
        profile.pull_kernel_ms += profile.ge_spmm_pull_kernel_ms;
        profile.ge_spmm_postprocess_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::ge_spmm_postprocess_kernel<Policy, vertex_type>
                  <<<detail::grid_for(pair_capacity, threads), threads, 0,
                     stream>>>(
                      thrust::raw_pointer_cast(ge_spmm_out.data()), vertex_count,
                      query_count, m_eff, thrust::raw_pointer_cast(values.data()),
                      level, thrust::raw_pointer_cast(next_frontier_dense.data()),
                      thrust::raw_pointer_cast(out_count.data()),
                      shared_push ? thrust::raw_pointer_cast(visited_mask.data())
                                  : nullptr,
                      shared_push
                          ? thrust::raw_pointer_cast(next_frontier_mask.data())
                          : nullptr,
                      shared_push
                          ? thrust::raw_pointer_cast(shared_frontier_b.data())
                          : nullptr,
                      shared_push
                          ? thrust::raw_pointer_cast(shared_pair_count.data())
                          : nullptr,
                      shared_push);
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "ge_spmm_postprocess_kernel");
      } else if (use_pull) {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_bitmap.data()),
                            0, pair_capacity * sizeof(unsigned char), stream),
            "cudaMemsetAsync(next_frontier_bitmap)");
        if (shared_push) {
          detail::throw_if_cuda_error(
              cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()),
                              0, vertex_count * sizeof(query_mask_t), stream),
              "cudaMemsetAsync(next_frontier_mask)");
        }
        std::size_t warp_count = pair_capacity;
        int pull_blocks =
            detail::grid_for(warp_count * static_cast<std::size_t>(32), threads);
        profile.pull_kernel_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::pull_expand_kernel<Policy, graph_t, vertex_type>
                  <<<pull_blocks, threads, 0, stream>>>(
                      graph, query_count,
                      thrust::raw_pointer_cast(frontier_bitmap.data()),
                      thrust::raw_pointer_cast(next_frontier_bitmap.data()),
                      thrust::raw_pointer_cast(out_count.data()),
                      shared_push ? thrust::raw_pointer_cast(visited_mask.data())
                                  : nullptr,
                      shared_push
                          ? thrust::raw_pointer_cast(next_frontier_mask.data())
                          : nullptr,
                      shared_push
                          ? thrust::raw_pointer_cast(shared_frontier_b.data())
                          : nullptr,
                      shared_push
                          ? thrust::raw_pointer_cast(shared_pair_count.data())
                          : nullptr,
                      shared_push, thrust::raw_pointer_cast(values.data()),
                      level);
            });
        detail::throw_if_cuda_error(cudaGetLastError(), "pull_expand_kernel");
      } else if (shared_push &&
                 (actual_edges > 0 ||
                  (skip_shared_direct_edge_count && current_unique_count > 0))) {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()),
                            0, vertex_count * sizeof(query_mask_t), stream),
            "cudaMemsetAsync(next_frontier_mask)");
        int vertex_blocks =
            static_cast<int>(std::min<std::size_t>(
                std::max<std::size_t>(current_unique_count, 1), 65535));
        profile.shared_push_kernel_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              if (options.push_strategy ==
                  push_strategy_t::shared_node_query_parallel) {
                detail::expand_shared_node_query_parallel_kernel<
                    Policy, graph_t, vertex_type>
                    <<<vertex_blocks, threads, 0, stream>>>(
                        graph,
                        thrust::raw_pointer_cast(shared_frontier_a.data()),
                        thrust::raw_pointer_cast(frontier_mask.data()),
                        current_unique_count,
                        thrust::raw_pointer_cast(visited_mask.data()),
                        thrust::raw_pointer_cast(next_frontier_mask.data()),
                        thrust::raw_pointer_cast(shared_frontier_b.data()),
                        thrust::raw_pointer_cast(out_count.data()),
                        thrust::raw_pointer_cast(shared_pair_count.data()),
                        thrust::raw_pointer_cast(values.data()), query_count,
                        level);
              } else if (options.push_strategy ==
                         push_strategy_t::shared_node_warp) {
                int warp_blocks =
                    detail::grid_for(current_unique_count * 32ULL, threads);
                detail::expand_shared_node_warp_kernel<
                    Policy, graph_t, vertex_type>
                    <<<warp_blocks, threads, 0, stream>>>(
                        graph,
                        thrust::raw_pointer_cast(shared_frontier_a.data()),
                        thrust::raw_pointer_cast(frontier_mask.data()),
                        current_unique_count,
                        thrust::raw_pointer_cast(visited_mask.data()),
                        thrust::raw_pointer_cast(next_frontier_mask.data()),
                        thrust::raw_pointer_cast(shared_frontier_b.data()),
                        thrust::raw_pointer_cast(out_count.data()),
                        thrust::raw_pointer_cast(shared_pair_count.data()),
                        thrust::raw_pointer_cast(values.data()), query_count,
                        level);
              } else if (options.push_strategy ==
                         push_strategy_t::shared_node_degree) {
                if constexpr (std::is_same<Policy,
                                            algorithms::bfs_policy>::value) {
                  detail::launch_expand_shared_node_degree_bfs<
                      graph_t, vertex_type>(
                      graph,
                      thrust::raw_pointer_cast(shared_frontier_a.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(shared_frontier_b.data()),
                      thrust::raw_pointer_cast(out_count.data()),
                      thrust::raw_pointer_cast(shared_pair_count.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      level, threads, stream);
                } else {
                  int warp_blocks =
                      detail::grid_for(current_unique_count * 32ULL, threads);
                  detail::expand_shared_node_warp_kernel<
                      Policy, graph_t, vertex_type>
                      <<<warp_blocks, threads, 0, stream>>>(
                          graph,
                          thrust::raw_pointer_cast(shared_frontier_a.data()),
                          thrust::raw_pointer_cast(frontier_mask.data()),
                          current_unique_count,
                          thrust::raw_pointer_cast(visited_mask.data()),
                          thrust::raw_pointer_cast(next_frontier_mask.data()),
                          thrust::raw_pointer_cast(shared_frontier_b.data()),
                          thrust::raw_pointer_cast(out_count.data()),
                          thrust::raw_pointer_cast(shared_pair_count.data()),
                          thrust::raw_pointer_cast(values.data()), query_count,
                          level);
                }
              } else {
                detail::expand_shared_node_kernel<Policy, graph_t, vertex_type>
                    <<<vertex_blocks, threads, 0, stream>>>(
                        graph,
                        thrust::raw_pointer_cast(shared_frontier_a.data()),
                        thrust::raw_pointer_cast(frontier_mask.data()),
                        current_unique_count,
                        thrust::raw_pointer_cast(visited_mask.data()),
                        thrust::raw_pointer_cast(next_frontier_mask.data()),
                        thrust::raw_pointer_cast(shared_frontier_b.data()),
                        thrust::raw_pointer_cast(out_count.data()),
                        thrust::raw_pointer_cast(shared_pair_count.data()),
                        thrust::raw_pointer_cast(values.data()), query_count,
                        level);
              }
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "expand_shared_node_kernel");
        profile.push_kernel_ms += profile.shared_push_kernel_ms;
        result.shared_push_kernel_ms += profile.shared_push_kernel_ms;
        detail::clear_shared_frontier_mask_kernel<vertex_type>
            <<<detail::grid_for(current_unique_count, threads), threads, 0,
               stream>>>(thrust::raw_pointer_cast(shared_frontier_a.data()),
                          current_unique_count,
                          thrust::raw_pointer_cast(frontier_mask.data()));
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "clear_shared_frontier_mask_kernel");
      } else if (total_edges > 0) {
        unsigned long long edge_blocks_64 =
            (total_edges + static_cast<unsigned long long>(threads) - 1ULL) /
            static_cast<unsigned long long>(threads);
        int edge_blocks = static_cast<int>(
            std::min<unsigned long long>(edge_blocks_64, 65535ULL));
        profile.push_kernel_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::expand_edge_balanced_kernel<Policy, graph_t, vertex_type>
                  <<<edge_blocks, threads, 0, stream>>>(
                      graph, thrust::raw_pointer_cast(frontier_a.data()),
                      current_count, thrust::raw_pointer_cast(edge_offsets.data()),
                      thrust::raw_pointer_cast(edge_ends.data()), total_edges,
                      thrust::raw_pointer_cast(frontier_b.data()),
                      thrust::raw_pointer_cast(out_count.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      level);
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "expand_edge_balanced_kernel");
      }

      unsigned long long next_count = 0;
      unsigned long long next_unique_count = 0;
      auto sync_start = std::chrono::high_resolution_clock::now();
      if (shared_push) {
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&next_unique_count,
                            thrust::raw_pointer_cast(
                                use_pull ? shared_pair_count.data()
                                         : out_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(next_unique_count)");
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&next_count,
                            thrust::raw_pointer_cast(
                                use_pull ? out_count.data()
                                         : shared_pair_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(next_count)");
      } else {
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&next_count, thrust::raw_pointer_cast(out_count.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(next_count)");
      }
      context.synchronize();
      profile.count_sync_ms += detail::elapsed_ms(sync_start);
      if (!shared_push) {
        next_unique_count = next_count;
      }

      profile.mode = use_pull ? "pull" : "push";
      profile.edge_count = total_edges;
      profile.actual_edge_count =
          shared_push && !use_pull ? actual_edges : total_edges;
      profile.virtual_edge_count =
          shared_push && !use_pull ? virtual_edges : total_edges;
      profile.iteration_wall_ms = detail::elapsed_ms(iteration_start);
      result.iteration_edge_counts.push_back(total_edges);
      result.actual_iteration_edge_counts.push_back(profile.actual_edge_count);
      result.virtual_iteration_edge_counts.push_back(profile.virtual_edge_count);
      result.iteration_modes.push_back(profile.mode);
      result.iteration_wall_times_ms.push_back(profile.iteration_wall_ms);
      if (options.profile_iterations) {
        result.iteration_profiles.push_back(profile);
      }
      ++result.iterations;
      ++level;
      current_count = static_cast<std::size_t>(next_count);
      result.frontier_sizes.push_back(current_count);

      if (use_pull) {
        if (shared_push) {
          shared_frontier_a.swap(shared_frontier_b);
          frontier_mask.swap(next_frontier_mask);
          current_unique_count = static_cast<std::size_t>(next_unique_count);
          current_repr = frontier_repr_t::shared;
        } else if (options.pull_strategy == pull_strategy_t::ge_spmm) {
          frontier_dense.swap(next_frontier_dense);
          current_repr = frontier_repr_t::dense;
        } else {
          frontier_bitmap.swap(next_frontier_bitmap);
          current_repr = frontier_repr_t::bitmap;
        }
        if (!shared_push) {
          current_unique_count = current_count;
        }
      } else {
        if (shared_push) {
          shared_frontier_a.swap(shared_frontier_b);
          frontier_mask.swap(next_frontier_mask);
          current_unique_count = static_cast<std::size_t>(next_unique_count);
          current_repr = frontier_repr_t::shared;
        } else {
          frontier_a.swap(frontier_b);
          current_repr = frontier_repr_t::list;
        }
      }
      result.unique_frontier_sizes.push_back(shared_push ? current_unique_count
                                                         : current_count);
    }

    result.gpu_time_ms = total_timer.end(stream);
    result.wall_time_ms = detail::elapsed_ms(wall_start);
    result.queries.reserve(queries.size());
    for (std::size_t q = 0; q < queries.size(); ++q) {
      result.queries.push_back(
          {static_cast<vertex_type>(q), queries[q],
           static_cast<vertex_type>(std::max(0, static_cast<int>(level) - 1)),
           result.wall_time_ms});
    }
    result.values = std::move(values);
    return result;
  }
};

}  // namespace puercgp
