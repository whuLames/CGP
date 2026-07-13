#pragma once

// 异构 batch 执行引擎：支持 BFS + SSSP + WCC 在同一 batch 内混合执行
// 与同质 frontier_engine 并存，复用 detail 命名空间下算法无关的 device helper
//
// 设计要点（见 /home/zyl/.claude/plans/sunny-dancing-fog.md）：
//   - unified_value_t = float，统一表示 BFS 层级 / SSSP 距离 / WCC label
//   - BFS slot 保留 visited_mask 批量 atomicOr 优化
//   - SSSP/WCC slot 走 apply_min_reduce（与 SSSP 同构）
//   - WCC per-vertex init（label=vertex_id），区别于 BFS/SSSP 单源 init

#include <chrono>
#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/algorithms/hybrid.hxx>
#include <puercgp/backend/pull_graph_access.hxx>
#include <puercgp/core/query_descriptor.hxx>
#include <puercgp/core/reduce_ops.hxx>
#include <puercgp/core/types.hxx>
#include <puercgp/engine/frontier_engine.hxx>  // 复用 detail::
#include <puercgp/kernels/replenish/slot_io_kernels.hxx>
#include <puercgp/state/engine_workspace.hxx>
#include <puercgp/state/pull_workspace.hxx>

namespace puercgp {
namespace hybrid_detail {

// 复用同质引擎的算法无关 helper（不重写，DRY）
using puercgp::detail::atomic_or_query_mask;
using puercgp::detail::elapsed_ms;
using puercgp::detail::grid_for;
using puercgp::detail::mark_next_shared_frontier;
using puercgp::detail::mark_next_shared_frontier_with_signal;
using puercgp::detail::mask_ffs;
using puercgp::detail::mask_popcount;
using puercgp::detail::get_pull_edge_weight;
using puercgp::detail::get_pull_neighbor_vertex;
using puercgp::detail::get_pull_starting_edge;
using puercgp::detail::query_bit;
using puercgp::detail::throw_if_cuda_error;
using puercgp::detail::timed_gpu;
using puercgp::detail::value_index;

// ============================================================
// Step 4: 异构 init kernels
// ============================================================

// 填 unified_infinity 到整个 values 数组
__global__ void fill_unified_kernel(
    algorithms::unified_value_t* values, std::size_t total) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < total; i += stride) {
    values[i] = algorithms::unified_infinity();
  }
}

// BFS / SSSP 源点 init（WCC slot 跳过，由 init_wcc_all_vertices_kernel 处理）
//   BFS slot : values[src,q]=source_value, visited_mask 设 bit, frontier_mask 设 bit
//   SSSP slot: values[src,q]=source_value, frontier_mask 设 bit（无 visited）
//   WCC slot : 跳过
__global__ void init_hybrid_sources_kernel(
    const int* sources,
    const algorithms::algo_kind_t* slot_kinds,
    const algorithms::unified_value_t* slot_source_values,
    int query_count,
    algorithms::unified_value_t* values,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    int* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t q = tid; q < static_cast<std::size_t>(query_count);
       q += stride) {
    algorithms::algo_kind_t kind = slot_kinds[q];
    if (kind == algorithms::algo_kind_t::wcc) {
      continue;  // WCC 由专门 kernel 处理
    }
    int src = sources[q];
    query_mask_t bit = query_bit(static_cast<int>(q));
    values[value_index(static_cast<std::size_t>(src), q,
                       static_cast<std::size_t>(query_count))] =
        slot_source_values[q];
    if (kind == algorithms::algo_kind_t::bfs) {
      atomic_or_query_mask(visited_mask + src, bit);
    }
    query_mask_t old = atomic_or_query_mask(frontier_mask + src, bit);
    if (old == 0) {
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = src;
    }
  }
}

// WCC per-vertex init：每个顶点的 label = vertex_id，所有顶点入 frontier
// 仅处理 slot_kinds[q]==wcc 的 query；每个 thread 处理一个 (vertex, wcc_query) pair
template <typename graph_t>
__global__ void init_wcc_all_vertices_kernel(
    graph_t graph,
    const algorithms::algo_kind_t* slot_kinds,
    int query_count,
    algorithms::unified_value_t* values,
    query_mask_t* frontier_mask,
    int* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t total = vertex_count * static_cast<std::size_t>(query_count);
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t idx = tid; idx < total; idx += stride) {
    std::size_t q = idx % static_cast<std::size_t>(query_count);
    if (slot_kinds[q] != algorithms::algo_kind_t::wcc) {
      continue;
    }
    std::size_t v = idx / static_cast<std::size_t>(query_count);
    values[value_index(v, q, static_cast<std::size_t>(query_count))] =
        static_cast<algorithms::unified_value_t>(v);  // label = vertex id
    query_mask_t bit = query_bit(static_cast<int>(q));
    query_mask_t old = atomic_or_query_mask(frontier_mask + v, bit);
    if (old == 0) {
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = static_cast<int>(v);
    }
  }
}

// ============================================================
// Step 5: heterogeneous shared_node push kernel
// 双路设计：BFS slot 走 visited_mask 批量 atomicOr；
//           SSSP/WCC slot 走 apply_min_reduce
// bfs_slot_mask / nonbfs_slot_mask 由 host 端预算（hybrid_query_batch），
// 作为 kernel 参数传入，O(1) 寄存器常量，避免 inner loop 查表
// ============================================================
template <typename graph_t, bool use_start_levels = false>
__global__ void expand_shared_node_hybrid_kernel(
    graph_t graph,
    const int* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    int* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    algorithms::unified_value_t* values,
    const algorithms::algo_kind_t* slot_kinds,
    int query_count,
    int level,
    query_mask_t bfs_slot_mask,
    query_mask_t nonbfs_slot_mask,
    const int* bfs_start_levels,
    query_mask_t* active_union) {
  for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
    int source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source];
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);

    for (auto edge = begin + threadIdx.x; edge < end; edge += blockDim.x) {
      int neighbor = graph.get_destination_vertex(edge);
      float weight = graph.get_edge_weight(edge);
      query_mask_t improved = 0;

      // ===== 路径 A：BFS slot 批量 atomicOr visited（与现有 637-653 一致）=====
      query_mask_t active_bfs = active_mask & bfs_slot_mask;
      if (active_bfs != 0) {
        query_mask_t old_visited =
            atomic_or_query_mask(visited_mask + neighbor, active_bfs);
        query_mask_t newly_visited = active_bfs & ~old_visited;
        if (newly_visited != 0) {
          query_mask_t bits = newly_visited;
          while (bits != 0) {
            int q = mask_ffs(bits) - 1;
            if (q < query_count) {
              int local_level = level + 1;
              if constexpr (use_start_levels) {
                local_level = level - bfs_start_levels[q] + 1;
              }
              values[value_index(static_cast<std::size_t>(neighbor),
                                 static_cast<std::size_t>(q),
                                 static_cast<std::size_t>(query_count))] =
                  static_cast<algorithms::unified_value_t>(local_level);
            }
            bits &= (bits - 1);
          }
          improved |= newly_visited;
        }
      }

      // ===== 路径 B：SSSP/WCC slot 逐 query min-reduce =====
      query_mask_t active_nonbfs = active_mask & nonbfs_slot_mask;
      if (active_nonbfs != 0) {
        query_mask_t bits = active_nonbfs;
        while (bits != 0) {
          int q = mask_ffs(bits) - 1;
          if (q < query_count) {
            algorithms::algo_kind_t kind = slot_kinds[q];
            std::size_t src_pos = value_index(
                static_cast<std::size_t>(source),
                static_cast<std::size_t>(q),
                static_cast<std::size_t>(query_count));
            algorithms::unified_value_t src_val = values[src_pos];
            if (src_val != algorithms::unified_infinity()) {
              algorithms::unified_value_t cand =
                  algorithms::dispatch_candidate_push(kind, src_val, weight,
                                                      level);
              std::size_t nb_pos = value_index(
                  static_cast<std::size_t>(neighbor),
                  static_cast<std::size_t>(q),
                  static_cast<std::size_t>(query_count));
              apply_result_t res = apply_min_reduce(&values[nb_pos], cand);
              if (res.improved) {
                improved |= query_bit(q);
              }
            }
          }
          bits &= (bits - 1);
        }
      }

      if (improved == 0) {
        continue;
      }
      mark_next_shared_frontier_with_signal(
          neighbor, improved, next_frontier_mask, next_frontier_vertices,
          next_unique_count, next_pair_count, active_union);
    }
  }
}

// ============================================================
// expand_shared_node_warp_hybrid_kernel（P3：warp 级异构 push）
// 参照同质 expand_shared_node_warp_kernel:819-905，
// 每 warp 处理 1 个 frontier 顶点（32 lane 协作遍历出边）。
// update 段套用 hybrid 双路：路径 A BFS 批量 atomicOr + 路径 B 非 BFS 逐 query min-reduce
// 相比 block 版（1 block/vertex）：高出度顶点的并行度从 1 block → 32 lane/warp
// ============================================================
template <typename graph_t, bool use_start_levels = false>
__global__ void expand_shared_node_warp_hybrid_kernel(
    graph_t graph,
    const int* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    int* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    algorithms::unified_value_t* values,
    const algorithms::algo_kind_t* slot_kinds,
    int query_count,
    int level,
    query_mask_t bfs_slot_mask,
    query_mask_t nonbfs_slot_mask,
    const int* bfs_start_levels,
    query_mask_t* active_union) {
  constexpr int warp_size = 32;
  int lane = threadIdx.x & (warp_size - 1);
  std::size_t warp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  std::size_t warp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) >> 5;

  for (std::size_t i = warp_id; i < unique_count; i += warp_stride) {
    int source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source];
    if (active_mask == 0) {
      continue;
    }
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);

    // 32 lane 协作遍历 source 的出边：每 lane 处理 1 条（步幅 32）
    for (auto edge = begin + lane; edge < end; edge += warp_size) {
      int neighbor = graph.get_destination_vertex(edge);
      float weight = graph.get_edge_weight(edge);
      query_mask_t improved = 0;

      // ===== 路径 A：BFS slot 批量 atomicOr visited =====
      query_mask_t active_bfs = active_mask & bfs_slot_mask;
      if (active_bfs != 0) {
        query_mask_t old_visited =
            atomic_or_query_mask(visited_mask + neighbor, active_bfs);
        query_mask_t newly_visited = active_bfs & ~old_visited;
        if (newly_visited != 0) {
          query_mask_t bits = newly_visited;
          while (bits != 0) {
            int q = mask_ffs(bits) - 1;
            if (q < query_count) {
              int local_level = level + 1;
              if constexpr (use_start_levels) {
                local_level = level - bfs_start_levels[q] + 1;
              }
              values[value_index(static_cast<std::size_t>(neighbor),
                                 static_cast<std::size_t>(q),
                                 static_cast<std::size_t>(query_count))] =
                  static_cast<algorithms::unified_value_t>(local_level);
            }
            bits &= (bits - 1);
          }
          improved |= newly_visited;
        }
      }

      // ===== 路径 B：SSSP/WCC slot 逐 query min-reduce =====
      query_mask_t active_nonbfs = active_mask & nonbfs_slot_mask;
      if (active_nonbfs != 0) {
        query_mask_t bits = active_nonbfs;
        while (bits != 0) {
          int q = mask_ffs(bits) - 1;
          if (q < query_count) {
            algorithms::algo_kind_t kind = slot_kinds[q];
            std::size_t src_pos = value_index(
                static_cast<std::size_t>(source),
                static_cast<std::size_t>(q),
                static_cast<std::size_t>(query_count));
            algorithms::unified_value_t src_val = values[src_pos];
            if (src_val != algorithms::unified_infinity()) {
              algorithms::unified_value_t cand =
                  algorithms::dispatch_candidate_push(kind, src_val, weight,
                                                      level);
              std::size_t nb_pos = value_index(
                  static_cast<std::size_t>(neighbor),
                  static_cast<std::size_t>(q),
                  static_cast<std::size_t>(query_count));
              apply_result_t res = apply_min_reduce(&values[nb_pos], cand);
              if (res.improved) {
                improved |= query_bit(q);
              }
            }
          }
          bits &= (bits - 1);
        }
      }

      if (improved == 0) {
        continue;
      }
      mark_next_shared_frontier_with_signal(
          neighbor, improved, next_frontier_mask, next_frontier_vertices,
          next_unique_count, next_pair_count, active_union);
    }
  }
}

// ============================================================
// Step 6: fused pull hybrid kernel
// pull 模式下 BFS 退化为 unweighted SSSP（nb_val+1），
// 所以全部 slot 走 algorithms::dispatch_candidate_pull，无双路
// ============================================================
template <typename graph_t>
__global__ void fused_pull_hybrid_simple_kernel(
    graph_t graph,
    int query_count,
    algorithms::unified_value_t* values,
    const algorithms::algo_kind_t* slot_kinds,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts,
    query_mask_t* active_union) {
  __shared__ query_mask_t thread_masks[128];

  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t query_stride = static_cast<std::size_t>(query_count);
  std::size_t vertex = blockIdx.x * blockDim.y + threadIdx.y;
  int query_id = threadIdx.x;
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  query_mask_t local_mask = 0;

  if (vertex < vertex_count && query_id < query_count) {
    algorithms::algo_kind_t kind = slot_kinds[query_id];
    algorithms::unified_value_t acc = algorithms::unified_infinity();
    auto begin = get_pull_starting_edge(graph, static_cast<int>(vertex));
    auto end = get_pull_starting_edge(graph, static_cast<int>(vertex + 1));
    for (auto edge = begin; edge < end; ++edge) {
      int neighbor = get_pull_neighbor_vertex(graph, edge);
      algorithms::unified_value_t nb_val = values[value_index(
          static_cast<std::size_t>(neighbor),
          static_cast<std::size_t>(query_id), query_stride)];
      if (nb_val != algorithms::unified_infinity()) {
        algorithms::unified_value_t candidate =
            algorithms::dispatch_candidate_pull(
                kind, nb_val, get_pull_edge_weight(graph, edge));
        if (candidate < acc) {
          acc = candidate;
        }
      }
    }
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query_id), query_stride);
    if (acc < values[value_pos]) {  // should_update: acc < cur（min 语义统一）
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
    atomic_or_query_mask(visited_mask + vertex, improved_mask);
    unique_flags[vertex] = improved_mask != 0 ? 1ULL : 0ULL;
    pair_counts[vertex] =
        static_cast<unsigned long long>(mask_popcount(improved_mask));
    atomic_or_query_mask(active_union, improved_mask);
  }
}

// ============================================================
// fused_pull_hybrid_smem_kernel（Q=33-64）
// 参照同质 fused_pull_smem_kernel(frontier_engine.hxx:1779-1879)，
// 核心差异：
//   - Policy::infinity() → algorithms::unified_infinity()
//   - Policy::relax(nb, w) → algorithms::dispatch_candidate_pull(kind, nb, w)
//   - 每 thread 读 slot_kinds[query0/1] 取自己的 kind
//   - should_update 统一为 acc < cur（min-reduce 语义）
// 每 thread 处理 2 条 query（query0=lane, query1=lane+32）覆盖 64 query
// ============================================================
template <int TILE_ROW, typename graph_t>
__global__ void fused_pull_hybrid_smem_kernel(
    graph_t graph,
    int query_count,
    algorithms::unified_value_t* values,
    const algorithms::algo_kind_t* slot_kinds,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    unsigned long long* unique_flags,
    unsigned long long* pair_counts,
    query_mask_t* active_union) {
  constexpr int warp_size = 32;
  using vertex_t = typename graph_t::vertex_type;
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

  // per-thread 读取自己的 kind（loop-invariant）
  algorithms::algo_kind_t kind0 =
      (query0 < query_count) ? slot_kinds[query0] : algorithms::algo_kind_t::bfs;
  algorithms::algo_kind_t kind1 =
      (query1 < query_count) ? slot_kinds[query1] : algorithms::algo_kind_t::bfs;

  algorithms::unified_value_t acc0 = algorithms::unified_infinity();
  algorithms::unified_value_t acc1 = algorithms::unified_infinity();

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
        algorithms::unified_value_t nb_val = values[value_index(
            static_cast<std::size_t>(neighbor),
            static_cast<std::size_t>(query0), query_stride)];
        if (nb_val != algorithms::unified_infinity()) {
          algorithms::unified_value_t candidate =
              algorithms::dispatch_candidate_pull(kind0, nb_val, weight);
          if (candidate < acc0) acc0 = candidate;
        }
      }
      if (query1 < query_count) {
        algorithms::unified_value_t nb_val = values[value_index(
            static_cast<std::size_t>(neighbor),
            static_cast<std::size_t>(query1), query_stride)];
        if (nb_val != algorithms::unified_infinity()) {
          algorithms::unified_value_t candidate =
              algorithms::dispatch_candidate_pull(kind1, nb_val, weight);
          if (candidate < acc1) acc1 = candidate;
        }
      }
    }
    __syncthreads();
  }

  query_mask_t local_mask = 0;
  if (row_valid && query0 < query_count) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query0), query_stride);
    if (acc0 < values[value_pos]) {
      values[value_pos] = acc0;
      local_mask |= query_bit(query0);
    }
  }
  if (row_valid && query1 < query_count) {
    std::size_t value_pos =
        value_index(vertex, static_cast<std::size_t>(query1), query_stride);
    if (acc1 < values[value_pos]) {
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
    atomic_or_query_mask(active_union, improved_mask);
  }
}

template <typename graph_t>
void launch_fused_pull_hybrid(graph_t graph,
                              int query_count,
                              algorithms::unified_value_t* values,
                              const algorithms::algo_kind_t* slot_kinds,
                              query_mask_t* visited_mask,
                              query_mask_t* next_frontier_mask,
                              unsigned long long* unique_flags,
                              unsigned long long* pair_counts,
                              query_mask_t* active_union,
                              cudaStream_t stream) {
  int vertex_count = static_cast<int>(graph.get_number_of_vertices());
  if (query_count <= 32) {
    int tile_row = std::max(1, 128 / std::max(1, query_count));
    int grid_x = (vertex_count + tile_row - 1) / tile_row;
    fused_pull_hybrid_simple_kernel<graph_t>
        <<<grid_x, dim3(query_count, tile_row), 0, stream>>>(
            graph, query_count, values, slot_kinds, visited_mask,
            next_frontier_mask, unique_flags, pair_counts, active_union);
  } else if (query_count <= 64) {
    constexpr int tile_row = 4;
    int grid_x = (vertex_count + tile_row - 1) / tile_row;
    fused_pull_hybrid_smem_kernel<tile_row, graph_t>
        <<<grid_x, dim3(32, tile_row), 0, stream>>>(
            graph, query_count, values, slot_kinds, visited_mask,
            next_frontier_mask, unique_flags, pair_counts, active_union);
  } else {
    throw std::invalid_argument(
        "fused_pull_hybrid supports at most 64 queries");
  }
}

}  // namespace hybrid_detail

// ============================================================
// hybrid_frontier_engine：异构 batch 的主循环（push-only 第一版）
// 支持 BFS + SSSP + WCC 混合，通过 shared_node push kernel 迭代至收敛
// pull/hybrid 模式与 frontier 表示转换留作后续（fused_pull_hybrid_kernel 已就位）
// ============================================================
class hybrid_frontier_engine {
 public:
  template <typename graph_t>
  auto run(graph_t& graph,
           const hybrid_query_batch& batch,
           execution_context& context,
           const run_options& options = run_options{}) const {
    using value_t = algorithms::unified_value_t;
    using result_t = run_result_t<int, value_t>;

    batch.validate(options.max_queries);
    const int Q = static_cast<int>(batch.size());
    const auto V = static_cast<std::size_t>(graph.get_number_of_vertices());
    if (Q > 64) {
      throw std::invalid_argument("hybrid supports at most 64 queries");
    }

    auto views = batch.upload_to_device();
    query_mask_t bfs_mask = batch.bfs_slot_mask();
    query_mask_t nonbfs_mask = batch.nonbfs_slot_mask();

    cudaStream_t stream = context.stream();
    constexpr int threads = 256;

    // 计时起点（参照 frontier_engine.hxx:2256-2258）
    auto wall_start = std::chrono::high_resolution_clock::now();
    detail::cuda_event_timer total_timer;
    total_timer.begin(stream);

    engine_workspace<int, value_t> workspace;
    workspace.resize(V, static_cast<std::size_t>(Q));
    auto& values = workspace.values_vector();
    auto& visited_mask = workspace.visited_mask_vector();
    auto& frontier_mask = workspace.frontier_mask_vector();
    auto& next_frontier_mask = workspace.next_frontier_mask_vector();
    auto& frontier_vertices = workspace.frontier_vertices_vector();
    auto& next_frontier_vertices = workspace.next_frontier_vertices_vector();
    auto& unique_count_dev = workspace.current_unique_count_vector();
    auto& next_unique_count_dev = workspace.next_unique_count_vector();
    auto& next_pair_count_dev = workspace.next_pair_count_vector();
    // replenishment 收敛信号：bit s=1 表示 slot s 本轮产生了 frontier 写入
    thrust::device_vector<query_mask_t> active_union_dev(1);
    // pull 路径专用 buffer（参照同质引擎 compact_shared_pull_frontier 模式）
    pull_workspace pull_state;
    pull_state.resize(V);
    auto& unique_flags = pull_state.unique_flags_vector();
    auto& pair_counts_buf = pull_state.pair_counts_vector();
    auto& unique_offsets = pull_state.unique_offsets_vector();

    auto mask_bytes = V * sizeof(query_mask_t);

    // ===== init =====
    hybrid_detail::fill_unified_kernel<<<hybrid_detail::grid_for(V * Q, threads),
                                         threads, 0, stream>>>(
        thrust::raw_pointer_cast(values.data()), V * static_cast<std::size_t>(Q));
    hybrid_detail::throw_if_cuda_error(cudaGetLastError(), "fill_unified");

    hybrid_detail::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0,
                        mask_bytes, stream),
        "memset visited");
    hybrid_detail::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                        mask_bytes, stream),
        "memset frontier");
    hybrid_detail::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                        mask_bytes, stream),
        "memset next_frontier");

    detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(unique_count_dev.data()));
    hybrid_detail::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(active_union_dev.data()), 0,
                        sizeof(query_mask_t), stream),
        "memset active_union (init)");

    hybrid_detail::init_hybrid_sources_kernel<<<hybrid_detail::grid_for(Q, threads),
                                                threads, 0, stream>>>(
        views.sources, views.kinds, views.source_values, Q,
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(visited_mask.data()),
        thrust::raw_pointer_cast(frontier_mask.data()),
        thrust::raw_pointer_cast(frontier_vertices.data()),
        thrust::raw_pointer_cast(unique_count_dev.data()));
    hybrid_detail::throw_if_cuda_error(cudaGetLastError(), "init_hybrid_sources");

    if (batch.has_wcc()) {
      hybrid_detail::init_wcc_all_vertices_kernel<<<hybrid_detail::grid_for(
          V * Q, threads), threads, 0, stream>>>(
          graph, views.kinds, Q,
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(frontier_mask.data()),
          thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(unique_count_dev.data()));
      hybrid_detail::throw_if_cuda_error(cudaGetLastError(), "init_wcc");
    }

    unsigned long long h_uc = 0;
    hybrid_detail::throw_if_cuda_error(
        cudaMemcpyAsync(&h_uc,
                        thrust::raw_pointer_cast(unique_count_dev.data()),
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream),
        "memcpy unique_count");
    context.synchronize();
    std::size_t current_unique = static_cast<std::size_t>(h_uc);

    // ===== 预算 pull 阈值（参照同质引擎 frontier_engine.hxx:2197-2201）=====
    // hybrid 无 list/bitmap capacity 概念，用 V 作 shared-frontier capacity；
    // hybrid batch 内 query 种类异构，无法用单一 edge_count 阈值精确刻画，
    // 用 current_unique（unique frontier 顶点数）作主要判据
    const double pull_frontier_threshold =
        options.pull_frontier_ratio * static_cast<double>(V);
    const std::size_t total_edges_static =
        static_cast<std::size_t>(graph.get_number_of_edges());
    const double pull_edge_threshold =
        options.pull_edge_ratio * static_cast<double>(Q) *
        static_cast<double>(total_edges_static);
    const bool has_pull_adjacency = detail::graph_has_pull_adjacency(graph);
    if (options.traversal_mode == traversal_mode_t::pull &&
        !has_pull_adjacency) {
      throw std::invalid_argument(
          "hybrid fused pull requires a graph view with incoming adjacency");
    }

    // ===== 主循环（push/pull/hybrid 三态分派）=====
    result_t result;
    result.options = options;
    result.effective_query_dim = Q;

    int level = 0;
    while (current_unique > 0 &&
           (options.max_iterations <= 0 || level < options.max_iterations)) {
      iteration_profile_t profile;
      profile.iteration = level;
      profile.frontier_size = current_unique;
      profile.unique_frontier_size = current_unique;
      profile.pull_frontier_threshold = pull_frontier_threshold;
      profile.pull_edge_threshold = pull_edge_threshold;
      auto iter_start = std::chrono::high_resolution_clock::now();

      // 决策：参照同质引擎 frontier_engine.hxx:2613-2621
      bool use_pull = false;
      if (options.traversal_mode == traversal_mode_t::pull) {
        use_pull = true;
      } else if (options.traversal_mode == traversal_mode_t::hybrid) {
        use_pull = has_pull_adjacency &&
            static_cast<double>(current_unique) >= pull_frontier_threshold;
      }
      // traversal_mode_t::push 时 use_pull = false（保持现有 push 路径）

      hybrid_detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                          mask_bytes, stream),
          "memset next_frontier (loop)");
      hybrid_detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(active_union_dev.data()), 0,
                          sizeof(query_mask_t), stream),
          "memset active_union (loop)");

      if (use_pull) {
        profile.mode = "pull";
        detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(next_unique_count_dev.data()));

        // 1) pull kernel：扫全图，输出 next_frontier_mask + unique_flags + pair_counts
        profile.pull_kernel_ms = detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              hybrid_detail::launch_fused_pull_hybrid(
                  graph, Q,
                  thrust::raw_pointer_cast(values.data()), views.kinds,
                  thrust::raw_pointer_cast(visited_mask.data()),
                  thrust::raw_pointer_cast(next_frontier_mask.data()),
                  thrust::raw_pointer_cast(unique_flags.data()),
                  thrust::raw_pointer_cast(pair_counts_buf.data()),
                  thrust::raw_pointer_cast(active_union_dev.data()), stream);
            });
        hybrid_detail::throw_if_cuda_error(cudaGetLastError(),
                                           "fused_pull_hybrid");

        // 2) inclusive_scan(unique_flags) → unique_offsets（cumulative unique count）
        profile.compact_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              auto par = thrust::cuda::par.on(stream);
              thrust::inclusive_scan(par, unique_flags.begin(),
                                     unique_flags.end(), unique_offsets.begin());
            });

        // 3) compact：从 per-vertex mask 重建 shared frontier_vertices list
        //    复用 detail::compact_shared_pull_frontier_kernel（与算法无关）
        profile.compact_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::compact_shared_pull_frontier_kernel<int>
                  <<<detail::grid_for(V, threads), threads, 0, stream>>>(
                      thrust::raw_pointer_cast(next_frontier_mask.data()), V,
                      thrust::raw_pointer_cast(unique_offsets.data()),
                      thrust::raw_pointer_cast(next_frontier_vertices.data()));
            });
        hybrid_detail::throw_if_cuda_error(cudaGetLastError(),
                                           "compact_shared_pull");

        // 4) 读回 next_unique_count = unique_offsets[V-1]（inclusive_scan 末尾元素）
        unsigned long long h_next_uc = 0;
        hybrid_detail::throw_if_cuda_error(
            cudaMemcpyAsync(&h_next_uc,
                            thrust::raw_pointer_cast(unique_offsets.data()) +
                                (V - 1),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "memcpy pull next_unique");
        query_mask_t h_union = 0;
        hybrid_detail::throw_if_cuda_error(
            cudaMemcpyAsync(&h_union,
                            thrust::raw_pointer_cast(active_union_dev.data()),
                            sizeof(query_mask_t), cudaMemcpyDeviceToHost, stream),
            "memcpy active_union (pull)");
        context.synchronize();
        profile.query_convergence_mask = h_union;

        workspace.swap_frontiers();
        current_unique = static_cast<std::size_t>(h_next_uc);
      } else {
        profile.mode = "push";
        detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(next_unique_count_dev.data()));
        detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(next_pair_count_dev.data()));

        // push_strategy 分派：shared_node_warp 走 warp 版（高出度顶点并行度更高），
        // 其他策略（含默认 shared_node）走 block 版
        const bool use_warp =
            options.push_strategy == push_strategy_t::shared_node_warp;

        auto push_start = std::chrono::high_resolution_clock::now();
        float push_gpu_ms = 0.0f;
        if (options.profile_iterations) {
          detail::cuda_event_timer kt;
          kt.begin(stream);
          if (use_warp) {
            hybrid_detail::expand_shared_node_warp_hybrid_kernel<graph_t>
                <<<hybrid_detail::grid_for(current_unique * 32, threads),
                   threads, 0, stream>>>(
                    graph,
                    thrust::raw_pointer_cast(frontier_vertices.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_vertices.data()),
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    thrust::raw_pointer_cast(next_pair_count_dev.data()),
                    thrust::raw_pointer_cast(values.data()), views.kinds, Q,
                    level, bfs_mask, nonbfs_mask, nullptr,
                    thrust::raw_pointer_cast(active_union_dev.data()));
          } else {
            hybrid_detail::expand_shared_node_hybrid_kernel<graph_t>
                <<<hybrid_detail::grid_for(current_unique, threads), threads,
                   0, stream>>>(
                    graph,
                    thrust::raw_pointer_cast(frontier_vertices.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_vertices.data()),
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    thrust::raw_pointer_cast(next_pair_count_dev.data()),
                    thrust::raw_pointer_cast(values.data()), views.kinds, Q,
                    level, bfs_mask, nonbfs_mask, nullptr,
                    thrust::raw_pointer_cast(active_union_dev.data()));
          }
          push_gpu_ms = kt.end(stream);
        } else {
          if (use_warp) {
            hybrid_detail::expand_shared_node_warp_hybrid_kernel<graph_t>
                <<<hybrid_detail::grid_for(current_unique * 32, threads),
                   threads, 0, stream>>>(
                    graph,
                    thrust::raw_pointer_cast(frontier_vertices.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_vertices.data()),
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    thrust::raw_pointer_cast(next_pair_count_dev.data()),
                    thrust::raw_pointer_cast(values.data()), views.kinds, Q,
                    level, bfs_mask, nonbfs_mask, nullptr,
                    thrust::raw_pointer_cast(active_union_dev.data()));
          } else {
            hybrid_detail::expand_shared_node_hybrid_kernel<graph_t>
                <<<hybrid_detail::grid_for(current_unique, threads), threads,
                   0, stream>>>(
                    graph,
                    thrust::raw_pointer_cast(frontier_vertices.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_vertices.data()),
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    thrust::raw_pointer_cast(next_pair_count_dev.data()),
                    thrust::raw_pointer_cast(values.data()), views.kinds, Q,
                    level, bfs_mask, nonbfs_mask, nullptr,
                    thrust::raw_pointer_cast(active_union_dev.data()));
          }
        }
        profile.shared_push_kernel_ms = push_gpu_ms;
        (void)push_start;
        hybrid_detail::throw_if_cuda_error(cudaGetLastError(), "expand_hybrid");

        unsigned long long h_next_uc = 0;
        hybrid_detail::throw_if_cuda_error(
            cudaMemcpyAsync(&h_next_uc,
                            thrust::raw_pointer_cast(next_unique_count_dev.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "memcpy next_unique");
        query_mask_t h_union = 0;
        hybrid_detail::throw_if_cuda_error(
            cudaMemcpyAsync(&h_union,
                            thrust::raw_pointer_cast(active_union_dev.data()),
                            sizeof(query_mask_t), cudaMemcpyDeviceToHost, stream),
            "memcpy active_union (push)");
        context.synchronize();
        profile.query_convergence_mask = h_union;

        workspace.swap_frontiers();
        current_unique = static_cast<std::size_t>(h_next_uc);
      }

      profile.iteration_wall_ms = detail::elapsed_ms(iter_start);
      ++level;

      if (options.profile_iterations) {
        result.iteration_profiles.push_back(std::move(profile));
        result.iteration_modes.push_back(profile.mode);
        result.frontier_sizes.push_back(profile.frontier_size);
        result.unique_frontier_sizes.push_back(profile.unique_frontier_size);
      }
    }

    // ===== 收集结果 =====
    result.iterations = level;
    result.values = std::move(values);
    result.gpu_time_ms = total_timer.end(stream);
    result.wall_time_ms = detail::elapsed_ms(wall_start);
    for (int q = 0; q < Q; ++q) {
      query_result_t<int> qr;
      qr.query_id = q;
      qr.source = batch[static_cast<std::size_t>(q)].source;
      qr.completion_level = level;
      result.queries.push_back(qr);
    }
    return result;
  }
};

// 异构 batch 入口
template <typename graph_t>
auto run_heterogeneous(graph_t& graph,
                       const hybrid_query_batch& batch,
                       execution_context& context,
                       const run_options& options = run_options{}) {
  return hybrid_frontier_engine{}.run(graph, batch, context, options);
}

}  // namespace puercgp
