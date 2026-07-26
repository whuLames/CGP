#pragma once

// replenish_engine.hxx —— 运行期 slot 动态补给（replenishment）引擎
//
// 解决两个痛点（对应 ICDE Minimum Bar 6 latency/fairness）：
//   1. ≤64 slot 硬限：N>64 query 一次提交，引擎内部 slot 复用
//   2. 长尾算法拖死整批：短轮次 query 收敛后，立刻把 pending 队列的新 query
//      装进腾出的 slot 继续跑，提升 GPU 利用率
//
// 设计（见 /home/zyl/.claude/plans/logical-tickling-cosmos.md）：
//   - Host 同步调度：复用现有每轮 synchronize 点，slot 收敛检测 + pending 注入
//   - 一次性提交队列 API：run_replenish_pipeline(graph, vector<descriptor>, ...)
//   - 收敛信号 active_union：bit s=0 即 slot s 本轮无写入（已收敛）
//   - slot 回收时立刻 snapshot values（避免同 slot 后续 query 覆盖）
//   - WCC 分流：WCC 全部进首批 slot（不进 pending）；WCC slot 收敛后可复用给 BFS/SSSP
//   - result.values 是 row-major [N*V]（values[q*V+v]），与 hybrid 的 vertex-major 不同
//
// 复用 hybrid_detail 的 init/push/pull kernel，slot I/O 通过 slot_io_manager 统一入口

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/swap.h>

#include <puercgp/algorithms/hybrid.hxx>
#include <puercgp/core/query_descriptor.hxx>
#include <puercgp/core/result.hxx>
#include <puercgp/core/types.hxx>
#include <puercgp/engine/frontier_engine.hxx>  // detail:: helper
#include <puercgp/engine/hybrid_engine.hxx>    // hybrid_detail:: helper + kernel
#include <puercgp/engine/slot_io_manager.hxx>

namespace puercgp {

class replenish_frontier_engine {
 public:
  template <typename graph_t>
  auto run(graph_t& graph,
           const std::vector<query_descriptor_t>& all_queries,
           execution_context& context,
           const run_options& options = run_options{}) const {
    using value_t = algorithms::unified_value_t;
    using result_t = run_result_t<int, value_t>;
    namespace hd = puercgp::hybrid_detail;
    using puercgp::detail::reset_counter_kernel;

    const std::size_t N = all_queries.size();
    if (N == 0) {
      throw std::invalid_argument("replenish: query list must be non-empty");
    }
    for (const auto& query : all_queries) {
      if (query.kind == algorithms::algo_kind_t::sswp) {
        throw std::invalid_argument(
            "replenish: SSWP slot reset is not implemented");
      }
    }
    const std::size_t slot_capacity =
        options.max_queries == 0
            ? std::size_t{64}
            : std::min(options.max_queries, std::size_t{64});
    if (slot_capacity == 0) {
      throw std::invalid_argument("replenish: max_queries must be positive");
    }
    const std::size_t Q = std::min(N, slot_capacity);
    const auto V = static_cast<std::size_t>(graph.get_number_of_vertices());
    int active_query_count = static_cast<int>(Q);

    // ===== WCC 分流：WCC 进首批 slot（不进 pending），非 WCC 填剩余 slot + pending =====
    std::vector<std::size_t> wcc_idx, nonwcc_idx;
    for (std::size_t i = 0; i < N; ++i) {
      if (all_queries[i].kind == algorithms::algo_kind_t::wcc) {
        wcc_idx.push_back(i);
      } else {
        nonwcc_idx.push_back(i);
      }
    }
    if (wcc_idx.size() > Q) {
      throw std::invalid_argument(
          "replenish: WCC count exceeds slot capacity (WCC cannot be pended)");
    }
    // slot 顺序：[WCC..., 非WCC 前 (Q-wcc) 个]；pending = 非WCC 剩余
    std::vector<query_descriptor_t> slot_descs;
    std::vector<query_descriptor_t> pending;
    std::vector<int> slot_orig_id(Q);       // slot -> original query id
    std::vector<int> pending_orig_id;       // pending 顺序 -> original query id
    slot_descs.reserve(Q);
    for (std::size_t i : wcc_idx) {
      slot_orig_id[slot_descs.size()] = static_cast<int>(i);
      slot_descs.push_back(all_queries[i]);
    }
    const std::size_t nonwcc_in_slot = Q - wcc_idx.size();
    for (std::size_t i = 0; i < nonwcc_idx.size(); ++i) {
      if (i < nonwcc_in_slot) {
        slot_orig_id[slot_descs.size()] = static_cast<int>(nonwcc_idx[i]);
        slot_descs.push_back(all_queries[nonwcc_idx[i]]);
      } else {
        pending_orig_id.push_back(static_cast<int>(nonwcc_idx[i]));
        pending.push_back(all_queries[nonwcc_idx[i]]);
      }
    }

    hybrid_query_batch batch(slot_descs);
    batch.validate(options.max_queries);
    auto views = batch.upload_to_device();
    query_mask_t bfs_mask = batch.bfs_slot_mask();
    query_mask_t nonbfs_mask = batch.nonbfs_slot_mask();

    cudaStream_t stream = context.stream();
    constexpr int threads = 256;

    auto wall_start = std::chrono::high_resolution_clock::now();
    detail::cuda_event_timer total_timer;
    total_timer.begin(stream);

    // ===== device buffer 分配（复用 hybrid_engine 模式）=====
    thrust::device_vector<value_t> values(V * Q);
    thrust::device_vector<query_mask_t> visited_mask(V);
    thrust::device_vector<query_mask_t> frontier_mask(V);
    thrust::device_vector<query_mask_t> next_frontier_mask(V);
    thrust::device_vector<int> frontier_vertices(V);
    thrust::device_vector<int> next_frontier_vertices(V);
    thrust::device_vector<unsigned long long> unique_count_dev(1);
    thrust::device_vector<unsigned long long> next_unique_count_dev(1);
    thrust::device_vector<unsigned long long> next_pair_count_dev(1);
    thrust::device_vector<query_mask_t> active_union_dev(1);
    thrust::device_vector<int> slot_start_levels(Q, 0);
    thrust::device_vector<unsigned long long> unique_flags(V);
    thrust::device_vector<unsigned long long> pair_counts_buf(V);
    thrust::device_vector<unsigned long long> unique_offsets(V);
    // 最终结果 buffer：row-major [N*V]，values[q*V + v]
    // discard_results=true 时不分配（省 N*V 显存，latency 基准用），传 nullptr 给 kernel
    thrust::device_vector<value_t> final_buffer;
    value_t* final_buffer_raw = nullptr;
    if (!options.discard_results) {
      final_buffer.assign(static_cast<std::size_t>(N) * V,
                          algorithms::unified_infinity());
      final_buffer_raw = thrust::raw_pointer_cast(final_buffer.data());
    }
    // replenishment 批量调度 device buffer（优化1+2+3）
    thrust::device_vector<int> conv_slots_dev(Q);        // 本轮收敛 slot 列表
    thrust::device_vector<int> conv_final_ids_dev(Q);    // 每个 slot 的 orig query id
    thrust::device_vector<algorithms::algo_kind_t> conv_kinds_dev(Q);
    thrust::device_vector<int> new_slots_dev(Q);         // 要注入的 slot
    thrust::device_vector<int> value_reset_slots_dev(Q);
    thrust::device_vector<algorithms::algo_kind_t> new_kinds_dev(Q);
    thrust::device_vector<int> new_sources_dev(Q);
    thrust::device_vector<algorithms::unified_value_t> new_sv_dev(Q);

    auto mask_bytes = V * sizeof(query_mask_t);

    // ===== init =====
    hd::fill_unified_kernel<<<hd::grid_for(V * Q, threads), threads, 0, stream>>>(
        thrust::raw_pointer_cast(values.data()), V * Q);
    hd::throw_if_cuda_error(cudaGetLastError(), "fill_unified");

    hd::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0,
                        mask_bytes, stream),
        "memset visited");
    hd::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                        mask_bytes, stream),
        "memset frontier");
    hd::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                        mask_bytes, stream),
        "memset next_frontier");
    hd::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(active_union_dev.data()), 0,
                        sizeof(query_mask_t), stream),
        "memset active_union (init)");

    reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(unique_count_dev.data()));

    hd::init_hybrid_sources_kernel<<<hd::grid_for(Q, threads), threads, 0,
                                     stream>>>(
        views.sources, views.kinds, views.source_values, active_query_count,
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(visited_mask.data()),
        thrust::raw_pointer_cast(frontier_mask.data()),
        thrust::raw_pointer_cast(frontier_vertices.data()),
        thrust::raw_pointer_cast(unique_count_dev.data()));
    hd::throw_if_cuda_error(cudaGetLastError(), "init_hybrid_sources");

    if (batch.has_wcc()) {
      hd::init_wcc_all_vertices_kernel<<<hd::grid_for(V * Q, threads), threads,
                                         0, stream>>>(
          graph, views.kinds, active_query_count,
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(frontier_mask.data()),
          thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(unique_count_dev.data()));
      hd::throw_if_cuda_error(cudaGetLastError(), "init_wcc");
    }

    unsigned long long h_uc = 0;
    hd::throw_if_cuda_error(
        cudaMemcpyAsync(&h_uc,
                        thrust::raw_pointer_cast(unique_count_dev.data()),
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                        stream),
        "memcpy unique_count (init)");
    context.synchronize();
    std::size_t current_unique = static_cast<std::size_t>(h_uc);

    // latency 计时起点：init 完成后（不含 buffer 分配/fill 等 init 开销）
    auto latency_start = std::chrono::high_resolution_clock::now();

    const double pull_frontier_threshold =
        options.pull_frontier_ratio * static_cast<double>(V);
    const std::size_t total_edges_static =
        static_cast<std::size_t>(graph.get_number_of_edges());
    (void)total_edges_static;
    const bool has_pull_adjacency = detail::graph_has_pull_adjacency(graph);
    if (options.traversal_mode == traversal_mode_t::pull &&
        !has_pull_adjacency) {
      throw std::invalid_argument(
          "replenish fused pull requires a graph view with incoming adjacency");
    }

    // ===== slot 调度状态 =====
    query_mask_t h_active_slot_mask =
        (Q == 64) ? ~query_mask_t{0} : ((query_mask_t{1} << Q) - 1);
    std::vector<int> completion_level(N, -1);   // per original query
    std::vector<float> completion_wall(N, -1.0f);  // per query 完成时刻（latency，相对 latency_start）
    std::size_t pending_head = 0;               // pending FIFO 出队指针
    std::vector<int> reusable_slots;
    reusable_slots.reserve(Q);
    const std::size_t replenish_batch_size =
        options.replenish_batch_size == 0
            ? std::max<std::size_t>(1, Q - wcc_idx.size())
            : std::min(options.replenish_batch_size, Q);

    // ===== 主循环（slot 复用调度）=====
    result_t result;
    result.options = options;
    result.effective_query_dim = static_cast<int>(N);

    int level = 0;
    while (h_active_slot_mask != 0 || pending_head < pending.size()) {
      if (options.max_iterations > 0 && level >= options.max_iterations) {
        break;
      }
      if (h_active_slot_mask == 0) {
        // 防御性：无 active slot 但 pending 非空（不应发生，每次回收都立即注入）
        break;
      }

      // ===== 单轮 push / pull（复用 hybrid kernel，传 active_union）=====
      hd::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()),
                          0, mask_bytes, stream),
          "memset next_frontier (loop)");
      hd::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(active_union_dev.data()), 0,
                          sizeof(query_mask_t), stream),
          "memset active_union (loop)");

      bool use_pull = false;
      if (options.traversal_mode == traversal_mode_t::pull) {
        use_pull = true;
      } else if (options.traversal_mode == traversal_mode_t::hybrid) {
        use_pull = has_pull_adjacency &&
            static_cast<double>(current_unique) >= pull_frontier_threshold;
      }

      if (use_pull) {
        reset_counter_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(next_unique_count_dev.data()));
        hd::launch_fused_pull_hybrid(
            graph, active_query_count, thrust::raw_pointer_cast(values.data()), views.kinds,
            thrust::raw_pointer_cast(visited_mask.data()),
            thrust::raw_pointer_cast(next_frontier_mask.data()),
            thrust::raw_pointer_cast(unique_flags.data()),
            thrust::raw_pointer_cast(pair_counts_buf.data()),
            thrust::raw_pointer_cast(active_union_dev.data()), stream);
        hd::throw_if_cuda_error(cudaGetLastError(), "fused_pull_hybrid");

        auto par = thrust::cuda::par.on(stream);
        thrust::inclusive_scan(par, unique_flags.begin(),
                               unique_flags.end(), unique_offsets.begin());
        detail::compact_shared_pull_frontier_kernel<int>
            <<<detail::grid_for(V, threads), threads, 0, stream>>>(
                thrust::raw_pointer_cast(next_frontier_mask.data()), V,
                thrust::raw_pointer_cast(unique_offsets.data()),
                thrust::raw_pointer_cast(next_frontier_vertices.data()));
        hd::throw_if_cuda_error(cudaGetLastError(), "compact_shared_pull");

        unsigned long long h_next_uc = 0;
        hd::throw_if_cuda_error(
            cudaMemcpyAsync(&h_next_uc,
                            thrust::raw_pointer_cast(unique_offsets.data()) +
                                (V - 1),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "memcpy pull next_unique");
        query_mask_t h_union = 0;
        hd::throw_if_cuda_error(
            cudaMemcpyAsync(&h_union,
                            thrust::raw_pointer_cast(active_union_dev.data()),
                            sizeof(query_mask_t), cudaMemcpyDeviceToHost,
                            stream),
            "memcpy active_union (pull)");
        context.synchronize();

        thrust::swap(frontier_mask, next_frontier_mask);
        thrust::swap(frontier_vertices, next_frontier_vertices);
        current_unique = static_cast<std::size_t>(h_next_uc);

        // 调度段
        h_active_slot_mask = dispatch_converged(
            h_active_slot_mask, h_union, level, active_query_count, V, graph,
            stream, context,
            views, batch, bfs_mask, nonbfs_mask, values, visited_mask,
            frontier_mask, next_frontier_mask, frontier_vertices,
            next_unique_count_dev, active_union_dev, final_buffer_raw, slot_orig_id,
            pending, pending_orig_id, pending_head, completion_level,
            completion_wall, latency_start, current_unique,
            conv_slots_dev, conv_final_ids_dev, new_slots_dev,
            value_reset_slots_dev, conv_kinds_dev, new_kinds_dev,
            new_sources_dev, new_sv_dev,
            options.traversal_mode != traversal_mode_t::push,
            reusable_slots, replenish_batch_size, slot_start_levels);
      } else {
        reset_counter_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(next_unique_count_dev.data()));
        reset_counter_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(next_pair_count_dev.data()));

        const bool use_warp =
            options.push_strategy == push_strategy_t::shared_node_warp;
        if (use_warp) {
          hd::expand_shared_node_warp_hybrid_kernel<graph_t, true>
              <<<hd::grid_for(current_unique * 32, threads), threads, 0,
                 stream>>>(
                  graph,
                  thrust::raw_pointer_cast(frontier_vertices.data()),
                  thrust::raw_pointer_cast(frontier_mask.data()), current_unique,
                  thrust::raw_pointer_cast(visited_mask.data()),
                  thrust::raw_pointer_cast(next_frontier_mask.data()),
                  thrust::raw_pointer_cast(next_frontier_vertices.data()),
                  thrust::raw_pointer_cast(next_unique_count_dev.data()),
                  thrust::raw_pointer_cast(next_pair_count_dev.data()),
                  thrust::raw_pointer_cast(values.data()), views.kinds,
                  active_query_count,
                  level, bfs_mask, nonbfs_mask,
                  thrust::raw_pointer_cast(slot_start_levels.data()),
                  thrust::raw_pointer_cast(active_union_dev.data()));
        } else {
          hd::expand_shared_node_hybrid_kernel<graph_t, true>
              <<<hd::grid_for(current_unique, threads), threads, 0, stream>>>(
                  graph,
                  thrust::raw_pointer_cast(frontier_vertices.data()),
                  thrust::raw_pointer_cast(frontier_mask.data()), current_unique,
                  thrust::raw_pointer_cast(visited_mask.data()),
                  thrust::raw_pointer_cast(next_frontier_mask.data()),
                  thrust::raw_pointer_cast(next_frontier_vertices.data()),
                  thrust::raw_pointer_cast(next_unique_count_dev.data()),
                  thrust::raw_pointer_cast(next_pair_count_dev.data()),
                  thrust::raw_pointer_cast(values.data()), views.kinds,
                  active_query_count,
                  level, bfs_mask, nonbfs_mask,
                  thrust::raw_pointer_cast(slot_start_levels.data()),
                  thrust::raw_pointer_cast(active_union_dev.data()));
        }
        hd::throw_if_cuda_error(cudaGetLastError(), "expand_hybrid");

        unsigned long long h_next_uc = 0;
        hd::throw_if_cuda_error(
            cudaMemcpyAsync(&h_next_uc,
                            thrust::raw_pointer_cast(next_unique_count_dev.data()),
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                            stream),
            "memcpy next_unique");
        query_mask_t h_union = 0;
        hd::throw_if_cuda_error(
            cudaMemcpyAsync(&h_union,
                            thrust::raw_pointer_cast(active_union_dev.data()),
                            sizeof(query_mask_t), cudaMemcpyDeviceToHost,
                            stream),
            "memcpy active_union (push)");
        context.synchronize();

        thrust::swap(frontier_mask, next_frontier_mask);
        thrust::swap(frontier_vertices, next_frontier_vertices);
        current_unique = static_cast<std::size_t>(h_next_uc);

        h_active_slot_mask = dispatch_converged(
            h_active_slot_mask, h_union, level, active_query_count, V, graph,
            stream, context,
            views, batch, bfs_mask, nonbfs_mask, values, visited_mask,
            frontier_mask, next_frontier_mask, frontier_vertices,
            next_unique_count_dev, active_union_dev, final_buffer_raw, slot_orig_id,
            pending, pending_orig_id, pending_head, completion_level,
            completion_wall, latency_start, current_unique,
            conv_slots_dev, conv_final_ids_dev, new_slots_dev,
            value_reset_slots_dev, conv_kinds_dev, new_kinds_dev,
            new_sources_dev, new_sv_dev,
            options.traversal_mode != traversal_mode_t::push,
            reusable_slots, replenish_batch_size, slot_start_levels);
      }

      if (options.profile_iterations) {
        result.frontier_sizes.push_back(current_unique);
      }
      ++level;
    }

    // ===== max_iterations 截断：仍 active 的 slot snapshot 当前（部分）结果 =====
    if (h_active_slot_mask != 0) {
      query_mask_t bits = h_active_slot_mask;
      while (bits != 0) {
        int s = __builtin_ffsll(static_cast<long long>(bits)) - 1;
        bits &= (bits - 1);
        int orig = slot_orig_id[s];
        if (final_buffer_raw != nullptr) {
          slot_io_manager::snapshot_slot(
              s, batch[static_cast<std::size_t>(s)].kind, active_query_count, V,
              thrust::raw_pointer_cast(values.data()),
              thrust::raw_pointer_cast(visited_mask.data()),
              final_buffer_raw + static_cast<std::size_t>(orig) * V,
              stream);
        }
        if (completion_level[orig] < 0) {
          completion_level[orig] = level;  // 截断轮次
          completion_wall[orig] = std::chrono::duration<float, std::milli>(
              std::chrono::high_resolution_clock::now() - latency_start).count();
        }
      }
      context.synchronize();
    }

    // ===== 收集结果 =====
    result.iterations = level;
    result.values = std::move(final_buffer);  // row-major [N*V]
    result.gpu_time_ms = total_timer.end(stream);
    result.wall_time_ms = detail::elapsed_ms(wall_start);
    for (std::size_t q = 0; q < N; ++q) {
      query_result_t<int> qr;
      qr.query_id = static_cast<int>(q);
      qr.source = all_queries[q].source;
      qr.completion_level = completion_level[q] < 0 ? level : completion_level[q];
      qr.completion_wall_time_ms = completion_wall[q] < 0 ? 0.0f : completion_wall[q];
      result.queries.push_back(qr);
    }
    return result;
  }

 private:
  // 调度段：处理本轮收敛的 slot（snapshot + 从 pending 注入新 query）
  // 返回更新后的 h_active_slot_mask
  template <typename graph_t>
  static query_mask_t dispatch_converged(
      query_mask_t h_active_slot_mask, query_mask_t h_union, int level,
      int& active_query_count, std::size_t V, graph_t& graph,
      cudaStream_t stream,
      execution_context& context,
      hybrid_query_batch::device_views& views, hybrid_query_batch& batch,
      query_mask_t& bfs_mask, query_mask_t& nonbfs_mask,
      thrust::device_vector<algorithms::unified_value_t>& values,
      thrust::device_vector<query_mask_t>& visited_mask,
      thrust::device_vector<query_mask_t>& frontier_mask,
      thrust::device_vector<query_mask_t>& next_frontier_mask,
      thrust::device_vector<int>& frontier_vertices,
      thrust::device_vector<unsigned long long>& unique_count_dev,
      thrust::device_vector<query_mask_t>& active_union_dev,
      algorithms::unified_value_t* final_buffer_raw,
      std::vector<int>& slot_orig_id,
      std::vector<query_descriptor_t>& pending,
      std::vector<int>& pending_orig_id, std::size_t& pending_head,
      std::vector<int>& completion_level,
      std::vector<float>& completion_wall,
      std::chrono::high_resolution_clock::time_point latency_start,
      std::size_t& current_unique,
      thrust::device_vector<int>& conv_slots_dev,
      thrust::device_vector<int>& conv_final_ids_dev,
      thrust::device_vector<int>& new_slots_dev,
      thrust::device_vector<int>& value_reset_slots_dev,
      thrust::device_vector<algorithms::algo_kind_t>& conv_kinds_dev,
      thrust::device_vector<algorithms::algo_kind_t>& new_kinds_dev,
      thrust::device_vector<int>& new_sources_dev,
      thrust::device_vector<algorithms::unified_value_t>& new_sv_dev,
      bool reset_all_values,
      std::vector<int>& reusable_slots,
      std::size_t replenish_batch_size,
      thrust::device_vector<int>& slot_start_levels) {
    namespace hd = puercgp::hybrid_detail;
    const int Qi = active_query_count;

    query_mask_t converged = h_active_slot_mask & ~h_union;
    if (converged == 0) {
      return h_active_slot_mask;  // 本轮无 slot 收敛
    }

    // ===== host 收集本轮收敛 slot + 决定注入哪些 pending（一趟遍历 mask）=====
    std::vector<int> h_conv_slots;
    std::vector<int> h_conv_ids;        // snapshot 目标 orig query id
    std::vector<algorithms::algo_kind_t> h_conv_kinds;
    std::vector<int> h_new_slots;       // 要 reinit 的 slot
    std::vector<query_descriptor_t> h_new_descs;
    std::vector<int> h_new_orig_ids;
    query_mask_t bits = converged;
    while (bits != 0) {
      int s = __builtin_ffsll(static_cast<long long>(bits)) - 1;
      bits &= (bits - 1);
      h_conv_slots.push_back(s);
      h_conv_ids.push_back(slot_orig_id[s]);
      h_conv_kinds.push_back(batch[static_cast<std::size_t>(s)].kind);
      int orig_id = slot_orig_id[s];
      completion_level[orig_id] = level + 1;
      completion_wall[orig_id] = std::chrono::duration<float, std::milli>(
          std::chrono::high_resolution_clock::now() - latency_start).count();
      h_active_slot_mask &= ~(query_mask_t{1} << s);
      reusable_slots.push_back(s);
    }

    const int k_conv = static_cast<int>(h_conv_slots.size());

    // Snapshot all completed queries, but reset only slots that will be reused.
    if (final_buffer_raw != nullptr) {
      conv_slots_dev = h_conv_slots;
      conv_final_ids_dev = h_conv_ids;
      conv_kinds_dev = h_conv_kinds;
      slot_io_manager::snapshot_slots(
          thrust::raw_pointer_cast(conv_slots_dev.data()),
          thrust::raw_pointer_cast(conv_final_ids_dev.data()),
          thrust::raw_pointer_cast(conv_kinds_dev.data()),
          k_conv, Qi, V,
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(visited_mask.data()),
          final_buffer_raw, V, stream);
    }

    const std::size_t pending_count = pending.size() - pending_head;
    if (h_active_slot_mask == 0 && pending_count > 0 &&
        pending_count < static_cast<std::size_t>(active_query_count)) {
      std::vector<query_descriptor_t> tail_descs;
      tail_descs.reserve(pending_count);
      for (std::size_t i = 0; i < pending_count; ++i) {
        tail_descs.push_back(pending[pending_head + i]);
        slot_orig_id[i] = pending_orig_id[pending_head + i];
      }
      pending_head += pending_count;
      active_query_count = static_cast<int>(pending_count);
      batch = hybrid_query_batch(std::move(tail_descs));
      batch.validate();
      views = batch.upload_to_device();
      bfs_mask = batch.bfs_slot_mask();
      nonbfs_mask = batch.nonbfs_slot_mask();
      reusable_slots.clear();
      thrust::fill(thrust::cuda::par.on(stream), slot_start_levels.begin(),
                   slot_start_levels.begin() + active_query_count, level + 1);

      constexpr int tail_threads = 256;
      const std::size_t tail_value_count =
          V * static_cast<std::size_t>(active_query_count);
      hd::fill_unified_kernel<<<hd::grid_for(tail_value_count, tail_threads),
                                tail_threads, 0, stream>>>(
          thrust::raw_pointer_cast(values.data()), tail_value_count);
      const std::size_t mask_bytes = V * sizeof(query_mask_t);
      hd::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0,
                          mask_bytes, stream),
          "memset visited (tail compact)");
      hd::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                          mask_bytes, stream),
          "memset frontier (tail compact)");
      hd::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                          mask_bytes, stream),
          "memset next frontier (tail compact)");
      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(unique_count_dev.data()));
      hd::init_hybrid_sources_kernel
          <<<hd::grid_for(static_cast<std::size_t>(active_query_count),
                          tail_threads),
             tail_threads, 0, stream>>>(
              views.sources, views.kinds, views.source_values,
              active_query_count, thrust::raw_pointer_cast(values.data()),
              thrust::raw_pointer_cast(visited_mask.data()),
              thrust::raw_pointer_cast(frontier_mask.data()),
              thrust::raw_pointer_cast(frontier_vertices.data()),
              thrust::raw_pointer_cast(unique_count_dev.data()));
      hd::throw_if_cuda_error(cudaGetLastError(), "init tail cohort");

      unsigned long long h_uc = 0;
      hd::throw_if_cuda_error(
          cudaMemcpyAsync(&h_uc,
                          thrust::raw_pointer_cast(unique_count_dev.data()),
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                          stream),
          "memcpy unique_count (tail compact)");
      context.synchronize();
      current_unique = static_cast<std::size_t>(h_uc);
      return active_query_count == 64
                 ? ~query_mask_t{0}
                 : ((query_mask_t{1} << active_query_count) - 1);
    }
    const bool launch_cohort =
        reusable_slots.size() >= replenish_batch_size ||
        h_active_slot_mask == 0;
    if (launch_cohort && pending_count > 0) {
      const std::size_t cohort_size =
          std::min(reusable_slots.size(), pending_count);
      for (std::size_t i = 0; i < cohort_size; ++i) {
        int s = reusable_slots[i];
        const query_descriptor_t& nd = pending[pending_head];
        if (nd.kind == algorithms::algo_kind_t::wcc) {
          throw std::logic_error("WCC must not enter pending queue");
        }
        h_new_slots.push_back(s);
        h_new_descs.push_back(nd);
        h_new_orig_ids.push_back(pending_orig_id[pending_head]);
        ++pending_head;
      }
      reusable_slots.erase(reusable_slots.begin(),
                           reusable_slots.begin() + cohort_size);
    }

    // ===== launch 2: set_sources_multi（仅对要注入的 slot）=====
    const int k_new = static_cast<int>(h_new_slots.size());
    if (k_new > 0) {
      std::vector<int> h_srcs(k_new);
      std::vector<algorithms::algo_kind_t> h_kinds(k_new);
      std::vector<algorithms::unified_value_t> h_sv(k_new);
      std::vector<int> h_value_reset_slots;
      query_mask_t visited_clear_bits = 0;
      for (int i = 0; i < k_new; ++i) {
        h_srcs[i] = h_new_descs[i].source;
        h_kinds[i] = h_new_descs[i].kind;
        h_sv[i] = h_new_descs[i].source_value;
        const int slot = h_new_slots[i];
        if (reset_all_values || h_new_descs[i].kind != algorithms::algo_kind_t::bfs) {
          h_value_reset_slots.push_back(slot);
        }
        if (h_new_descs[i].kind == algorithms::algo_kind_t::bfs) {
          visited_clear_bits |= (query_mask_t{1} << slot);
        }
      }
      new_slots_dev = h_new_slots;
      value_reset_slots_dev = h_value_reset_slots;
      new_kinds_dev = h_kinds;
      new_sources_dev = h_srcs;
      new_sv_dev = h_sv;

      const bool reset_full_cohort =
          k_new == Qi && h_active_slot_mask == 0;
      if (reset_full_cohort) {
        constexpr int reset_threads = 256;
        const std::size_t value_count = V * static_cast<std::size_t>(Qi);
        hd::fill_unified_kernel
            <<<hd::grid_for(value_count, reset_threads), reset_threads, 0,
               stream>>>(thrust::raw_pointer_cast(values.data()), value_count);
        hd::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0,
                            V * sizeof(query_mask_t), stream),
            "memset visited (full replenish cohort)");
      } else {
        slot_io_manager::reset_reused_slots(
            thrust::raw_pointer_cast(value_reset_slots_dev.data()),
            static_cast<int>(h_value_reset_slots.size()), visited_clear_bits,
            Qi, V, thrust::raw_pointer_cast(values.data()),
            thrust::raw_pointer_cast(visited_mask.data()), stream);
      }

      slot_io_manager::reinit_slots(
          thrust::raw_pointer_cast(new_slots_dev.data()),
          thrust::raw_pointer_cast(new_kinds_dev.data()),
          thrust::raw_pointer_cast(new_sources_dev.data()),
          thrust::raw_pointer_cast(new_sv_dev.data()),
          k_new, Qi,
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(visited_mask.data()),
          thrust::raw_pointer_cast(frontier_mask.data()),
          thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(unique_count_dev.data()),
          thrust::raw_pointer_cast(slot_start_levels.data()), level + 1,
          stream);

      // host 端更新 slot 元数据 + active mask + device slot_kinds
      for (int i = 0; i < k_new; ++i) {
        int s = h_new_slots[i];
        slot_orig_id[s] = h_new_orig_ids[i];
        h_active_slot_mask |= (query_mask_t{1} << s);
      }
      views = batch.update_slots(h_new_slots, h_new_descs);
      bfs_mask = batch.bfs_slot_mask();
      nonbfs_mask = batch.nonbfs_slot_mask();

      // 批量 sync：循环外读 unique_count 一次（set_sources_multi 已累加）
      unsigned long long h_uc = 0;
      hd::throw_if_cuda_error(
          cudaMemcpyAsync(&h_uc,
                          thrust::raw_pointer_cast(unique_count_dev.data()),
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                          stream),
          "memcpy unique_count (reinit batch)");
      context.synchronize();
      current_unique = static_cast<std::size_t>(h_uc);
    } else {
      // Only a result snapshot may be pending; discard mode has no device work.
      if (final_buffer_raw != nullptr) context.synchronize();
    }

    return h_active_slot_mask;
  }
};

// 顶层入口：一次性提交 N 个 query（N 可 > 64），引擎内部 slot 复用
// enable_replenishment=false 时回退：N<=64 走 run_heterogeneous，N>64 抛异常
template <typename graph_t>
auto run_replenish_pipeline(graph_t& graph,
                            const std::vector<query_descriptor_t>& all_queries,
                            execution_context& context,
                            const run_options& options = run_options{}) {
  if (!options.enable_replenishment) {
    if (all_queries.size() > 64) {
      throw std::invalid_argument(
          "enable_replenishment=false but query count > 64");
    }
    hybrid_query_batch batch(all_queries);
    return run_heterogeneous(graph, batch, context, options);
  }
  return replenish_frontier_engine{}.run(graph, all_queries, context, options);
}

}  // namespace puercgp
