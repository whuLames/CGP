#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace puercgp {

enum class execution_model_t { frontier, dense };

enum class reduction_kind_t { minimum, maximum, sum };

enum class traversal_mode_t { push, pull, hybrid };

enum class push_strategy_t {
  shared_node,
  shared_node_query_parallel,
  shared_node_warp
};

enum class pull_strategy_t { fused, degree_aware, degree_segmented };

enum class pull_sweep_t { forward, bidirectional };

enum class pull_update_mode_t { in_place, synchronous };

enum class frontier_repr_t { shared };

using query_mask_t = std::uint64_t;

struct run_options {
  traversal_mode_t traversal_mode = traversal_mode_t::hybrid;
  push_strategy_t push_strategy = push_strategy_t::shared_node_warp;
  pull_strategy_t pull_strategy = pull_strategy_t::fused;
  pull_update_mode_t pull_update_mode = pull_update_mode_t::in_place;
  int pull_degree_threshold_2 = 32;
  int pull_degree_threshold_4 = 64;
  int pull_degree_threshold_8 = 128;
  int pull_high_degree_segment_edges = 1024;
  int pull_high_degree_segment_threads = 256;
  std::vector<int> pull_degree_bucket_order{0, 1, 2, 3};
  double pull_frontier_ratio = 0.15;
  double pull_edge_ratio = 0.20;
  bool profile_iterations = false;
  bool emit_nvtx = false;
  int max_iterations = 0;
  // Keep dense algorithms active until max_iterations even if the convergence
  // test succeeds early. Frontier algorithms continue to use convergence.
  bool fixed_iterations = false;
  pull_sweep_t pull_sweep = pull_sweep_t::forward;
  // In bidirectional mode, add a reverse sweep every N outer iterations.
  // pull_bidirectional_rounds=0 leaves the periodic rule active to convergence.
  std::size_t pull_bidirectional_period = 1;
  std::size_t pull_bidirectional_rounds = 0;
  // Limit the extra reverse pass to [begin, end). end=0 selects all vertices.
  std::size_t pull_reverse_vertex_begin = 0;
  std::size_t pull_reverse_vertex_end = 0;
  // Optional experiment trace: aggregate the pull iterations in which each
  // vertex-query value improved. Disabled by default and never allocated on
  // production paths.
  bool trace_pull_updates = false;
  std::size_t max_queries = 0;
  // replenishment（运行期 slot 动态补给）开关：
  //   false（默认）→ 走现有 hybrid_frontier_engine::run，所有 replenish kernel 不调用
  //   true          → 走 run_replenish_pipeline，slot 收敛后从 pending 队列注入新 query
  // active_union 收敛信号的记录始终开启（与开关无关），开销可忽略，便于观测。
  bool enable_replenishment = false;
  // Minimum number of free slots accumulated before injecting a new cohort.
  // 0 selects an engine default; 1 preserves immediate per-slot replenishment.
  std::size_t replenish_batch_size = 0;
  // 是否丢弃完整结果（不分配 final_buffer N*V）：
  //   false（默认）→ 维护 final_buffer，slot 复用时 snapshot 存档，result.values 可用
  //   true          → 不分配 final_buffer，snapshot 的写结果跳过（clear 仍做），省 N*V 显存
  //                   用于 latency 基准（只需完成时间，不需结果值）
  bool discard_results = false;
};

inline const char* traversal_mode_name(traversal_mode_t mode) {
  switch (mode) {
    case traversal_mode_t::push:
      return "push";
    case traversal_mode_t::pull:
      return "pull";
    case traversal_mode_t::hybrid:
      return "hybrid";
  }
  return "unknown";
}

inline const char* push_strategy_name(push_strategy_t strategy) {
  switch (strategy) {
    case push_strategy_t::shared_node:
      return "shared_node";
    case push_strategy_t::shared_node_query_parallel:
      return "shared_node_query_parallel";
    case push_strategy_t::shared_node_warp:
      return "shared_node_warp";
  }
  return "unknown";
}

inline const char* pull_strategy_name(pull_strategy_t strategy) {
  switch (strategy) {
    case pull_strategy_t::fused:
      return "fused";
    case pull_strategy_t::degree_aware:
      return "degree_aware";
    case pull_strategy_t::degree_segmented:
      return "degree_segmented";
  }
  return "unknown";
}

inline const char* pull_update_mode_name(pull_update_mode_t mode) {
  switch (mode) {
    case pull_update_mode_t::in_place:
      return "in_place";
    case pull_update_mode_t::synchronous:
      return "synchronous";
  }
  return "unknown";
}

inline const char* frontier_repr_name(frontier_repr_t repr) {
  switch (repr) {
    case frontier_repr_t::shared:
      return "shared";
  }
  return "unknown";
}

}  // namespace puercgp
