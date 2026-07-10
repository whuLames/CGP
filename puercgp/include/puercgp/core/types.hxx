#pragma once

#include <cstddef>
#include <cstdint>

namespace puercgp {

enum class execution_model_t { frontier, dense };

enum class traversal_mode_t { push, pull, hybrid };

enum class push_strategy_t {
  edge_balanced,
  shared_node,
  shared_node_query_parallel,
  shared_node_warp,
  shared_node_degree
};

enum class pull_strategy_t { bitmap, ge_spmm };

enum class frontier_repr_t { list, shared, bitmap, dense };

using query_mask_t = std::uint64_t;

struct run_options {
  traversal_mode_t traversal_mode = traversal_mode_t::hybrid;
  push_strategy_t push_strategy = push_strategy_t::edge_balanced;
  pull_strategy_t pull_strategy = pull_strategy_t::bitmap;
  double pull_frontier_ratio = 0.15;
  double pull_edge_ratio = 0.20;
  bool profile_iterations = false;
  bool emit_nvtx = false;
  int max_iterations = 0;
  std::size_t max_queries = 0;
  // replenishment（运行期 slot 动态补给）开关：
  //   false（默认）→ 走现有 hybrid_frontier_engine::run，所有 replenish kernel 不调用
  //   true          → 走 run_replenish_pipeline，slot 收敛后从 pending 队列注入新 query
  // active_union 收敛信号的记录始终开启（与开关无关），开销可忽略，便于观测。
  bool enable_replenishment = false;
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
    case push_strategy_t::edge_balanced:
      return "edge_balanced";
    case push_strategy_t::shared_node:
      return "shared_node";
    case push_strategy_t::shared_node_query_parallel:
      return "shared_node_query_parallel";
    case push_strategy_t::shared_node_warp:
      return "shared_node_warp";
    case push_strategy_t::shared_node_degree:
      return "shared_node_degree";
  }
  return "unknown";
}

inline const char* pull_strategy_name(pull_strategy_t strategy) {
  switch (strategy) {
    case pull_strategy_t::bitmap:
      return "bitmap";
    case pull_strategy_t::ge_spmm:
      return "ge_spmm";
  }
  return "unknown";
}

inline const char* frontier_repr_name(frontier_repr_t repr) {
  switch (repr) {
    case frontier_repr_t::list:
      return "list";
    case frontier_repr_t::shared:
      return "shared";
    case frontier_repr_t::bitmap:
      return "bitmap";
    case frontier_repr_t::dense:
      return "dense";
  }
  return "unknown";
}

}  // namespace puercgp
