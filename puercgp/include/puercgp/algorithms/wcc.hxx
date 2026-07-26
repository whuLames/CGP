#pragma once

#include <limits>

#include <puercgp/algorithms/algorithm_traits.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace algorithms {

// WCC via Label Propagation（同质 policy，用于单算法基线对比）
// 与 sssp_policy 同构：
//   - value_type = float（label = vertex id，cast 后 < 2^24 无损）
//   - relax 退化为 identity（直接 propagate source 的 label）
//   - should_update = candidate < current（min-reduce 语义）
//
// 注意：WCC 的 init 不是单源 source_value，而是 per-vertex label=vertex_id
//       （所有顶点初始为自己的连通分量代表）。
//       frontier_engine 的 init 段对 wcc_policy 有专门分支调
//       init_wcc_labels_kernel，不使用 source_value()。
struct wcc_policy {
  using vertex_type = int;
  using value_type = float;

  static constexpr execution_model_t execution_model =
      execution_model_t::frontier;
  static constexpr algo_kind_t algorithm_kind = algo_kind_t::wcc;
  static constexpr init_mode_t init_mode =
      algorithm_traits<algorithm_kind>::init_mode;
  static constexpr reduction_kind_t reduction =
      algorithm_traits<algorithm_kind>::reduction;

  static constexpr value_type infinity() {
    return std::numeric_limits<value_type>::infinity();
  }

  // 仅满足 policy 接口契约；WCC 实际不使用（init 走 per-vertex 路径）
  static constexpr value_type source_value() { return 0.0f; }

  template <typename weight_t>
  static constexpr value_type relax(value_type source_label, weight_t) {
    return source_label;  // identity：label propagation 的 relax 退化
  }

  static constexpr bool should_update(value_type candidate, value_type current) {
    return candidate < current;
  }
};

}  // namespace algorithms
}  // namespace puercgp
