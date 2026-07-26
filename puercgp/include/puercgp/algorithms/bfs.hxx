#pragma once

#include <limits>

#include <puercgp/algorithms/algorithm_traits.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace algorithms {

// we should to add a abstract for application policy
struct bfs_policy {
  using vertex_type = int;
  using value_type = int;

  static constexpr execution_model_t execution_model =
      execution_model_t::frontier;
  static constexpr algo_kind_t algorithm_kind = algo_kind_t::bfs;
  static constexpr init_mode_t init_mode =
      algorithm_traits<algorithm_kind>::init_mode;
  static constexpr reduction_kind_t reduction =
      algorithm_traits<algorithm_kind>::reduction;

  static constexpr value_type infinity() {
    return std::numeric_limits<value_type>::max();
  }

  static constexpr value_type source_value() { return 0; }

  template <typename weight_t>
  static constexpr value_type relax(value_type source_distance,  // 这个名字叫 relax 可能不太合适？
                                    weight_t /*weight*/) {
    return source_distance + 1;  
  }

  static constexpr bool should_update(value_type candidate,
                                      value_type current) {
    return candidate < current;
  }
};

}  // namespace algorithms
}  // namespace puercgp
