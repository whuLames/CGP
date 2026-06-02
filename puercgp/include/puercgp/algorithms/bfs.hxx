#pragma once

#include <limits>

#include <puercgp/core/types.hxx>

namespace puercgp {
namespace algorithms {

// we should to add a abstract for application policy
struct bfs_policy {
  using vertex_type = int;
  using value_type = int;

  static constexpr execution_model_t execution_model =
      execution_model_t::frontier;

  static constexpr value_type infinity() {
    return std::numeric_limits<value_type>::max();
  }

  static constexpr value_type source_value() { return 0; } // 感觉这样的抽象完全不需要s

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
