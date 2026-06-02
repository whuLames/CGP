#pragma once

#include <limits>

#include <puercgp/core/types.hxx>

namespace puercgp {
namespace algorithms {

struct sssp_policy {
  using vertex_type = int;
  using value_type = float;

  static constexpr execution_model_t execution_model =
      execution_model_t::frontier;
  static constexpr bool assumes_nonnegative_weights = true;

  static constexpr value_type infinity() {
    return std::numeric_limits<value_type>::infinity();
  }

  static constexpr value_type source_value() { return 0.0f; }

  template <typename weight_t>
  static constexpr value_type relax(value_type source_distance, weight_t weight) {
    return source_distance + static_cast<value_type>(weight);
  }

  static constexpr bool should_update(value_type candidate,
                                      value_type current) {
    return candidate < current;
  }
};

}  // namespace algorithms
}  // namespace puercgp
