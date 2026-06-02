#pragma once

#include <puercgp/core/types.hxx>

namespace puercgp {
namespace algorithms {

struct pagerank_policy {
  using vertex_type = int;
  using value_type = float;

  static constexpr execution_model_t execution_model = execution_model_t::dense;
  static constexpr value_type default_damping_factor = 0.85f;

  static constexpr value_type initial_rank(int vertex_count) {
    return vertex_count > 0 ? value_type{1.0f} / static_cast<value_type>(vertex_count)
                            : value_type{0.0f};
  }
};

}  // namespace algorithms
}  // namespace puercgp
