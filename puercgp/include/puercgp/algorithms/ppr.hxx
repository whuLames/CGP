#pragma once

#include <cstddef>
#include <stdexcept>

#include <puercgp/algorithms/rank.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {
namespace algorithms {

struct ppr_policy {
  using vertex_type = int;
  using value_type = float;
  using query_type = ppr_query;

  static constexpr execution_model_t execution_model = execution_model_t::dense;
  static constexpr reduction_kind_t reduction = reduction_kind_t::sum;
  static constexpr personalization_kind_t personalization =
      personalization_kind_t::source;

  static void validate_query(const query_type &query,
                             std::size_t vertex_count) {
    validate_rank_parameters(query.damping_factor, query.epsilon);
    if (query.source < 0 ||
        static_cast<std::size_t>(query.source) >= vertex_count) {
      throw std::invalid_argument("PPR source is outside the graph");
    }
  }

  static constexpr int query_source(const query_type &query) {
    return query.source;
  }

  template <typename vertex_t>
  __host__ __device__ static value_type
  personalization_value(vertex_t vertex, int query_id,
                        std::size_t /*vertex_count*/, const vertex_t *sources) {
    return vertex == sources[query_id] ? 1.0f : 0.0f;
  }
};

} // namespace algorithms
} // namespace puercgp
