#pragma once

#include <stdexcept>

#include <puercgp/core/algorithm_query_batch.hxx>

namespace puercgp {
namespace algorithms {

enum class personalization_kind_t { uniform, source };

struct pagerank_query {
  float damping_factor = 0.85f;
  float epsilon = 1.0e-8f;
};

struct ppr_query {
  int source = 0;
  float damping_factor = 0.85f;
  float epsilon = 1.0e-8f;
};

using pagerank_query_batch = algorithm_query_batch<pagerank_query>;
using ppr_query_batch = algorithm_query_batch<ppr_query>;

inline void validate_rank_parameters(float damping_factor, float epsilon) {
  if (!(damping_factor >= 0.0f && damping_factor < 1.0f)) {
    throw std::invalid_argument(
        "damping_factor must be in the interval [0, 1)");
  }
  if (!(epsilon > 0.0f)) {
    throw std::invalid_argument("epsilon must be positive");
  }
}

} // namespace algorithms
} // namespace puercgp
