#pragma once

#include <puercgp/algorithms/algorithm_traits.hxx>

namespace puercgp {
namespace algorithms {

__device__ __forceinline__ unified_value_t dispatch_candidate_push(
    algo_kind_t kind,
    unified_value_t source_value,
    float edge_weight,
    int level) {
  switch (kind) {
    case algo_kind_t::bfs:
      return algorithm_traits<algo_kind_t::bfs>::candidate_push(
          source_value, edge_weight, level);
    case algo_kind_t::sssp:
      return algorithm_traits<algo_kind_t::sssp>::candidate_push(
          source_value, edge_weight, level);
    case algo_kind_t::wcc:
      return algorithm_traits<algo_kind_t::wcc>::candidate_push(
          source_value, edge_weight, level);
  }
  return unified_infinity();
}

__device__ __forceinline__ unified_value_t dispatch_candidate_pull(
    algo_kind_t kind,
    unified_value_t neighbor_value,
    float edge_weight) {
  switch (kind) {
    case algo_kind_t::bfs:
      return algorithm_traits<algo_kind_t::bfs>::candidate_pull(
          neighbor_value, edge_weight);
    case algo_kind_t::sssp:
      return algorithm_traits<algo_kind_t::sssp>::candidate_pull(
          neighbor_value, edge_weight);
    case algo_kind_t::wcc:
      return algorithm_traits<algo_kind_t::wcc>::candidate_pull(
          neighbor_value, edge_weight);
  }
  return unified_infinity();
}

}  // namespace algorithms
}  // namespace puercgp
