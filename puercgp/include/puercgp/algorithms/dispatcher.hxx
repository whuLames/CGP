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
    case algo_kind_t::sswp:
      return algorithm_traits<algo_kind_t::sswp>::candidate_push(
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
    case algo_kind_t::sswp:
      return algorithm_traits<algo_kind_t::sswp>::candidate_pull(
          neighbor_value, edge_weight);
  }
  return unified_infinity();
}

__host__ __device__ __forceinline__ unified_value_t dispatch_infinity(
    algo_kind_t kind) {
  switch (kind) {
    case algo_kind_t::bfs:
      return algorithm_traits<algo_kind_t::bfs>::infinity();
    case algo_kind_t::sssp:
      return algorithm_traits<algo_kind_t::sssp>::infinity();
    case algo_kind_t::wcc:
      return algorithm_traits<algo_kind_t::wcc>::infinity();
    case algo_kind_t::sswp:
      return algorithm_traits<algo_kind_t::sswp>::infinity();
  }
  return unified_infinity();
}

__host__ __device__ __forceinline__ unified_value_t dispatch_source_value(
    algo_kind_t kind) {
  switch (kind) {
    case algo_kind_t::bfs:
      return algorithm_traits<algo_kind_t::bfs>::source_value();
    case algo_kind_t::sssp:
      return algorithm_traits<algo_kind_t::sssp>::source_value();
    case algo_kind_t::wcc:
      return algorithm_traits<algo_kind_t::wcc>::source_value();
    case algo_kind_t::sswp:
      return algorithm_traits<algo_kind_t::sswp>::source_value();
  }
  return unified_source_value();
}

__host__ __device__ __forceinline__ bool dispatch_is_reachable(
    algo_kind_t kind,
    unified_value_t value) {
  return value != dispatch_infinity(kind);
}

__host__ __device__ __forceinline__ bool dispatch_should_update(
    algo_kind_t kind,
    unified_value_t candidate,
    unified_value_t current) {
  switch (kind) {
    case algo_kind_t::bfs:
      return algorithm_traits<algo_kind_t::bfs>::should_update(candidate,
                                                                current);
    case algo_kind_t::sssp:
      return algorithm_traits<algo_kind_t::sssp>::should_update(candidate,
                                                                 current);
    case algo_kind_t::wcc:
      return algorithm_traits<algo_kind_t::wcc>::should_update(candidate,
                                                                current);
    case algo_kind_t::sswp:
      return algorithm_traits<algo_kind_t::sswp>::should_update(candidate,
                                                                 current);
  }
  return false;
}

}  // namespace algorithms
}  // namespace puercgp
