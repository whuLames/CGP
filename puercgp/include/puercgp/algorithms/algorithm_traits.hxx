#pragma once

#include <puercgp/core/algorithm_id.hxx>

namespace puercgp {
namespace algorithms {

enum class init_mode_t {
  single_source,
  all_vertices_label
};

template <algo_kind_t Kind>
struct algorithm_traits;

template <>
struct algorithm_traits<algo_kind_t::bfs> {
  using value_type = unified_value_t;
  static constexpr init_mode_t init_mode = init_mode_t::single_source;
  static constexpr bool mark_source_visited = true;

  __host__ __device__ __forceinline__ static constexpr value_type infinity() {
    return unified_infinity();
  }

  __host__ __device__ __forceinline__ static constexpr value_type
  source_value() {
    return unified_source_value();
  }

  __host__ __device__ __forceinline__ static constexpr value_type
  initial_value(int vertex, int source, int /*query_id*/) {
    return vertex == source ? source_value() : infinity();
  }

  __host__ __device__ __forceinline__ static value_type candidate_push(
      value_type source_value,
      float /*edge_weight*/,
      int /*level*/) {
    return source_value + value_type{1};
  }

  __host__ __device__ __forceinline__ static value_type candidate_pull(
      value_type neighbor_value,
      float /*edge_weight*/) {
    return neighbor_value + value_type{1};
  }

  __host__ __device__ __forceinline__ static constexpr bool should_update(
      value_type candidate,
      value_type current) {
    return candidate < current;
  }
};

template <>
struct algorithm_traits<algo_kind_t::sssp> {
  using value_type = unified_value_t;
  static constexpr init_mode_t init_mode = init_mode_t::single_source;
  static constexpr bool mark_source_visited = false;

  __host__ __device__ __forceinline__ static constexpr value_type infinity() {
    return unified_infinity();
  }

  __host__ __device__ __forceinline__ static constexpr value_type
  source_value() {
    return unified_source_value();
  }

  __host__ __device__ __forceinline__ static constexpr value_type
  initial_value(int vertex, int source, int /*query_id*/) {
    return vertex == source ? source_value() : infinity();
  }

  __host__ __device__ __forceinline__ static value_type candidate_push(
      value_type source_value,
      float edge_weight,
      int /*level*/) {
    return source_value + edge_weight;
  }

  __host__ __device__ __forceinline__ static value_type candidate_pull(
      value_type neighbor_value,
      float edge_weight) {
    return neighbor_value + edge_weight;
  }

  __host__ __device__ __forceinline__ static constexpr bool should_update(
      value_type candidate,
      value_type current) {
    return candidate < current;
  }
};

template <>
struct algorithm_traits<algo_kind_t::wcc> {
  using value_type = unified_value_t;
  static constexpr init_mode_t init_mode = init_mode_t::all_vertices_label;
  static constexpr bool mark_source_visited = false;

  __host__ __device__ __forceinline__ static constexpr value_type infinity() {
    return unified_infinity();
  }

  __host__ __device__ __forceinline__ static constexpr value_type
  source_value() {
    return unified_source_value();
  }

  __host__ __device__ __forceinline__ static constexpr value_type
  initial_value(int vertex, int /*source*/, int /*query_id*/) {
    return static_cast<value_type>(vertex);
  }

  __host__ __device__ __forceinline__ static value_type candidate_push(
      value_type source_value,
      float /*edge_weight*/,
      int /*level*/) {
    return source_value;
  }

  __host__ __device__ __forceinline__ static value_type candidate_pull(
      value_type neighbor_value,
      float /*edge_weight*/) {
    return neighbor_value;
  }

  __host__ __device__ __forceinline__ static constexpr bool should_update(
      value_type candidate,
      value_type current) {
    return candidate < current;
  }
};

}  // namespace algorithms
}  // namespace puercgp
