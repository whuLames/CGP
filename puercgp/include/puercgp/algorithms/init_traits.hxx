#pragma once

#include <puercgp/algorithms/algorithm_traits.hxx>
#include <puercgp/algorithms/bfs.hxx>
#include <puercgp/algorithms/sssp.hxx>
#include <puercgp/algorithms/sswp.hxx>
#include <puercgp/algorithms/wcc.hxx>

namespace puercgp {
namespace algorithms {

template <typename Policy>
struct policy_algorithm_traits;

template <>
struct policy_algorithm_traits<bfs_policy> {
  static constexpr algo_kind_t kind = algo_kind_t::bfs;
  static constexpr init_mode_t init_mode =
      algorithm_traits<kind>::init_mode;
  static constexpr bool mark_source_visited =
      algorithm_traits<kind>::mark_source_visited;
};

template <>
struct policy_algorithm_traits<sssp_policy> {
  static constexpr algo_kind_t kind = algo_kind_t::sssp;
  static constexpr init_mode_t init_mode =
      algorithm_traits<kind>::init_mode;
  static constexpr bool mark_source_visited =
      algorithm_traits<kind>::mark_source_visited;
};

template <>
struct policy_algorithm_traits<wcc_policy> {
  static constexpr algo_kind_t kind = algo_kind_t::wcc;
  static constexpr init_mode_t init_mode =
      algorithm_traits<kind>::init_mode;
  static constexpr bool mark_source_visited =
      algorithm_traits<kind>::mark_source_visited;
};

template <>
struct policy_algorithm_traits<sswp_policy> {
  static constexpr algo_kind_t kind = algo_kind_t::sswp;
  static constexpr init_mode_t init_mode =
      algorithm_traits<kind>::init_mode;
  static constexpr bool mark_source_visited =
      algorithm_traits<kind>::mark_source_visited;
};

}  // namespace algorithms
}  // namespace puercgp
