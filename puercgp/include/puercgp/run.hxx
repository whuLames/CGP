#pragma once

#include <type_traits>

#include <puercgp/core/types.hxx>
#include <puercgp/engine/dense_engine.hxx>
#include <puercgp/engine/frontier_engine.hxx>
#include <puercgp/scheduling/slot_start_schedule.hxx>

namespace puercgp {

template <typename Policy, typename graph_t, typename query_batch_t>
auto run(graph_t& graph,
         const query_batch_t& queries,
         execution_context& context,
         const run_options& options = {}) {
  if constexpr (Policy::execution_model == execution_model_t::frontier) {
    return frontier_engine<Policy>{}.run(graph, queries, context, options);
  } else {
    static_assert(Policy::execution_model == execution_model_t::dense,
                  "unknown puercgp execution model");
    return dense_engine<Policy>{}.run(graph, queries, context, options);
  }
}

template <typename Policy, typename graph_t, typename query_batch_t>
auto run_scheduled(
    graph_t& graph, const query_batch_t& queries, execution_context& context,
    const scheduling::slot_start_schedule& start_schedule,
    const run_options& options = {}) {
  static_assert(Policy::execution_model == execution_model_t::frontier,
                "slot start schedules require a frontier algorithm");
  return frontier_engine<Policy>{}.run(graph, queries, context, options,
                                       start_schedule);
}
}  // namespace puercgp
