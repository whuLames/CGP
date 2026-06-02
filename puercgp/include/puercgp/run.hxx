#pragma once

#include <type_traits>

#include <puercgp/core/types.hxx>
#include <puercgp/engine/dense_engine.hxx>
#include <puercgp/engine/frontier_engine.hxx>

namespace puercgp {

template <typename Policy, typename graph_t>
auto run(graph_t& graph,
         const query_batch<typename Policy::vertex_type>& queries,
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
}  // namespace puercgp
