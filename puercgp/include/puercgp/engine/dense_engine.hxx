#pragma once

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include <puercgp/backend/graph_adapter.hxx>
#include <puercgp/core/query_batch.hxx>
#include <puercgp/core/result.hxx>
#include <puercgp/state/value_matrix.hxx>

namespace puercgp {

template <typename Policy>
class dense_engine {
 public:
  using vertex_type = typename Policy::vertex_type;
  using value_type = typename Policy::value_type;
  using result_type = run_result_t<vertex_type, value_type>;

  template <typename graph_t>
  result_type run(graph_t& graph,
                  const query_batch<vertex_type>& queries,
                  execution_context& context,
                  const run_options& options) const {
    (void)context;
    static_assert(std::is_same<vertex_type, int>::value,
                  "puercgp dense policies currently require int vertices");

    queries.validate(options.max_queries);
    if (options.max_iterations < 0) {
      throw std::invalid_argument("max_iterations must be non-negative");
    }

    const auto vertex_count =
        static_cast<std::size_t>(graph.get_number_of_vertices());

    value_matrix<value_type> values;
    values.resize(queries.size(), vertex_count);

    result_type result;
    result.options = options;
    result.effective_query_dim = static_cast<int>(queries.size());
    result.values.swap(values.values());
    result.iterations = options.max_iterations;

    result.queries.reserve(queries.size());
    for (std::size_t query_id = 0; query_id < queries.size(); ++query_id) {
      query_result_t<vertex_type> query_result;
      query_result.query_id = static_cast<vertex_type>(query_id);
      query_result.source = queries[query_id];
      query_result.completion_level =
          static_cast<vertex_type>(options.max_iterations);
      result.queries.push_back(query_result);
    }

    if (options.profile_iterations) {
      result.iteration_profiles.reserve(
          static_cast<std::size_t>(options.max_iterations));
      for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        iteration_profile_t profile;
        profile.iteration = iteration;
        profile.mode = "dense";
        result.iteration_profiles.push_back(profile);
        result.iteration_modes.push_back(profile.mode);
        result.iteration_wall_times_ms.push_back(0.0f);
      }
    }

    return result;
  }
};

}  // namespace puercgp
