#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

template <typename value_t>
bool equal_value(value_t first, value_t second) {
  if constexpr (std::is_floating_point<value_t>::value) {
    if (std::isinf(first) || std::isinf(second)) {
      return first == second;
    }
    return std::fabs(first - second) <= 1.0e-5f;
  } else {
    return first == second;
  }
}

template <typename Policy, typename graph_t>
bool validate_policy(graph_t& graph, int vertex_count,
                     puercgp::traversal_mode_t mode, int query_count) {
  std::vector<int> sources;
  std::vector<int> offsets;
  sources.reserve(query_count);
  offsets.reserve(query_count);
  for (int query = 0; query < query_count; ++query) {
    sources.push_back(query % vertex_count);
    offsets.push_back(query % 4);
  }

  puercgp::query_batch<int> queries(sources);
  puercgp::scheduling::slot_start_schedule schedule(offsets);
  puercgp::execution_context context;
  puercgp::run_options options;
  options.traversal_mode = mode;
  options.max_queries = 64;

  auto baseline = puercgp::run<Policy>(graph, queries, context, options);
  auto scheduled =
      puercgp::run_scheduled<Policy>(graph, queries, context, schedule, options);
  thrust::host_vector<typename Policy::value_type> baseline_values(
      baseline.values);
  thrust::host_vector<typename Policy::value_type> scheduled_values(
      scheduled.values);

  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < baseline_values.size(); ++index) {
    if (!equal_value(baseline_values[index], scheduled_values[index])) {
      ++mismatches;
      if (mismatches <= 4) {
        std::cerr << "mismatch index=" << index
                  << " baseline=" << baseline_values[index]
                  << " scheduled=" << scheduled_values[index] << '\n';
      }
    }
  }
  std::cout << "algorithm=" << static_cast<int>(Policy::algorithm_kind)
            << " mode=" << puercgp::traversal_mode_name(mode)
            << " Q=" << query_count << " mismatches=" << mismatches
            << " baseline_iterations=" << baseline.iterations
            << " scheduled_iterations=" << scheduled.iterations << '\n';
  return mismatches == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string graph_path =
      argc > 1 ? argv[1] : "examples/matrices/weighted.mm";
  auto graph_storage = puercgp_examples::load_graph_auto(graph_path, true);
  auto graph = graph_storage.view();
  bool correct = true;
  for (auto mode : {puercgp::traversal_mode_t::push,
                    puercgp::traversal_mode_t::pull,
                    puercgp::traversal_mode_t::hybrid}) {
    for (int query_count : {1, 32, 64}) {
      correct &= validate_policy<puercgp::algorithms::bfs_policy>(
          graph, graph_storage.vertices, mode, query_count);
      correct &= validate_policy<puercgp::algorithms::sssp_policy>(
          graph, graph_storage.vertices, mode, query_count);
      correct &= validate_policy<puercgp::algorithms::sswp_policy>(
          graph, graph_storage.vertices, mode, query_count);
    }
  }
  std::cout << "online_runner_correct=" << (correct ? "yes" : "no") << '\n';
  return correct ? 0 : 1;
}
