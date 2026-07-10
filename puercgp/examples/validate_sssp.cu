#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

puercgp::traversal_mode_t parse_traversal_mode(const std::string& value) {
  if (value == "push") {
    return puercgp::traversal_mode_t::push;
  }
  if (value == "pull") {
    return puercgp::traversal_mode_t::pull;
  }
  if (value == "hybrid") {
    return puercgp::traversal_mode_t::hybrid;
  }
  throw std::invalid_argument("traversal_mode must be push, pull, or hybrid");
}

puercgp::push_strategy_t parse_push_strategy(const std::string& value) {
  if (value == "shared_node") {
    return puercgp::push_strategy_t::shared_node;
  }
  if (value == "shared_node_query_parallel") {
    return puercgp::push_strategy_t::shared_node_query_parallel;
  }
  if (value == "shared_node_warp") {
    return puercgp::push_strategy_t::shared_node_warp;
  }
  throw std::invalid_argument(
      "push_strategy must be shared_node, shared_node_query_parallel, or "
      "shared_node_warp");
}

puercgp::pull_strategy_t parse_pull_strategy(const std::string& value) {
  if (value == "fused") {
    return puercgp::pull_strategy_t::fused;
  }
  throw std::invalid_argument("pull_strategy must be fused");
}

int main(int argc, char** argv) {
  std::string matrix =
      "/home/zyl/Projects/ocgp/puercgp/examples/matrices/weighted.mm";
  std::string source_text = "0,1,2,3";
  int repeats = 7;
  std::string traversal_mode = "push";
  std::string push_strategy = "shared_node_warp";
  std::string pull_strategy = "fused";
  if (argc > 1) {
    matrix = argv[1];
  }
  if (argc > 2) {
    source_text = argv[2];
  }
  if (argc > 3) {
    repeats = std::max(1, std::stoi(argv[3]));
  }
  if (argc > 4) {
    traversal_mode = argv[4];
  }
  if (argc > 5) {
    push_strategy = argv[5];
  }
  if (argc > 6) {
    pull_strategy = argv[6];
  }

  const bool build_pull_adjacency = traversal_mode != "push";
  auto graph =
      puercgp_examples::load_matrix_market(matrix, build_pull_adjacency);
  auto graph_view = graph.view();
  auto sources = puercgp_examples::parse_sources(source_text);
  if (sources.empty()) {
    std::cerr << "validate_sssp requires at least one source\n";
    return 2;
  }

  puercgp::execution_context context;
  puercgp::query_batch<int> queries(sources);
  puercgp::run_options options;
  options.traversal_mode = parse_traversal_mode(traversal_mode);
  options.push_strategy = parse_push_strategy(push_strategy);
  options.pull_strategy = parse_pull_strategy(pull_strategy);
  options.profile_iterations = true;

  auto warmup = puercgp::run<puercgp::algorithms::sssp_policy>(
      graph_view, queries, context, options);
  (void)warmup;

  std::vector<float> wall_times;
  std::vector<float> gpu_times;
  wall_times.reserve(static_cast<std::size_t>(repeats));
  gpu_times.reserve(static_cast<std::size_t>(repeats));

  std::vector<std::vector<float>> references;
  references.reserve(sources.size());
  for (int source : sources) {
    references.push_back(puercgp_examples::cpu_sssp(graph, source));
  }
  std::size_t mismatches = 0;
  float max_abs_delta = 0.0f;

  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto result = puercgp::run<puercgp::algorithms::sssp_policy>(
        graph_view, queries, context, options);
    wall_times.push_back(result.wall_time_ms);
    gpu_times.push_back(result.gpu_time_ms);

    thrust::host_vector<float> distances(result.values);
    for (std::size_t query_id = 0; query_id < sources.size(); ++query_id) {
      for (int vertex = 0; vertex < graph.vertices; ++vertex) {
        auto index = static_cast<std::size_t>(vertex) * sources.size() +
                     query_id;
        float actual = distances[index];
        float expected = references[query_id][static_cast<std::size_t>(vertex)];
        bool both_infinite = std::isinf(actual) && std::isinf(expected);
        float delta = both_infinite ? 0.0f : std::fabs(actual - expected);
        max_abs_delta = std::max(max_abs_delta, delta);
        if (!both_infinite && delta > 1.0e-5f) {
          ++mismatches;
        }
      }
    }
  }

  std::cout << "matrix=" << matrix << "\n";
  std::cout << "sources=" << source_text << "\n";
  std::cout << "traversal_mode=" << traversal_mode << "\n";
  std::cout << "push_strategy=" << push_strategy << "\n";
  std::cout << "pull_strategy=" << pull_strategy << "\n";
  std::cout << "repeats=" << repeats << "\n";
  std::cout << "vertices=" << graph.vertices << " edges=" << graph.edges
            << "\n";
  std::cout << "distance_mismatches=" << mismatches
            << " max_abs_delta=" << max_abs_delta << "\n";
  std::cout << "wall_ms_median="
            << puercgp_examples::median(wall_times) << "\n";
  std::cout << "gpu_ms_median=" << puercgp_examples::median(gpu_times)
            << "\n";

  return mismatches == 0 ? 0 : 1;
}
