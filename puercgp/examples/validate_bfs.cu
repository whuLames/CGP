#include <algorithm>
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
  if (value == "edge_balanced") {
    return puercgp::push_strategy_t::edge_balanced;
  }
  if (value == "shared_node") {
    return puercgp::push_strategy_t::shared_node;
  }
  if (value == "shared_node_query_parallel") {
    return puercgp::push_strategy_t::shared_node_query_parallel;
  }
  if (value == "shared_node_warp") {
    return puercgp::push_strategy_t::shared_node_warp;
  }
  if (value == "shared_node_degree") {
    return puercgp::push_strategy_t::shared_node_degree;
  }
  throw std::invalid_argument(
      "push_strategy must be edge_balanced, shared_node, "
      "shared_node_query_parallel, shared_node_warp, or shared_node_degree");
}

puercgp::pull_strategy_t parse_pull_strategy(const std::string& value) {
  if (value == "bitmap") {
    return puercgp::pull_strategy_t::bitmap;
  }
  if (value == "ge_spmm") {
    return puercgp::pull_strategy_t::ge_spmm;
  }
  throw std::invalid_argument("pull_strategy must be bitmap or ge_spmm");
}

int main(int argc, char** argv) {
  std::string matrix =
      "/home/zyl/Projects/ocgp/puercgp/examples/matrices/unweighted.mm";
  std::string source_text = "0,1,2,3";
  int repeats = 7;
  std::string traversal_mode = "push";
  std::string push_strategy = "edge_balanced";
  std::string pull_strategy = "bitmap";
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

  auto graph = puercgp_examples::load_graph_auto(matrix);
  auto graph_view = graph.view();
  auto sources = puercgp_examples::parse_sources(source_text);
  
  if (sources.empty()) {
    std::cerr << "validate_bfs requires at least one source\n";
    return 2;
  }

  puercgp::execution_context context;
  puercgp::query_batch<int> queries(sources);
  puercgp::run_options options;
  options.traversal_mode = parse_traversal_mode(traversal_mode);
  options.push_strategy = parse_push_strategy(push_strategy);
  options.pull_strategy = parse_pull_strategy(pull_strategy);
  options.profile_iterations = true;
  options.max_queries = 128;

  if (graph.vertices <= 1000000) {
    auto warmup = puercgp::run<puercgp::algorithms::bfs_policy>(
        graph_view, queries, context, options);
    (void)warmup;
  }

  std::vector<float> wall_times;
  std::vector<float> gpu_times;
  wall_times.reserve(static_cast<std::size_t>(repeats));
  gpu_times.reserve(static_cast<std::size_t>(repeats));

  bool run_cpu_check = graph.vertices <= 1000000;
  std::vector<std::vector<int>> references;
  if (run_cpu_check) {
    references.reserve(sources.size());
    for (int source : sources) {
      references.push_back(puercgp_examples::cpu_bfs(graph, source));
    }
  }
  std::size_t mismatches = 0;

  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto result = puercgp::run<puercgp::algorithms::bfs_policy>(
        graph_view, queries, context, options);
    wall_times.push_back(result.wall_time_ms);
    gpu_times.push_back(result.gpu_time_ms);

    if (run_cpu_check) {
      thrust::host_vector<int> distances(result.values);
      for (std::size_t query_id = 0; query_id < sources.size(); ++query_id) {
        for (int vertex = 0; vertex < graph.vertices; ++vertex) {
          auto index = static_cast<std::size_t>(vertex) * sources.size() +
                       query_id;
          if (distances[index] !=
              references[query_id][static_cast<std::size_t>(vertex)]) {
            ++mismatches;
          }
        }
      }
    }
    if (repeat + 1 == repeats) {
      std::cout << "iterations=" << result.iterations << "\n";
      std::cout << "frontier_sizes=";
      for (std::size_t i = 0; i < result.frontier_sizes.size(); ++i) {
        if (i != 0) {
          std::cout << ",";
        }
        std::cout << result.frontier_sizes[i];
      }
      std::cout << "\n";
      std::cout << "iteration_modes=";
      for (std::size_t i = 0; i < result.iteration_modes.size(); ++i) {
        if (i != 0) {
          std::cout << ",";
        }
        std::cout << result.iteration_modes[i];
      }
      std::cout << "\n";
      std::cout << "effective_query_dim=" << result.effective_query_dim
                << "\n";
      std::cout << "iteration_profile="
                << "iter,mode,frontier,unique,pull_ms,ge_pull_ms,"
                   "dense_build_ms,postprocess_ms,degree_scan_ms,"
                   "shared_push_ms,iteration_wall_ms\n";
      for (const auto& profile : result.iteration_profiles) {
        std::cout << profile.iteration << "," << profile.mode << ","
                  << profile.frontier_size << ","
                  << profile.unique_frontier_size << ","
                  << profile.pull_kernel_ms << ","
                  << profile.ge_spmm_pull_kernel_ms << ","
                  << profile.dense_build_ms << ","
                  << profile.ge_spmm_postprocess_ms << ","
                  << profile.degree_scan_ms << ","
                  << profile.shared_push_kernel_ms << ","
                  << profile.iteration_wall_ms << "\n";
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
  std::cout << "cpu_reference_check=" << (run_cpu_check ? "yes" : "no")
            << "\n";
  std::cout << "distance_mismatches=" << mismatches << "\n";
  std::cout << "wall_ms_median="
            << puercgp_examples::median(wall_times) << "\n";
  std::cout << "gpu_ms_median=" << puercgp_examples::median(gpu_times)
            << "\n";

  return mismatches == 0 ? 0 : 1;
}
