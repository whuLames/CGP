#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

puercgp::traversal_mode_t parse_mode(const std::string &mode) {
  if (mode == "push")
    return puercgp::traversal_mode_t::push;
  if (mode == "pull")
    return puercgp::traversal_mode_t::pull;
  if (mode == "hybrid")
    return puercgp::traversal_mode_t::hybrid;
  throw std::invalid_argument("mode must be push, pull, or hybrid");
}

} // namespace

int main(int argc, char **argv) {
  std::string matrix =
      "/home/zyl/Projects/ocgp/puercgp/examples/matrices/rank_validation.mm";
  std::string source_text = "0,1,2,3,4,5";
  float damping_factor = 0.85f;
  float epsilon = 1.0e-8f;
  int repeats = 3;
  std::string mode = "push";
  if (argc > 1)
    matrix = argv[1];
  if (argc > 2)
    source_text = argv[2];
  if (argc > 3)
    damping_factor = std::stof(argv[3]);
  if (argc > 4)
    epsilon = std::stof(argv[4]);
  if (argc > 5)
    repeats = std::max(1, std::stoi(argv[5]));
  if (argc > 6)
    mode = argv[6];

  auto graph = puercgp_examples::load_graph_auto(matrix, mode != "push");
  auto graph_view = graph.view();
  auto sources = puercgp_examples::parse_sources(source_text);
  std::vector<puercgp::algorithms::ppr_query> descriptors;
  for (int source : sources) {
    descriptors.push_back({source, damping_factor, epsilon});
  }
  puercgp::algorithms::ppr_query_batch queries(std::move(descriptors));
  puercgp::execution_context context;
  puercgp::run_options options;
  options.traversal_mode = parse_mode(mode);
  options.profile_iterations = true;
  options.max_iterations = 10000;
  options.max_queries = 64;

  std::vector<std::vector<float>> references;
  for (int source : sources) {
    references.push_back(
        puercgp_examples::cpu_ppr(graph, source, damping_factor));
  }

  std::size_t mismatches = 0;
  float max_abs_delta = 0.0f;
  float max_l1_delta = 0.0f;
  std::vector<float> gpu_times;
  std::vector<std::string> iteration_modes;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto result = puercgp::run<puercgp::algorithms::ppr_policy>(
        graph_view, queries, context, options);
    gpu_times.push_back(result.gpu_time_ms);
    iteration_modes = result.iteration_modes;
    thrust::host_vector<float> ranks(result.values);
    for (std::size_t query_id = 0; query_id < sources.size(); ++query_id) {
      float l1_delta = 0.0f;
      for (int vertex = 0; vertex < graph.vertices; ++vertex) {
        std::size_t position =
            static_cast<std::size_t>(vertex) * sources.size() + query_id;
        float delta =
            std::fabs(ranks[position] -
                      references[query_id][static_cast<std::size_t>(vertex)]);
        max_abs_delta = std::max(max_abs_delta, delta);
        l1_delta += delta;
        if (delta > 1.0e-4f)
          ++mismatches;
      }
      max_l1_delta = std::max(max_l1_delta, l1_delta);
    }
  }

  std::cout << "algorithm=ppr\n";
  std::cout << "traversal_mode=" << mode << "\n";
  std::cout << "matrix=" << matrix << "\n";
  std::cout << "sources=" << source_text << "\n";
  std::cout << "damping_factor=" << damping_factor << "\n";
  std::cout << "epsilon=" << epsilon << "\n";
  std::cout << "mismatches=" << mismatches << " max_abs_delta=" << max_abs_delta
            << " max_l1_delta=" << max_l1_delta << "\n";
  std::cout << "iteration_modes=";
  for (std::size_t i = 0; i < iteration_modes.size(); ++i) {
    std::cout << (i == 0 ? "" : ",") << iteration_modes[i];
  }
  std::cout << "\n";
  std::cout << "gpu_ms_median=" << puercgp_examples::median(gpu_times) << "\n";
  return mismatches == 0 ? 0 : 1;
}
