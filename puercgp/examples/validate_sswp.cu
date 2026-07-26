#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

puercgp::traversal_mode_t parse_traversal_mode(const std::string &value) {
  if (value == "push")
    return puercgp::traversal_mode_t::push;
  if (value == "pull")
    return puercgp::traversal_mode_t::pull;
  if (value == "hybrid")
    return puercgp::traversal_mode_t::hybrid;
  throw std::invalid_argument("traversal_mode must be push, pull, or hybrid");
}

puercgp::push_strategy_t parse_push_strategy(const std::string &value) {
  if (value == "shared_node")
    return puercgp::push_strategy_t::shared_node;
  if (value == "shared_node_query_parallel")
    return puercgp::push_strategy_t::shared_node_query_parallel;
  if (value == "shared_node_warp")
    return puercgp::push_strategy_t::shared_node_warp;
  throw std::invalid_argument("unknown push strategy");
}

} // namespace

int main(int argc, char **argv) {
  std::string matrix =
      "/home/zyl/Projects/ocgp/puercgp/examples/matrices/weighted.mm";
  std::string source_text = "0,1,2,3";
  int repeats = 3;
  std::string traversal_mode = "push";
  std::string push_strategy = "shared_node_warp";
  bool print_validation_summary = false;
  if (argc > 1)
    matrix = argv[1];
  if (argc > 2)
    source_text = argv[2];
  if (argc > 3)
    repeats = std::max(1, std::stoi(argv[3]));
  if (argc > 4)
    traversal_mode = argv[4];
  if (argc > 5)
    push_strategy = argv[5];
  if (argc > 6)
    print_validation_summary = std::string(argv[6]) == "summary";

  bool build_pull_adjacency = traversal_mode != "push";
  auto graph = puercgp_examples::load_graph_auto(matrix, build_pull_adjacency);
  auto graph_view = graph.view();
  auto sources = puercgp_examples::parse_sources(source_text);
  puercgp::query_batch<int> queries(sources);
  puercgp::execution_context context;
  puercgp::run_options options;
  options.traversal_mode = parse_traversal_mode(traversal_mode);
  options.push_strategy = parse_push_strategy(push_strategy);
  options.profile_iterations = true;
  options.max_queries = 64;

  std::vector<std::vector<float>> references;
  for (int source : sources) {
    references.push_back(puercgp_examples::cpu_sswp(graph, source));
  }

  std::size_t mismatches = 0;
  float max_abs_delta = 0.0f;
  std::vector<float> gpu_times;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto result = puercgp::run<puercgp::algorithms::sswp_policy>(
        graph_view, queries, context, options);
    gpu_times.push_back(result.gpu_time_ms);
    thrust::host_vector<float> widths(result.values);
    for (std::size_t query_id = 0; query_id < sources.size(); ++query_id) {
      for (int vertex = 0; vertex < graph.vertices; ++vertex) {
        std::size_t position =
            static_cast<std::size_t>(vertex) * sources.size() + query_id;
        float actual = widths[position];
        float expected = references[query_id][static_cast<std::size_t>(vertex)];
        bool matching_infinity = std::isinf(actual) && std::isinf(expected) &&
                                 std::signbit(actual) == std::signbit(expected);
        float delta = matching_infinity ? 0.0f : std::fabs(actual - expected);
        max_abs_delta = std::max(max_abs_delta, delta);
        if (!matching_infinity && delta > 1.0e-5f) {
          ++mismatches;
        }
      }
    }
    if (repeat + 1 == repeats && print_validation_summary) {
      for (std::size_t query_id = 0; query_id < sources.size(); ++query_id) {
        unsigned long long reached = 0;
        unsigned long long width_sum = 0;
        unsigned long long weighted_sum = 0;
        for (int vertex_id = 0; vertex_id < graph.vertices; ++vertex_id) {
          const float value = widths[
              static_cast<std::size_t>(vertex_id) * sources.size() + query_id];
          const bool is_source = vertex_id == sources[query_id];
          if (is_source || (std::isfinite(value) && value > 0.0f)) {
            const auto width = is_source
                                   ? 0ULL
                                   : static_cast<unsigned long long>(
                                         std::llround(value));
            ++reached;
            width_sum += width;
            weighted_sum +=
                (static_cast<unsigned long long>(vertex_id) + 1) *
                (width + 1);
          }
        }
        std::cout << "validation_summary query=" << query_id
                  << " reached=" << reached << " value_sum=" << width_sum
                  << " weighted_sum=" << weighted_sum << "\n";
      }
    }
  }

  std::cout << "algorithm=sswp\n";
  std::cout << "matrix=" << matrix << "\n";
  std::cout << "sources=" << source_text << "\n";
  std::cout << "traversal_mode=" << traversal_mode << "\n";
  std::cout << "push_strategy=" << push_strategy << "\n";
  std::cout << "mismatches=" << mismatches << " max_abs_delta=" << max_abs_delta
            << "\n";
  std::cout << "gpu_ms_median=" << puercgp_examples::median(gpu_times) << "\n";
  return mismatches == 0 ? 0 : 1;
}
