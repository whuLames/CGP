#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

std::vector<int> make_sources(int vertex_count, unsigned int seed) {
  std::mt19937 rng(seed);
  std::unordered_set<int> used;
  std::vector<int> sources;
  while (sources.size() < 64) {
    int source = static_cast<int>(rng() % vertex_count);
    if (used.insert(source).second) sources.push_back(source);
  }
  return sources;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: trace_hybrid_q64 <graph> <output-dir> [seed]\n";
    return 2;
  }
  unsigned int seed = argc > 3 ? std::stoul(argv[3]) : 42;
  std::filesystem::path output = argv[2];
  std::filesystem::create_directories(output);
  auto storage = puercgp_examples::load_graph_auto(argv[1], true);
  auto sources = make_sources(storage.vertices, seed);
  puercgp::query_batch<int> queries(sources);
  puercgp::execution_context context;
  puercgp::run_options options;
  options.traversal_mode = puercgp::traversal_mode_t::hybrid;
  options.push_strategy = puercgp::push_strategy_t::shared_node_warp;
  options.profile_iterations = true;
  auto graph = storage.view();
  auto result = puercgp::run<puercgp::algorithms::bfs_policy>(
      graph, queries, context, options);

  std::ofstream source_file(output / "sources.csv");
  source_file << "query_id,source\n";
  for (int q = 0; q < 64; ++q) source_file << q << ',' << sources[q] << '\n';
  std::ofstream trace(output / "hybrid_trace.csv");
  trace << "iteration,mode,unique_frontier,actual_edges,virtual_edges,sharing,"
           "push_ms,pull_ms,compact_ms,wall_ms\n";
  for (const auto& p : result.iteration_profiles) {
    double sharing = p.actual_edge_count == 0 ? 1.0 :
        static_cast<double>(p.virtual_edge_count) / p.actual_edge_count;
    trace << p.iteration << ',' << p.mode << ',' << p.unique_frontier_size
          << ',' << p.actual_edge_count << ',' << p.virtual_edge_count << ','
          << sharing << ',' << p.shared_push_kernel_ms << ','
          << p.pull_kernel_ms << ',' << p.compact_ms << ','
          << p.iteration_wall_ms << '\n';
  }
  std::cout << "iterations=" << result.iterations
            << " wall_ms=" << result.wall_time_ms << '\n';
  return 0;
}
