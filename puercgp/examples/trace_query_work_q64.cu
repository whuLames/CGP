#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>
#include <puercgp/engine/frontier_engine.hxx>
#include <puercgp/engine/push_executor.hxx>

#include "mtx_loader.hxx"

namespace {

constexpr int kQueries = 64;
constexpr int kThreads = 256;
using mask_t = puercgp::query_mask_t;
using policy_t = puercgp::algorithms::bfs_policy;

std::vector<int> make_sources(int vertex_count, unsigned int seed) {
  std::mt19937 rng(seed);
  std::unordered_set<int> used;
  std::vector<int> sources;
  while (sources.size() < kQueries) {
    int source = static_cast<int>(rng() % vertex_count);
    if (used.insert(source).second) sources.push_back(source);
  }
  return sources;
}

template <typename graph_t>
__global__ void query_work_kernel(
    graph_t graph, const int* frontier_vertices, const mask_t* frontier_mask,
    std::size_t unique_count, unsigned long long* edges,
    unsigned long long* vertices) {
  int q = threadIdx.x;
  mask_t bit = mask_t{1} << q;
  unsigned long long local_edges = 0;
  unsigned long long local_vertices = 0;
  for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
    int v = frontier_vertices[i];
    if ((frontier_mask[v] & bit) == 0) continue;
    local_edges += graph.get_starting_edge(v + 1) - graph.get_starting_edge(v);
    ++local_vertices;
  }
  if (local_edges) atomicAdd(edges + q, local_edges);
  if (local_vertices) atomicAdd(vertices + q, local_vertices);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: trace_query_work_q64 <graph> <output-dir> [seed]\n";
    return 2;
  }
  unsigned int seed = argc > 3 ? std::stoul(argv[3]) : 42;
  std::filesystem::path output = argv[2];
  std::filesystem::create_directories(output);
  auto storage = puercgp_examples::load_graph_auto(argv[1], false);
  auto graph = storage.view();
  auto sources = make_sources(storage.vertices, seed);
  const std::size_t V = storage.vertices;
  const std::size_t value_count = V * kQueries;
  const std::size_t mask_bytes = V * sizeof(mask_t);
  thrust::device_vector<int> device_sources(sources);
  thrust::device_vector<int> values(value_count);
  thrust::device_vector<mask_t> visited(V, 0), frontier_mask(V, 0),
      next_frontier_mask(V, 0);
  thrust::device_vector<int> frontier_vertices(V), next_frontier_vertices(V);
  thrust::device_vector<unsigned long long> unique_count(1, 0),
      next_unique_count(1, 0), next_pair_count(1, 0), query_edges(kQueries, 0),
      query_vertices(kQueries, 0);

  puercgp::detail::fill_values_kernel<policy_t>
      <<<puercgp::detail::grid_for(value_count, kThreads), kThreads>>>(
          thrust::raw_pointer_cast(values.data()), value_count);
  puercgp::detail::init_shared_sources_kernel<policy_t><<<1, 64>>>(
      graph, thrust::raw_pointer_cast(device_sources.data()), kQueries,
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited.data()),
      thrust::raw_pointer_cast(frontier_mask.data()),
      thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(unique_count.data()));

  unsigned long long unique = 0;
  cudaMemcpy(&unique, thrust::raw_pointer_cast(unique_count.data()),
             sizeof(unique), cudaMemcpyDeviceToHost);
  std::ofstream csv(output / "query_work.csv");
  csv << "iteration,query_id,source,frontier_vertices,frontier_edges,"
         "preferred_mode\n";
  const double threshold = 0.20 * static_cast<double>(storage.edges);
  int iteration = 0;
  while (unique > 0 && iteration < 4096) {
    cudaMemset(thrust::raw_pointer_cast(query_edges.data()), 0,
               kQueries * sizeof(unsigned long long));
    cudaMemset(thrust::raw_pointer_cast(query_vertices.data()), 0,
               kQueries * sizeof(unsigned long long));
    int blocks = std::max(1, std::min(1024, static_cast<int>(unique)));
    query_work_kernel<<<blocks, kQueries>>>(
        graph, thrust::raw_pointer_cast(frontier_vertices.data()),
        thrust::raw_pointer_cast(frontier_mask.data()), unique,
        thrust::raw_pointer_cast(query_edges.data()),
        thrust::raw_pointer_cast(query_vertices.data()));
    std::array<unsigned long long, kQueries> host_edges{}, host_vertices{};
    cudaMemcpy(host_edges.data(), thrust::raw_pointer_cast(query_edges.data()),
               sizeof(host_edges), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vertices.data(),
               thrust::raw_pointer_cast(query_vertices.data()),
               sizeof(host_vertices), cudaMemcpyDeviceToHost);
    for (int q = 0; q < kQueries; ++q) {
      csv << iteration << ',' << q << ',' << sources[q] << ','
          << host_vertices[q] << ',' << host_edges[q] << ','
          << (host_edges[q] >= threshold ? "pull" : "push") << '\n';
    }

    cudaMemset(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
               mask_bytes);
    cudaMemset(thrust::raw_pointer_cast(next_unique_count.data()), 0,
               sizeof(unsigned long long));
    cudaMemset(thrust::raw_pointer_cast(next_pair_count.data()), 0,
               sizeof(unsigned long long));
    puercgp::detail::launch_shared_push_warp<policy_t, decltype(graph), int>(
        graph, thrust::raw_pointer_cast(frontier_vertices.data()),
        thrust::raw_pointer_cast(frontier_mask.data()), unique,
        thrust::raw_pointer_cast(visited.data()),
        thrust::raw_pointer_cast(next_frontier_mask.data()),
        thrust::raw_pointer_cast(next_frontier_vertices.data()),
        thrust::raw_pointer_cast(next_unique_count.data()),
        thrust::raw_pointer_cast(next_pair_count.data()),
        thrust::raw_pointer_cast(values.data()), kQueries, ~mask_t{0},
        iteration, kThreads, nullptr);
    cudaMemcpy(&unique, thrust::raw_pointer_cast(next_unique_count.data()),
               sizeof(unique), cudaMemcpyDeviceToHost);
    thrust::swap(frontier_mask, next_frontier_mask);
    thrust::swap(frontier_vertices, next_frontier_vertices);
    ++iteration;
  }
  std::cout << "iterations=" << iteration << " threshold=" << threshold
            << '\n';
  return 0;
}
