#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>
#include <puercgp/engine/push_executor.hxx>
#include <puercgp/kernels/pull/fused_pull_kernels.hxx>

namespace {

constexpr int kVertices = 8;
constexpr int kQueries = 64;
using mask_t = puercgp::query_mask_t;
using policy_t = puercgp::algorithms::bfs_policy;

struct output_t {
  thrust::host_vector<int> values;
  thrust::host_vector<mask_t> visited;
  thrust::host_vector<mask_t> frontier;
};

puercgp::csr_graph_storage<int, int, float> make_graph() {
  thrust::host_vector<int> rows{0, 2, 4, 6, 8, 8, 8, 8, 8};
  thrust::host_vector<int> cols{4, 5, 5, 6, 6, 7, 7, 4};
  return {kVertices, rows, cols, {}, true};
}

void make_initial(thrust::host_vector<int>& values,
                  thrust::host_vector<mask_t>& visited,
                  thrust::host_vector<mask_t>& frontier) {
  values.assign(kVertices * kQueries, policy_t::infinity());
  visited.assign(kVertices, 0);
  frontier.assign(kVertices, 0);
  for (int q = 0; q < kQueries; ++q) {
    int source = q % 4;
    values[source * kQueries + q] = 0;
    visited[source] |= mask_t{1} << q;
    frontier[source] |= mask_t{1} << q;
  }
}

template <typename graph_t>
output_t run_pull(graph_t graph, mask_t active_slots) {
  thrust::host_vector<int> values_host;
  thrust::host_vector<mask_t> visited_host;
  thrust::host_vector<mask_t> unused_frontier;
  make_initial(values_host, visited_host, unused_frontier);
  thrust::device_vector<int> values(values_host);
  thrust::device_vector<mask_t> visited(visited_host);
  thrust::device_vector<mask_t> next(kVertices, 0);
  thrust::device_vector<unsigned long long> flags(kVertices, 0);
  thrust::device_vector<unsigned long long> pairs(kVertices, 0);
  puercgp::detail::launch_fused_pull<policy_t, graph_t, int>(
      graph, kQueries, thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited.data()),
      thrust::raw_pointer_cast(next.data()),
      thrust::raw_pointer_cast(flags.data()),
      thrust::raw_pointer_cast(pairs.data()), active_slots, nullptr);
  cudaDeviceSynchronize();
  return {values, visited, next};
}

template <typename graph_t>
output_t run_push(graph_t graph, mask_t active_slots) {
  thrust::host_vector<int> values_host;
  thrust::host_vector<mask_t> visited_host;
  thrust::host_vector<mask_t> frontier_host;
  make_initial(values_host, visited_host, frontier_host);
  thrust::device_vector<int> values(values_host);
  thrust::device_vector<mask_t> visited(visited_host);
  thrust::device_vector<mask_t> frontier(frontier_host);
  thrust::device_vector<mask_t> next(kVertices, 0);
  thrust::device_vector<int> frontier_vertices(kVertices);
  thrust::device_vector<int> next_vertices(kVertices);
  thrust::device_vector<unsigned long long> unique_count(1, 0);
  thrust::device_vector<unsigned long long> pair_count(1, 0);
  thrust::host_vector<int> ids(kVertices);
  for (int v = 0; v < kVertices; ++v) ids[v] = v;
  frontier_vertices = ids;
  puercgp::detail::launch_shared_push_warp<policy_t, graph_t, int>(
      graph, thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(frontier.data()), kVertices,
      thrust::raw_pointer_cast(visited.data()),
      thrust::raw_pointer_cast(next.data()),
      thrust::raw_pointer_cast(next_vertices.data()),
      thrust::raw_pointer_cast(unique_count.data()),
      thrust::raw_pointer_cast(pair_count.data()),
      thrust::raw_pointer_cast(values.data()), kQueries, active_slots, 0, 256,
      nullptr);
  cudaDeviceSynchronize();
  return {values, visited, next};
}

void validate_split(const output_t& full, const output_t& first,
                    const output_t& second, mask_t first_mask,
                    const char* label) {
  for (int v = 0; v < kVertices; ++v) {
    if ((first.frontier[v] | second.frontier[v]) != full.frontier[v])
      throw std::runtime_error(std::string(label) + " frontier mismatch");
    if ((first.visited[v] | second.visited[v]) != full.visited[v])
      throw std::runtime_error(std::string(label) + " visited mismatch");
    for (int q = 0; q < kQueries; ++q) {
      const auto& selected = (first_mask & (mask_t{1} << q)) ? first : second;
      int index = v * kQueries + q;
      if (selected.values[index] != full.values[index])
        throw std::runtime_error(std::string(label) + " values mismatch");
    }
  }
}

template <typename runner_t>
void validate_runner(runner_t&& runner, const char* label) {
  mask_t all = ~mask_t{0};
  mask_t low = 0x00000000FFFFFFFFULL;
  mask_t even = 0x5555555555555555ULL;
  output_t full = runner(all);
  output_t low_result = runner(low);
  output_t high_result = runner(all ^ low);
  validate_split(full, low_result, high_result, low, label);
  output_t even_result = runner(even);
  output_t odd_result = runner(all ^ even);
  validate_split(full, even_result, odd_result, even, label);
}

}  // namespace

int main() {
  auto storage = make_graph();
  auto graph = storage.view();
  validate_runner([&](mask_t mask) { return run_push(graph, mask); }, "push");
  validate_runner([&](mask_t mask) { return run_pull(graph, mask); }, "pull");
  std::cout << "query partition kernels: PASS (Q=64, contiguous + interleaved)\n";
  return 0;
}
