#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

struct source_record {
  int source = -1;
  unsigned long long reached = 0;
  unsigned long long distance_sum = 0;
};

std::vector<source_record> load_records(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open source metadata");
  std::vector<source_record> records;
  std::string line;
  std::getline(file, line);
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream row(line);
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(row, field, ',')) fields.push_back(field);
    if (fields.size() < 6) throw std::runtime_error("invalid source metadata");
    records.push_back({std::stoi(fields[1]), std::stoull(fields[3]),
                       std::stoull(fields[4])});
  }
  return records;
}

__global__ void result_fingerprint_kernel(
    const puercgp::algorithms::unified_value_t* values, std::size_t V,
    unsigned long long* counts, unsigned long long* sums) {
  int q = blockIdx.x;
  unsigned long long local_count = 0;
  unsigned long long local_sum = 0;
  for (std::size_t v = threadIdx.x; v < V; v += blockDim.x) {
    auto value = values[static_cast<std::size_t>(q) * V + v];
    if (isinf(value)) continue;
    ++local_count;
    local_sum += static_cast<unsigned long long>(llrintf(value));
  }
  if (local_count) atomicAdd(counts + q, local_count);
  if (local_sum) atomicAdd(sums + q, local_sum);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: validate_roadnet_replenish <csr-dir> <sources.csv>\n";
    return 2;
  }
  auto records = load_records(argv[2]);
  auto storage = puercgp_examples::load_graph_auto(argv[1], false);
  auto graph = storage.view();
  std::vector<puercgp::query_descriptor_t> queries;
  queries.reserve(records.size());
  for (const auto& record : records)
    queries.push_back({record.source, puercgp::algorithms::algo_kind_t::bfs,
                       0.0f});
  puercgp::execution_context context;
  puercgp::run_options options;
  options.enable_replenishment = true;
  options.traversal_mode = puercgp::traversal_mode_t::push;
  options.push_strategy = puercgp::push_strategy_t::shared_node_warp;
  options.max_queries = 32;
  options.replenish_batch_size = 1;
  options.max_iterations = 30000;
  options.discard_results = false;
  auto result = puercgp::run_replenish_pipeline(graph, queries, context,
                                                 options);
  int N = static_cast<int>(records.size());
  thrust::device_vector<unsigned long long> counts(N, 0), sums(N, 0);
  result_fingerprint_kernel<<<N, 256>>>(
      thrust::raw_pointer_cast(result.values.data()), storage.vertices,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(sums.data()));
  std::vector<unsigned long long> host_counts(N), host_sums(N);
  cudaMemcpy(host_counts.data(), thrust::raw_pointer_cast(counts.data()),
             N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_sums.data(), thrust::raw_pointer_cast(sums.data()),
             N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  int mismatches = 0;
  for (int q = 0; q < N; ++q) {
    if (host_counts[q] == records[q].reached &&
        host_sums[q] == records[q].distance_sum)
      continue;
    if (mismatches < 10)
      std::cerr << "q=" << q << " source=" << records[q].source
                << " expected=" << records[q].reached << '/'
                << records[q].distance_sum << " actual=" << host_counts[q]
                << '/' << host_sums[q] << '\n';
    ++mismatches;
  }
  std::cout << "queries=" << N << " iterations=" << result.iterations
            << " mismatches=" << mismatches << '\n';
  return mismatches == 0 ? 0 : 1;
}
