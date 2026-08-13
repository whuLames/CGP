#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

constexpr std::int32_t kReorderMagic = 0x52454f52;

// BFS preprocessing stores each shortcut's original hop count as its weight.
// This policy preserves that distance while using the production frontier engine.
struct weighted_bfs_policy {
  using vertex_type = int;
  using value_type = puercgp::algorithms::unified_value_t;

  static constexpr puercgp::execution_model_t execution_model =
      puercgp::execution_model_t::frontier;
  static constexpr puercgp::algorithms::algo_kind_t algorithm_kind =
      puercgp::algorithms::algo_kind_t::bfs;
  static constexpr puercgp::algorithms::init_mode_t init_mode =
      puercgp::algorithms::init_mode_t::single_source;
  static constexpr puercgp::reduction_kind_t reduction =
      puercgp::reduction_kind_t::minimum;

  __host__ __device__ static constexpr value_type infinity() {
    return puercgp::algorithms::unified_infinity();
  }

  __host__ __device__ static constexpr value_type source_value() {
    return puercgp::algorithms::unified_source_value();
  }

  template <typename weight_t>
  __host__ __device__ static value_type relax(value_type source_distance,
                                               weight_t weight) {
    return source_distance + static_cast<value_type>(weight);
  }

  __host__ __device__ static constexpr bool should_update(
      value_type candidate, value_type current) {
    return candidate < current;
  }
};

struct options_t {
  std::string graph_path;
  std::filesystem::path sources_path;
  std::filesystem::path mapping_path;
  std::filesystem::path fingerprint_path;
  std::filesystem::path oracle_mappings_prefix;
  std::filesystem::path pull_trace_mappings_prefix;
  std::filesystem::path dependency_weights_path;
  std::filesystem::path values_output_path;
  std::string algorithm = "bfs";
  std::string mode = "push";
  std::string push_strategy = "shared_node";
  std::string pull_strategy = "fused";
  std::string pull_update = "in-place";
  std::vector<int> pull_degree_order{0, 1, 2, 3};
  std::vector<int> pull_degree_thresholds{32, 64, 128};
  int pull_high_degree_segment_edges = 1024;
  int pull_high_degree_segment_threads = 256;
  double pull_edge_ratio = 0.20;
  std::vector<double> pull_edge_ratio_grid;
  int query_count = 64;
  int total_queries = 0;
  int warmups = 2;
  int repeats = 7;
  int max_iterations = 10000;
  int pull_bidirectional_period = 1;
  int pull_bidirectional_rounds = 0;
  double pull_reverse_begin = 0.0;
  double pull_reverse_end = 1.0;
  std::vector<int> pull_period_grid;
  std::vector<int> pull_bidirectional_rounds_grid;
  std::vector<int> pull_bidirectional_rounds_by_batch;
  std::string pull_sweep = "forward";
  bool profile_iterations = false;
  bool bfs_edge_weights = false;
};

struct fingerprint_t {
  unsigned long long sum = 0;
  unsigned long long xor_value = 0;
};

struct mapping_t {
  int strategy = 0;
  std::vector<int> new_to_old;
  std::vector<int> old_to_new;
};

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

options_t parse_options(int argc, char **argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "Usage: bench_reordering <graph> <sources-file> "
        "[--algorithm=bfs|sssp|sswp] [--mode=push|pull|hybrid] [--mapping=path] "
        "[--fingerprints=path] [--queries=64] [--total-queries=64] "
        "[--warmups=2] [--repeats=7] "
        "[--max-iterations=10000] "
        "[--pull-sweep=forward|bidirectional] "
        "[--pull-bidirectional-period=1] "
        "[--pull-period-grid=3,4,6,8] "
        "[--pull-bidirectional-rounds=0] "
        "[--pull-bidirectional-rounds-grid=2,4,6] "
        "[--pull-bidirectional-rounds-by-batch=3,5,3,3] "
        "[--pull-reverse-range=0.0:1.0] "
        "[--oracle-mappings-prefix=path] "
        "[--pull-trace-mappings-prefix=path] "
        "[--dependency-weights=path] "
        "[--values-out=path] "
        "[--profile-iterations=0|1] "
        "[--bfs-edge-weights] "
        "[--push-strategy=shared_node|shared_node_warp] "
        "[--pull-strategy=fused|degree-aware|degree-segmented] "
        "[--pull-update=in-place|synchronous] "
        "[--pull-high-degree-segment-edges=1024] "
        "[--pull-high-degree-segment-threads=256] "
        "[--pull-edge-ratio=0.20] "
        "[--pull-edge-ratio-grid=0.005,0.02,0.08,0.20] "
        "[--pull-degree-thresholds=32,64,128]");
  }
  options_t options;
  options.graph_path = argv[1];
  options.sources_path = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value_after = [&](const std::string &prefix) {
      return argument.substr(prefix.size());
    };
    if (argument.rfind("--algorithm=", 0) == 0)
      options.algorithm = value_after("--algorithm=");
    else if (argument.rfind("--mode=", 0) == 0)
      options.mode = value_after("--mode=");
    else if (argument.rfind("--mapping=", 0) == 0)
      options.mapping_path = value_after("--mapping=");
    else if (argument.rfind("--fingerprints=", 0) == 0)
      options.fingerprint_path = value_after("--fingerprints=");
    else if (argument.rfind("--oracle-mappings-prefix=", 0) == 0)
      options.oracle_mappings_prefix =
          value_after("--oracle-mappings-prefix=");
    else if (argument.rfind("--pull-trace-mappings-prefix=", 0) == 0)
      options.pull_trace_mappings_prefix =
          value_after("--pull-trace-mappings-prefix=");
    else if (argument.rfind("--dependency-weights=", 0) == 0)
      options.dependency_weights_path = value_after("--dependency-weights=");
    else if (argument.rfind("--values-out=", 0) == 0)
      options.values_output_path = value_after("--values-out=");
    else if (argument.rfind("--queries=", 0) == 0)
      options.query_count = std::stoi(value_after("--queries="));
    else if (argument.rfind("--total-queries=", 0) == 0)
      options.total_queries = std::stoi(value_after("--total-queries="));
    else if (argument.rfind("--warmups=", 0) == 0)
      options.warmups = std::stoi(value_after("--warmups="));
    else if (argument.rfind("--repeats=", 0) == 0)
      options.repeats = std::stoi(value_after("--repeats="));
    else if (argument.rfind("--max-iterations=", 0) == 0)
      options.max_iterations = std::stoi(value_after("--max-iterations="));
    else if (argument.rfind("--pull-bidirectional-period=", 0) == 0)
      options.pull_bidirectional_period =
          std::stoi(value_after("--pull-bidirectional-period="));
    else if (argument.rfind("--pull-period-grid=", 0) == 0) {
      std::stringstream values(value_after("--pull-period-grid="));
      std::string value;
      while (std::getline(values, value, ',')) {
        options.pull_period_grid.push_back(std::stoi(value));
      }
    }
    else if (argument.rfind("--pull-bidirectional-rounds=", 0) == 0)
      options.pull_bidirectional_rounds =
          std::stoi(value_after("--pull-bidirectional-rounds="));
    else if (argument.rfind("--pull-bidirectional-rounds-grid=", 0) == 0) {
      std::stringstream values(
          value_after("--pull-bidirectional-rounds-grid="));
      std::string value;
      while (std::getline(values, value, ',')) {
        options.pull_bidirectional_rounds_grid.push_back(std::stoi(value));
      }
    } else if (argument.rfind("--pull-reverse-range=", 0) == 0) {
      const auto range = value_after("--pull-reverse-range=");
      const auto separator = range.find(':');
      if (separator == std::string::npos) {
        throw std::invalid_argument(
            "pull reverse range must use begin:end");
      }
      options.pull_reverse_begin = std::stod(range.substr(0, separator));
      options.pull_reverse_end = std::stod(range.substr(separator + 1));
    } else if (argument.rfind("--pull-bidirectional-rounds-by-batch=", 0) ==
               0) {
      std::stringstream values(
          value_after("--pull-bidirectional-rounds-by-batch="));
      std::string value;
      while (std::getline(values, value, ',')) {
        options.pull_bidirectional_rounds_by_batch.push_back(std::stoi(value));
      }
    }
    else if (argument.rfind("--pull-sweep=", 0) == 0)
      options.pull_sweep = value_after("--pull-sweep=");
    else if (argument.rfind("--profile-iterations=", 0) == 0) {
      const auto value = value_after("--profile-iterations=");
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--profile-iterations must be 0 or 1");
      }
      options.profile_iterations = value == "1";
    } else if (argument == "--bfs-edge-weights") {
      options.bfs_edge_weights = true;
    } else if (argument.rfind("--push-strategy=", 0) == 0)
      options.push_strategy = value_after("--push-strategy=");
    else if (argument.rfind("--pull-strategy=", 0) == 0)
      options.pull_strategy = value_after("--pull-strategy=");
    else if (argument.rfind("--pull-update=", 0) == 0)
      options.pull_update = value_after("--pull-update=");
    else if (argument.rfind("--pull-high-degree-segment-edges=", 0) == 0)
      options.pull_high_degree_segment_edges =
          std::stoi(value_after("--pull-high-degree-segment-edges="));
    else if (argument.rfind("--pull-high-degree-segment-threads=", 0) == 0)
      options.pull_high_degree_segment_threads =
          std::stoi(value_after("--pull-high-degree-segment-threads="));
    else if (argument.rfind("--pull-edge-ratio=", 0) == 0)
      options.pull_edge_ratio =
          std::stod(value_after("--pull-edge-ratio="));
    else if (argument.rfind("--pull-edge-ratio-grid=", 0) == 0) {
      std::stringstream values(value_after("--pull-edge-ratio-grid="));
      std::string value;
      while (std::getline(values, value, ',')) {
        options.pull_edge_ratio_grid.push_back(std::stod(value));
      }
    }
    else if (argument.rfind("--pull-degree-thresholds=", 0) == 0) {
      options.pull_degree_thresholds.clear();
      std::stringstream values(value_after("--pull-degree-thresholds="));
      std::string value;
      while (std::getline(values, value, ',')) {
        options.pull_degree_thresholds.push_back(std::stoi(value));
      }
    }
    else if (argument.rfind("--pull-degree-order=", 0) == 0) {
      options.pull_degree_order.clear();
      const auto order = value_after("--pull-degree-order=");
      for (char value : order) {
        if (value < '0' || value > '3') {
          throw std::invalid_argument("degree order must contain digits 0..3");
        }
        options.pull_degree_order.push_back(value - '0');
      }
    }
    else
      throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.total_queries == 0)
    options.total_queries = options.query_count;
  if ((options.algorithm != "bfs" && options.algorithm != "sssp" &&
       options.algorithm != "sswp") ||
      (options.mode != "push" && options.mode != "pull" &&
       options.mode != "hybrid") ||
      options.query_count <= 0 || options.query_count > 64 ||
      options.total_queries <= 0 || options.warmups < 0 ||
      options.repeats <= 0 || options.max_iterations <= 0 ||
      options.pull_edge_ratio < 0.0 ||
      options.pull_bidirectional_period <= 0 ||
      options.pull_bidirectional_rounds < 0 ||
      options.pull_reverse_begin < 0.0 ||
      options.pull_reverse_begin >= options.pull_reverse_end ||
      options.pull_reverse_end > 1.0 ||
      options.pull_high_degree_segment_edges <= 0 ||
      (options.pull_high_degree_segment_threads != 128 &&
       options.pull_high_degree_segment_threads != 256 &&
       options.pull_high_degree_segment_threads != 512 &&
       options.pull_high_degree_segment_threads != 1024) ||
      options.pull_degree_thresholds.size() != 3 ||
      options.pull_degree_thresholds[0] <= 0 ||
      options.pull_degree_thresholds[0] >=
          options.pull_degree_thresholds[1] ||
      options.pull_degree_thresholds[1] >=
          options.pull_degree_thresholds[2] ||
      (options.pull_sweep != "forward" &&
       options.pull_sweep != "bidirectional")) {
    throw std::invalid_argument("invalid benchmark options");
  }
  if (options.push_strategy != "shared_node" &&
      options.push_strategy != "shared_node_warp") {
    throw std::invalid_argument("invalid push strategy");
  }
  if (options.pull_strategy != "fused" &&
      options.pull_strategy != "degree-aware" &&
      options.pull_strategy != "degree-segmented") {
    throw std::invalid_argument("invalid pull strategy");
  }
  if (options.pull_update != "in-place" &&
      options.pull_update != "synchronous") {
    throw std::invalid_argument("invalid pull update mode");
  }
  if (options.pull_update == "synchronous" &&
      options.pull_sweep == "bidirectional") {
    throw std::invalid_argument(
        "synchronous pull currently requires a forward sweep");
  }
  if (options.bfs_edge_weights && options.algorithm != "bfs") {
    throw std::invalid_argument("--bfs-edge-weights requires --algorithm=bfs");
  }
  if (std::any_of(options.pull_bidirectional_rounds_grid.begin(),
                  options.pull_bidirectional_rounds_grid.end(),
                  [](int rounds) { return rounds <= 0; })) {
    throw std::invalid_argument(
        "pull bidirectional rounds grid requires positive values");
  }
  if (std::any_of(options.pull_period_grid.begin(),
                  options.pull_period_grid.end(),
                  [](int period) { return period <= 0; })) {
    throw std::invalid_argument("pull period grid requires positive values");
  }
  if (std::any_of(options.pull_edge_ratio_grid.begin(),
                  options.pull_edge_ratio_grid.end(),
                  [](double ratio) { return ratio < 0.0; })) {
    throw std::invalid_argument(
        "pull edge ratio grid requires non-negative values");
  }
  if (std::any_of(options.pull_bidirectional_rounds_by_batch.begin(),
                  options.pull_bidirectional_rounds_by_batch.end(),
                  [](int rounds) { return rounds < 0; })) {
    throw std::invalid_argument(
        "per-batch pull bidirectional rounds must be non-negative");
  }
  if (!options.pull_bidirectional_rounds_grid.empty() &&
      !options.pull_bidirectional_rounds_by_batch.empty()) {
    throw std::invalid_argument(
        "rounds grid and per-batch rounds cannot be combined");
  }
  if (!options.pull_period_grid.empty() &&
      (!options.pull_bidirectional_rounds_grid.empty() ||
       !options.pull_bidirectional_rounds_by_batch.empty())) {
    throw std::invalid_argument(
        "pull period grid cannot be combined with a rounds grid");
  }
  const int grid_count = !options.pull_period_grid.empty() +
                         !options.pull_bidirectional_rounds_grid.empty() +
                         !options.pull_edge_ratio_grid.empty();
  if (grid_count > 1) {
    throw std::invalid_argument(
        "only one pull parameter grid can be specified");
  }
  return options;
}

std::vector<int> load_sources(const std::filesystem::path &path, int count) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open sources file: " + path.string());
  }
  std::vector<int> sources;
  std::string line;
  while (std::getline(input, line) &&
         static_cast<int>(sources.size()) < count) {
    if (line.empty() || line[0] == '#')
      continue;
    std::stringstream row(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(row, field, ','))
      fields.push_back(field);
    try {
      sources.push_back(std::stoi(fields.size() >= 2 ? fields[1] : fields[0]));
    } catch (const std::exception &) {
      if (sources.empty())
        continue;
      throw;
    }
  }
  if (static_cast<int>(sources.size()) != count) {
    throw std::runtime_error("sources file has fewer entries than requested");
  }
  return sources;
}

std::vector<int> load_query_weights(const std::filesystem::path &path,
                                    int query_count) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open dependency weights: " +
                             path.string());
  }
  std::vector<int> weights;
  int weight = 0;
  while (input >> weight) {
    if (weight < 0)
      throw std::runtime_error("dependency weights must be non-negative");
    weights.push_back(weight);
  }
  if (static_cast<int>(weights.size()) != query_count) {
    throw std::runtime_error("dependency weight count does not match Q");
  }
  return weights;
}

mapping_t load_mapping(const std::filesystem::path &path, int vertices) {
  mapping_t mapping;
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open mapping file: " + path.string());
  }
  std::int32_t magic = 0;
  std::int32_t count = 0;
  std::int32_t strategy = 0;
  input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  input.read(reinterpret_cast<char *>(&strategy), sizeof(strategy));
  if (!input || magic != kReorderMagic || count != vertices) {
    throw std::runtime_error("mapping header does not match graph");
  }
  mapping.strategy = strategy;
  mapping.new_to_old.resize(static_cast<std::size_t>(vertices));
  input.read(
      reinterpret_cast<char *>(mapping.new_to_old.data()),
      static_cast<std::streamsize>(mapping.new_to_old.size() * sizeof(int)));
  if (!input)
    throw std::runtime_error("truncated mapping file");

  mapping.old_to_new.assign(static_cast<std::size_t>(vertices), -1);
  for (int new_id = 0; new_id < vertices; ++new_id) {
    const int old_id = mapping.new_to_old[static_cast<std::size_t>(new_id)];
    if (old_id < 0 || old_id >= vertices ||
        mapping.old_to_new[static_cast<std::size_t>(old_id)] != -1) {
      throw std::runtime_error("mapping is not a permutation");
    }
    mapping.old_to_new[static_cast<std::size_t>(old_id)] = new_id;
  }
  return mapping;
}

puercgp::traversal_mode_t traversal_mode(const std::string &mode) {
  if (mode == "push")
    return puercgp::traversal_mode_t::push;
  if (mode == "pull")
    return puercgp::traversal_mode_t::pull;
  return puercgp::traversal_mode_t::hybrid;
}

puercgp::push_strategy_t push_strategy(const std::string &strategy) {
  return strategy == "shared_node_warp"
             ? puercgp::push_strategy_t::shared_node_warp
             : puercgp::push_strategy_t::shared_node;
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

template <typename value_t>
__device__ __forceinline__ unsigned int value_bits(value_t value) {
  if constexpr (std::is_same<value_t, float>::value) {
    return __float_as_uint(value);
  } else {
    return static_cast<unsigned int>(value);
  }
}

template <typename value_t>
__global__ void
fingerprint_kernel(const value_t *values, std::size_t vertex_count,
                   int query_count, const int *new_to_old,
                   unsigned long long *sums, unsigned long long *xor_values) {
  const int query = threadIdx.x;
  if (query >= query_count)
    return;
  unsigned long long local_sum = 0;
  unsigned long long local_xor = 0;
  for (std::size_t new_vertex = blockIdx.x; new_vertex < vertex_count;
       new_vertex += gridDim.x) {
    const auto old_vertex = static_cast<unsigned long long>(
        new_to_old == nullptr ? new_vertex : new_to_old[new_vertex]);
    const auto bits = value_bits(
        values[new_vertex * static_cast<std::size_t>(query_count) + query]);
    const auto mixed = mix64((old_vertex << 32) ^ bits ^ 0x9e3779b97f4a7c15ULL);
    local_sum += mixed;
    local_xor ^= mixed;
  }
  atomicAdd(sums + query, local_sum);
  atomicXor(xor_values + query, local_xor);
}

template <typename value_t>
std::vector<fingerprint_t>
fingerprint_values(const thrust::device_vector<value_t> &values,
                   std::size_t vertex_count, int query_count,
                   const thrust::device_vector<int> *new_to_old,
                   cudaStream_t stream) {
  thrust::device_vector<unsigned long long> sums(query_count);
  thrust::device_vector<unsigned long long> xor_values(query_count);
  check_cuda(cudaMemsetAsync(thrust::raw_pointer_cast(sums.data()), 0,
                             sums.size() * sizeof(unsigned long long), stream),
             "clear fingerprint sums");
  check_cuda(cudaMemsetAsync(thrust::raw_pointer_cast(xor_values.data()), 0,
                             xor_values.size() * sizeof(unsigned long long),
                             stream),
             "clear fingerprint xors");
  const int blocks =
      std::max(1, std::min(4096, static_cast<int>(vertex_count)));
  const int *map_ptr = new_to_old == nullptr
                           ? nullptr
                           : thrust::raw_pointer_cast(new_to_old->data());
  fingerprint_kernel<<<blocks, query_count, 0, stream>>>(
      thrust::raw_pointer_cast(values.data()), vertex_count, query_count,
      map_ptr, thrust::raw_pointer_cast(sums.data()),
      thrust::raw_pointer_cast(xor_values.data()));
  check_cuda(cudaGetLastError(), "launch fingerprint kernel");

  std::vector<unsigned long long> host_sums(query_count);
  std::vector<unsigned long long> host_xors(query_count);
  check_cuda(cudaMemcpyAsync(host_sums.data(),
                             thrust::raw_pointer_cast(sums.data()),
                             sums.size() * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, stream),
             "copy fingerprint sums");
  check_cuda(cudaMemcpyAsync(host_xors.data(),
                             thrust::raw_pointer_cast(xor_values.data()),
                             xor_values.size() * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, stream),
             "copy fingerprint xors");
  check_cuda(cudaStreamSynchronize(stream), "synchronize fingerprints");

  std::vector<fingerprint_t> fingerprints(query_count);
  for (int query = 0; query < query_count; ++query) {
    fingerprints[query] = {host_sums[query], host_xors[query]};
  }
  return fingerprints;
}

void save_fingerprints(const std::filesystem::path &path,
                       const std::vector<int> &original_sources,
                       const std::vector<fingerprint_t> &fingerprints) {
  if (path.empty())
    return;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot create fingerprint file: " +
                             path.string());
  }
  output << "query,source,sum,xor\n";
  for (std::size_t query = 0; query < fingerprints.size(); ++query) {
    output << query << ',' << original_sources[query] << ','
           << fingerprints[query].sum << ',' << fingerprints[query].xor_value
           << '\n';
  }
}

template <typename value_t>
void save_device_values(const std::filesystem::path &path,
                        const thrust::device_vector<value_t> &values) {
  if (path.empty()) {
    return;
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  thrust::host_vector<value_t> host_values(values);
  const auto temporary = std::filesystem::path(path.string() + ".tmp");
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(host_values.data()),
               static_cast<std::streamsize>(host_values.size() *
                                            sizeof(value_t)));
  output.close();
  if (!output) {
    throw std::runtime_error("cannot write values: " + path.string());
  }
  std::filesystem::rename(temporary, path);
}

void save_mapping(const std::filesystem::path &path, int strategy,
                  const std::vector<int> &new_to_old) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const auto temporary = std::filesystem::path(path.string() + ".tmp");
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create mapping file: " + path.string());
  }
  const std::int32_t magic = kReorderMagic;
  const std::int32_t count = static_cast<std::int32_t>(new_to_old.size());
  const std::int32_t tag = strategy;
  output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  output.write(reinterpret_cast<const char *>(&count), sizeof(count));
  output.write(reinterpret_cast<const char *>(&tag), sizeof(tag));
  output.write(reinterpret_cast<const char *>(new_to_old.data()),
               static_cast<std::streamsize>(new_to_old.size() * sizeof(int)));
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write mapping file: " + path.string());
  }
  std::filesystem::rename(temporary, path);
}

void save_segment_offsets(const std::filesystem::path &path,
                          const std::vector<std::size_t> &offsets) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create segment offsets: " +
                             path.string());
  }
  for (std::size_t offset : offsets)
    output << offset << '\n';
}

void save_pull_trace_mappings(
    const std::filesystem::path &prefix,
    const thrust::device_vector<unsigned long long> &iteration_sum,
    const thrust::device_vector<unsigned int> &update_count,
    const thrust::device_vector<unsigned int> &first_iteration,
    const thrust::device_vector<unsigned int> &last_iteration,
    const std::vector<int> *new_to_old) {
  if (iteration_sum.empty() || iteration_sum.size() != update_count.size() ||
      iteration_sum.size() != first_iteration.size() ||
      iteration_sum.size() != last_iteration.size()) {
    throw std::runtime_error("pull update trace is unavailable or malformed");
  }
  thrust::host_vector<unsigned long long> host_sum = iteration_sum;
  thrust::host_vector<unsigned int> host_count = update_count;
  thrust::host_vector<unsigned int> host_first = first_iteration;
  thrust::host_vector<unsigned int> host_last = last_iteration;
  std::vector<int> order(iteration_sum.size());
  std::iota(order.begin(), order.end(), 0);

  auto emit = [&](const std::string &suffix, int tag, auto score,
                  auto phase_key, bool descending) {
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
      const auto left_score = score(static_cast<std::size_t>(left));
      const auto right_score = score(static_cast<std::size_t>(right));
      if (left_score != right_score) {
        return descending ? left_score > right_score : left_score < right_score;
      }
      return left < right;
    });
    std::vector<int> original_order(order.size());
    for (std::size_t position = 0; position < order.size(); ++position) {
      const auto internal_vertex = static_cast<std::size_t>(order[position]);
      original_order[position] = new_to_old == nullptr
                                     ? order[position]
                                     : (*new_to_old)[internal_vertex];
    }
    save_mapping(prefix.string() + "." + suffix + ".bin", tag,
                 original_order);

    std::vector<std::size_t> offsets{0};
    if (!order.empty()) {
      auto previous = phase_key(static_cast<std::size_t>(order.front()));
      for (std::size_t position = 1; position < order.size(); ++position) {
        const auto current =
            phase_key(static_cast<std::size_t>(order[position]));
        if (current != previous) {
          offsets.push_back(position);
          previous = current;
        }
      }
      offsets.push_back(order.size());
    }
    save_segment_offsets(prefix.string() + "." + suffix + ".offsets",
                         offsets);
    std::iota(order.begin(), order.end(), 0);
  };

  auto mean_phase = [&](std::size_t vertex) {
    return host_count[vertex] == 0
               ? 0.0
               : static_cast<double>(host_sum[vertex]) /
                     static_cast<double>(host_count[vertex]);
  };
  emit("update-first", 90,
       [&](std::size_t vertex) { return host_first[vertex]; },
       [&](std::size_t vertex) { return host_first[vertex]; }, false);
  emit("update-last", 91,
       [&](std::size_t vertex) { return host_last[vertex]; },
       [&](std::size_t vertex) { return host_last[vertex]; }, false);
  emit("update-mean", 92, mean_phase,
       [&](std::size_t vertex) {
         return static_cast<unsigned int>(mean_phase(vertex));
       },
       false);
  emit("update-count", 93,
       [&](std::size_t vertex) { return host_count[vertex]; },
       [&](std::size_t vertex) { return host_count[vertex]; }, true);
}

template <typename graph_t>
__global__ void sssp_oracle_scores_kernel(
    graph_t graph, const float *values, std::size_t vertex_count,
    int query_count, float *distance_sum, float *distance_min,
    float *distance_max, long long *dependency_balance,
    float *dependency_normalized, const int *query_weights) {
  constexpr int warp_size = 32;
  constexpr int warps_per_block = 8;
  const int lane = threadIdx.x & (warp_size - 1);
  const int warp = threadIdx.x / warp_size;
  const std::size_t vertex =
      static_cast<std::size_t>(blockIdx.x) * warps_per_block + warp;
  if (vertex >= vertex_count)
    return;

  float local_sum = 0.0f;
  float local_min = std::numeric_limits<float>::infinity();
  float local_max = 0.0f;
  for (int query = lane; query < query_count; query += warp_size) {
    const float value =
        values[vertex * static_cast<std::size_t>(query_count) + query];
    if (isfinite(value)) {
      local_sum += value;
      local_min = fminf(local_min, value);
      local_max = fmaxf(local_max, value);
    }
  }
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
    local_min = fminf(
        local_min, __shfl_down_sync(0xffffffffu, local_min, offset));
    local_max = fmaxf(
        local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
  }

  long long local_balance = 0;
  auto edge_begin = graph.get_starting_edge(static_cast<int>(vertex));
  auto edge_end = graph.get_starting_edge(static_cast<int>(vertex + 1));
  for (auto edge = edge_begin + lane; edge < edge_end; edge += warp_size) {
    const auto neighbor = graph.get_destination_vertex(edge);
    const float weight = graph.get_edge_weight(edge);
    const std::size_t neighbor_offset =
        static_cast<std::size_t>(neighbor) * query_count;
    const std::size_t vertex_offset = vertex * query_count;
    for (int query = 0; query < query_count; ++query) {
      const float own = values[vertex_offset + query];
      const float other = values[neighbor_offset + query];
      if (!isfinite(own) || !isfinite(other))
        continue;
      const float tolerance = 1.0e-4f * fmaxf(1.0f, fmaxf(own, other));
      const int vote_weight = query_weights == nullptr ? 1 : query_weights[query];
      if (fabsf(own + weight - other) <= tolerance) {
        local_balance += vote_weight;
      } else if (fabsf(other + weight - own) <= tolerance) {
        local_balance -= vote_weight;
      }
    }
  }
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    local_balance +=
        __shfl_down_sync(0xffffffffu, local_balance, offset);
  }

  if (lane == 0) {
    distance_sum[vertex] = local_sum;
    distance_min[vertex] = local_min;
    distance_max[vertex] = local_max;
    dependency_balance[vertex] = local_balance;
    const auto degree = edge_end - edge_begin;
    dependency_normalized[vertex] =
        static_cast<float>(local_balance) /
        static_cast<float>(degree > 0 ? degree : 1);
  }
}

template <typename graph_t>
void save_sssp_oracle_mappings(
    const std::filesystem::path &prefix, graph_t graph,
    const thrust::device_vector<float> &values, std::size_t vertex_count,
    int query_count, const int *query_weights, cudaStream_t stream) {
  thrust::device_vector<float> distance_sum(vertex_count);
  thrust::device_vector<float> distance_min(vertex_count);
  thrust::device_vector<float> distance_max(vertex_count);
  thrust::device_vector<long long> dependency_balance(vertex_count);
  thrust::device_vector<float> dependency_normalized(vertex_count);
  constexpr int threads = 256;
  constexpr int warps_per_block = threads / 32;
  const int blocks = static_cast<int>(
      (vertex_count + warps_per_block - 1) / warps_per_block);
  sssp_oracle_scores_kernel<<<blocks, threads, 0, stream>>>(
      graph, thrust::raw_pointer_cast(values.data()), vertex_count, query_count,
      thrust::raw_pointer_cast(distance_sum.data()),
      thrust::raw_pointer_cast(distance_min.data()),
      thrust::raw_pointer_cast(distance_max.data()),
      thrust::raw_pointer_cast(dependency_balance.data()),
      thrust::raw_pointer_cast(dependency_normalized.data()), query_weights);
  check_cuda(cudaGetLastError(), "launch SSSP oracle score kernel");
  check_cuda(cudaStreamSynchronize(stream), "compute SSSP oracle scores");

  thrust::host_vector<float> host_sum = distance_sum;
  thrust::host_vector<float> host_min = distance_min;
  thrust::host_vector<float> host_max = distance_max;
  thrust::host_vector<long long> host_balance = dependency_balance;
  thrust::host_vector<float> host_normalized = dependency_normalized;
  std::vector<int> order(vertex_count);
  std::iota(order.begin(), order.end(), 0);

  auto emit = [&](const std::string &suffix, int tag, auto score,
                  bool descending) {
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
      const auto left_score = score(static_cast<std::size_t>(left));
      const auto right_score = score(static_cast<std::size_t>(right));
      if (left_score != right_score) {
        return descending ? left_score > right_score : left_score < right_score;
      }
      return left < right;
    });
    save_mapping(prefix.string() + "." + suffix + ".bin", tag, order);
    std::iota(order.begin(), order.end(), 0);
  };

  emit("distance-mean", 80,
       [&](std::size_t vertex) { return host_sum[vertex]; }, false);
  emit("distance-max", 81,
       [&](std::size_t vertex) { return host_max[vertex]; }, false);
  emit("distance-min", 82,
       [&](std::size_t vertex) { return host_min[vertex]; }, false);
  emit("dependency-balance", 83,
       [&](std::size_t vertex) { return host_balance[vertex]; }, true);
  emit("dependency-normalized", 84,
       [&](std::size_t vertex) { return host_normalized[vertex]; },
       true);
}

template <typename policy_t, typename graph_t>
int run_benchmark(const options_t &options, graph_t graph,
                  std::size_t vertex_count,
                  const std::vector<int> &original_sources,
                  const std::vector<int> &execution_sources,
                  const std::vector<int> *new_to_old, int mapping_strategy) {
  puercgp::execution_context context;
  puercgp::run_options run_options;
  run_options.traversal_mode = traversal_mode(options.mode);
  run_options.push_strategy = push_strategy(options.push_strategy);
  run_options.pull_strategy = options.pull_strategy == "degree-aware"
      ? puercgp::pull_strategy_t::degree_aware
      : options.pull_strategy == "degree-segmented"
          ? puercgp::pull_strategy_t::degree_segmented
          : puercgp::pull_strategy_t::fused;
  run_options.pull_update_mode = options.pull_update == "synchronous"
      ? puercgp::pull_update_mode_t::synchronous
      : puercgp::pull_update_mode_t::in_place;
  run_options.pull_degree_bucket_order = options.pull_degree_order;
  run_options.pull_degree_threshold_2 = options.pull_degree_thresholds[0];
  run_options.pull_degree_threshold_4 = options.pull_degree_thresholds[1];
  run_options.pull_degree_threshold_8 = options.pull_degree_thresholds[2];
  run_options.pull_high_degree_segment_edges =
      options.pull_high_degree_segment_edges;
  run_options.pull_high_degree_segment_threads =
      options.pull_high_degree_segment_threads;
  run_options.pull_edge_ratio = options.pull_edge_ratio;
  run_options.max_queries = 64;
  run_options.max_iterations = options.max_iterations;
  run_options.profile_iterations = options.profile_iterations;
  run_options.trace_pull_updates =
      !options.pull_trace_mappings_prefix.empty();
  run_options.pull_bidirectional_period =
      static_cast<std::size_t>(options.pull_bidirectional_period);
  run_options.pull_bidirectional_rounds =
      static_cast<std::size_t>(options.pull_bidirectional_rounds);
  run_options.pull_reverse_vertex_begin = static_cast<std::size_t>(
      options.pull_reverse_begin * static_cast<double>(vertex_count));
  run_options.pull_reverse_vertex_end = static_cast<std::size_t>(
      options.pull_reverse_end * static_cast<double>(vertex_count));
  if (options.pull_sweep == "bidirectional") {
    run_options.pull_sweep = puercgp::pull_sweep_t::bidirectional;
  }

  const int batch_count =
      (options.total_queries + options.query_count - 1) / options.query_count;
  if (!options.values_output_path.empty() && batch_count != 1) {
    throw std::invalid_argument("--values-out requires exactly one batch");
  }
  if (!options.pull_bidirectional_rounds_by_batch.empty() &&
      static_cast<int>(options.pull_bidirectional_rounds_by_batch.size()) !=
          batch_count) {
    throw std::invalid_argument(
        "per-batch bidirectional rounds count must match batch count");
  }
  auto options_for_batch = [&](int batch) {
    auto batch_options = run_options;
    if (!options.pull_bidirectional_rounds_by_batch.empty()) {
      batch_options.pull_bidirectional_rounds = static_cast<std::size_t>(
          options.pull_bidirectional_rounds_by_batch[batch]);
      batch_options.pull_sweep =
          batch_options.pull_bidirectional_rounds == 0
              ? puercgp::pull_sweep_t::forward
              : puercgp::pull_sweep_t::bidirectional;
    }
    return batch_options;
  };
  if (!options.oracle_mappings_prefix.empty() &&
      (options.algorithm != "sssp" || batch_count != 1 ||
       new_to_old != nullptr)) {
    throw std::invalid_argument(
        "oracle mappings require one original-layout SSSP batch");
  }
  if (!options.pull_trace_mappings_prefix.empty() &&
      (options.algorithm != "sssp" || options.mode != "pull" ||
       batch_count != 1)) {
    throw std::invalid_argument(
        "pull trace mappings require one original-layout SSSP pull batch");
  }

  auto make_batch = [&](int batch) {
    const int first = batch * options.query_count;
    const int last =
        std::min(first + options.query_count, options.total_queries);
    return std::vector<int>(execution_sources.begin() + first,
                            execution_sources.begin() + last);
  };

  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    for (int batch = 0; batch < batch_count; ++batch) {
      puercgp::query_batch<int> queries(make_batch(batch));
      auto batch_options = options_for_batch(batch);
      auto result =
          puercgp::run<policy_t>(graph, queries, context, batch_options);
      (void)result;
    }
  }

  std::vector<float> gpu_times;
  std::vector<float> wall_times;
  std::vector<int> iteration_counts;
  std::vector<int> push_iteration_counts;
  std::vector<int> pull_iteration_counts;
  gpu_times.reserve(static_cast<std::size_t>(options.repeats));
  wall_times.reserve(static_cast<std::size_t>(options.repeats));
  iteration_counts.reserve(static_cast<std::size_t>(options.repeats));
  push_iteration_counts.reserve(static_cast<std::size_t>(options.repeats));
  pull_iteration_counts.reserve(static_cast<std::size_t>(options.repeats));
  std::vector<fingerprint_t> fingerprints(
      static_cast<std::size_t>(options.total_queries));

  thrust::device_vector<int> device_new_to_old;
  if (new_to_old != nullptr)
    device_new_to_old = *new_to_old;
  thrust::device_vector<int> dependency_weights;
  if (!options.dependency_weights_path.empty()) {
    if (options.oracle_mappings_prefix.empty()) {
      throw std::invalid_argument(
          "dependency weights require --oracle-mappings-prefix");
    }
    dependency_weights =
        load_query_weights(options.dependency_weights_path, options.query_count);
  }

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    float total_gpu_ms = 0.0f;
    float total_wall_ms = 0.0f;
    int total_iterations = 0;
    int total_push_iterations = 0;
    int total_pull_iterations = 0;
    for (int batch = 0; batch < batch_count; ++batch) {
      auto batch_sources = make_batch(batch);
      puercgp::query_batch<int> queries(batch_sources);
      auto batch_options = options_for_batch(batch);
      auto result =
          puercgp::run<policy_t>(graph, queries, context, batch_options);
      total_gpu_ms += result.gpu_time_ms;
      total_wall_ms += result.wall_time_ms;
      total_iterations += result.iterations;
      const int batch_pull_iterations = static_cast<int>(std::count(
          result.iteration_modes.begin(), result.iteration_modes.end(),
          "pull"));
      const int batch_push_iterations =
          result.iterations - batch_pull_iterations;
      total_push_iterations += batch_push_iterations;
      total_pull_iterations += batch_pull_iterations;
      std::cout << "BATCH repeat=" << repeat << " batch=" << batch
                << " bidirectional_rounds="
                << batch_options.pull_bidirectional_rounds
                << " gpu_ms=" << result.gpu_time_ms
                << " wall_ms=" << result.wall_time_ms
                << " iterations=" << result.iterations
                << " push_iterations=" << batch_push_iterations
                << " pull_iterations=" << batch_pull_iterations << '\n';
      if (options.profile_iterations && repeat + 1 == options.repeats) {
        for (const auto &profile : result.iteration_profiles) {
          std::cout << "ITER batch=" << batch
                    << " iteration=" << profile.iteration
                    << " mode=" << profile.mode
                    << " frontier_pairs=" << profile.frontier_size
                    << " frontier_vertices=" << profile.unique_frontier_size
                    << " actual_edges=" << profile.actual_edge_count
                    << " virtual_edges=" << profile.virtual_edge_count
                    << " kernel_ms="
                    << (profile.mode == "pull" ? profile.pull_kernel_ms
                                               : profile.shared_push_kernel_ms)
                    << " wall_ms=" << profile.iteration_wall_ms << '\n';
        }
      }
      if (repeat + 1 == options.repeats) {
        save_device_values(options.values_output_path, result.values);
        auto batch_fingerprints = fingerprint_values(
            result.values, vertex_count, static_cast<int>(batch_sources.size()),
            new_to_old == nullptr ? nullptr : &device_new_to_old,
            context.stream());
        const auto first =
            static_cast<std::size_t>(batch * options.query_count);
        std::copy(batch_fingerprints.begin(), batch_fingerprints.end(),
                  fingerprints.begin() + first);
        if (!options.oracle_mappings_prefix.empty()) {
          save_sssp_oracle_mappings(
              options.oracle_mappings_prefix, graph, result.values,
              vertex_count, static_cast<int>(batch_sources.size()),
              dependency_weights.empty()
                  ? nullptr
                  : thrust::raw_pointer_cast(dependency_weights.data()),
              context.stream());
        }
        if (!options.pull_trace_mappings_prefix.empty()) {
          save_pull_trace_mappings(
              options.pull_trace_mappings_prefix,
              result.pull_update_iteration_sum, result.pull_update_count,
              result.pull_update_first, result.pull_update_last, new_to_old);
        }
      }
    }
    gpu_times.push_back(total_gpu_ms);
    wall_times.push_back(total_wall_ms);
    iteration_counts.push_back(total_iterations);
    push_iteration_counts.push_back(total_push_iterations);
    pull_iteration_counts.push_back(total_pull_iterations);
    std::cout << "repeat=" << repeat << " gpu_ms=" << total_gpu_ms
              << " wall_ms=" << total_wall_ms
              << " iterations=" << total_iterations
              << " push_iterations=" << total_push_iterations
              << " pull_iterations=" << total_pull_iterations << '\n';
  }

  save_fingerprints(options.fingerprint_path, original_sources, fingerprints);
  const auto gpu_median = puercgp_examples::median(gpu_times);
  const auto wall_median = puercgp_examples::median(wall_times);
  const auto gpu_min = *std::min_element(gpu_times.begin(), gpu_times.end());
  const auto gpu_max = *std::max_element(gpu_times.begin(), gpu_times.end());
  std::sort(iteration_counts.begin(), iteration_counts.end());
  std::sort(push_iteration_counts.begin(), push_iteration_counts.end());
  std::sort(pull_iteration_counts.begin(), pull_iteration_counts.end());
  const int iterations = iteration_counts[iteration_counts.size() / 2];
  const int iterations_min = iteration_counts.front();
  const int iterations_max = iteration_counts.back();
  const int push_iterations =
      push_iteration_counts[push_iteration_counts.size() / 2];
  const int pull_iterations =
      pull_iteration_counts[pull_iteration_counts.size() / 2];
  std::cout << std::setprecision(9) << "RESULT graph=" << options.graph_path
            << " algorithm=" << options.algorithm << " mode=" << options.mode
            << " vertices=" << vertex_count
            << " edges=" << graph.get_number_of_edges()
            << " Q=" << options.query_count
            << " mapping_strategy=" << mapping_strategy
            << " pull_sweep=" << options.pull_sweep
            << " pull_strategy=" << options.pull_strategy
            << " pull_update=" << options.pull_update
            << " pull_degree_order=";
  for (int bucket : options.pull_degree_order) {
    std::cout << bucket;
  }
  std::cout
            << " pull_bidirectional_period="
            << options.pull_bidirectional_period
            << " pull_bidirectional_rounds="
            << options.pull_bidirectional_rounds
            << " pull_reverse_range=" << options.pull_reverse_begin << ':'
            << options.pull_reverse_end
            << " pull_edge_ratio=" << options.pull_edge_ratio
            << " iterations=" << iterations
            << " push_iterations=" << push_iterations
            << " pull_iterations=" << pull_iterations
            << " iterations_min=" << iterations_min
            << " iterations_max=" << iterations_max
            << " gpu_ms_median=" << gpu_median << " gpu_ms_min=" << gpu_min
            << " gpu_ms_max=" << gpu_max << " wall_ms_median=" << wall_median
            << " N=" << options.total_queries << " batches=" << batch_count
            << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    auto options = parse_options(argc, argv);
    const bool build_pull = options.mode != "push";
    auto graph_storage =
        puercgp_examples::load_graph_auto(options.graph_path, build_pull);
    auto original_sources =
        load_sources(options.sources_path, options.total_queries);

    mapping_t mapping;
    std::vector<int> execution_sources = original_sources;
    const std::vector<int> *new_to_old = nullptr;
    int mapping_strategy = 0;
    if (!options.mapping_path.empty()) {
      mapping = load_mapping(options.mapping_path, graph_storage.vertices);
      mapping_strategy = mapping.strategy;
      new_to_old = &mapping.new_to_old;
      for (int &source : execution_sources) {
        if (source < 0 || source >= graph_storage.vertices) {
          throw std::runtime_error("source is outside graph");
        }
        source = mapping.old_to_new[static_cast<std::size_t>(source)];
      }
    }
    for (int source : execution_sources) {
      if (source < 0 || source >= graph_storage.vertices) {
        throw std::runtime_error("source is outside graph");
      }
    }

    std::cout << "vertices=" << graph_storage.vertices
              << " edges=" << graph_storage.edges
              << " algorithm=" << options.algorithm << " mode=" << options.mode
              << " Q=" << options.query_count << " N=" << options.total_queries
              << " warmups=" << options.warmups
              << " repeats=" << options.repeats
              << " pull_sweep=" << options.pull_sweep << " mapping="
              << (options.mapping_path.empty() ? "identity"
                                               : options.mapping_path.string())
              << '\n';

    auto graph = graph_storage.view();
    auto run_once = [&](const options_t &run_options) {
      if (run_options.algorithm == "bfs") {
        if (run_options.bfs_edge_weights) {
          return run_benchmark<weighted_bfs_policy>(
              run_options, graph, graph_storage.vertices, original_sources,
              execution_sources, new_to_old, mapping_strategy);
        }
        return run_benchmark<puercgp::algorithms::bfs_policy>(
            run_options, graph, graph_storage.vertices, original_sources,
            execution_sources, new_to_old, mapping_strategy);
      }
      if (run_options.algorithm == "sswp") {
        return run_benchmark<puercgp::algorithms::sswp_policy>(
            run_options, graph, graph_storage.vertices, original_sources,
            execution_sources, new_to_old, mapping_strategy);
      }
      return run_benchmark<puercgp::algorithms::sssp_policy>(
          run_options, graph, graph_storage.vertices, original_sources,
          execution_sources, new_to_old, mapping_strategy);
    };
    if (options.pull_bidirectional_rounds_grid.empty() &&
        options.pull_period_grid.empty() &&
        options.pull_edge_ratio_grid.empty()) {
      return run_once(options);
    }
    if (!options.pull_edge_ratio_grid.empty()) {
      const auto grid = options.pull_edge_ratio_grid;
      for (double ratio : grid) {
        options.pull_edge_ratio = ratio;
        options.pull_edge_ratio_grid.clear();
        std::cout << "CONFIG pull_edge_ratio=" << ratio << '\n';
        const int status = run_once(options);
        if (status != 0)
          return status;
      }
      return 0;
    }
    if (!options.pull_period_grid.empty()) {
      const auto grid = options.pull_period_grid;
      for (int period : grid) {
        options.pull_bidirectional_period = period;
        options.pull_period_grid.clear();
        std::cout << "CONFIG pull_period=" << period << '\n';
        const int status = run_once(options);
        if (status != 0)
          return status;
      }
      return 0;
    }
    const auto grid = options.pull_bidirectional_rounds_grid;
    for (int rounds : grid) {
      options.pull_bidirectional_rounds = rounds;
      options.pull_bidirectional_rounds_grid.clear();
      std::cout << "CONFIG pull_bidirectional_rounds=" << rounds << '\n';
      const int status = run_once(options);
      if (status != 0)
        return status;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
