#include <algorithm>
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
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

constexpr std::int32_t kReorderMagic = 0x52454f52;
constexpr int kThreadsPerBlock = 128;

enum class algorithm_t { bfs, sssp, sswp };

const char* algorithm_name(algorithm_t algorithm) {
  if (algorithm == algorithm_t::bfs) return "bfs";
  return algorithm == algorithm_t::sssp ? "sssp" : "sswp";
}

struct options_t {
  std::string graph_path;
  std::filesystem::path sources_path;
  std::filesystem::path mapping_path;
  std::filesystem::path fingerprints_path;
  int query_count = 64;
  int total_queries = 256;
  int warmups = 1;
  int repeats = 3;
  int max_iterations = 10000;
  algorithm_t algorithm = algorithm_t::sssp;
};

struct mapping_t {
  std::vector<int> new_to_old;
  std::vector<int> old_to_new;
};

struct fingerprint_t {
  unsigned long long sum = 0;
  unsigned long long xor_value = 0;

  bool operator==(const fingerprint_t& other) const {
    return sum == other.sum && xor_value == other.xor_value;
  }
};

struct measurement_t {
  float gpu_ms = 0.0f;
  int sweeps = 0;
  std::vector<fingerprint_t> fingerprints;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

options_t parse_options(int argc, char** argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "Usage: bench_pull_update_ablation <graph> <sources-file> "
        "[--mapping=path] [--fingerprints=path] [--queries=64] "
        "[--total-queries=256] [--warmups=1] [--repeats=3] "
        "[--algorithm=bfs|sssp|sswp]");
  }
  options_t options;
  options.graph_path = argv[1];
  options.sources_path = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value_after = [&](const std::string& prefix) {
      return argument.substr(prefix.size());
    };
    if (argument.rfind("--mapping=", 0) == 0) {
      options.mapping_path = value_after("--mapping=");
    } else if (argument.rfind("--fingerprints=", 0) == 0) {
      options.fingerprints_path = value_after("--fingerprints=");
    } else if (argument.rfind("--queries=", 0) == 0) {
      options.query_count = std::stoi(value_after("--queries="));
    } else if (argument.rfind("--total-queries=", 0) == 0) {
      options.total_queries = std::stoi(value_after("--total-queries="));
    } else if (argument.rfind("--warmups=", 0) == 0) {
      options.warmups = std::stoi(value_after("--warmups="));
    } else if (argument.rfind("--repeats=", 0) == 0) {
      options.repeats = std::stoi(value_after("--repeats="));
    } else if (argument.rfind("--max-iterations=", 0) == 0) {
      options.max_iterations = std::stoi(value_after("--max-iterations="));
    } else if (argument == "--algorithm=bfs") {
      options.algorithm = algorithm_t::bfs;
    } else if (argument == "--algorithm=sssp") {
      options.algorithm = algorithm_t::sssp;
    } else if (argument == "--algorithm=sswp") {
      options.algorithm = algorithm_t::sswp;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.query_count <= 0 || options.query_count > 64 ||
      kThreadsPerBlock % options.query_count != 0 ||
      options.total_queries <= 0 ||
      options.total_queries % options.query_count != 0 ||
      options.warmups < 0 || options.repeats <= 0 ||
      options.max_iterations <= 0) {
    throw std::invalid_argument("invalid benchmark options");
  }
  return options;
}

std::vector<int> load_sources(const std::filesystem::path& path, int count) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open sources file: " + path.string());
  }
  std::vector<int> sources;
  std::string line;
  while (std::getline(input, line) && static_cast<int>(sources.size()) < count) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream row(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(row, field, ',')) {
      fields.push_back(field);
    }
    try {
      sources.push_back(std::stoi(fields.size() >= 2 ? fields[1] : fields[0]));
    } catch (const std::exception&) {
      if (sources.empty()) {
        continue;
      }
      throw;
    }
  }
  if (static_cast<int>(sources.size()) != count) {
    throw std::runtime_error("sources file has fewer entries than requested");
  }
  return sources;
}

mapping_t load_mapping(const std::filesystem::path& path, int vertices) {
  mapping_t mapping;
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open mapping file: " + path.string());
  }
  std::int32_t magic = 0;
  std::int32_t count = 0;
  std::int32_t strategy = 0;
  input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  input.read(reinterpret_cast<char*>(&strategy), sizeof(strategy));
  if (!input || magic != kReorderMagic || count != vertices) {
    throw std::runtime_error("mapping header does not match graph");
  }
  mapping.new_to_old.resize(static_cast<std::size_t>(vertices));
  input.read(reinterpret_cast<char*>(mapping.new_to_old.data()),
             static_cast<std::streamsize>(mapping.new_to_old.size() *
                                          sizeof(int)));
  if (!input) {
    throw std::runtime_error("truncated mapping file");
  }
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

template <algorithm_t Algorithm>
__global__ void initialize_values_kernel(float* values,
                                         std::size_t value_count,
                                         const int* sources, int query_count,
                                         std::size_t vertex_count) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t index = tid; index < value_count; index += stride) {
    const std::size_t vertex = index / static_cast<std::size_t>(query_count);
    const int query = static_cast<int>(index % query_count);
    if constexpr (Algorithm != algorithm_t::sswp) {
      values[index] = vertex == static_cast<std::size_t>(sources[query])
                          ? 0.0f
                          : std::numeric_limits<float>::infinity();
    } else {
      values[index] = vertex == static_cast<std::size_t>(sources[query])
                          ? std::numeric_limits<float>::infinity()
                          : -std::numeric_limits<float>::infinity();
    }
  }
}

template <algorithm_t Algorithm, bool Synchronous, typename graph_t>
__global__ void pull_relax_kernel(graph_t graph, int query_count,
                                  const float* input_values,
                                  float* output_values,
                                  unsigned long long* changed_pairs) {
  __shared__ unsigned int changed_flags[kThreadsPerBlock];
  const std::size_t rows_per_block = blockDim.y;
  const std::size_t vertex =
      static_cast<std::size_t>(blockIdx.x) * rows_per_block + threadIdx.y;
  const int query = threadIdx.x;
  const int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  bool changed = false;

  if (vertex < static_cast<std::size_t>(graph.get_number_of_vertices()) &&
      query < query_count) {
    const std::size_t position =
        vertex * static_cast<std::size_t>(query_count) + query;
    const float current = input_values[position];
    float best = current;
    const auto begin = graph.get_starting_pull_edge(static_cast<int>(vertex));
    const auto end =
        graph.get_starting_pull_edge(static_cast<int>(vertex + 1));
    for (auto edge = begin; edge < end; ++edge) {
      const int neighbor = graph.get_pull_neighbor_vertex(edge);
      const float neighbor_value =
          input_values[static_cast<std::size_t>(neighbor) * query_count + query];
      if constexpr (Algorithm != algorithm_t::sswp) {
        if (isfinite(neighbor_value)) {
          best = fminf(best,
                       neighbor_value + graph.get_pull_edge_weight(edge));
        }
      } else {
        best = fmaxf(
            best, fminf(neighbor_value, graph.get_pull_edge_weight(edge)));
      }
    }
    if constexpr (Algorithm != algorithm_t::sswp) {
      changed = best < current;
    } else {
      changed = best > current;
    }
    if constexpr (Synchronous) {
      output_values[position] = best;
    } else if (changed) {
      output_values[position] = best;
    }
  }

  changed_flags[shared_index] = changed ? 1U : 0U;
  __syncthreads();
  if (shared_index == 0) {
    unsigned int block_changes = 0;
    for (int index = 0; index < blockDim.x * blockDim.y; ++index) {
      block_changes += changed_flags[index];
    }
    if (block_changes != 0) {
      atomicAdd(changed_pairs,
                static_cast<unsigned long long>(block_changes));
    }
  }
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

__global__ void fingerprint_kernel(const float* values,
                                   std::size_t vertex_count, int query_count,
                                   const int* new_to_old,
                                   unsigned long long* sums,
                                   unsigned long long* xor_values) {
  const int query = threadIdx.x;
  if (query >= query_count) {
    return;
  }
  unsigned long long local_sum = 0;
  unsigned long long local_xor = 0;
  for (std::size_t vertex = blockIdx.x; vertex < vertex_count;
       vertex += gridDim.x) {
    const auto old_vertex = static_cast<unsigned long long>(
        new_to_old == nullptr ? vertex : new_to_old[vertex]);
    const auto bits = static_cast<unsigned long long>(
        __float_as_uint(values[vertex * static_cast<std::size_t>(query_count) +
                               query]));
    const auto mixed = mix64((old_vertex << 32) ^ bits ^
                             0x9e3779b97f4a7c15ULL);
    local_sum += mixed;
    local_xor ^= mixed;
  }
  atomicAdd(sums + query, local_sum);
  atomicXor(xor_values + query, local_xor);
}

std::vector<fingerprint_t> fingerprint_values(
    const float* values, std::size_t vertex_count, int query_count,
    const thrust::device_vector<int>* new_to_old, cudaStream_t stream) {
  thrust::device_vector<unsigned long long> sums(query_count, 0);
  thrust::device_vector<unsigned long long> xor_values(query_count, 0);
  const int blocks =
      std::max(1, std::min(4096, static_cast<int>(vertex_count)));
  fingerprint_kernel<<<blocks, query_count, 0, stream>>>(
      values, vertex_count, query_count,
      new_to_old == nullptr ? nullptr
                            : thrust::raw_pointer_cast(new_to_old->data()),
      thrust::raw_pointer_cast(sums.data()),
      thrust::raw_pointer_cast(xor_values.data()));
  check_cuda(cudaGetLastError(), "launch fingerprint kernel");
  thrust::host_vector<unsigned long long> host_sums = sums;
  thrust::host_vector<unsigned long long> host_xors = xor_values;
  std::vector<fingerprint_t> result(static_cast<std::size_t>(query_count));
  for (int query = 0; query < query_count; ++query) {
    result[query] = {host_sums[query], host_xors[query]};
  }
  return result;
}

template <algorithm_t Algorithm, bool Synchronous, typename graph_t>
measurement_t run_once(graph_t graph, const std::vector<int>& sources,
                       const thrust::device_vector<int>* new_to_old,
                       int max_iterations, bool collect_fingerprints,
                       cudaStream_t stream) {
  const int query_count = static_cast<int>(sources.size());
  const std::size_t vertex_count =
      static_cast<std::size_t>(graph.get_number_of_vertices());
  const std::size_t value_count = vertex_count * query_count;
  thrust::device_vector<float> values_a(value_count);
  thrust::device_vector<float> values_b(Synchronous ? value_count : 0);
  thrust::device_vector<int> device_sources(sources);
  thrust::device_vector<unsigned long long> changed_pairs(1);
  const int init_blocks = std::max(
      1, std::min(65535, static_cast<int>((value_count + 255) / 256)));
  initialize_values_kernel<Algorithm><<<init_blocks, 256, 0, stream>>>(
      thrust::raw_pointer_cast(values_a.data()), value_count,
      thrust::raw_pointer_cast(device_sources.data()), query_count,
      vertex_count);
  check_cuda(cudaGetLastError(), "launch initialize values kernel");
  check_cuda(cudaStreamSynchronize(stream), "initialize values");

  float* current = thrust::raw_pointer_cast(values_a.data());
  float* next = Synchronous ? thrust::raw_pointer_cast(values_b.data()) : current;
  const dim3 block(static_cast<unsigned int>(query_count),
                   static_cast<unsigned int>(kThreadsPerBlock / query_count));
  const int grid = static_cast<int>(
      (vertex_count + block.y - 1) / static_cast<std::size_t>(block.y));

  cudaEvent_t start;
  cudaEvent_t stop;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");
  check_cuda(cudaEventRecord(start, stream), "record start event");

  measurement_t result;
  bool converged = false;
  for (; result.sweeps < max_iterations; ++result.sweeps) {
    check_cuda(cudaMemsetAsync(thrust::raw_pointer_cast(changed_pairs.data()), 0,
                               sizeof(unsigned long long), stream),
               "reset changed pair count");
    pull_relax_kernel<Algorithm, Synchronous><<<grid, block, 0, stream>>>(
        graph, query_count, current, next,
        thrust::raw_pointer_cast(changed_pairs.data()));
    check_cuda(cudaGetLastError(), "launch pull relaxation kernel");
    unsigned long long host_changed_pairs = 0;
    check_cuda(cudaMemcpyAsync(&host_changed_pairs,
                               thrust::raw_pointer_cast(changed_pairs.data()),
                               sizeof(host_changed_pairs),
                               cudaMemcpyDeviceToHost, stream),
               "copy changed pair count");
    check_cuda(cudaStreamSynchronize(stream), "synchronize pull iteration");
    if constexpr (Synchronous) {
      std::swap(current, next);
    }
    if (host_changed_pairs == 0) {
      ++result.sweeps;
      converged = true;
      break;
    }
  }
  check_cuda(cudaEventRecord(stop, stream), "record stop event");
  check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
  check_cuda(cudaEventElapsedTime(&result.gpu_ms, start, stop),
             "measure GPU time");
  check_cuda(cudaEventDestroy(start), "destroy start event");
  check_cuda(cudaEventDestroy(stop), "destroy stop event");
  if (!converged) {
    throw std::runtime_error("pull relaxation did not converge");
  }
  if (collect_fingerprints) {
    result.fingerprints = fingerprint_values(
        current, vertex_count, query_count, new_to_old, stream);
  }
  return result;
}

template <algorithm_t Algorithm, bool Synchronous, typename graph_t>
measurement_t run_all_batches(graph_t graph,
                              const std::vector<int>& execution_sources,
                              const thrust::device_vector<int>* new_to_old,
                              const options_t& options,
                              bool collect_fingerprints,
                              cudaStream_t stream) {
  measurement_t total;
  for (int first = 0; first < options.total_queries;
       first += options.query_count) {
    std::vector<int> batch(execution_sources.begin() + first,
                           execution_sources.begin() + first +
                               options.query_count);
    auto result = run_once<Algorithm, Synchronous>(
        graph, batch, new_to_old, options.max_iterations,
        collect_fingerprints, stream);
    total.gpu_ms += result.gpu_ms;
    total.sweeps += result.sweeps;
    total.fingerprints.insert(total.fingerprints.end(),
                              result.fingerprints.begin(),
                              result.fingerprints.end());
  }
  return total;
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int median(std::vector<int> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void save_fingerprints(const std::filesystem::path& path,
                       const std::vector<int>& original_sources,
                       const std::vector<fingerprint_t>& inplace,
                       const std::vector<fingerprint_t>& synchronous) {
  if (path.empty()) {
    return;
  }
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  output << "query,source,inplace_sum,inplace_xor,sync_sum,sync_xor,match\n";
  for (std::size_t query = 0; query < inplace.size(); ++query) {
    const bool match = inplace[query].sum == synchronous[query].sum &&
                       inplace[query].xor_value ==
                           synchronous[query].xor_value;
    output << query << ',' << original_sources[query] << ','
           << inplace[query].sum << ',' << inplace[query].xor_value << ','
           << synchronous[query].sum << ',' << synchronous[query].xor_value
           << ',' << (match ? "PASS" : "FAIL") << '\n';
  }
}

template <algorithm_t Algorithm, typename graph_t>
int run_benchmark(const options_t& options, graph_t graph,
                  const std::vector<int>& original_sources,
                  const mapping_t* mapping, cudaStream_t stream) {
  std::vector<int> execution_sources = original_sources;
  thrust::device_vector<int> device_new_to_old;
  if (mapping != nullptr) {
    for (int& source : execution_sources) {
      source = mapping->old_to_new[static_cast<std::size_t>(source)];
    }
    device_new_to_old = mapping->new_to_old;
  }
  const auto* device_mapping =
      mapping == nullptr ? nullptr : &device_new_to_old;

  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    (void)run_all_batches<Algorithm, false>(
        graph, execution_sources, device_mapping, options, false, stream);
    (void)run_all_batches<Algorithm, true>(
        graph, execution_sources, device_mapping, options, false, stream);
  }

  std::vector<float> inplace_times;
  std::vector<float> synchronous_times;
  std::vector<int> inplace_sweeps;
  std::vector<int> synchronous_sweeps;
  measurement_t last_inplace;
  measurement_t last_synchronous;
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    const bool collect = repeat + 1 == options.repeats;
    last_inplace = run_all_batches<Algorithm, false>(
        graph, execution_sources, device_mapping, options, collect, stream);
    last_synchronous = run_all_batches<Algorithm, true>(
        graph, execution_sources, device_mapping, options, collect, stream);
    inplace_times.push_back(last_inplace.gpu_ms);
    synchronous_times.push_back(last_synchronous.gpu_ms);
    inplace_sweeps.push_back(last_inplace.sweeps);
    synchronous_sweeps.push_back(last_synchronous.sweeps);
    std::cout << "REPEAT repeat=" << repeat
              << " inplace_sweeps=" << last_inplace.sweeps
              << " inplace_gpu_ms=" << last_inplace.gpu_ms
              << " synchronous_sweeps=" << last_synchronous.sweeps
              << " synchronous_gpu_ms=" << last_synchronous.gpu_ms << '\n';
  }

  const bool correct =
      last_inplace.fingerprints == last_synchronous.fingerprints;
  save_fingerprints(options.fingerprints_path, original_sources,
                    last_inplace.fingerprints,
                    last_synchronous.fingerprints);
  std::cout << std::setprecision(9)
            << "RESULT graph=" << options.graph_path
            << " algorithm="
            << algorithm_name(Algorithm)
            << " vertices=" << graph.get_number_of_vertices()
            << " edges=" << graph.get_number_of_edges()
            << " N=" << options.total_queries << " Q=" << options.query_count
            << " inplace_sweeps=" << median(inplace_sweeps)
            << " inplace_gpu_ms=" << median(inplace_times)
            << " synchronous_sweeps=" << median(synchronous_sweeps)
            << " synchronous_gpu_ms=" << median(synchronous_times)
            << " correctness=" << (correct ? "PASS" : "FAIL") << '\n';
  return correct ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    auto graph_storage =
        puercgp_examples::load_graph_auto(options.graph_path, true);
    const auto original_sources =
        load_sources(options.sources_path, options.total_queries);
    mapping_t mapping;
    const mapping_t* mapping_ptr = nullptr;
    if (!options.mapping_path.empty()) {
      mapping = load_mapping(options.mapping_path, graph_storage.vertices);
      mapping_ptr = &mapping;
    }
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "create stream");
    int status = 0;
    if (options.algorithm == algorithm_t::bfs) {
      status = run_benchmark<algorithm_t::bfs>(
          options, graph_storage.view(), original_sources, mapping_ptr, stream);
    } else if (options.algorithm == algorithm_t::sssp) {
      status = run_benchmark<algorithm_t::sssp>(
          options, graph_storage.view(), original_sources, mapping_ptr, stream);
    } else {
      status = run_benchmark<algorithm_t::sswp>(
          options, graph_storage.view(), original_sources, mapping_ptr, stream);
    }
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    return status;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
