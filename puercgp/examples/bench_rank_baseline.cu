#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

struct benchmark_options {
  std::string graph_path;
  std::filesystem::path output_dir;
  std::filesystem::path sources_path;
  std::string algorithm = "pagerank";
  int total_queries = 256;
  int batch_size = 32;
  int iterations = 10;
  int warmups = 2;
  int repeats = 5;
};

struct measurement {
  std::string method;
  int repeat = -1;
  float gpu_ms = 0.0f;
  float engine_wall_ms = 0.0f;
  double runner_wall_ms = 0.0;
  int iterations = 0;
  bool correct = true;
  float max_sample_abs_delta = 0.0f;
  float max_sample_rel_delta = 0.0f;
  std::vector<float> samples;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

benchmark_options parse_options(int argc, char** argv) {
  if (argc < 3) {
    throw std::invalid_argument(
        "usage: bench_rank_baseline GRAPH OUTPUT_DIR "
        "--algorithm=pagerank|ppr --sources=PATH [--total-queries=256] "
        "[--batch-size=32] [--iterations=10] [--warmups=2] [--repeats=5]");
  }
  benchmark_options options;
  options.graph_path = argv[1];
  options.output_dir = argv[2];
  for (int index = 3; index < argc; ++index) {
    const std::string argument = argv[index];
    auto integer_value = [&](const std::string& prefix) {
      return std::stoi(argument.substr(prefix.size()));
    };
    if (argument.rfind("--algorithm=", 0) == 0) {
      options.algorithm = argument.substr(12);
    } else if (argument.rfind("--sources=", 0) == 0) {
      options.sources_path = argument.substr(10);
    } else if (argument.rfind("--total-queries=", 0) == 0) {
      options.total_queries = integer_value("--total-queries=");
    } else if (argument.rfind("--batch-size=", 0) == 0) {
      options.batch_size = integer_value("--batch-size=");
    } else if (argument.rfind("--iterations=", 0) == 0) {
      options.iterations = integer_value("--iterations=");
    } else if (argument.rfind("--warmups=", 0) == 0) {
      options.warmups = integer_value("--warmups=");
    } else if (argument.rfind("--repeats=", 0) == 0) {
      options.repeats = integer_value("--repeats=");
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.algorithm != "pagerank" && options.algorithm != "ppr") {
    throw std::invalid_argument("algorithm must be pagerank or ppr");
  }
  if (options.sources_path.empty() || options.total_queries <= 0 ||
      options.batch_size <= 0 || options.batch_size > 64 ||
      options.total_queries % options.batch_size != 0 ||
      options.iterations <= 0 || options.warmups < 0 || options.repeats <= 0) {
    throw std::invalid_argument("invalid rank benchmark configuration");
  }
  return options;
}

std::vector<int> load_sources(const std::filesystem::path& path,
                              int total_queries) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open sources file: " + path.string());
  }
  std::vector<int> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream row(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(row, field, ',')) fields.push_back(field);
    try {
      result.push_back(std::stoi(fields.size() >= 2 ? fields[1] : fields[0]));
    } catch (const std::exception&) {
      if (result.empty()) continue;
      throw;
    }
  }
  if (static_cast<int>(result.size()) != total_queries) {
    throw std::runtime_error("sources file does not contain the requested N");
  }
  return result;
}

bool sampled_csr_is_symmetric(const puercgp_examples::host_csr_graph& graph) {
  constexpr std::size_t target_samples = 100000;
  const std::size_t edge_count = static_cast<std::size_t>(graph.edges);
  const std::size_t stride = std::max<std::size_t>(1, edge_count / target_samples);
  std::size_t next_sample = 0;
  for (int source = 0; source < graph.vertices && next_sample < edge_count;
       ++source) {
    const std::size_t begin = graph.row_offsets[source];
    const std::size_t end = graph.row_offsets[source + 1];
    while (next_sample < begin) next_sample += stride;
    for (; next_sample < end; next_sample += stride) {
      const int destination = graph.column_indices[next_sample];
      const auto reverse_begin = graph.column_indices.begin() +
          graph.row_offsets[static_cast<std::size_t>(destination)];
      const auto reverse_end = graph.column_indices.begin() +
          graph.row_offsets[static_cast<std::size_t>(destination) + 1];
      if (!std::binary_search(reverse_begin, reverse_end, source)) return false;
    }
  }
  return true;
}

std::vector<float> sample_values(const thrust::device_vector<float>& values,
                                 std::size_t vertex_count, int query_count,
                                 cudaStream_t stream) {
  constexpr std::size_t sample_count = 16;
  std::vector<float> samples(sample_count * query_count);
  const float* data = thrust::raw_pointer_cast(values.data());
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const std::size_t vertex =
        sample * (vertex_count - 1) / (sample_count - 1);
    check_cuda(cudaMemcpyAsync(samples.data() + sample * query_count,
                               data + vertex * query_count,
                               query_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "copy sampled rank values");
  }
  check_cuda(cudaStreamSynchronize(stream), "synchronize sampled rank values");
  return samples;
}

template <typename Policy>
auto make_queries(const std::vector<int>& sources, int first, int last) {
  if constexpr (std::is_same_v<Policy, puercgp::algorithms::pagerank_policy>) {
    std::vector<puercgp::algorithms::pagerank_query> descriptors(last - first);
    for (auto& query : descriptors) {
      query.damping_factor = 0.85f;
      query.epsilon = 1.0e-30f;
    }
    return puercgp::algorithms::pagerank_query_batch(std::move(descriptors));
  } else {
    std::vector<puercgp::algorithms::ppr_query> descriptors;
    descriptors.reserve(last - first);
    for (int query = first; query < last; ++query) {
      descriptors.push_back({sources[query], 0.85f, 1.0e-30f});
    }
    return puercgp::algorithms::ppr_query_batch(std::move(descriptors));
  }
}

template <typename Policy, typename Graph>
measurement execute(Graph& graph, const std::vector<int>& sources,
                    const benchmark_options& benchmark,
                    puercgp::traversal_mode_t mode, int repeat,
                    bool collect_samples) {
  puercgp::execution_context context;
  puercgp::run_options options;
  options.traversal_mode = mode;
  options.profile_iterations = false;
  options.max_iterations = benchmark.iterations;
  options.fixed_iterations = true;
  options.max_queries = 64;

  measurement result;
  result.method = puercgp::traversal_mode_name(mode);
  result.repeat = repeat;
  const auto wall_start = std::chrono::steady_clock::now();
  for (int first = 0; first < benchmark.total_queries;
       first += benchmark.batch_size) {
    const int last = first + benchmark.batch_size;
    auto queries = make_queries<Policy>(sources, first, last);
    auto batch = puercgp::run<Policy>(graph, queries, context, options);
    if (batch.iterations != benchmark.iterations) {
      throw std::runtime_error(result.method + " stopped before fixed iteration count");
    }
    result.gpu_ms += batch.gpu_time_ms;
    result.engine_wall_ms += batch.wall_time_ms;
    result.iterations += batch.iterations;
    if (collect_samples) {
      auto batch_samples = sample_values(
          batch.values, static_cast<std::size_t>(graph.get_number_of_vertices()),
          benchmark.batch_size, context.stream());
      result.samples.insert(result.samples.end(), batch_samples.begin(),
                            batch_samples.end());
    }
  }
  result.runner_wall_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - wall_start)
                              .count();
  return result;
}

void compare_samples(const measurement& reference, measurement& candidate) {
  if (reference.samples.size() != candidate.samples.size()) {
    candidate.correct = false;
    return;
  }
  for (std::size_t index = 0; index < reference.samples.size(); ++index) {
    const float absolute =
        std::fabs(reference.samples[index] - candidate.samples[index]);
    const float scale = std::max(
        {std::fabs(reference.samples[index]), std::fabs(candidate.samples[index]),
         1.0e-30f});
    candidate.max_sample_abs_delta =
        std::max(candidate.max_sample_abs_delta, absolute);
    candidate.max_sample_rel_delta =
        std::max(candidate.max_sample_rel_delta, absolute / scale);
  }
  candidate.correct = candidate.max_sample_abs_delta <= 1.0e-5f ||
      candidate.max_sample_rel_delta <= 1.0e-4f;
}

template <typename Policy, typename Graph>
int run_benchmark(Graph& graph, const std::vector<int>& sources,
                  const benchmark_options& options) {
  const std::vector<puercgp::traversal_mode_t> modes = {
      puercgp::traversal_mode_t::push,
      puercgp::traversal_mode_t::hybrid,
      puercgp::traversal_mode_t::pull};
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    for (const auto mode : modes) {
      (void)execute<Policy>(graph, sources, options, mode, -1, false);
    }
  }

  std::vector<measurement> measurements;
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    std::vector<measurement> current;
    current.reserve(modes.size());
    if ((repeat & 1) == 0) {
      for (const auto mode : modes) {
        current.push_back(execute<Policy>(graph, sources, options, mode,
                                          repeat, repeat == 0));
      }
    } else {
      for (auto iterator = modes.rbegin(); iterator != modes.rend(); ++iterator) {
        current.push_back(execute<Policy>(graph, sources, options, *iterator,
                                          repeat, false));
      }
    }
    if (repeat == 0) {
      auto reference = std::find_if(current.begin(), current.end(),
          [](const measurement& value) { return value.method == "push"; });
      for (auto& value : current) {
        if (value.method != "push") compare_samples(*reference, value);
        if (!value.correct) {
          throw std::runtime_error(value.method + " rank sample mismatch");
        }
      }
    }
    for (const auto& value : current) {
      std::cout << "algorithm=" << options.algorithm
                << " method=" << value.method << " repeat=" << repeat
                << " gpu_ms=" << value.gpu_ms
                << " wall_ms=" << value.runner_wall_ms
                << " correct=" << (value.correct ? "yes" : "no") << '\n';
    }
    measurements.insert(measurements.end(), current.begin(), current.end());
  }

  std::filesystem::create_directories(options.output_dir);
  std::ofstream output(options.output_dir / "runs.csv");
  output << "algorithm,method,repeat,gpu_ms,engine_wall_ms,runner_wall_ms,"
            "iterations,correct,max_sample_abs_delta,max_sample_rel_delta\n";
  for (const auto& value : measurements) {
    output << options.algorithm << ',' << value.method << ',' << value.repeat
           << ',' << value.gpu_ms << ',' << value.engine_wall_ms << ','
           << value.runner_wall_ms << ',' << value.iterations << ','
           << (value.correct ? 1 : 0) << ',' << value.max_sample_abs_delta
           << ',' << value.max_sample_rel_delta << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    auto sources = load_sources(options.sources_path, options.total_queries);
    auto storage = puercgp_examples::load_graph_auto(options.graph_path, false);
    if (!sampled_csr_is_symmetric(storage)) {
      throw std::runtime_error("rank baseline requires symmetric CSR input");
    }
    auto graph = storage.view();
    graph.pull_row_offsets = graph.row_offsets;
    graph.pull_column_indices = graph.column_indices;
    graph.pull_edge_weights = graph.edge_weights;
    for (int source : sources) {
      if (source < 0 || source >= storage.vertices) {
        throw std::runtime_error("source is outside graph");
      }
    }
    if (options.algorithm == "pagerank") {
      return run_benchmark<puercgp::algorithms::pagerank_policy>(
          graph, sources, options);
    }
    return run_benchmark<puercgp::algorithms::ppr_policy>(
        graph, sources, options);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
