#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

struct benchmark_options {
  std::string graph_path;
  std::filesystem::path index_path;
  std::filesystem::path output_dir;
  std::filesystem::path sources_path;
  std::string algorithm = "bfs";
  std::string mode = "hybrid";
  std::string online_policy = "auto";
  std::string measure = "both";
  int total_queries = 256;
  int batch_size = 64;
  int repeats = 5;
  int warmups = 2;
  int landmarks = 32;
  int batch_swaps = 16;
  int max_offset = 16;
  unsigned int seed = 42;
  bool rebuild_index = false;
  bool validate = true;
  bool include_zero_offset = false;
  bool full_ablation = false;
};

struct query_fingerprint {
  unsigned long long sum = 0;
  unsigned long long xor_value = 0;

  bool operator==(const query_fingerprint& other) const {
    return sum == other.sum && xor_value == other.xor_value;
  }
};

struct run_measurement {
  std::string method;
  int repeat = 0;
  double gpu_ms = 0.0;
  double runner_wall_ms = 0.0;
  double evaluator_ms = 0.0;
  double grouping_ms = 0.0;
  double offset_ms = 0.0;
  std::uint64_t alignment_score = 0;
  int iterations = 0;
  int push_iterations = 0;
  int pull_iterations = 0;
  bool correct = true;

  double end_to_end_ms() const { return gpu_ms + evaluator_ms; }
};

__device__ __forceinline__ unsigned long long mix64(
    unsigned long long value) {
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
__global__ void fingerprint_values_kernel(
    const value_t* values, std::size_t vertex_count, int query_count,
    unsigned long long* sums, unsigned long long* xor_values) {
  int query = threadIdx.x;
  if (query >= query_count) return;
  unsigned long long local_sum = 0;
  unsigned long long local_xor = 0;
  for (std::size_t vertex = blockIdx.x; vertex < vertex_count;
       vertex += gridDim.x) {
    const auto bits = value_bits(values[vertex * query_count + query]);
    const auto mixed = mix64(
        (static_cast<unsigned long long>(vertex) << 32) ^ bits ^
        0x9e3779b97f4a7c15ULL);
    local_sum += mixed;
    local_xor ^= mixed;
  }
  atomicAdd(sums + query, local_sum);
  atomicXor(xor_values + query, local_xor);
}

template <typename value_t>
std::vector<query_fingerprint> fingerprint_values(
  const thrust::device_vector<value_t>& values, std::size_t vertex_count,
    int query_count, cudaStream_t stream) {
  thrust::device_vector<unsigned long long> sums(query_count);
  thrust::device_vector<unsigned long long> xor_values(query_count);
  cudaMemsetAsync(thrust::raw_pointer_cast(sums.data()), 0,
                  query_count * sizeof(unsigned long long), stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(xor_values.data()), 0,
                  query_count * sizeof(unsigned long long), stream);
  const int blocks = std::max(
      1, std::min(4096, static_cast<int>(vertex_count)));
  fingerprint_values_kernel<<<blocks, query_count, 0, stream>>>(
      thrust::raw_pointer_cast(values.data()), vertex_count, query_count,
      thrust::raw_pointer_cast(sums.data()),
      thrust::raw_pointer_cast(xor_values.data()));
  auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
  std::vector<unsigned long long> host_sums(query_count);
  std::vector<unsigned long long> host_xor(query_count);
  cudaMemcpyAsync(host_sums.data(), thrust::raw_pointer_cast(sums.data()),
                  query_count * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(host_xor.data(),
                  thrust::raw_pointer_cast(xor_values.data()),
                  query_count * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::vector<query_fingerprint> result(query_count);
  for (int query = 0; query < query_count; ++query) {
    result[query] = {host_sums[query], host_xor[query]};
  }
  return result;
}

puercgp::traversal_mode_t parse_mode(const std::string& value) {
  if (value == "push") return puercgp::traversal_mode_t::push;
  if (value == "pull") return puercgp::traversal_mode_t::pull;
  if (value == "hybrid") return puercgp::traversal_mode_t::hybrid;
  throw std::invalid_argument("mode must be push, pull, or hybrid");
}

benchmark_options parse_options(int argc, char** argv) {
  if (argc < 4) {
    throw std::invalid_argument(
        "Usage: bench_online_runner <graph> <phase-index> <output-dir> "
        "[--algorithm=bfs|sssp|sswp] [--mode=push|pull|hybrid] "
        "[--online-policy=auto|full|offset-only|batch-only] "
        "[--measure=both|baseline-only|online-only] "
        "[--total-queries=256] [--batch-size=64] [--repeats=5] "
        "[--warmups=2] [--sources=path] [--seed=42] [--landmarks=32] "
        "[--batch-swaps=16] [--max-offset=16] [--rebuild-index] "
        "[--no-validate] [--include-zero-offset] [--full-ablation]");
  }
  benchmark_options options;
  options.graph_path = argv[1];
  options.index_path = argv[2];
  options.output_dir = argv[3];
  for (int i = 4; i < argc; ++i) {
    const std::string argument = argv[i];
    auto integer_value = [&](const std::string& prefix) {
      return std::stoi(argument.substr(prefix.size()));
    };
    if (argument.rfind("--algorithm=", 0) == 0)
      options.algorithm = argument.substr(12);
    else if (argument.rfind("--mode=", 0) == 0)
      options.mode = argument.substr(7);
    else if (argument.rfind("--online-policy=", 0) == 0)
      options.online_policy = argument.substr(16);
    else if (argument.rfind("--measure=", 0) == 0)
      options.measure = argument.substr(10);
    else if (argument.rfind("--total-queries=", 0) == 0)
      options.total_queries = integer_value("--total-queries=");
    else if (argument.rfind("--batch-size=", 0) == 0)
      options.batch_size = integer_value("--batch-size=");
    else if (argument.rfind("--repeats=", 0) == 0)
      options.repeats = integer_value("--repeats=");
    else if (argument.rfind("--warmups=", 0) == 0)
      options.warmups = integer_value("--warmups=");
    else if (argument.rfind("--sources=", 0) == 0)
      options.sources_path = argument.substr(10);
    else if (argument.rfind("--seed=", 0) == 0)
      options.seed = static_cast<unsigned int>(integer_value("--seed="));
    else if (argument.rfind("--landmarks=", 0) == 0)
      options.landmarks = integer_value("--landmarks=");
    else if (argument.rfind("--batch-swaps=", 0) == 0)
      options.batch_swaps = integer_value("--batch-swaps=");
    else if (argument.rfind("--max-offset=", 0) == 0)
      options.max_offset = integer_value("--max-offset=");
    else if (argument == "--rebuild-index")
      options.rebuild_index = true;
    else if (argument == "--no-validate")
      options.validate = false;
    else if (argument == "--include-zero-offset")
      options.include_zero_offset = true;
    else if (argument == "--full-ablation")
      options.full_ablation = true;
    else
      throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.total_queries <= 0 || options.batch_size <= 0 ||
      options.batch_size > 64 || options.repeats <= 0 ||
      options.warmups < 0 || options.landmarks <= 0 ||
      options.batch_swaps < 0 || options.max_offset < 0) {
    throw std::invalid_argument("invalid benchmark configuration");
  }
  parse_mode(options.mode);
  if (options.online_policy != "auto" && options.online_policy != "full" &&
      options.online_policy != "offset-only" &&
      options.online_policy != "batch-only") {
    throw std::invalid_argument("invalid online policy");
  }
  if (options.measure != "both" && options.measure != "baseline-only" &&
      options.measure != "online-only") {
    throw std::invalid_argument("invalid measure selection");
  }
  if (options.full_ablation && options.measure != "both") {
    throw std::invalid_argument("full ablation requires --measure=both");
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

std::vector<int> generate_sources(
    const puercgp_examples::host_csr_graph& graph, int total_queries,
    unsigned int seed) {
  std::vector<int> candidates;
  candidates.reserve(graph.vertices);
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    if (graph.row_offsets[vertex] != graph.row_offsets[vertex + 1]) {
      candidates.push_back(vertex);
    }
  }
  if (total_queries > static_cast<int>(candidates.size())) {
    throw std::invalid_argument("N exceeds the non-isolated vertex count");
  }
  std::mt19937 random(seed);
  std::shuffle(candidates.begin(), candidates.end(), random);
  candidates.resize(total_queries);
  return candidates;
}

void save_sources(const std::filesystem::path& path,
                  const std::vector<int>& sources) {
  std::ofstream output(path);
  output << "query_id,source\n";
  for (int query = 0; query < static_cast<int>(sources.size()); ++query) {
    output << query << ',' << sources[query] << '\n';
  }
}

template <typename Policy, typename graph_t>
run_measurement execute_batches(
    graph_t& graph, std::size_t vertex_count, puercgp::execution_context& context,
    const puercgp::run_options& run_options,
    const std::vector<puercgp::scheduling::online_execution_batch>& batches,
    const std::string& method, int repeat, bool scheduled,
    bool collect_fingerprints,
    std::vector<query_fingerprint>* fingerprints) {
  run_measurement measurement;
  measurement.method = method;
  measurement.repeat = repeat;
  auto wall_start = std::chrono::steady_clock::now();
  for (int batch_id = 0; batch_id < static_cast<int>(batches.size());
       ++batch_id) {
    const auto& batch = batches[batch_id];
    puercgp::query_batch<int> queries(batch.sources);
    auto result = scheduled
        ? puercgp::run_scheduled<Policy>(graph, queries, context,
                                         batch.start_schedule, run_options)
        : puercgp::run<Policy>(graph, queries, context, run_options);
    measurement.gpu_ms += result.gpu_time_ms;
    measurement.iterations += result.iterations;
    for (const auto& iteration_mode : result.iteration_modes) {
      if (iteration_mode == "pull")
        ++measurement.pull_iterations;
      else
        ++measurement.push_iterations;
    }
    if (collect_fingerprints) {
      auto batch_fingerprints = fingerprint_values(
          result.values, vertex_count, static_cast<int>(batch.sources.size()),
          context.stream());
      for (int slot = 0; slot < static_cast<int>(batch.query_ids.size());
           ++slot) {
        (*fingerprints)[batch.query_ids[slot]] = batch_fingerprints[slot];
      }
    }
    std::cout << "method=" << method << " repeat=" << repeat
              << " batch=" << batch_id + 1 << '/' << batches.size()
              << " gpu_ms=" << result.gpu_time_ms
              << " iterations=" << result.iterations << '\n';
  }
  measurement.runner_wall_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - wall_start)
                                   .count();
  return measurement;
}

template <typename Policy, typename graph_t>
int run_benchmark(
    const benchmark_options& options,
    const puercgp_examples::host_csr_graph& graph_storage, graph_t& graph,
    const std::vector<int>& sources,
    const puercgp::scheduling::landmark_phase_index& phase_index,
    double preprocessing_ms, bool built_index) {
  puercgp::execution_context context;
  puercgp::run_options run_options;
  run_options.traversal_mode = parse_mode(options.mode);
  run_options.max_queries = 64;

  puercgp::scheduling::online_offset_evaluator evaluator(
      phase_index, options.max_offset);
  auto sequential = puercgp::scheduling::make_sequential_execution_plan(
      sources, options.batch_size);
  const auto gate_start = std::chrono::steady_clock::now();
  double mean_predicted_length = 0.0;
  for (int source : sources) {
    mean_predicted_length += phase_index.estimate_phase_length(source);
  }
  mean_predicted_length /= static_cast<double>(sources.size());
  const double gate_ms = options.online_policy == "auto"
      ? std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gate_start)
            .count()
      : 0.0;
  std::string selected_policy = options.online_policy;
  if (selected_policy == "auto") {
    if (options.algorithm != "bfs" ||
        mean_predicted_length < static_cast<double>(options.max_offset)) {
      selected_policy = "disabled";
    } else {
      selected_policy = "full";
    }
  }
  auto make_selected_plan = [&]() {
    if (selected_policy == "disabled") {
      puercgp::scheduling::online_execution_plan plan;
      plan.batches = sequential;
      plan.grouping_ms = gate_ms;
      return plan;
    }
    if (selected_policy == "full") {
      auto plan = puercgp::scheduling::make_online_execution_plan(
          evaluator, sources, options.batch_size, options.batch_swaps);
      plan.grouping_ms += gate_ms;
      return plan;
    }
    if (selected_policy == "offset-only") {
      auto plan = puercgp::scheduling::make_online_offset_execution_plan(
          evaluator, sequential);
      plan.grouping_ms += gate_ms;
      return plan;
    }
    auto plan = puercgp::scheduling::make_online_batching_execution_plan(
        evaluator, sources, options.batch_size, options.batch_swaps);
    plan.grouping_ms += gate_ms;
    return plan;
  };
  const bool selected_policy_is_scheduled =
      selected_policy != "batch-only" && selected_policy != "disabled";
  std::cout << "selected_online_policy=" << selected_policy
            << " mean_predicted_length=" << mean_predicted_length
            << " gate_ms=" << gate_ms << '\n';

  auto run_warmup = [&]() {
    if (options.measure != "online-only") {
      auto baseline = execute_batches<Policy>(
          graph, graph_storage.vertices, context, run_options, sequential,
          "warmup_baseline", -1, false, false, nullptr);
      (void)baseline;
    }
    if (options.measure != "baseline-only") {
      auto plan = make_selected_plan();
      auto online = execute_batches<Policy>(
          graph, graph_storage.vertices, context, run_options, plan.batches,
          "warmup_online", -1, selected_policy_is_scheduled, false, nullptr);
      (void)online;
    }
  };
  for (int warmup = 0; warmup < options.warmups; ++warmup) run_warmup();

  std::vector<run_measurement> measurements;
  std::vector<query_fingerprint> baseline_fingerprints(
      options.total_queries);
  bool baseline_fingerprints_ready = false;
  puercgp::scheduling::online_execution_plan recorded_plan;

  if (options.measure == "online-only" && options.validate) {
    auto reference = execute_batches<Policy>(
        graph, graph_storage.vertices, context, run_options, sequential,
        "correctness_reference", -1, false, true, &baseline_fingerprints);
    (void)reference;
    baseline_fingerprints_ready = true;
  }

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    auto run_baseline = [&]() {
      const bool collect = options.validate && !baseline_fingerprints_ready;
      auto measurement = execute_batches<Policy>(
          graph, graph_storage.vertices, context, run_options, sequential,
          "baseline", repeat, false, collect, &baseline_fingerprints);
      if (collect) baseline_fingerprints_ready = true;
      if (collect && !baseline_fingerprints.empty()) {
        std::cout << "baseline_fingerprint_q0="
                  << baseline_fingerprints[0].sum << ':'
                  << baseline_fingerprints[0].xor_value << '\n';
      }
      measurements.push_back(measurement);
      std::cout << "repeat=" << repeat
                << " baseline_end_to_end_ms=" << measurement.end_to_end_ms()
                << '\n';
      if (repeat == 0 && options.include_zero_offset) {
        std::vector<query_fingerprint> zero_fingerprints(
            options.total_queries);
        auto zero = execute_batches<Policy>(
            graph, graph_storage.vertices, context, run_options, sequential,
            "scheduled_zero_offset", repeat, true, options.validate,
            &zero_fingerprints);
        if (options.validate) {
          std::cout << "scheduled_zero_fingerprint_q0="
                    << zero_fingerprints[0].sum << ':'
                    << zero_fingerprints[0].xor_value << '\n';
          int mismatch_count = 0;
          for (int query = 0; query < options.total_queries; ++query) {
            mismatch_count +=
                !(baseline_fingerprints[query] == zero_fingerprints[query]);
          }
          zero.correct = mismatch_count == 0;
          std::cout << "scheduled_zero_offset_mismatches=" << mismatch_count
                    << '\n';
        }
        measurements.push_back(zero);
      }
    };

    auto run_online = [&]() {
      auto plan = make_selected_plan();
      if (recorded_plan.batches.empty()) recorded_plan = plan;
      std::vector<query_fingerprint> online_fingerprints(
          options.total_queries);
      const bool collect = options.validate && baseline_fingerprints_ready;
      auto measurement = execute_batches<Policy>(
          graph, graph_storage.vertices, context, run_options, plan.batches,
          "online", repeat, selected_policy_is_scheduled, collect,
          &online_fingerprints);
      measurement.grouping_ms = plan.grouping_ms;
      measurement.offset_ms = plan.offset_evaluator_ms;
      measurement.evaluator_ms = plan.evaluator_ms();
      measurement.alignment_score = plan.grouping_alignment_score;
      if (collect) {
        if (!online_fingerprints.empty()) {
          std::cout << "online_fingerprint_q0="
                    << online_fingerprints[0].sum << ':'
                    << online_fingerprints[0].xor_value << '\n';
        }
        int mismatch_count = 0;
        for (int query = 0; query < options.total_queries; ++query) {
          if (!(baseline_fingerprints[query] == online_fingerprints[query])) {
            measurement.correct = false;
            if (mismatch_count < 16) {
              std::cerr << "fingerprint_mismatch query=" << query
                        << " source=" << sources[query] << '\n';
            }
            ++mismatch_count;
          }
        }
        if (mismatch_count != 0) {
          std::cerr << "fingerprint_mismatch_count=" << mismatch_count
                    << '\n';
        }
      }
      measurements.push_back(measurement);
      std::cout << "repeat=" << repeat
                << " online_gpu_ms=" << measurement.gpu_ms
                << " evaluator_ms=" << measurement.evaluator_ms
                << " online_end_to_end_ms=" << measurement.end_to_end_ms()
                << " correct=" << (measurement.correct ? "yes" : "no")
                << '\n';
      if (!measurement.correct) {
        throw std::runtime_error("online result fingerprint mismatch");
      }
    };

    auto run_batch_only = [&]() {
      auto plan = puercgp::scheduling::make_online_batching_execution_plan(
          evaluator, sources, options.batch_size, options.batch_swaps);
      auto measurement = execute_batches<Policy>(
          graph, graph_storage.vertices, context, run_options, plan.batches,
          "online_batching_only", repeat, false, false, nullptr);
      measurement.grouping_ms = plan.grouping_ms;
      measurement.evaluator_ms = plan.grouping_ms;
      measurement.alignment_score = plan.grouping_alignment_score;
      measurements.push_back(measurement);
      std::cout << "repeat=" << repeat
                << " batching_only_end_to_end_ms="
                << measurement.end_to_end_ms() << '\n';
    };

    auto run_offset_only = [&]() {
      auto plan = puercgp::scheduling::make_online_offset_execution_plan(
          evaluator, sequential);
      auto measurement = execute_batches<Policy>(
          graph, graph_storage.vertices, context, run_options, plan.batches,
          "online_offset_only", repeat, true, false, nullptr);
      measurement.offset_ms = plan.offset_evaluator_ms;
      measurement.evaluator_ms = plan.offset_evaluator_ms;
      measurements.push_back(measurement);
      std::cout << "repeat=" << repeat
                << " offset_only_end_to_end_ms="
                << measurement.end_to_end_ms() << '\n';
    };

    if (options.measure == "baseline-only") {
      run_baseline();
    } else if (options.measure == "online-only") {
      run_online();
    } else if (options.full_ablation) {
      run_baseline();
      run_batch_only();
      run_offset_only();
      run_online();
    } else if ((repeat & 1) == 0 || !baseline_fingerprints_ready) {
      run_baseline();
      run_online();
    } else {
      run_online();
      run_baseline();
    }
  }

  std::ofstream raw(options.output_dir / "runs.csv");
  raw << "method,repeat,gpu_ms,evaluator_ms,end_to_end_ms,runner_wall_ms,"
         "grouping_ms,offset_ms,alignment_score,iterations,push_iterations,"
         "pull_iterations,correct\n";
  for (const auto& value : measurements) {
    raw << value.method << ',' << value.repeat << ',' << value.gpu_ms << ','
        << value.evaluator_ms << ',' << value.end_to_end_ms() << ','
        << value.runner_wall_ms << ',' << value.grouping_ms << ','
        << value.offset_ms << ',' << value.alignment_score << ','
        << value.iterations << ',' << value.push_iterations << ','
        << value.pull_iterations << ',' << (value.correct ? 1 : 0) << '\n';
  }

  std::ofstream plans(options.output_dir / "online_plan.csv");
  plans << "batch,slot,query_id,source,predicted_length,offset,"
           "alignment_score,offset_evaluator_ms\n";
  for (int batch_id = 0;
       batch_id < static_cast<int>(recorded_plan.batches.size()); ++batch_id) {
    const auto& batch = recorded_plan.batches[batch_id];
    for (int slot = 0; slot < static_cast<int>(batch.query_ids.size()); ++slot) {
      const int predicted_length = slot <
              static_cast<int>(batch.predicted_lengths.size())
          ? batch.predicted_lengths[slot]
          : -1;
      plans << batch_id << ',' << slot << ',' << batch.query_ids[slot] << ','
            << batch.sources[slot] << ',' << predicted_length
            << ',' << batch.start_schedule.offsets()[slot] << ','
            << batch.alignment_score << ',' << batch.offset_evaluator_ms
            << '\n';
    }
  }

  std::ofstream metadata(options.output_dir / "metadata.csv");
  metadata << "graph,algorithm,mode,online_policy,measure,mean_predicted_length,vertices,edges,N,Q,repeats,warmups,seed,"
              "index,index_built,index_landmarks,index_preprocessing_ms,"
              "index_bytes,max_offset,batch_swaps\n";
  metadata << options.graph_path << ',' << options.algorithm << ','
           << options.mode << ',' << selected_policy << ','
           << options.measure << ',' << mean_predicted_length << ','
           << graph_storage.vertices << ','
           << graph_storage.edges << ',' << options.total_queries << ','
           << options.batch_size << ',' << options.repeats << ','
           << options.warmups << ',' << options.seed << ','
           << options.index_path.string() << ',' << (built_index ? 1 : 0)
           << ',' << phase_index.landmark_count() << ',' << preprocessing_ms
           << ',' << std::filesystem::file_size(options.index_path) << ','
           << options.max_offset << ',' << options.batch_swaps << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    std::filesystem::create_directories(options.output_dir);
    const bool build_pull = options.mode != "push";
    auto graph_storage =
        puercgp_examples::load_graph_auto(options.graph_path, build_pull);
    auto graph = graph_storage.view();

    auto sources = options.sources_path.empty()
        ? generate_sources(graph_storage, options.total_queries, options.seed)
        : load_sources(options.sources_path, options.total_queries);
    for (int source : sources) {
      if (source < 0 || source >= graph_storage.vertices) {
        throw std::runtime_error("source vertex is outside the graph");
      }
    }
    save_sources(options.output_dir / "sources.csv", sources);

    const auto preprocessing_start = std::chrono::steady_clock::now();
    const bool built_index = options.rebuild_index ||
        !std::filesystem::is_regular_file(options.index_path);
    puercgp::scheduling::landmark_phase_index phase_index;
    if (built_index) {
      phase_index = puercgp::scheduling::landmark_phase_index::build(
          graph_storage.row_offsets, graph_storage.column_indices,
          options.landmarks);
      phase_index.save(options.index_path);
    } else {
      phase_index = puercgp::scheduling::landmark_phase_index::load(
          options.index_path);
    }
    if (phase_index.vertex_count() != graph_storage.vertices) {
      throw std::runtime_error("phase index does not match the graph");
    }
    const double preprocessing_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preprocessing_start)
            .count();

    std::cout << "algorithm=" << options.algorithm << " mode=" << options.mode
              << " N=" << options.total_queries << " Q=" << options.batch_size
              << " V=" << graph_storage.vertices << " E=" << graph_storage.edges
              << '\n';
    if (options.algorithm == "bfs") {
      return run_benchmark<puercgp::algorithms::bfs_policy>(
          options, graph_storage, graph, sources, phase_index,
          preprocessing_ms, built_index);
    }
    if (options.algorithm == "sssp") {
      return run_benchmark<puercgp::algorithms::sssp_policy>(
          options, graph_storage, graph, sources, phase_index,
          preprocessing_ms, built_index);
    }
    if (options.algorithm == "sswp") {
      return run_benchmark<puercgp::algorithms::sswp_policy>(
          options, graph_storage, graph, sources, phase_index,
          preprocessing_ms, built_index);
    }
    throw std::invalid_argument("algorithm must be bfs, sssp, or sswp");
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
