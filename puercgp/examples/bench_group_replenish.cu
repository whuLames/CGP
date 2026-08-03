/*
 * Group-level replenish experiment.
 *
 * The global queue is divided into fixed-size cohorts. Two independent group
 * workers consume alternating cohorts and never replace an individual slot.
 * This keeps each Q=32 pull/push state phase-aligned while allowing the workers
 * to overlap on ordinary streams or disjoint Green Context partitions.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <puercgp/core/green_context.hxx>
#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

using puercgp::algorithms::algo_kind_t;
using puercgp::algorithms::unified_value_t;
using puercgp::execution_context;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::run_options;

namespace {

using steady_clock_t = std::chrono::steady_clock;

class start_barrier {
 public:
  explicit start_barrier(int count) : count_(count) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (++arrived_ == count_) {
      ready_ = true;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return ready_; });
  }

 private:
  int count_;
  int arrived_ = 0;
  bool ready_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
};

struct group_input {
  std::vector<query_descriptor_t> queries;
  std::vector<int> original_ids;
};

struct group_output {
  float engine_wall_ms = 0.0f;
  float gpu_ms = 0.0f;
  float start_offset_ms = 0.0f;
  int iterations = 0;
  std::vector<float> completion_ms;
  std::vector<unsigned long long> reached;
  std::vector<double> value_sum;
};

struct round_output {
  float host_wall_ms = 0.0f;
  float engine_makespan_ms = 0.0f;
  std::vector<group_output> groups;
  std::vector<float> completion_ms;
};

float elapsed_ms(steady_clock_t::time_point start,
                 steady_clock_t::time_point stop) {
  return std::chrono::duration<float, std::milli>(stop - start).count();
}

float median(std::vector<float> values) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

float percentile(std::vector<float> values, double p) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  const double position = p * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(position);
  const std::size_t hi = std::min(lo + 1, values.size() - 1);
  const double fraction = position - static_cast<double>(lo);
  return static_cast<float>(values[lo] * (1.0 - fraction) +
                            values[hi] * fraction);
}

std::vector<int> random_unique_sources(int vertex_count, int count,
                                       unsigned int seed) {
  std::mt19937 rng(seed);
  std::unordered_set<int> seen;
  std::vector<int> sources;
  sources.reserve(static_cast<std::size_t>(count));
  while (static_cast<int>(sources.size()) < count) {
    const int source = static_cast<int>(rng() % vertex_count);
    if (seen.insert(source).second) sources.push_back(source);
  }
  return sources;
}

std::vector<int> load_sources(const std::string& path, int expected) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open sources file: " + path);
  std::vector<int> sources;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const bool csv = line.find(',') != std::string::npos;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream parser(line);
    std::string first;
    parser >> first;
    if (first.empty()) continue;
    char* end = nullptr;
    const long first_value = std::strtol(first.c_str(), &end, 10);
    if (end == first.c_str() || *end != '\0') continue;
    long source = first_value;
    if (!csv) {
      source = first_value;
    } else {
      parser >> source;
    }
    sources.push_back(static_cast<int>(source));
  }
  if (static_cast<int>(sources.size()) < expected) {
    throw std::runtime_error("sources file has fewer entries than requested");
  }
  sources.resize(static_cast<std::size_t>(expected));
  return sources;
}

std::vector<group_input> split_into_groups(
    const std::vector<query_descriptor_t>& queries, int groups,
    int cohort_size) {
  std::vector<group_input> result(static_cast<std::size_t>(groups));
  int cohort = 0;
  for (std::size_t begin = 0; begin < queries.size();
       begin += static_cast<std::size_t>(cohort_size), ++cohort) {
    const int group = cohort % groups;
    const std::size_t end = std::min(
        begin + static_cast<std::size_t>(cohort_size), queries.size());
    for (std::size_t i = begin; i < end; ++i) {
      result[static_cast<std::size_t>(group)].queries.push_back(queries[i]);
      result[static_cast<std::size_t>(group)].original_ids.push_back(
          static_cast<int>(i));
    }
  }
  return result;
}

void fingerprint_row_major(
    const thrust::device_vector<unified_value_t>& values, std::size_t count,
    std::size_t vertices, std::vector<unsigned long long>* reached,
    std::vector<double>* sums) {
  thrust::host_vector<unified_value_t> host(values);
  reached->assign(count, 0);
  sums->assign(count, 0.0);
  for (std::size_t q = 0; q < count; ++q) {
    for (std::size_t v = 0; v < vertices; ++v) {
      const double value = static_cast<double>(host[q * vertices + v]);
      if (!std::isfinite(value)) continue;
      ++(*reached)[q];
      (*sums)[q] += value;
    }
  }
}

puercgp::traversal_mode_t parse_mode(const std::string& text) {
  if (text == "push") return puercgp::traversal_mode_t::push;
  if (text == "pull") return puercgp::traversal_mode_t::pull;
  if (text == "hybrid") return puercgp::traversal_mode_t::hybrid;
  throw std::invalid_argument("mode must be push, pull, or hybrid");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr
        << "Usage: " << argv[0]
        << " <graph> --method=static64|cohort64|immediate64|group_serial|"
           "group_stream|group_green [--n=256] [--sources-file=path]"
           " [--seed=42] [--groups=2] [--group-size=32] [--sm-per-group=40]"
           " [--mode=push] [--warmup=1] [--repeats=5] [--verify]\n";
    return 2;
  }

  const std::string graph_path = argv[1];
  std::string method = "static64";
  std::string mode_text = "push";
  std::string source_path;
  int total_queries = 256;
  int seed = 42;
  int group_count = 2;
  int group_size = 32;
  int sms_per_group = 0;
  int warmups = 1;
  int repeats = 5;
  int max_iterations = 20000;
  bool verify = false;
  bool discard = true;
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    auto integer_after = [&](const std::string& prefix) {
      return std::atoi(argument.c_str() + prefix.size());
    };
    if (argument.rfind("--method=", 0) == 0)
      method = argument.substr(9);
    else if (argument.rfind("--mode=", 0) == 0)
      mode_text = argument.substr(7);
    else if (argument.rfind("--sources-file=", 0) == 0)
      source_path = argument.substr(15);
    else if (argument.rfind("--n=", 0) == 0)
      total_queries = integer_after("--n=");
    else if (argument.rfind("--seed=", 0) == 0)
      seed = integer_after("--seed=");
    else if (argument.rfind("--groups=", 0) == 0)
      group_count = integer_after("--groups=");
    else if (argument.rfind("--group-size=", 0) == 0)
      group_size = integer_after("--group-size=");
    else if (argument.rfind("--first-sms=", 0) == 0)
      sms_per_group = integer_after("--first-sms=");
    else if (argument.rfind("--sm-per-group=", 0) == 0)
      sms_per_group = integer_after("--sm-per-group=");
    else if (argument.rfind("--warmup=", 0) == 0)
      warmups = integer_after("--warmup=");
    else if (argument.rfind("--repeats=", 0) == 0)
      repeats = integer_after("--repeats=");
    else if (argument.rfind("--max-iterations=", 0) == 0)
      max_iterations = integer_after("--max-iterations=");
    else if (argument == "--verify") {
      verify = true;
      discard = false;
    } else if (argument == "--no-discard") {
      discard = false;
    }
  }

  const bool grouped = method.rfind("group_", 0) == 0;
  const bool concurrent = method == "group_stream" || method == "group_green";
  if (group_size <= 0 || group_size > 32)
    throw std::invalid_argument("group-size must be in [1,32]");
  if (grouped && group_count < 2)
    throw std::invalid_argument("grouped methods require at least two groups");
  if (total_queries <= 0)
    throw std::invalid_argument("n must be positive");
  if (method != "static64" && method != "cohort64" &&
      method != "immediate64" && method != "group_serial" &&
      method != "group_stream" && method != "group_green") {
    throw std::invalid_argument("unknown method: " + method);
  }

  const auto traversal_mode = parse_mode(mode_text);
  auto graph_storage = puercgp_examples::load_graph_auto(
      graph_path, traversal_mode != puercgp::traversal_mode_t::push);
  auto graph = graph_storage.view();
  std::vector<int> sources =
      source_path.empty()
          ? random_unique_sources(graph_storage.vertices, total_queries,
                                  static_cast<unsigned int>(seed))
          : load_sources(source_path, total_queries);
  for (int source : sources) {
    if (source < 0 || source >= graph_storage.vertices)
      throw std::out_of_range("source outside graph");
  }

  std::vector<query_descriptor_t> queries;
  queries.reserve(static_cast<std::size_t>(total_queries));
  for (int source : sources)
    queries.push_back({source, algo_kind_t::bfs, 0.0f});

  run_options options;
  options.traversal_mode = traversal_mode;
  options.push_strategy = puercgp::push_strategy_t::shared_node_warp;
  options.max_iterations = max_iterations;
  options.discard_results = discard;

  std::unique_ptr<puercgp::detail::green_context_group> green_groups;
  std::vector<std::unique_ptr<execution_context>> contexts;
  if (method == "group_green") {
    if (sms_per_group == 0) sms_per_group = 80 / group_count;
    std::vector<unsigned int> partitions(
        static_cast<std::size_t>(group_count),
        static_cast<unsigned int>(sms_per_group));
    green_groups =
        std::make_unique<puercgp::detail::green_context_group>(0, partitions);
    for (int group = 0; group < group_count; ++group) {
      contexts.push_back(std::make_unique<execution_context>(
          green_groups->stream(static_cast<std::size_t>(group))));
    }
  } else {
    const int context_count = grouped ? group_count : 1;
    for (int i = 0; i < context_count; ++i)
      contexts.push_back(std::make_unique<execution_context>());
  }

  const auto group_inputs =
      split_into_groups(queries, group_count, group_size);
  if (grouped && std::any_of(group_inputs.begin(), group_inputs.end(),
                             [](const group_input& input) {
                               return input.queries.empty();
                             })) {
    throw std::invalid_argument(
        "n must provide at least one cohort to every group");
  }
  auto run_group = [&](int group, steady_clock_t::time_point epoch,
                       group_output* output) {
    output->start_offset_ms = elapsed_ms(epoch, steady_clock_t::now());
    run_options local_options = options;
    local_options.enable_replenishment = true;
    local_options.max_queries = static_cast<std::size_t>(group_size);
    local_options.replenish_batch_size = static_cast<std::size_t>(group_size);
    const auto& input = group_inputs[static_cast<std::size_t>(group)];
    auto result = puercgp::run_replenish_pipeline(
        graph, input.queries, *contexts[static_cast<std::size_t>(group)],
        local_options);
    output->engine_wall_ms = result.wall_time_ms;
    output->gpu_ms = result.gpu_time_ms;
    output->iterations = result.iterations;
    output->completion_ms.resize(result.queries.size());
    for (std::size_t i = 0; i < result.queries.size(); ++i) {
      output->completion_ms[i] =
          output->start_offset_ms + result.queries[i].completion_wall_time_ms;
    }
    if (verify) {
      fingerprint_row_major(result.values, input.queries.size(),
                            static_cast<std::size_t>(graph_storage.vertices),
                            &output->reached, &output->value_sum);
    }
  };

  auto run_round = [&]() {
    round_output output;
    const auto epoch = steady_clock_t::now();
    if (method == "static64") {
      output.completion_ms.resize(queries.size(), 0.0f);
      float engine_sum = 0.0f;
      for (std::size_t begin = 0; begin < queries.size(); begin += 64) {
        const std::size_t end = std::min(begin + 64, queries.size());
        std::vector<query_descriptor_t> batch_queries(queries.begin() + begin,
                                                       queries.begin() + end);
        hybrid_query_batch batch(std::move(batch_queries));
        auto result = puercgp::run_heterogeneous(graph, batch, *contexts[0],
                                                  options);
        engine_sum += result.wall_time_ms;
        const float done = elapsed_ms(epoch, steady_clock_t::now());
        std::fill(output.completion_ms.begin() + begin,
                  output.completion_ms.begin() + end, done);
      }
      output.engine_makespan_ms = engine_sum;
    } else if (method == "cohort64" || method == "immediate64") {
      run_options local_options = options;
      local_options.enable_replenishment = true;
      local_options.max_queries = 64;
      local_options.replenish_batch_size = method == "cohort64" ? 64 : 1;
      auto result = puercgp::run_replenish_pipeline(
          graph, queries, *contexts[0], local_options);
      output.engine_makespan_ms = result.wall_time_ms;
      output.completion_ms.resize(result.queries.size());
      for (std::size_t i = 0; i < result.queries.size(); ++i)
        output.completion_ms[i] = result.queries[i].completion_wall_time_ms;
      if (verify) {
        group_output single;
        fingerprint_row_major(
            result.values, queries.size(),
            static_cast<std::size_t>(graph_storage.vertices), &single.reached,
            &single.value_sum);
        output.groups.push_back(std::move(single));
      }
    } else {
      output.groups.resize(static_cast<std::size_t>(group_count));
      if (concurrent) {
        start_barrier barrier(group_count);
        std::vector<std::thread> workers;
        for (int group = 0; group < group_count; ++group) {
          workers.emplace_back([&, group] {
            barrier.wait();
            run_group(group, epoch,
                      &output.groups[static_cast<std::size_t>(group)]);
          });
        }
        for (auto& worker : workers) worker.join();
      } else {
        for (int group = 0; group < group_count; ++group)
          run_group(group, epoch,
                    &output.groups[static_cast<std::size_t>(group)]);
      }
      output.completion_ms.resize(queries.size(), 0.0f);
      float end = 0.0f;
      for (int group = 0; group < group_count; ++group) {
        const auto& input = group_inputs[static_cast<std::size_t>(group)];
        const auto& group_result = output.groups[static_cast<std::size_t>(group)];
        end = std::max(end, group_result.start_offset_ms +
                               group_result.engine_wall_ms);
        for (std::size_t i = 0; i < input.original_ids.size(); ++i) {
          output.completion_ms[static_cast<std::size_t>(input.original_ids[i])] =
              group_result.completion_ms[i];
        }
      }
      output.engine_makespan_ms = end;
    }
    output.host_wall_ms = elapsed_ms(epoch, steady_clock_t::now());
    return output;
  };

  std::cout << "graph=" << graph_path << " V=" << graph_storage.vertices
            << " E=" << graph_storage.edges << " method=" << method
            << " mode=" << mode_text << " N=" << total_queries
            << " groups=" << (grouped ? group_count : 1)
            << " group_size=" << group_size
            << " source=" << (source_path.empty() ? "seed" : source_path)
            << " discard=" << (discard ? 1 : 0) << "\n";
  if (green_groups) {
    std::cout << "sm_partition=";
    for (std::size_t group = 0; group < green_groups->size(); ++group) {
      if (group != 0) std::cout << "+";
      std::cout << green_groups->sm_count(group);
    }
    std::cout << "\n";
  }

  for (int warmup = 0; warmup < warmups; ++warmup) (void)run_round();
  std::vector<float> host_times;
  std::vector<float> engine_times;
  std::vector<round_output> rounds;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto result = run_round();
    std::cout << "run," << repeat << ",host_ms=" << result.host_wall_ms
              << ",engine_ms=" << result.engine_makespan_ms;
    for (std::size_t group = 0; group < result.groups.size(); ++group) {
      std::cout << ",group" << group
                << "_ms=" << result.groups[group].engine_wall_ms;
    }
    std::cout << "\n";
    host_times.push_back(result.host_wall_ms);
    engine_times.push_back(result.engine_makespan_ms);
    rounds.push_back(std::move(result));
  }

  const std::size_t median_index = [&] {
    std::vector<std::size_t> order(rounds.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return rounds[a].host_wall_ms < rounds[b].host_wall_ms;
    });
    return order[order.size() / 2];
  }();
  const auto& representative = rounds[median_index];
  const float median_host = median(host_times);
  std::cout << std::fixed << std::setprecision(3)
            << "RESULT,method=" << method << ",median_host_ms=" << median_host
            << ",median_engine_ms=" << median(engine_times)
            << ",qps=" << (1000.0 * total_queries / median_host)
            << ",p50_ms=" << percentile(representative.completion_ms, 0.50)
            << ",p95_ms=" << percentile(representative.completion_ms, 0.95)
            << ",p99_ms=" << percentile(representative.completion_ms, 0.99)
            << "\n";

  if (verify) {
    if (method == "cohort64" || method == "immediate64") {
      const auto& single = representative.groups[0];
      for (std::size_t i = 0; i < single.reached.size(); ++i)
        std::cout << "fp," << i << "," << single.reached[i] << ","
                  << std::setprecision(9) << single.value_sum[i] << "\n";
    } else if (grouped) {
      for (int group = 0; group < group_count; ++group) {
        const auto& input = group_inputs[static_cast<std::size_t>(group)];
        const auto& result = representative.groups[static_cast<std::size_t>(group)];
        for (std::size_t i = 0; i < input.original_ids.size(); ++i)
          std::cout << "fp," << input.original_ids[i] << ","
                    << result.reached[i] << "," << std::setprecision(9)
                    << result.value_sum[i] << "\n";
      }
    }
  }
  return 0;
}
