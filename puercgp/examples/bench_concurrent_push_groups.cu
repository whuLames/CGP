/*
 * bench_concurrent_push_groups.cu
 * E1 decision experiment: SM-partitioned push groups versus one full-GPU
 * batch.
 *
 * Split the same N BFS queries into G contiguous groups. Each group runs the
 * production shared-node-warp push engine on a disjoint Green Context.
 *
 * Usage:
 *   bench_concurrent_push_groups <graph> [--n=64] [--seed=42] [--groups=1]
 *       [--sms=40,40] [--sequential] [--alone=g] [--repeats=5] [--warmup=1]
 *       [--max-iterations=20000] [--verify]
 *
 * Example configurations:
 *   A0: --groups=1                         (one full-GPU batch)
 *   A1: --groups=2 --sms=40,40             (two isolated groups)
 *   A2: --groups=2 --sms=16,16             (48 SMs intentionally idle)
 *   A3: --groups=4 --sms=20,20,20,20
 *   A4: --groups=2 --sequential            (grouping cost without overlap)
 *   Probe: --groups=2 --sms=40,40 --alone=0
 *
 * Makespan spans the first group start through the last group finish. Host
 * threads use a common barrier. --verify emits per-query fingerprints.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <thrust/device_vector.h>

#include <puercgp/core/green_context.hxx>
#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

using puercgp::algorithms::algo_kind_t;
using puercgp::execution_context;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::run_heterogeneous;
using puercgp::run_options;
using value_t = puercgp::algorithms::unified_value_t;

namespace {

// C++17-compatible reusable barrier.
class spin_barrier {
 public:
  explicit spin_barrier(int count) : threshold_(count) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;
    if (++arrived_ == threshold_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation_ != gen; });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int threshold_;
  int arrived_ = 0;
  int generation_ = 0;
};

// Fingerprint vertex-major values[v*Q+q] for cross-configuration checks.
__global__ void fingerprint_kernel(const value_t* values, std::size_t V, int Q,
                                   unsigned long long* counts,
                                   unsigned long long* level_sums) {
  int q = threadIdx.x;
  if (q >= Q) return;
  unsigned long long local_count = 0;
  unsigned long long local_sum = 0;
  for (std::size_t v = blockIdx.x; v < V; v += gridDim.x) {
    value_t val = values[v * static_cast<std::size_t>(Q) +
                         static_cast<std::size_t>(q)];
    if (isinf(val)) continue;
    ++local_count;
    local_sum += static_cast<unsigned long long>(val);
  }
  if (local_count != 0) atomicAdd(counts + q, local_count);
  if (local_sum != 0) atomicAdd(level_sums + q, local_sum);
}

std::vector<int> make_unique_sources(int vertex_count, int n,
                                     unsigned int seed) {
  std::mt19937 rng(seed);
  std::unordered_set<int> used;
  std::vector<int> sources;
  sources.reserve(static_cast<std::size_t>(n));
  while (sources.size() < static_cast<std::size_t>(n)) {
    int source = static_cast<int>(rng() % vertex_count);
    if (used.insert(source).second) sources.push_back(source);
  }
  return sources;
}

float medianf(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0f : v[v.size() / 2];
}

struct group_run {
  float wall_ms = 0.0f;
  int iterations = 0;
  std::vector<unsigned long long> reached;
  std::vector<unsigned long long> level_sum;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <graph> [--n=64] [--seed=42] [--groups=1] [--sms=40,40]"
              << " [--sequential] [--alone=g] [--repeats=5] [--warmup=1]"
              << " [--max-iterations=20000] [--verify]\n";
    return 2;
  }
  const std::string matrix = argv[1];
  int n_total = 64, seed = 42, groups = 1, repeats = 5, warmup = 1;
  int max_iterations = 20000, alone = -1;
  bool sequential = false, verify = false;
  std::vector<unsigned int> sms;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--n=", 0) == 0) n_total = std::atoi(a.c_str() + 4);
    else if (a.rfind("--seed=", 0) == 0) seed = std::atoi(a.c_str() + 7);
    else if (a.rfind("--groups=", 0) == 0) groups = std::atoi(a.c_str() + 9);
    else if (a.rfind("--repeats=", 0) == 0) repeats = std::atoi(a.c_str() + 10);
    else if (a.rfind("--warmup=", 0) == 0) warmup = std::atoi(a.c_str() + 9);
    else if (a.rfind("--max-iterations=", 0) == 0)
      max_iterations = std::atoi(a.c_str() + 17);
    else if (a.rfind("--alone=", 0) == 0) alone = std::atoi(a.c_str() + 8);
    else if (a == "--sequential") sequential = true;
    else if (a == "--verify") verify = true;
    else if (a.rfind("--sms=", 0) == 0) {
      std::stringstream ss(a.substr(6));
      std::string tok;
      while (std::getline(ss, tok, ','))
        sms.push_back(static_cast<unsigned int>(std::atoi(tok.c_str())));
    }
  }
  if (groups < 1 || n_total % groups != 0)
    throw std::invalid_argument("n must be divisible by groups");
  if (!sms.empty() && static_cast<int>(sms.size()) != groups)
    throw std::invalid_argument("--sms partition count must equal --groups");
  if (sequential && !sms.empty())
    throw std::invalid_argument("--sequential runs on full GPU (omit --sms)");
  const int per_group = n_total / groups;

  auto graph_storage = puercgp_examples::load_graph_auto(matrix, false);
  auto graph = graph_storage.view();
  auto sources = make_unique_sources(graph_storage.vertices, n_total,
                                     static_cast<unsigned int>(seed));

  // Optional SM partition.
  std::unique_ptr<puercgp::detail::green_context_group> partition;
  if (!sms.empty()) {
    partition =
        std::make_unique<puercgp::detail::green_context_group>(0, sms);
  }

  std::cout << "graph=" << matrix << " V=" << graph_storage.vertices
            << " E=" << graph_storage.edges << " n=" << n_total
            << " groups=" << groups << " per_group=" << per_group
            << " seed=" << seed << " sequential=" << (sequential ? 1 : 0)
            << " alone=" << alone << "\n";
  if (partition) {
    std::cout << "sm_partition=";
    for (std::size_t g = 0; g < partition->size(); ++g)
      std::cout << (g ? "+" : "") << partition->sm_count(g);
    std::cout << "\n";
  } else {
    std::cout << "sm_partition=full_gpu\n";
  }

  run_options opt;
  opt.traversal_mode = puercgp::traversal_mode_t::push;
  opt.push_strategy = puercgp::push_strategy_t::shared_node_warp;
  opt.max_iterations = max_iterations;

  // Query descriptors for each contiguous group.
  std::vector<std::vector<query_descriptor_t>> group_descs(
      static_cast<std::size_t>(groups));
  for (int g = 0; g < groups; ++g)
    for (int i = 0; i < per_group; ++i)
      group_descs[static_cast<std::size_t>(g)].push_back(
          {sources[static_cast<std::size_t>(g * per_group + i)],
           algo_kind_t::bfs, 0.0f});

  // Run one group on its assigned or independently owned stream.
  auto run_group = [&](int g, group_run& out) {
    std::unique_ptr<execution_context> owned;
    execution_context* ctx;
    if (partition) {
      owned = std::make_unique<execution_context>(
          partition->stream(static_cast<std::size_t>(g)));
    } else {
      owned = std::make_unique<execution_context>();
    }
    ctx = owned.get();
    hybrid_query_batch batch(group_descs[static_cast<std::size_t>(g)]);
    batch.validate();
    auto result = run_heterogeneous(graph, batch, *ctx, opt);
    out.wall_ms = result.wall_time_ms;
    out.iterations = result.iterations;
    if (verify) {
      const std::size_t V = static_cast<std::size_t>(graph_storage.vertices);
      thrust::device_vector<unsigned long long> counts(
          static_cast<std::size_t>(per_group), 0ULL);
      thrust::device_vector<unsigned long long> sums(
          static_cast<std::size_t>(per_group), 0ULL);
      int blocks = static_cast<int>(std::min<std::size_t>(4096, V));
      fingerprint_kernel<<<blocks, per_group, 0, ctx->stream()>>>(
          thrust::raw_pointer_cast(result.values.data()), V, per_group,
          thrust::raw_pointer_cast(counts.data()),
          thrust::raw_pointer_cast(sums.data()));
      ctx->synchronize();
      out.reached.resize(static_cast<std::size_t>(per_group));
      out.level_sum.resize(static_cast<std::size_t>(per_group));
      thrust::copy(counts.begin(), counts.end(), out.reached.begin());
      thrust::copy(sums.begin(), sums.end(), out.level_sum.begin());
    }
  };

  // Active group list; --alone selects one interference probe.
  std::vector<int> active_groups;
  for (int g = 0; g < groups; ++g)
    if (alone < 0 || g == alone) active_groups.push_back(g);
  const int k = static_cast<int>(active_groups.size());

  // One round runs every active group and returns makespan plus group times.
  auto run_round = [&]() {
    std::vector<group_run> runs(static_cast<std::size_t>(groups));
    std::vector<double> start_ms(static_cast<std::size_t>(groups), 0.0);
    std::vector<double> end_ms(static_cast<std::size_t>(groups), 0.0);
    auto epoch = std::chrono::steady_clock::now();
    if (sequential) {
      for (int g : active_groups) {
        auto t0 = std::chrono::steady_clock::now();
        run_group(g, runs[static_cast<std::size_t>(g)]);
        auto t1 = std::chrono::steady_clock::now();
        start_ms[g] = std::chrono::duration<double, std::milli>(t0 - epoch).count();
        end_ms[g] = std::chrono::duration<double, std::milli>(t1 - epoch).count();
      }
    } else {
      spin_barrier barrier(k);
      std::vector<std::thread> threads;
      for (int g : active_groups) {
        threads.emplace_back([&, g] {
          barrier.arrive_and_wait();
          auto t0 = std::chrono::steady_clock::now();
          run_group(g, runs[static_cast<std::size_t>(g)]);
          auto t1 = std::chrono::steady_clock::now();
          start_ms[g] = std::chrono::duration<double, std::milli>(t0 - epoch).count();
          end_ms[g] = std::chrono::duration<double, std::milli>(t1 - epoch).count();
        });
      }
      for (auto& t : threads) t.join();
    }
    double min_start = 1e300, max_end = 0.0;
    for (int g : active_groups) {
      min_start = std::min(min_start, start_ms[static_cast<std::size_t>(g)]);
      max_end = std::max(max_end, end_ms[static_cast<std::size_t>(g)]);
    }
    return std::make_pair(static_cast<float>(max_end - min_start),
                          std::move(runs));
  };

  for (int w = 0; w < warmup; ++w) (void)run_round();

  std::vector<float> makespans;
  std::vector<std::vector<float>> group_walls(
      static_cast<std::size_t>(groups));
  std::vector<group_run> last_runs;
  for (int r = 0; r < repeats; ++r) {
    auto [ms, runs] = run_round();
    makespans.push_back(ms);
    for (int g : active_groups)
      group_walls[static_cast<std::size_t>(g)].push_back(
          runs[static_cast<std::size_t>(g)].wall_ms);
    last_runs = std::move(runs);
  }

  std::cout << "\n=== Result (median over " << repeats << ") ===\n";
  std::cout << "makespan_ms=" << medianf(makespans) << "\n";
  for (int g : active_groups) {
    std::cout << "group" << g << "_wall_ms="
              << medianf(group_walls[static_cast<std::size_t>(g)])
              << " iterations="
              << last_runs[static_cast<std::size_t>(g)].iterations << "\n";
  }
  if (verify) {
    std::cout << "\n=== Fingerprints (query,reached,level_sum) ===\n";
    for (int g : active_groups) {
      const auto& run = last_runs[static_cast<std::size_t>(g)];
      for (int i = 0; i < per_group; ++i)
        std::cout << "fp," << (g * per_group + i) << ","
                  << run.reached[static_cast<std::size_t>(i)] << ","
                  << run.level_sum[static_cast<std::size_t>(i)] << "\n";
    }
  }
  return 0;
}
