#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <puercgp/puercgp.hxx>
#include <puercgp/core/green_context.hxx>
#include <puercgp/engine/frontier_engine.hxx>
#include <puercgp/engine/pull_executor.hxx>
#include <puercgp/engine/push_executor.hxx>
#include <puercgp/kernels/pull/fused_pull_kernels.hxx>

#include "mtx_loader.hxx"

namespace {

constexpr int kQueries = 64;
constexpr int kThreads = 256;
using mask_t = puercgp::query_mask_t;
using policy_t = puercgp::algorithms::bfs_policy;

struct fingerprint_t {
  std::array<unsigned long long, kQueries> counts{};
  std::array<unsigned long long, kQueries> sums{};
  unsigned long long unique = 0;
  unsigned long long pairs = 0;
};

struct measurement_t {
  std::string method;
  int push_sms = 0;
  int pull_sms = 0;
  float push_ms = 0;
  float pull_ms = 0;
  float concurrent_ms = 0;
  float post_ms = 0;
  float total_ms = 0;
  fingerprint_t fingerprint;
};

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
__global__ void query_edges_kernel(
    graph_t graph, const int* frontier_vertices, const mask_t* frontier_mask,
    std::size_t unique_count, unsigned long long* edges) {
  int q = threadIdx.x;
  mask_t bit = mask_t{1} << q;
  unsigned long long local = 0;
  for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
    int v = frontier_vertices[i];
    if ((frontier_mask[v] & bit) != 0)
      local += graph.get_starting_edge(v + 1) - graph.get_starting_edge(v);
  }
  if (local) atomicAdd(edges + q, local);
}

__global__ void merge_frontiers_kernel(
    const mask_t* push_mask, const mask_t* pull_mask, std::size_t count,
    mask_t* combined, unsigned long long* flags,
    unsigned long long* pairs) {
  std::size_t tid = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                    threadIdx.x;
  std::size_t stride = gridDim.x * static_cast<std::size_t>(blockDim.x);
  for (std::size_t v = tid; v < count; v += stride) {
    mask_t value = push_mask[v] | pull_mask[v];
    combined[v] = value;
    flags[v] = value != 0 ? 1ULL : 0ULL;
    pairs[v] = __popcll(value);
  }
}

__global__ void fingerprint_kernel(
    const mask_t* visited, const int* values, std::size_t vertex_count,
    unsigned long long* counts, unsigned long long* sums) {
  int q = threadIdx.x;
  mask_t bit = mask_t{1} << q;
  unsigned long long count = 0;
  unsigned long long sum = 0;
  for (std::size_t v = blockIdx.x; v < vertex_count; v += gridDim.x) {
    if ((visited[v] & bit) == 0) continue;
    ++count;
    sum += static_cast<unsigned long long>(values[v * kQueries + q]);
  }
  if (count) atomicAdd(counts + q, count);
  if (sum) atomicAdd(sums + q, sum);
}

template <typename graph_t>
class replay_workspace {
 public:
  replay_workspace(graph_t graph, std::vector<int> sources)
      : graph_(graph), sources_(sources), V_(graph.get_number_of_vertices()),
        value_count_(V_ * kQueries), mask_bytes_(V_ * sizeof(mask_t)),
        device_sources_(sources), values_(value_count_), visited_(V_),
        frontier_mask_(V_), next_frontier_mask_(V_), frontier_vertices_(V_),
        next_frontier_vertices_(V_), push_mask_(V_), pull_mask_(V_),
        combined_mask_(V_), scratch_vertices_(V_), unique_count_(1),
        next_unique_count_(1), next_pair_count_(1), flags_(V_), offsets_(V_),
        pairs_(V_), query_edges_(kQueries), fp_counts_(kQueries),
        fp_sums_(kQueries) {}

  void prepare(int target_step) {
    puercgp::detail::fill_values_kernel<policy_t>
        <<<puercgp::detail::grid_for(value_count_, kThreads), kThreads>>>(
            thrust::raw_pointer_cast(values_.data()), value_count_);
    cudaMemset(thrust::raw_pointer_cast(visited_.data()), 0, mask_bytes_);
    cudaMemset(thrust::raw_pointer_cast(frontier_mask_.data()), 0, mask_bytes_);
    cudaMemset(thrust::raw_pointer_cast(next_frontier_mask_.data()), 0,
               mask_bytes_);
    cudaMemset(thrust::raw_pointer_cast(unique_count_.data()), 0,
               sizeof(unsigned long long));
    puercgp::detail::init_shared_sources_kernel<policy_t><<<1, 64>>>(
        graph_, thrust::raw_pointer_cast(device_sources_.data()), kQueries,
        thrust::raw_pointer_cast(values_.data()),
        thrust::raw_pointer_cast(visited_.data()),
        thrust::raw_pointer_cast(frontier_mask_.data()),
        thrust::raw_pointer_cast(frontier_vertices_.data()),
        thrust::raw_pointer_cast(unique_count_.data()));
    cudaMemcpy(&current_unique_, thrust::raw_pointer_cast(unique_count_.data()),
               sizeof(current_unique_), cudaMemcpyDeviceToHost);
    for (int step = 0; step < target_step; ++step) {
      clear_push_output(nullptr);
      puercgp::detail::launch_shared_push_warp<policy_t, graph_t, int>(
          graph_, thrust::raw_pointer_cast(frontier_vertices_.data()),
          thrust::raw_pointer_cast(frontier_mask_.data()), current_unique_,
          thrust::raw_pointer_cast(visited_.data()),
          thrust::raw_pointer_cast(next_frontier_mask_.data()),
          thrust::raw_pointer_cast(next_frontier_vertices_.data()),
          thrust::raw_pointer_cast(next_unique_count_.data()),
          thrust::raw_pointer_cast(next_pair_count_.data()),
          thrust::raw_pointer_cast(values_.data()), kQueries, ~mask_t{0}, step,
          kThreads, nullptr);
      cudaMemcpy(&current_unique_,
                 thrust::raw_pointer_cast(next_unique_count_.data()),
                 sizeof(current_unique_), cudaMemcpyDeviceToHost);
      thrust::swap(frontier_mask_, next_frontier_mask_);
      thrust::swap(frontier_vertices_, next_frontier_vertices_);
    }
    cudaDeviceSynchronize();
  }

  mask_t classify(double ratio, std::array<unsigned long long, kQueries>& work) {
    cudaMemset(thrust::raw_pointer_cast(query_edges_.data()), 0,
               kQueries * sizeof(unsigned long long));
    int blocks = std::max(1, std::min(1024, static_cast<int>(current_unique_)));
    query_edges_kernel<<<blocks, kQueries>>>(
        graph_, thrust::raw_pointer_cast(frontier_vertices_.data()),
        thrust::raw_pointer_cast(frontier_mask_.data()), current_unique_,
        thrust::raw_pointer_cast(query_edges_.data()));
    cudaMemcpy(work.data(), thrust::raw_pointer_cast(query_edges_.data()),
               sizeof(work), cudaMemcpyDeviceToHost);
    double threshold = ratio * graph_.get_number_of_edges();
    mask_t push = 0;
    for (int q = 0; q < kQueries; ++q)
      if (static_cast<double>(work[q]) < threshold) push |= mask_t{1} << q;
    return push;
  }

  measurement_t run_all_pull(int target_step) {
    prepare(target_step);
    clear_pull_output(nullptr);
    measurement_t result;
    result.method = "all_pull";
    result.pull_sms = 80;
    puercgp::detail::cuda_event_timer timer;
    timer.begin(nullptr);
    launch_pull(~mask_t{0}, nullptr);
    result.pull_ms = timer.end(nullptr);
    result.post_ms = postprocess(pull_mask_, pull_mask_, false, nullptr);
    result.total_ms = result.pull_ms + result.post_ms;
    result.fingerprint = fingerprint();
    return result;
  }

  measurement_t run_all_push(int target_step) {
    prepare(target_step);
    cudaMemset(thrust::raw_pointer_cast(push_mask_.data()), 0, mask_bytes_);
    cudaMemset(thrust::raw_pointer_cast(next_unique_count_.data()), 0,
               sizeof(unsigned long long));
    cudaMemset(thrust::raw_pointer_cast(next_pair_count_.data()), 0,
               sizeof(unsigned long long));
    measurement_t result;
    result.method = "all_push";
    result.push_sms = 80;
    puercgp::detail::cuda_event_timer timer;
    timer.begin(nullptr);
    launch_push(~mask_t{0}, target_step, nullptr);
    result.push_ms = timer.end(nullptr);
    result.total_ms = result.push_ms;
    result.fingerprint = fingerprint();
    return result;
  }

  measurement_t run_split_serial(int target_step, mask_t push_slots) {
    prepare(target_step);
    clear_split_outputs(nullptr, nullptr);
    measurement_t result;
    result.method = "split_serial";
    result.push_sms = 80;
    result.pull_sms = 80;
    puercgp::detail::cuda_event_timer timer;
    timer.begin(nullptr);
    launch_push(push_slots, target_step, nullptr);
    result.push_ms = timer.end(nullptr);
    timer.begin(nullptr);
    launch_pull(~push_slots, nullptr);
    result.pull_ms = timer.end(nullptr);
    result.concurrent_ms = result.push_ms + result.pull_ms;
    result.post_ms = postprocess(push_mask_, pull_mask_, true, nullptr);
    result.total_ms = result.concurrent_ms + result.post_ms;
    result.fingerprint = fingerprint();
    return result;
  }

  measurement_t run_split_concurrent(int target_step, mask_t push_slots,
                                     int push_sms) {
    prepare(target_step);
    puercgp::detail::green_context_pair contexts(0, push_sms);
    auto push_stream = contexts.first_stream();
    auto pull_stream = contexts.second_stream();
    clear_split_outputs(push_stream, pull_stream);
    cudaStreamSynchronize(push_stream);
    cudaStreamSynchronize(pull_stream);
    measurement_t result;
    result.method = "split_concurrent";
    result.push_sms = contexts.first_sm_count();
    result.pull_sms = contexts.second_sm_count();
    puercgp::detail::cuda_event_timer push_timer, pull_timer;
    auto wall_start = std::chrono::steady_clock::now();
    push_timer.begin(push_stream);
    launch_push(push_slots, target_step, push_stream);
    pull_timer.begin(pull_stream);
    launch_pull(~push_slots, pull_stream);
    result.push_ms = push_timer.end(push_stream);
    result.pull_ms = pull_timer.end(pull_stream);
    cudaStreamSynchronize(push_stream);
    cudaStreamSynchronize(pull_stream);
    result.concurrent_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - wall_start).count();
    result.post_ms = postprocess(push_mask_, pull_mask_, true, pull_stream);
    result.total_ms = result.concurrent_ms + result.post_ms;
    result.fingerprint = fingerprint(pull_stream);
    return result;
  }

 private:
  void clear_push_output(cudaStream_t stream) {
    cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask_.data()), 0,
                    mask_bytes_, stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(next_unique_count_.data()), 0,
                    sizeof(unsigned long long), stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(next_pair_count_.data()), 0,
                    sizeof(unsigned long long), stream);
  }

  void clear_pull_output(cudaStream_t stream) {
    cudaMemsetAsync(thrust::raw_pointer_cast(pull_mask_.data()), 0, mask_bytes_,
                    stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(flags_.data()), 0,
                    V_ * sizeof(unsigned long long), stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(pairs_.data()), 0,
                    V_ * sizeof(unsigned long long), stream);
  }

  void clear_split_outputs(cudaStream_t push_stream, cudaStream_t pull_stream) {
    cudaMemsetAsync(thrust::raw_pointer_cast(push_mask_.data()), 0, mask_bytes_,
                    push_stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(next_unique_count_.data()), 0,
                    sizeof(unsigned long long), push_stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(next_pair_count_.data()), 0,
                    sizeof(unsigned long long), push_stream);
    clear_pull_output(pull_stream);
  }

  void launch_push(mask_t slots, int level, cudaStream_t stream) {
    puercgp::detail::launch_shared_push_warp<policy_t, graph_t, int>(
        graph_, thrust::raw_pointer_cast(frontier_vertices_.data()),
        thrust::raw_pointer_cast(frontier_mask_.data()), current_unique_,
        thrust::raw_pointer_cast(visited_.data()),
        thrust::raw_pointer_cast(push_mask_.data()),
        thrust::raw_pointer_cast(scratch_vertices_.data()),
        thrust::raw_pointer_cast(next_unique_count_.data()),
        thrust::raw_pointer_cast(next_pair_count_.data()),
        thrust::raw_pointer_cast(values_.data()), kQueries, slots, level,
        kThreads, stream);
  }

  void launch_pull(mask_t slots, cudaStream_t stream) {
    puercgp::detail::launch_fused_pull<policy_t, graph_t, int>(
        graph_, kQueries, thrust::raw_pointer_cast(values_.data()),
        thrust::raw_pointer_cast(visited_.data()),
        thrust::raw_pointer_cast(pull_mask_.data()),
        thrust::raw_pointer_cast(flags_.data()),
        thrust::raw_pointer_cast(pairs_.data()), slots, stream);
  }

  float postprocess(const thrust::device_vector<mask_t>& first,
                    const thrust::device_vector<mask_t>& second, bool merge,
                    cudaStream_t stream) {
    puercgp::detail::cuda_event_timer timer;
    timer.begin(stream);
    if (merge) {
      merge_frontiers_kernel<<<puercgp::detail::grid_for(V_, kThreads),
                               kThreads, 0, stream>>>(
          thrust::raw_pointer_cast(first.data()),
          thrust::raw_pointer_cast(second.data()), V_,
          thrust::raw_pointer_cast(combined_mask_.data()),
          thrust::raw_pointer_cast(flags_.data()),
          thrust::raw_pointer_cast(pairs_.data()));
    }
    auto policy = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(policy, flags_.begin(), flags_.end(),
                           offsets_.begin());
    thrust::inclusive_scan(policy, pairs_.begin(), pairs_.end(), pairs_.begin());
    const auto& mask = merge ? combined_mask_ : pull_mask_;
    puercgp::detail::launch_pull_frontier_compact<int>(
        thrust::raw_pointer_cast(mask.data()), V_,
        thrust::raw_pointer_cast(offsets_.data()),
        thrust::raw_pointer_cast(scratch_vertices_.data()), kThreads, stream);
    return timer.end(stream);
  }

  fingerprint_t fingerprint(cudaStream_t stream = nullptr) {
    cudaStreamSynchronize(stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(fp_counts_.data()), 0,
                    kQueries * sizeof(unsigned long long), stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(fp_sums_.data()), 0,
                    kQueries * sizeof(unsigned long long), stream);
    fingerprint_kernel<<<std::min<std::size_t>(1024, V_), kQueries, 0,
                         stream>>>(
        thrust::raw_pointer_cast(visited_.data()),
        thrust::raw_pointer_cast(values_.data()), V_,
        thrust::raw_pointer_cast(fp_counts_.data()),
        thrust::raw_pointer_cast(fp_sums_.data()));
    fingerprint_t result;
    cudaMemcpyAsync(result.counts.data(),
                    thrust::raw_pointer_cast(fp_counts_.data()),
                    sizeof(result.counts), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(result.sums.data(), thrust::raw_pointer_cast(fp_sums_.data()),
                    sizeof(result.sums), cudaMemcpyDeviceToHost, stream);
    if (V_ > 0) {
      cudaMemcpyAsync(&result.unique,
                      thrust::raw_pointer_cast(offsets_.data()) + V_ - 1,
                      sizeof(result.unique), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(&result.pairs,
                      thrust::raw_pointer_cast(pairs_.data()) + V_ - 1,
                      sizeof(result.pairs), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    return result;
  }

  graph_t graph_;
  std::vector<int> sources_;
  std::size_t V_;
  std::size_t value_count_;
  std::size_t mask_bytes_;
  thrust::device_vector<int> device_sources_, values_;
  thrust::device_vector<mask_t> visited_, frontier_mask_, next_frontier_mask_;
  thrust::device_vector<int> frontier_vertices_, next_frontier_vertices_;
  thrust::device_vector<mask_t> push_mask_, pull_mask_, combined_mask_;
  thrust::device_vector<int> scratch_vertices_;
  thrust::device_vector<unsigned long long> unique_count_, next_unique_count_,
      next_pair_count_, flags_, offsets_, pairs_, query_edges_, fp_counts_,
      fp_sums_;
  unsigned long long current_unique_ = 0;
};

bool same_fingerprint(const fingerprint_t& a, const fingerprint_t& b) {
  return a.counts == b.counts && a.sums == b.sums && a.unique == b.unique &&
         a.pairs == b.pairs;
}

bool same_query_fingerprint(const fingerprint_t& a, const fingerprint_t& b,
                            mask_t slots) {
  for (int q = 0; q < kQueries; ++q) {
    if ((slots & (mask_t{1} << q)) == 0) continue;
    if (a.counts[q] != b.counts[q] || a.sums[q] != b.sums[q]) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: bench_query_partition_replay_q64 <graph> "
                 "<target-step> <output.csv> [seed]\n";
    return 2;
  }
  int target_step = std::stoi(argv[2]);
  unsigned int seed = 42;
  int forced_push_count = -1;
  int single_push_sms = -1;
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--seed=", 0) == 0)
      seed = std::stoul(arg.substr(7));
    else if (arg.rfind("--push-count=", 0) == 0)
      forced_push_count = std::stoi(arg.substr(13));
    else if (arg.rfind("--push-sms=", 0) == 0)
      single_push_sms = std::stoi(arg.substr(11));
    else
      seed = std::stoul(arg);
  }
  auto storage = puercgp_examples::load_graph_auto(argv[1], true);
  auto graph = storage.view();
  auto sources = make_sources(storage.vertices, seed);
  replay_workspace workspace(graph, sources);
  workspace.prepare(target_step);
  std::array<unsigned long long, kQueries> query_work{};
  mask_t push_slots = workspace.classify(0.20, query_work);
  if (forced_push_count >= 0) {
    std::array<int, kQueries> order{};
    for (int q = 0; q < kQueries; ++q) order[q] = q;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return query_work[a] < query_work[b];
    });
    push_slots = 0;
    for (int i = 0; i < std::min(forced_push_count, kQueries); ++i)
      push_slots |= mask_t{1} << order[i];
  }
  int push_queries = __builtin_popcountll(push_slots);
  std::cout << "target_step=" << target_step
            << " push_queries=" << push_queries
            << " pull_queries=" << (kQueries - push_queries) << '\n';

  std::vector<measurement_t> results;
  results.push_back(workspace.run_all_pull(target_step));
  results.push_back(workspace.run_all_push(target_step));
  results.push_back(workspace.run_split_serial(target_step, push_slots));
  std::vector<int> sm_options = single_push_sms > 0
      ? std::vector<int>{single_push_sms}
      : std::vector<int>{8, 16, 24, 32};
  for (int push_sms : sm_options)
    results.push_back(
        workspace.run_split_concurrent(target_step, push_slots, push_sms));
  const auto& serial = results[2].fingerprint;
  for (std::size_t i = 3; i < results.size(); ++i) {
    if (!same_query_fingerprint(serial, results[i].fingerprint, push_slots)) {
      for (int q = 0; q < kQueries; ++q) {
        if ((push_slots & (mask_t{1} << q)) == 0) continue;
        if (serial.counts[q] != results[i].fingerprint.counts[q] ||
            serial.sums[q] != results[i].fingerprint.sums[q]) {
          std::cerr << "push fingerprint mismatch method=" << i << " q=" << q
                    << " serial=" << serial.counts[q] << '/'
                    << serial.sums[q] << " concurrent="
                    << results[i].fingerprint.counts[q] << '/'
                    << results[i].fingerprint.sums[q] << '\n';
          break;
        }
      }
    }
  }

  std::filesystem::path output = argv[3];
  std::filesystem::create_directories(output.parent_path());
  std::ofstream csv(output);
  csv << "method,target_step,push_queries,pull_queries,push_sms,pull_sms,"
         "push_ms,pull_ms,concurrent_ms,post_ms,total_ms,push_slots_exact,"
         "full_iteration_exact\n";
  for (const auto& r : results) {
    csv << r.method << ',' << target_step << ',' << push_queries << ','
        << (kQueries - push_queries) << ',' << r.push_sms << ',' << r.pull_sms
        << ',' << r.push_ms << ',' << r.pull_ms << ',' << r.concurrent_ms
        << ',' << r.post_ms << ',' << r.total_ms << ','
        << (r.method == "all_pull" || r.method == "all_push" ||
            same_query_fingerprint(serial, r.fingerprint, push_slots))
        << ',' << (r.method == "all_pull" || r.method == "all_push" ||
                    same_fingerprint(serial, r.fingerprint)) << '\n';
    std::cout << r.method << " sm=" << r.push_sms << '/' << r.pull_sms
              << " total_ms=" << r.total_ms << '\n';
  }
  return 0;
}
