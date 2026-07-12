#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>
#include <puercgp/engine/frontier_engine.hxx>
#include <puercgp/kernels/common/frontier_metrics.hxx>
#include <puercgp/kernels/push/shared_push_kernels.hxx>

#include "mtx_loader.hxx"

namespace {

constexpr int kQueryCount = 32;
constexpr int kThreads = 256;
using mask_t = puercgp::query_mask_t;
using policy_t = puercgp::algorithms::bfs_policy;

struct sample_record {
  int vertex;
  unsigned int degree;
  mask_t mask;
};

struct step_trace {
  int step = 0;
  mask_t alive_mask = 0;
  mask_t scheduled_mask = 0;
  std::array<unsigned long long, kQueryCount> query_edges{};
  std::array<unsigned long long, kQueryCount> query_vertices{};
  unsigned long long ready_union_edges = 0;
  unsigned long long ready_virtual_edges = 0;
  unsigned long long scheduled_union_edges = 0;
  unsigned long long scheduled_virtual_edges = 0;
  std::size_t unique_vertices = 0;
  float push_ms = 0.0f;
  std::array<int, kQueryCount> local_iterations{};
  std::vector<sample_record> samples;
};

struct replay_result {
  std::string strategy;
  float gpu_ms = 0.0f;
  float push_ms = 0.0f;
  int global_steps = 0;
  std::array<unsigned long long, kQueryCount> visited_counts{};
  std::array<unsigned long long, kQueryCount> distance_sums{};
  std::vector<step_trace> steps;
};

struct trace_model {
  int max_levels = 0;
  std::array<int, kQueryCount> query_levels{};
  std::array<int, kQueryCount> peak_levels{};
  std::vector<unsigned long long> overlap;

  std::size_t index(int q, int r, int lq, int lr) const {
    return (((static_cast<std::size_t>(q) * kQueryCount + r) * max_levels +
             lq) *
            max_levels + lr);
  }

  unsigned long long get_overlap(int q, int r, int lq, int lr) const {
    if (lq < 0 || lr < 0 || lq >= query_levels[q] ||
        lr >= query_levels[r]) {
      return 0;
    }
    return overlap[index(q, r, lq, lr)];
  }
};

struct schedule_t {
  std::string name;
  std::vector<mask_t> masks;
  std::vector<mask_t> alive_masks;
  std::vector<std::array<int, kQueryCount>> local_before;
  std::array<int, kQueryCount> offsets{};
};

template <typename graph_t>
__global__ void query_frontier_stats_kernel(
    graph_t graph, const int* frontier_vertices, const mask_t* frontier_mask,
    std::size_t unique_count, mask_t active_mask,
    unsigned long long* query_edges,
    unsigned long long* query_vertices) {
  int q = threadIdx.x;
  if (q >= kQueryCount) return;
  unsigned long long local_edges = 0;
  unsigned long long local_vertices = 0;
  mask_t bit = mask_t{1} << q;
  if ((active_mask & bit) != 0) {
    for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
      int v = frontier_vertices[i];
      if ((frontier_mask[v] & bit) == 0) continue;
      local_edges += static_cast<unsigned long long>(
          graph.get_starting_edge(v + 1) - graph.get_starting_edge(v));
      ++local_vertices;
    }
  }
  if (local_edges != 0) atomicAdd(query_edges + q, local_edges);
  if (local_vertices != 0) atomicAdd(query_vertices + q, local_vertices);
}

template <typename graph_t>
__global__ void sample_frontier_kernel(
    graph_t graph, const int* frontier_vertices, const mask_t* frontier_mask,
    std::size_t unique_count, mask_t active_mask, unsigned int sample_mask,
    sample_record* records, unsigned long long capacity,
    unsigned long long* record_count) {
  std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < unique_count; i += stride) {
    int v = frontier_vertices[i];
    unsigned int hash = static_cast<unsigned int>(v) * 2654435761U;
    if ((hash & sample_mask) != 0) continue;
    mask_t bits = frontier_mask[v] & active_mask;
    if (bits == 0) continue;
    unsigned long long pos = atomicAdd(record_count, 1ULL);
    if (pos < capacity) {
      records[pos] = sample_record{
          v,
          static_cast<unsigned int>(graph.get_starting_edge(v + 1) -
                                    graph.get_starting_edge(v)),
          bits};
    }
  }
}

__global__ void carry_paused_frontier_kernel(
    const int* frontier_vertices, const mask_t* frontier_mask,
    std::size_t unique_count, mask_t paused_mask, mask_t* next_frontier_mask,
    int* next_frontier_vertices, unsigned long long* next_unique_count) {
  std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < unique_count; i += stride) {
    int v = frontier_vertices[i];
    mask_t paused = frontier_mask[v] & paused_mask;
    if (paused == 0) continue;
    mask_t old = puercgp::detail::atomic_or_query_mask(
        next_frontier_mask + v, paused);
    if (old == 0) {
      unsigned long long pos = atomicAdd(next_unique_count, 1ULL);
      next_frontier_vertices[pos] = v;
    }
  }
}

template <typename graph_t>
__global__ void phase_expand_shared_node_warp_kernel(
    graph_t graph, const int* frontier_vertices,
    const mask_t* frontier_mask, std::size_t unique_count,
    mask_t* visited_mask, mask_t* next_frontier_mask,
    int* next_frontier_vertices, unsigned long long* next_unique_count,
    unsigned long long* next_pair_count, int* values,
    const int* next_levels, mask_t scheduled_mask) {
  constexpr int warp_size = 32;
  int lane = threadIdx.x & (warp_size - 1);
  std::size_t warp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  std::size_t warp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) >> 5;
  for (std::size_t i = warp_id; i < unique_count; i += warp_stride) {
    int source = frontier_vertices[i];
    mask_t active = frontier_mask[source] & scheduled_mask;
    if (active == 0) continue;
    auto begin = graph.get_starting_edge(source);
    auto end = graph.get_starting_edge(source + 1);
    for (auto edge = begin + lane; edge < end; edge += warp_size) {
      int neighbor = graph.get_destination_vertex(edge);
      mask_t old = puercgp::detail::atomic_or_query_mask(
          visited_mask + neighbor, active);
      mask_t improved = active & ~old;
      mask_t bits = improved;
      while (bits != 0) {
        int q = __ffsll(static_cast<long long>(bits)) - 1;
        values[puercgp::detail::value_index(
            static_cast<std::size_t>(neighbor), static_cast<std::size_t>(q),
            static_cast<std::size_t>(kQueryCount))] = next_levels[q];
        bits &= bits - 1;
      }
      if (improved != 0) {
        puercgp::detail::mark_next_shared_frontier(
            neighbor, improved, next_frontier_mask, next_frontier_vertices,
            next_unique_count, next_pair_count);
      }
    }
  }
}

__global__ void result_fingerprint_kernel(
    const mask_t* visited_mask, const int* values, std::size_t vertex_count,
    unsigned long long* counts, unsigned long long* distance_sums) {
  int q = threadIdx.x;
  mask_t bit = mask_t{1} << q;
  unsigned long long local_count = 0;
  unsigned long long local_sum = 0;
  for (std::size_t v = blockIdx.x; v < vertex_count; v += gridDim.x) {
    if ((visited_mask[v] & bit) == 0) continue;
    ++local_count;
    local_sum += static_cast<unsigned long long>(values[
        puercgp::detail::value_index(v, static_cast<std::size_t>(q),
                                     static_cast<std::size_t>(kQueryCount))]);
  }
  if (local_count != 0) atomicAdd(counts + q, local_count);
  if (local_sum != 0) atomicAdd(distance_sums + q, local_sum);
}

unsigned int choose_sample_mod(int vertex_count) {
  unsigned int ratio =
      static_cast<unsigned int>(std::max(1, vertex_count / 100000));
  unsigned int mod = 1;
  while ((mod << 1) <= ratio) mod <<= 1;
  return mod;
}

std::vector<int> make_unique_sources(int vertex_count, unsigned int seed) {
  std::mt19937 rng(seed);
  std::unordered_set<int> used;
  std::vector<int> sources;
  sources.reserve(kQueryCount);
  while (sources.size() < kQueryCount) {
    int source = static_cast<int>(rng() % vertex_count);
    if (used.insert(source).second) sources.push_back(source);
  }
  return sources;
}

mask_t valid_mask() { return (mask_t{1} << kQueryCount) - 1; }

template <typename graph_t>
void collect_step_metrics(
    graph_t graph, const thrust::device_vector<int>& frontier_vertices,
    const thrust::device_vector<mask_t>& frontier_mask,
    std::size_t unique_count, mask_t alive_mask, mask_t scheduled_mask,
    thrust::device_vector<unsigned long long>& query_edges_dev,
    thrust::device_vector<unsigned long long>& query_vertices_dev,
    thrust::device_vector<unsigned long long>& counters_dev,
    step_trace& trace, cudaStream_t stream) {
  cudaMemsetAsync(thrust::raw_pointer_cast(query_edges_dev.data()), 0,
                  kQueryCount * sizeof(unsigned long long), stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(query_vertices_dev.data()), 0,
                  kQueryCount * sizeof(unsigned long long), stream);
  int blocks = std::max(1, std::min(1024, static_cast<int>(unique_count)));
  query_frontier_stats_kernel<<<blocks, kQueryCount, 0, stream>>>(
      graph, thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(frontier_mask.data()), unique_count, alive_mask,
      thrust::raw_pointer_cast(query_edges_dev.data()),
      thrust::raw_pointer_cast(query_vertices_dev.data()));

  auto run_union_metrics = [&](mask_t mask, unsigned long long* counters) {
    cudaMemsetAsync(counters, 0, 3 * sizeof(unsigned long long), stream);
    puercgp::detail::launch_compute_shared_frontier_metrics(
        graph, thrust::raw_pointer_cast(frontier_vertices.data()),
        thrust::raw_pointer_cast(frontier_mask.data()), unique_count, mask,
        counters, counters + 1, counters + 2, kThreads, stream);
  };

  run_union_metrics(alive_mask,
                    thrust::raw_pointer_cast(counters_dev.data()));
  run_union_metrics(scheduled_mask,
                    thrust::raw_pointer_cast(counters_dev.data()) + 3);

  std::array<unsigned long long, 6> counters{};
  cudaMemcpyAsync(trace.query_edges.data(),
                  thrust::raw_pointer_cast(query_edges_dev.data()),
                  kQueryCount * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(trace.query_vertices.data(),
                  thrust::raw_pointer_cast(query_vertices_dev.data()),
                  kQueryCount * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(counters.data(),
                  thrust::raw_pointer_cast(counters_dev.data()),
                  counters.size() * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  trace.ready_union_edges = counters[0];
  trace.ready_virtual_edges = counters[1];
  trace.scheduled_union_edges = counters[3];
  trace.scheduled_virtual_edges = counters[4];
}

template <typename graph_t>
replay_result run_schedule(
    graph_t graph, const std::vector<int>& sources,
    const schedule_t* schedule, bool collect_trace, bool collect_samples,
    unsigned int sample_mod, cudaStream_t stream = nullptr,
    bool use_production_push = false) {
  const std::size_t V = graph.get_number_of_vertices();
  const std::size_t value_count = V * kQueryCount;
  const std::size_t mask_bytes = V * sizeof(mask_t);
  thrust::device_vector<int> device_sources(sources);
  thrust::device_vector<int> values(value_count);
  thrust::device_vector<mask_t> visited_mask(V);
  thrust::device_vector<mask_t> frontier_mask(V);
  thrust::device_vector<mask_t> next_frontier_mask(V);
  thrust::device_vector<int> frontier_vertices(V);
  thrust::device_vector<int> next_frontier_vertices(V);
  thrust::device_vector<unsigned long long> unique_count_dev(1);
  thrust::device_vector<unsigned long long> next_unique_count_dev(1);
  thrust::device_vector<unsigned long long> next_pair_count_dev(1);
  thrust::device_vector<unsigned long long> query_edges_dev(kQueryCount);
  thrust::device_vector<unsigned long long> query_vertices_dev(kQueryCount);
  thrust::device_vector<unsigned long long> counters_dev(6);
  thrust::device_vector<unsigned long long> visited_counts_dev(kQueryCount);
  thrust::device_vector<unsigned long long> distance_sums_dev(kQueryCount);
  thrust::device_vector<int> next_levels_dev(kQueryCount);

  unsigned long long sample_capacity =
      static_cast<unsigned long long>(V / sample_mod + 2048);
  thrust::device_vector<sample_record> sample_records(
      collect_samples ? sample_capacity : 1);
  thrust::device_vector<unsigned long long> sample_count_dev(1);

  puercgp::detail::fill_values_kernel<policy_t>
      <<<puercgp::detail::grid_for(value_count, kThreads), kThreads, 0,
         stream>>>(
          thrust::raw_pointer_cast(values.data()), value_count);
  cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0, mask_bytes,
                  stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0, mask_bytes,
                  stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                  mask_bytes, stream);
  puercgp::detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
      thrust::raw_pointer_cast(unique_count_dev.data()));
  puercgp::detail::init_shared_sources_kernel<policy_t>
      <<<1, 64, 0, stream>>>(graph, thrust::raw_pointer_cast(device_sources.data()),
                  kQueryCount, thrust::raw_pointer_cast(values.data()),
                  thrust::raw_pointer_cast(visited_mask.data()),
                  thrust::raw_pointer_cast(frontier_mask.data()),
                  thrust::raw_pointer_cast(frontier_vertices.data()),
                  thrust::raw_pointer_cast(unique_count_dev.data()));

  unsigned long long unique_raw = 0;
  cudaMemcpyAsync(&unique_raw,
                  thrust::raw_pointer_cast(unique_count_dev.data()),
                  sizeof(unique_raw), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::size_t current_unique = static_cast<std::size_t>(unique_raw);

  replay_result result;
  result.strategy = schedule == nullptr ? "baseline" : schedule->name;
  std::array<int, kQueryCount> local{};
  mask_t alive_mask = valid_mask();
  int step = 0;
  puercgp::detail::cuda_event_timer total_timer;
  total_timer.begin(stream);

  while (current_unique > 0) {
    mask_t scheduled_mask = alive_mask;
    if (schedule != nullptr) {
      if (step >= static_cast<int>(schedule->masks.size())) {
        throw std::runtime_error("schedule ended before traversal converged");
      }
      alive_mask = schedule->alive_masks[step];
      scheduled_mask = schedule->masks[step] & alive_mask;
    }

    step_trace trace;
    trace.step = step;
    trace.alive_mask = alive_mask;
    trace.scheduled_mask = scheduled_mask;
    trace.unique_vertices = current_unique;
    trace.local_iterations = local;
    if (collect_trace) {
      collect_step_metrics(
          graph, frontier_vertices, frontier_mask, current_unique, alive_mask,
          scheduled_mask, query_edges_dev, query_vertices_dev, counters_dev,
          trace, stream);
    }

    if (collect_samples) {
      cudaMemsetAsync(thrust::raw_pointer_cast(sample_count_dev.data()), 0,
                      sizeof(unsigned long long), stream);
      sample_frontier_kernel<<<
          puercgp::detail::grid_for(current_unique, kThreads), kThreads>>>(
          graph, thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(frontier_mask.data()), current_unique,
          alive_mask, sample_mod - 1,
          thrust::raw_pointer_cast(sample_records.data()), sample_capacity,
          thrust::raw_pointer_cast(sample_count_dev.data()));
      unsigned long long sample_count = 0;
      cudaMemcpyAsync(&sample_count,
                      thrust::raw_pointer_cast(sample_count_dev.data()),
                      sizeof(sample_count), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      if (sample_count > sample_capacity) {
        throw std::runtime_error("sample buffer capacity exceeded");
      }
      trace.samples.resize(static_cast<std::size_t>(sample_count));
      cudaMemcpyAsync(trace.samples.data(),
                      thrust::raw_pointer_cast(sample_records.data()),
                      sample_count * sizeof(sample_record),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    }

    cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                    mask_bytes, stream);
    puercgp::detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(next_unique_count_dev.data()));
    puercgp::detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(next_pair_count_dev.data()));

    mask_t paused_mask = alive_mask & ~scheduled_mask;
    if (paused_mask != 0) {
      carry_paused_frontier_kernel<<<
          puercgp::detail::grid_for(current_unique, kThreads), kThreads>>>(
          thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(frontier_mask.data()), current_unique,
          paused_mask,
          thrust::raw_pointer_cast(next_frontier_mask.data()),
          thrust::raw_pointer_cast(next_frontier_vertices.data()),
          thrust::raw_pointer_cast(next_unique_count_dev.data()));
    }

    puercgp::detail::cuda_event_timer push_timer;
    std::array<int, kQueryCount> next_levels{};
    for (int q = 0; q < kQueryCount; ++q) next_levels[q] = local[q] + 1;
    cudaMemcpyAsync(thrust::raw_pointer_cast(next_levels_dev.data()),
                    next_levels.data(), kQueryCount * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    push_timer.begin(stream);
    if (scheduled_mask != 0 && use_production_push) {
      puercgp::detail::expand_shared_node_warp_kernel<policy_t, graph_t, int>
          <<<puercgp::detail::grid_for(current_unique * 32, kThreads),
             kThreads, 0, stream>>>(
              graph, thrust::raw_pointer_cast(frontier_vertices.data()),
              thrust::raw_pointer_cast(frontier_mask.data()), current_unique,
              thrust::raw_pointer_cast(visited_mask.data()),
              thrust::raw_pointer_cast(next_frontier_mask.data()),
              thrust::raw_pointer_cast(next_frontier_vertices.data()),
              thrust::raw_pointer_cast(next_unique_count_dev.data()),
              thrust::raw_pointer_cast(next_pair_count_dev.data()),
              thrust::raw_pointer_cast(values.data()), kQueryCount,
              scheduled_mask, step);
    } else if (scheduled_mask != 0) {
      phase_expand_shared_node_warp_kernel
          <<<puercgp::detail::grid_for(current_unique * 32, kThreads),
             kThreads, 0, stream>>>(
              graph, thrust::raw_pointer_cast(frontier_vertices.data()),
              thrust::raw_pointer_cast(frontier_mask.data()), current_unique,
              thrust::raw_pointer_cast(visited_mask.data()),
              thrust::raw_pointer_cast(next_frontier_mask.data()),
              thrust::raw_pointer_cast(next_frontier_vertices.data()),
              thrust::raw_pointer_cast(next_unique_count_dev.data()),
              thrust::raw_pointer_cast(next_pair_count_dev.data()),
              thrust::raw_pointer_cast(values.data()),
              thrust::raw_pointer_cast(next_levels_dev.data()), scheduled_mask);
    }
    trace.push_ms = push_timer.end(stream);
    result.push_ms += trace.push_ms;

    cudaMemcpyAsync(&unique_raw,
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    sizeof(unique_raw), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    current_unique = static_cast<std::size_t>(unique_raw);
    thrust::swap(frontier_mask, next_frontier_mask);
    thrust::swap(frontier_vertices, next_frontier_vertices);

    if (schedule == nullptr) {
      if (collect_trace) {
        for (int q = 0; q < kQueryCount; ++q) {
          if (trace.query_vertices[q] != 0) ++local[q];
        }
      }
    } else {
      for (int q = 0; q < kQueryCount; ++q) {
        if ((scheduled_mask & (mask_t{1} << q)) != 0) ++local[q];
      }
    }
    if (collect_trace) result.steps.push_back(std::move(trace));
    ++step;
    if (step > 4096) throw std::runtime_error("phase schedule did not converge");
  }

  result.gpu_ms = total_timer.end(stream);
  result.global_steps = step;
  cudaMemsetAsync(thrust::raw_pointer_cast(visited_counts_dev.data()), 0,
                  kQueryCount * sizeof(unsigned long long), stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(distance_sums_dev.data()), 0,
                  kQueryCount * sizeof(unsigned long long), stream);
  int fingerprint_blocks =
      std::max(1, std::min(1024, static_cast<int>(V)));
  result_fingerprint_kernel<<<fingerprint_blocks, kQueryCount, 0, stream>>>(
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(values.data()), V,
      thrust::raw_pointer_cast(visited_counts_dev.data()),
      thrust::raw_pointer_cast(distance_sums_dev.data()));
  cudaMemcpyAsync(result.visited_counts.data(),
                  thrust::raw_pointer_cast(visited_counts_dev.data()),
                  kQueryCount * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(result.distance_sums.data(),
                  thrust::raw_pointer_cast(distance_sums_dev.data()),
                  kQueryCount * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return result;
}

trace_model build_trace_model(const replay_result& baseline) {
  trace_model model;
  model.max_levels = static_cast<int>(baseline.steps.size());
  for (int q = 0; q < kQueryCount; ++q) {
    unsigned long long peak_work = 0;
    for (const auto& step : baseline.steps) {
      if (step.query_vertices[q] != 0) {
        model.query_levels[q] = step.step + 1;
      }
      if (step.query_edges[q] > peak_work) {
        peak_work = step.query_edges[q];
        model.peak_levels[q] = step.step;
      }
    }
  }

  struct sampled_vertex_state {
    unsigned int degree = 0;
    std::array<short, kQueryCount> levels;
    sampled_vertex_state() { levels.fill(-1); }
  };
  std::unordered_map<int, sampled_vertex_state> sampled;
  for (const auto& step : baseline.steps) {
    for (const auto& record : step.samples) {
      auto& state = sampled[record.vertex];
      state.degree = record.degree;
      mask_t bits = record.mask;
      while (bits != 0) {
        int q = __builtin_ffsll(static_cast<long long>(bits)) - 1;
        state.levels[q] = static_cast<short>(step.step);
        bits &= bits - 1;
      }
    }
  }

  std::size_t overlap_size = static_cast<std::size_t>(kQueryCount) *
      kQueryCount * model.max_levels * model.max_levels;
  model.overlap.assign(overlap_size, 0);
  for (const auto& item : sampled) {
    const auto& state = item.second;
    for (int q = 0; q < kQueryCount; ++q) {
      int lq = state.levels[q];
      if (lq < 0) continue;
      for (int r = q + 1; r < kQueryCount; ++r) {
        int lr = state.levels[r];
        if (lr < 0) continue;
        model.overlap[model.index(q, r, lq, lr)] += state.degree;
        model.overlap[model.index(r, q, lr, lq)] += state.degree;
      }
    }
  }
  return model;
}

schedule_t make_offset_schedule(
    const std::string& name, const trace_model& model,
    const std::array<int, kQueryCount>& offsets) {
  schedule_t schedule;
  schedule.name = name;
  schedule.offsets = offsets;
  int total_steps = 0;
  for (int q = 0; q < kQueryCount; ++q) {
    total_steps = std::max(total_steps, offsets[q] + model.query_levels[q]);
  }
  std::array<int, kQueryCount> local{};
  for (int t = 0; t < total_steps; ++t) {
    std::array<int, kQueryCount> before;
    before.fill(-1);
    mask_t mask = 0;
    mask_t alive = 0;
    for (int q = 0; q < kQueryCount; ++q) {
      if (local[q] < model.query_levels[q]) {
        before[q] = local[q];
        alive |= mask_t{1} << q;
        if (t >= offsets[q]) mask |= mask_t{1} << q;
      }
    }
    schedule.local_before.push_back(before);
    schedule.alive_masks.push_back(alive);
    schedule.masks.push_back(mask);
    for (int q = 0; q < kQueryCount; ++q) {
      if ((mask & (mask_t{1} << q)) != 0) ++local[q];
    }
  }
  return schedule;
}

schedule_t make_baseline_schedule(const trace_model& model) {
  std::array<int, kQueryCount> offsets{};
  return make_offset_schedule("baseline", model, offsets);
}

schedule_t make_heavy_schedule(const trace_model& model) {
  int latest_peak =
      *std::max_element(model.peak_levels.begin(), model.peak_levels.end());
  std::array<int, kQueryCount> offsets{};
  for (int q = 0; q < kQueryCount; ++q) {
    offsets[q] = latest_peak - model.peak_levels[q];
  }
  int min_offset = *std::min_element(offsets.begin(), offsets.end());
  for (int& offset : offsets) offset -= min_offset;
  return make_offset_schedule("heavy_alignment", model, offsets);
}

unsigned long long aligned_pair_score(
    const trace_model& model, int q, int r, int offset_q, int offset_r) {
  unsigned long long score = 0;
  int delta = offset_q - offset_r;
  for (int lq = 0; lq < model.query_levels[q]; ++lq) {
    int lr = lq + delta;
    score += model.get_overlap(q, r, lq, lr);
  }
  return score;
}

unsigned long long offset_score(
    const trace_model& model,
    const std::array<int, kQueryCount>& offsets) {
  unsigned long long score = 0;
  for (int q = 0; q < kQueryCount; ++q) {
    for (int r = q + 1; r < kQueryCount; ++r) {
      score += aligned_pair_score(model, q, r, offsets[q], offsets[r]);
    }
  }
  return score;
}

schedule_t make_offset_search_schedule(
    const trace_model& model, const schedule_t& heavy, unsigned int seed) {
  int max_delay = std::min(16, model.max_levels);
  std::mt19937 rng(seed);
  std::vector<std::array<int, kQueryCount>> starts;
  starts.push_back({});
  starts.push_back(heavy.offsets);
  for (int i = 0; i < 6; ++i) {
    std::array<int, kQueryCount> candidate{};
    for (int& value : candidate) value = static_cast<int>(rng() % (max_delay + 1));
    starts.push_back(candidate);
  }

  std::array<int, kQueryCount> best{};
  unsigned long long best_score = 0;
  for (auto offsets : starts) {
    for (int pass = 0; pass < 6; ++pass) {
      bool changed = false;
      for (int q = 0; q < kQueryCount; ++q) {
        int best_offset = offsets[q];
        unsigned long long best_local = 0;
        for (int candidate = 0; candidate <= max_delay; ++candidate) {
          unsigned long long local_score = 0;
          for (int r = 0; r < kQueryCount; ++r) {
            if (r == q) continue;
            local_score += aligned_pair_score(
                model, q, r, candidate, offsets[r]);
          }
          if (local_score > best_local) {
            best_local = local_score;
            best_offset = candidate;
          }
        }
        changed |= best_offset != offsets[q];
        offsets[q] = best_offset;
      }
      if (!changed) break;
    }
    int min_offset = *std::min_element(offsets.begin(), offsets.end());
    for (int& offset : offsets) offset -= min_offset;
    unsigned long long score = offset_score(model, offsets);
    if (score > best_score) {
      best_score = score;
      best = offsets;
    }
  }
  return make_offset_schedule("offset_search", model, best);
}

schedule_t make_greedy_pause_schedule(const trace_model& model) {
  schedule_t schedule;
  schedule.name = "greedy_pause";
  std::array<int, kQueryCount> local{};
  std::array<int, kQueryCount> pause_streak{};
  int remaining = kQueryCount;
  while (remaining > 0) {
    std::array<int, kQueryCount> before;
    before.fill(-1);
    std::array<int, kQueryCount> planned_next{};
    mask_t scheduled = 0;
    int scheduled_count = 0;
    for (int q = 0; q < kQueryCount; ++q) {
      if (local[q] < model.query_levels[q]) {
        before[q] = local[q];
        planned_next[q] = local[q] + 1;
        scheduled |= mask_t{1} << q;
        ++scheduled_count;
      } else {
        planned_next[q] = local[q];
      }
    }

    while (scheduled_count > 16) {
      int best_q = -1;
      long long best_gain = 0;
      for (int q = 0; q < kQueryCount; ++q) {
        if ((scheduled & (mask_t{1} << q)) == 0 || pause_streak[q] >= 3)
          continue;
        long long gain = 0;
        for (int r = 0; r < kQueryCount; ++r) {
          if (r == q || before[r] < 0) continue;
          gain += static_cast<long long>(model.get_overlap(
                      q, r, local[q], planned_next[r])) -
                  static_cast<long long>(model.get_overlap(
                      q, r, local[q] + 1, planned_next[r]));
        }
        if (gain > best_gain) {
          best_gain = gain;
          best_q = q;
        }
      }
      if (best_q < 0) break;
      scheduled &= ~(mask_t{1} << best_q);
      planned_next[best_q] = local[best_q];
      --scheduled_count;
    }

    schedule.local_before.push_back(before);
    mask_t alive = 0;
    for (int q = 0; q < kQueryCount; ++q) {
      if (before[q] >= 0) alive |= mask_t{1} << q;
    }
    schedule.alive_masks.push_back(alive);
    schedule.masks.push_back(scheduled);
    remaining = 0;
    for (int q = 0; q < kQueryCount; ++q) {
      if (before[q] < 0) continue;
      if ((scheduled & (mask_t{1} << q)) != 0) {
        ++local[q];
        pause_streak[q] = 0;
      } else {
        ++pause_streak[q];
      }
      if (local[q] < model.query_levels[q]) ++remaining;
    }
    if (schedule.masks.size() > 4096) {
      throw std::runtime_error("greedy pause schedule did not terminate");
    }
  }
  return schedule;
}

void write_sources(const std::filesystem::path& output,
                   const std::vector<int>& sources) {
  std::ofstream file(output / "sources.csv");
  file << "query_id,source\n";
  for (int q = 0; q < kQueryCount; ++q) file << q << ',' << sources[q] << '\n';
}

void write_schedule(const std::filesystem::path& output,
                    const schedule_t& schedule) {
  std::ofstream file(output / (schedule.name + "_schedule.csv"));
  file << "global_step,alive_mask,scheduled_mask,local_iterations\n";
  for (std::size_t t = 0; t < schedule.masks.size(); ++t) {
    file << t << ",0x" << std::hex << std::setw(8) << std::setfill('0')
         << schedule.alive_masks[t] << ",0x" << std::setw(8)
         << schedule.masks[t] << std::dec << std::setfill(' ') << ",\"";
    for (int q = 0; q < kQueryCount; ++q) {
      if (q != 0) file << ';';
      file << schedule.local_before[t][q];
    }
    file << "\"\n";
  }
}

void write_step_trace(const std::filesystem::path& output,
                      const replay_result& result) {
  std::ofstream file(output / (result.strategy + "_steps.csv"));
  file << "step,alive_mask,scheduled_mask,unique_vertices,ready_union_edges,"
          "ready_virtual_edges,ready_sharing,scheduled_union_edges,"
          "scheduled_virtual_edges,scheduled_sharing,push_ms,local_iterations\n";
  for (const auto& step : result.steps) {
    double ready_sharing = step.ready_union_edges == 0
        ? 1.0
        : static_cast<double>(step.ready_virtual_edges) /
              step.ready_union_edges;
    double scheduled_sharing = step.scheduled_union_edges == 0
        ? 1.0
        : static_cast<double>(step.scheduled_virtual_edges) /
              step.scheduled_union_edges;
    file << step.step << ",0x" << std::hex << std::setw(8)
         << std::setfill('0') << step.alive_mask << ",0x" << std::setw(8)
         << step.scheduled_mask << std::dec << std::setfill(' ') << ','
         << step.unique_vertices << ',' << step.ready_union_edges << ','
         << step.ready_virtual_edges << ',' << ready_sharing << ','
         << step.scheduled_union_edges << ',' << step.scheduled_virtual_edges
         << ',' << scheduled_sharing << ',' << step.push_ms << ",\"";
    for (int q = 0; q < kQueryCount; ++q) {
      if (q != 0) file << ';';
      file << step.local_iterations[q];
    }
    file << "\"\n";
  }
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: bench_phase_schedule <graph> <output-dir> "
                 "[--seed=42] [--repeats=3]\n";
    return 2;
  }
  std::string graph_path = argv[1];
  std::filesystem::path output = argv[2];
  unsigned int seed = 42;
  int repeats = 3;
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--seed=", 0) == 0)
      seed = static_cast<unsigned int>(std::stoul(arg.substr(7)));
    else if (arg.rfind("--repeats=", 0) == 0)
      repeats = std::stoi(arg.substr(10));
  }
  std::filesystem::create_directories(output);

  auto graph_storage = puercgp_examples::load_graph_auto(graph_path, false);
  auto graph = graph_storage.view();
  auto sources = make_unique_sources(graph_storage.vertices, seed);
  unsigned int sample_mod = choose_sample_mod(graph_storage.vertices);
  write_sources(output, sources);

  std::cout << "graph=" << graph_path << " V=" << graph_storage.vertices
            << " E=" << graph_storage.edges << " Q=N=" << kQueryCount
            << " seed=" << seed << " sample_mod=" << sample_mod << '\n';

  replay_result baseline_trace =
      run_schedule(graph, sources, nullptr, true, true, sample_mod);
  trace_model model = build_trace_model(baseline_trace);
  schedule_t baseline = make_baseline_schedule(model);
  schedule_t heavy = make_heavy_schedule(model);
  schedule_t offset = make_offset_search_schedule(model, heavy, seed + 1);
  schedule_t greedy = make_greedy_pause_schedule(model);
  std::vector<schedule_t> schedules = {baseline, heavy, offset, greedy};

  for (const auto& schedule : schedules) write_schedule(output, schedule);

  std::ofstream summary(output / "summary.csv");
  summary << "strategy,gpu_ms,push_ms,global_steps,correct\n";
  std::array<unsigned long long, kQueryCount> reference_counts{};
  std::array<unsigned long long, kQueryCount> reference_distance_sums{};
  for (const auto& schedule : schedules) {
    replay_result traced =
        run_schedule(graph, sources, &schedule, true, false, sample_mod);
    write_step_trace(output, traced);
    std::vector<float> gpu_times;
    std::vector<float> push_times;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      replay_result timed =
          run_schedule(graph, sources, &schedule, false, false, sample_mod);
      gpu_times.push_back(timed.gpu_ms);
      push_times.push_back(timed.push_ms);
    }
    bool correct = true;
    if (schedule.name == "baseline") {
      reference_counts = traced.visited_counts;
      reference_distance_sums = traced.distance_sums;
    } else {
      correct = traced.visited_counts == reference_counts &&
                traced.distance_sums == reference_distance_sums;
    }
    float gpu_ms = median(gpu_times);
    float push_ms = median(push_times);
    summary << schedule.name << ',' << gpu_ms << ',' << push_ms << ','
            << traced.global_steps << ',' << (correct ? 1 : 0) << '\n';
    std::cout << schedule.name << ": gpu_ms=" << gpu_ms
              << " push_ms=" << push_ms
              << " steps=" << traced.global_steps
              << " correct=" << (correct ? "yes" : "no") << '\n';
    if (!correct) return 1;
  }
  return 0;
}
