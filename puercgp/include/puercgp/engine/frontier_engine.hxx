#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <puercgp/algorithms/bfs.hxx>
#include <puercgp/algorithms/sssp.hxx>
#include <puercgp/algorithms/sswp.hxx>
#include <puercgp/algorithms/wcc.hxx>
#include <puercgp/backend/graph_adapter.hxx>
#include <puercgp/backend/pull_graph_access.hxx>
#include <puercgp/core/atomics.hxx>
#include <puercgp/core/cuda_utils.hxx>
#include <puercgp/core/layout.hxx>
#include <puercgp/core/mask.hxx>
#include <puercgp/core/query_batch.hxx>
#include <puercgp/core/result.hxx>
#include <puercgp/engine/query_partition.hxx>
#include <puercgp/engine/pull_executor.hxx>
#include <puercgp/engine/push_executor.hxx>
#include <puercgp/kernels/common/frontier_metrics.hxx>
#include <puercgp/kernels/common/pull_postprocess.hxx>
#include <puercgp/kernels/pull/fused_pull_kernels.hxx>
#include <puercgp/kernels/push/shared_push_kernels.hxx>
#include <puercgp/scheduling/slot_start_schedule.hxx>
#include <puercgp/state/engine_workspace.hxx>
#include <puercgp/state/pull_workspace.hxx>

namespace puercgp {
namespace detail {

__global__ void reset_counter_kernel(unsigned long long* counter) {
  *counter = 0ULL;
}

template <typename Policy, typename value_t>
__global__ void fill_values_kernel(value_t* values, std::size_t total) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = tid; i < total; i += stride) {
    values[i] = Policy::infinity();
  }
}

template <typename Policy, typename graph_t, typename vertex_t>
__global__ void init_shared_sources_kernel(
    graph_t graph,
    const vertex_t* sources,
    int query_count,
    typename Policy::value_type* values,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    vertex_t* frontier_vertices,
    unsigned long long* unique_count) {
  (void)graph;
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t q = tid; q < static_cast<std::size_t>(query_count);
       q += stride) {
    vertex_t source = sources[q];
    query_mask_t bit = query_bit(static_cast<int>(q));
    values[value_index(static_cast<std::size_t>(source), q,
                       static_cast<std::size_t>(query_count))] =
        Policy::source_value();
    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      atomic_or_query_mask(visited_mask + source, bit);
    }
    query_mask_t old = atomic_or_query_mask(frontier_mask + source, bit);
    if (old == 0) {
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = source;
    }
  }
}

template <typename Policy, typename vertex_t>
__global__ void activate_shared_sources_kernel(
    const vertex_t* sources,
    int query_count,
    query_mask_t activation_mask,
    typename Policy::value_type* values,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    vertex_t* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t q = tid; q < static_cast<std::size_t>(query_count);
       q += stride) {
    query_mask_t bit = query_bit(static_cast<int>(q));
    if ((activation_mask & bit) == 0) {
      continue;
    }
    vertex_t source = sources[q];
    values[value_index(static_cast<std::size_t>(source), q,
                       static_cast<std::size_t>(query_count))] =
        Policy::source_value();
    if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
      atomic_or_query_mask(visited_mask + source, bit);
    }
    query_mask_t old = atomic_or_query_mask(frontier_mask + source, bit);
    if (old == 0) {
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = source;
    }
  }
}

template <typename graph_t, typename vertex_t>
__global__ void init_wcc_labels_kernel(
    graph_t graph,
    int query_count,
    float* values,
    query_mask_t* frontier_mask,
    vertex_t* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t vertex_count = graph.get_number_of_vertices();
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t vertex = tid; vertex < vertex_count; vertex += stride) {
    for (int q = 0; q < query_count; ++q) {
      values[value_index(vertex, static_cast<std::size_t>(q),
                         static_cast<std::size_t>(query_count))] =
          static_cast<float>(static_cast<int>(vertex));
    }
    for (int q = 0; q < query_count; ++q) {
      query_mask_t bit = query_bit(q);
      query_mask_t old =
          atomic_or_query_mask(frontier_mask + static_cast<vertex_t>(vertex),
                               bit);
      if (old == 0) {
        unsigned long long position = atomicAdd(unique_count, 1ULL);
        frontier_vertices[position] = static_cast<vertex_t>(vertex);
      }
    }
  }
}

}  // namespace detail

template <typename Policy>
class frontier_engine {
 public:
  using vertex_type = typename Policy::vertex_type;
  using value_type = typename Policy::value_type;
  using result_type = run_result_t<vertex_type, value_type>;

  template <typename graph_t>
  result_type run(graph_t& graph,
                  const query_batch<vertex_type>& queries,
                  execution_context& context,
                  const run_options& options) const {
    return run_impl(graph, queries, context, options, nullptr);
  }

  template <typename graph_t>
  result_type run(
      graph_t& graph, const query_batch<vertex_type>& queries,
      execution_context& context, const run_options& options,
      const scheduling::slot_start_schedule& start_schedule) const {
    return run_impl(graph, queries, context, options, &start_schedule);
  }

 private:
  template <typename graph_t>
  result_type run_impl(
      graph_t& graph, const query_batch<vertex_type>& queries,
      execution_context& context, const run_options& options,
      const scheduling::slot_start_schedule* start_schedule) const {
    static_assert(std::is_same<vertex_type, int>::value,
                  "puercgp frontier policies currently require int vertices");

    queries.validate(options.max_queries);
    if (options.pull_bidirectional_period == 0) {
      throw std::invalid_argument("pull_bidirectional_period must be positive");
    }
    const bool degree_partitioned_pull =
        options.pull_strategy == pull_strategy_t::degree_aware ||
        options.pull_strategy == pull_strategy_t::degree_segmented;
    if (degree_partitioned_pull &&
        (!(options.pull_degree_threshold_2 > 0 &&
           options.pull_degree_threshold_2 < options.pull_degree_threshold_4 &&
           options.pull_degree_threshold_4 < options.pull_degree_threshold_8) ||
         (options.pull_sweep != pull_sweep_t::forward &&
          options.pull_sweep != pull_sweep_t::bidirectional) ||
         options.pull_degree_bucket_order.size() != 4 ||
         options.pull_high_degree_segment_edges <= 0 ||
         (options.pull_high_degree_segment_threads != 128 &&
          options.pull_high_degree_segment_threads != 256 &&
          options.pull_high_degree_segment_threads != 512 &&
          options.pull_high_degree_segment_threads != 1024))) {
      throw std::invalid_argument("invalid degree-aware pull configuration");
    }
    if (degree_partitioned_pull) {
      auto order = options.pull_degree_bucket_order;
      std::sort(order.begin(), order.end());
      if (order != std::vector<int>({0, 1, 2, 3})) {
        throw std::invalid_argument(
            "degree-aware bucket order must be a permutation of 0..3");
      }
    }
    const int query_count = static_cast<int>(queries.size());
    if (query_count > 64) {
      throw std::invalid_argument("shared frontier supports at most 64 queries");
    }
    const bool scheduled = start_schedule != nullptr;
    if (scheduled) {
      start_schedule->validate(query_count);
      if constexpr (Policy::init_mode !=
                    algorithms::init_mode_t::single_source) {
        throw std::invalid_argument(
            "slot start schedules require a single-source algorithm");
      }
    }

    const auto vertex_count =
        static_cast<std::size_t>(graph.get_number_of_vertices());
    if (options.pull_sweep == pull_sweep_t::bidirectional) {
      const auto reverse_end = options.pull_reverse_vertex_end == 0
          ? vertex_count
          : options.pull_reverse_vertex_end;
      if (options.pull_reverse_vertex_begin >= reverse_end ||
          reverse_end > vertex_count) {
        throw std::invalid_argument(
            "pull reverse vertex range must be within [0, V)");
      }
    }
    const std::size_t value_count =
        vertex_count * static_cast<std::size_t>(query_count);

    cudaStream_t stream = context.stream();
    auto thrust_policy = thrust::cuda::par.on(stream);
    constexpr int threads = 256;
    const auto mask_bytes = vertex_count * sizeof(query_mask_t);

    thrust::device_vector<vertex_type> device_sources(queries.sources());
    engine_workspace<vertex_type, value_type> workspace;
    workspace.resize(vertex_count, static_cast<std::size_t>(query_count));
    auto& values = workspace.values_vector();
    thrust::device_vector<value_type> pull_input_values(
        options.pull_update_mode == pull_update_mode_t::synchronous
            ? value_count
            : 0);
    auto& visited_mask = workspace.visited_mask_vector();
    auto& frontier_mask = workspace.frontier_mask_vector();
    auto& next_frontier_mask = workspace.next_frontier_mask_vector();
    auto& frontier_vertices = workspace.frontier_vertices_vector();
    auto& next_frontier_vertices = workspace.next_frontier_vertices_vector();
    auto& unique_count_dev = workspace.current_unique_count_vector();
    auto& next_unique_count_dev = workspace.next_unique_count_vector();
    auto& next_pair_count_dev = workspace.next_pair_count_vector();
    thrust::device_vector<unsigned long long> frontier_metric_counters(3);
    thrust::device_vector<query_mask_t> active_union_dev(scheduled ? 1 : 0);
    thrust::device_vector<unsigned long long> pull_update_iteration_sum(
        options.trace_pull_updates ? vertex_count : 0);
    thrust::device_vector<unsigned int> pull_update_count(
        options.trace_pull_updates ? vertex_count : 0);
    thrust::device_vector<unsigned int> pull_update_first(
        options.trace_pull_updates ? vertex_count : 0);
    thrust::device_vector<unsigned int> pull_update_last(
        options.trace_pull_updates ? vertex_count : 0);
    thrust::device_vector<int> device_start_levels;
    if (scheduled) {
      device_start_levels = start_schedule->offsets();
    }

    pull_workspace pull_state;
    pull_state.resize(vertex_count);
    auto& unique_flags = pull_state.unique_flags_vector();
    auto& pair_counts = pull_state.pair_counts_vector();
    auto& unique_offsets = pull_state.unique_offsets_vector();

    std::array<std::size_t, 5> pull_degree_offsets{};
    thrust::device_vector<vertex_type> pull_degree_vertices;
    using edge_type = typename graph_t::edge_type;
    thrust::device_vector<vertex_type> pull_segment_vertices;
    thrust::device_vector<edge_type> pull_segment_begins;
    thrust::device_vector<edge_type> pull_segment_ends;
    std::size_t pull_segment_count = 0;
    if (degree_partitioned_pull &&
        options.traversal_mode != traversal_mode_t::push) {
      thrust::device_vector<unsigned long long> bucket_counts(4);
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(bucket_counts.data()), 0,
                          4 * sizeof(unsigned long long), stream),
          "cudaMemsetAsync(pull_degree_bucket_counts)");
      detail::count_pull_degree_buckets_kernel<graph_t, vertex_type>
          <<<detail::grid_for(vertex_count, threads), threads, 0, stream>>>(
              graph, options.pull_degree_threshold_2,
              options.pull_degree_threshold_4,
              options.pull_degree_threshold_8,
              thrust::raw_pointer_cast(bucket_counts.data()));
      detail::throw_if_cuda_error(cudaGetLastError(),
                                  "count_pull_degree_buckets_kernel");
      std::array<unsigned long long, 4> host_counts{};
      detail::throw_if_cuda_error(
          cudaMemcpyAsync(host_counts.data(),
                          thrust::raw_pointer_cast(bucket_counts.data()),
                          host_counts.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync(pull_degree_bucket_counts)");
      context.synchronize();
      for (std::size_t bucket = 0; bucket < host_counts.size(); ++bucket) {
        pull_degree_offsets[bucket + 1] =
            pull_degree_offsets[bucket] +
            static_cast<std::size_t>(host_counts[bucket]);
      }
      pull_degree_vertices.resize(vertex_count);
      thrust::device_vector<unsigned long long> bucket_offsets(4);
      thrust::device_vector<unsigned long long> bucket_positions(4);
      std::array<unsigned long long, 4> host_offsets{};
      for (std::size_t bucket = 0; bucket < host_offsets.size(); ++bucket) {
        host_offsets[bucket] = pull_degree_offsets[bucket];
      }
      detail::throw_if_cuda_error(
          cudaMemcpyAsync(thrust::raw_pointer_cast(bucket_offsets.data()),
                          host_offsets.data(),
                          host_offsets.size() * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync(pull_degree_bucket_offsets)");
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(bucket_positions.data()), 0,
                          4 * sizeof(unsigned long long), stream),
          "cudaMemsetAsync(pull_degree_bucket_positions)");
      detail::fill_pull_degree_buckets_kernel<graph_t, vertex_type>
          <<<detail::grid_for(vertex_count, threads), threads, 0, stream>>>(
              graph, options.pull_degree_threshold_2,
              options.pull_degree_threshold_4,
              options.pull_degree_threshold_8,
              thrust::raw_pointer_cast(bucket_offsets.data()),
              thrust::raw_pointer_cast(bucket_positions.data()),
              thrust::raw_pointer_cast(pull_degree_vertices.data()));
      detail::throw_if_cuda_error(cudaGetLastError(),
                                  "fill_pull_degree_buckets_kernel");
      for (std::size_t bucket = 0; bucket < 4; ++bucket) {
        thrust::sort(thrust_policy,
                     pull_degree_vertices.begin() +
                         static_cast<std::ptrdiff_t>(
                             pull_degree_offsets[bucket]),
                     pull_degree_vertices.begin() +
                         static_cast<std::ptrdiff_t>(
                             pull_degree_offsets[bucket + 1]));
      }
      if (options.pull_strategy == pull_strategy_t::degree_segmented) {
        const std::size_t high_degree_count =
            pull_degree_offsets[4] - pull_degree_offsets[3];
        thrust::device_vector<unsigned int> segment_counts(high_degree_count);
        thrust::device_vector<unsigned int> segment_offsets(
            high_degree_count + 1);
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(segment_offsets.data()), 0,
                            sizeof(unsigned int), stream),
            "cudaMemsetAsync(pull_segment_offsets)");
        detail::count_pull_high_degree_segments_kernel<graph_t, vertex_type>
            <<<detail::grid_for(high_degree_count, threads), threads, 0,
               stream>>>(
                graph,
                thrust::raw_pointer_cast(pull_degree_vertices.data()) +
                    pull_degree_offsets[3],
                high_degree_count, options.pull_high_degree_segment_edges,
                thrust::raw_pointer_cast(segment_counts.data()));
        detail::throw_if_cuda_error(
            cudaGetLastError(), "count_pull_high_degree_segments_kernel");
        thrust::inclusive_scan(
            thrust_policy, segment_counts.begin(), segment_counts.end(),
            segment_offsets.begin() + 1);
        unsigned int host_segment_count = 0;
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(
                &host_segment_count,
                thrust::raw_pointer_cast(segment_offsets.data()) +
                    high_degree_count,
                sizeof(host_segment_count), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync(pull_segment_count)");
        context.synchronize();
        pull_segment_count = host_segment_count;
        pull_segment_vertices.resize(pull_segment_count);
        pull_segment_begins.resize(pull_segment_count);
        pull_segment_ends.resize(pull_segment_count);
        detail::fill_pull_high_degree_segments_kernel<graph_t, vertex_type>
            <<<detail::grid_for(high_degree_count, threads), threads, 0,
               stream>>>(
                graph,
                thrust::raw_pointer_cast(pull_degree_vertices.data()) +
                    pull_degree_offsets[3],
                high_degree_count, options.pull_high_degree_segment_edges,
                thrust::raw_pointer_cast(segment_offsets.data()),
                thrust::raw_pointer_cast(pull_segment_vertices.data()),
                thrust::raw_pointer_cast(pull_segment_begins.data()),
                thrust::raw_pointer_cast(pull_segment_ends.data()));
        detail::throw_if_cuda_error(
            cudaGetLastError(), "fill_pull_high_degree_segments_kernel");
        context.synchronize();
      }
    }

    auto wall_start = std::chrono::high_resolution_clock::now();
    detail::cuda_event_timer total_timer;
    total_timer.begin(stream);

    detail::fill_values_kernel<Policy>
        <<<detail::grid_for(value_count, threads), threads, 0, stream>>>(
            thrust::raw_pointer_cast(values.data()), value_count);
    detail::throw_if_cuda_error(cudaGetLastError(), "fill_values_kernel");

    detail::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0,
                        mask_bytes, stream),
        "cudaMemsetAsync(visited_mask)");
    detail::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                        mask_bytes, stream),
        "cudaMemsetAsync(frontier_mask)");
    detail::throw_if_cuda_error(
        cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                        mask_bytes, stream),
        "cudaMemsetAsync(next_frontier_mask)");
    if (options.trace_pull_updates && vertex_count != 0) {
      detail::throw_if_cuda_error(
          cudaMemsetAsync(
              thrust::raw_pointer_cast(pull_update_iteration_sum.data()), 0,
              vertex_count * sizeof(unsigned long long), stream),
          "cudaMemsetAsync(pull_update_iteration_sum)");
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(pull_update_count.data()), 0,
                          vertex_count * sizeof(unsigned int), stream),
          "cudaMemsetAsync(pull_update_count)");
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(pull_update_first.data()), 0,
                          vertex_count * sizeof(unsigned int), stream),
          "cudaMemsetAsync(pull_update_first)");
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(pull_update_last.data()), 0,
                          vertex_count * sizeof(unsigned int), stream),
          "cudaMemsetAsync(pull_update_last)");
    }
    detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(unique_count_dev.data()));
    detail::throw_if_cuda_error(cudaGetLastError(), "reset_counter_kernel");

    if (!scheduled) {
      if constexpr (std::is_same<Policy, algorithms::wcc_policy>::value) {
        detail::init_wcc_labels_kernel<graph_t, vertex_type>
            <<<detail::grid_for(vertex_count, threads), threads, 0, stream>>>(
                graph, query_count, thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(frontier_mask.data()),
                thrust::raw_pointer_cast(frontier_vertices.data()),
                thrust::raw_pointer_cast(unique_count_dev.data()));
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "init_wcc_labels_kernel");
      } else {
        detail::init_shared_sources_kernel<Policy>
            <<<detail::grid_for(queries.size(), threads), threads, 0, stream>>>(
                graph, thrust::raw_pointer_cast(device_sources.data()),
                query_count, thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(visited_mask.data()),
                thrust::raw_pointer_cast(frontier_mask.data()),
                thrust::raw_pointer_cast(frontier_vertices.data()),
                thrust::raw_pointer_cast(unique_count_dev.data()));
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "init_shared_sources_kernel");
      }
    }

    unsigned long long current_unique_raw = 0;
    detail::throw_if_cuda_error(
        cudaMemcpyAsync(&current_unique_raw,
                        thrust::raw_pointer_cast(unique_count_dev.data()),
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                        stream),
        "cudaMemcpyAsync(initial_unique_count)");
    context.synchronize();

    std::size_t current_unique_count =
        static_cast<std::size_t>(current_unique_raw);
    std::size_t current_count = scheduled ? 0 : queries.size();
    if constexpr (std::is_same<Policy, algorithms::wcc_policy>::value) {
      if (!scheduled) {
        current_count =
            current_unique_count * static_cast<std::size_t>(query_count);
      }
    }

    const double pull_frontier_threshold =
        options.pull_frontier_ratio * static_cast<double>(value_count);
    const double default_pull_edge_threshold =
        options.pull_edge_ratio * static_cast<double>(query_count) *
        static_cast<double>(graph.get_number_of_edges());
    const bool has_pull_adjacency = detail::graph_has_pull_adjacency(graph);
    if (options.traversal_mode == traversal_mode_t::pull &&
        !has_pull_adjacency) {
      throw std::invalid_argument(
          "fused pull requires a graph view with incoming adjacency");
    }

    result_type result;
    result.options = options;
    result.effective_query_dim = query_count;
    result.frontier_sizes.push_back(current_count);
    result.unique_frontier_sizes.push_back(current_unique_count);
    const query_partition_t default_partition =
        query_partition_t::all_slots(query_count, options.traversal_mode);

    vertex_type level = 0;
    std::size_t pull_iteration_count = 0;
    query_mask_t live_slots = scheduled ? query_mask_t{0}
                                        : default_partition.active_slots;
    std::vector<int> query_completion_levels(
        static_cast<std::size_t>(query_count), -1);
    std::vector<float> query_completion_wall_ms(
        static_cast<std::size_t>(query_count), 0.0f);
    const int last_start_step = scheduled ? start_schedule->max_offset() : -1;
    bool bfs_pull_phase = false;
    while ((current_unique_count > 0 ||
            (scheduled && static_cast<int>(level) <= last_start_step)) &&
           (options.max_iterations <= 0 ||
            result.iterations < options.max_iterations)) {
      if (scheduled) {
        const query_mask_t activation_mask =
            start_schedule->activation_mask(static_cast<int>(level));
        if (activation_mask != 0) {
          current_unique_raw =
              static_cast<unsigned long long>(current_unique_count);
          detail::throw_if_cuda_error(
              cudaMemcpyAsync(
                  thrust::raw_pointer_cast(unique_count_dev.data()),
                  &current_unique_raw, sizeof(current_unique_raw),
                  cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(current_unique_count_for_activation)");
          detail::activate_shared_sources_kernel<Policy, vertex_type>
              <<<detail::grid_for(queries.size(), threads), threads, 0,
                 stream>>>(
                  thrust::raw_pointer_cast(device_sources.data()), query_count,
                  activation_mask, thrust::raw_pointer_cast(values.data()),
                  thrust::raw_pointer_cast(visited_mask.data()),
                  thrust::raw_pointer_cast(frontier_mask.data()),
                  thrust::raw_pointer_cast(frontier_vertices.data()),
                  thrust::raw_pointer_cast(unique_count_dev.data()));
          detail::throw_if_cuda_error(cudaGetLastError(),
                                      "activate_shared_sources_kernel");
          detail::throw_if_cuda_error(
              cudaMemcpyAsync(
                  &current_unique_raw,
                  thrust::raw_pointer_cast(unique_count_dev.data()),
                  sizeof(current_unique_raw), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(current_unique_count_after_activation)");
          context.synchronize();
          current_unique_count =
              static_cast<std::size_t>(current_unique_raw);
          current_count += static_cast<std::size_t>(
              __builtin_popcountll(
                  static_cast<unsigned long long>(activation_mask)));
          live_slots |= activation_mask;
        }
        if (current_unique_count == 0) {
          ++level;
          continue;
        }
      }

      const query_mask_t iteration_active_slots =
          scheduled ? live_slots : default_partition.active_slots;
      const int active_slot_count = scheduled
          ? __builtin_popcountll(
                static_cast<unsigned long long>(iteration_active_slots))
          : query_count;
      const double iteration_pull_edge_threshold = scheduled
          ? options.pull_edge_ratio * static_cast<double>(active_slot_count) *
                static_cast<double>(graph.get_number_of_edges())
          : default_pull_edge_threshold;

      iteration_profile_t profile;
      profile.iteration = static_cast<int>(level);
      profile.frontier_size = current_count;
      profile.unique_frontier_size = current_unique_count;
      profile.pull_frontier_threshold = pull_frontier_threshold;
      profile.pull_edge_threshold = iteration_pull_edge_threshold;
      auto iteration_start = std::chrono::high_resolution_clock::now();

      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()),
                          0, mask_bytes, stream),
          "cudaMemsetAsync(next_frontier_mask)");
      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(next_unique_count_dev.data()));
      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(next_pair_count_dev.data()));
      if (scheduled) {
        detail::throw_if_cuda_error(
            cudaMemsetAsync(thrust::raw_pointer_cast(active_union_dev.data()),
                            0, sizeof(query_mask_t), stream),
            "cudaMemsetAsync(active_union)");
      }
      detail::throw_if_cuda_error(cudaGetLastError(),
                                  "reset_frontier_counters");

      unsigned long long actual_edges = 0;
      unsigned long long virtual_edges = 0;
      unsigned long long active_pairs = 0;
      bool frontier_metrics_ready = false;

      auto compute_frontier_metrics = [&]() {
        if (frontier_metrics_ready) {
          return;
        }
        detail::throw_if_cuda_error(
            cudaMemsetAsync(
                thrust::raw_pointer_cast(frontier_metric_counters.data()), 0,
                frontier_metric_counters.size() * sizeof(unsigned long long),
                stream),
            "cudaMemsetAsync(frontier_metric_counters)");
        profile.degree_scan_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::launch_compute_shared_frontier_metrics<graph_t,
                                                             vertex_type>(
                  graph, thrust::raw_pointer_cast(frontier_vertices.data()),
                  thrust::raw_pointer_cast(frontier_mask.data()),
                  current_unique_count, iteration_active_slots,
                  thrust::raw_pointer_cast(frontier_metric_counters.data()),
                  thrust::raw_pointer_cast(frontier_metric_counters.data()) + 1,
                  thrust::raw_pointer_cast(frontier_metric_counters.data()) + 2,
                  threads, stream);
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "compute_shared_frontier_metrics_kernel");

        unsigned long long host_metrics[3] = {0ULL, 0ULL, 0ULL};
        auto sync_start = std::chrono::high_resolution_clock::now();
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(host_metrics,
                            thrust::raw_pointer_cast(
                                frontier_metric_counters.data()),
                            sizeof(host_metrics), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(frontier_metric_counters)");
        context.synchronize();
        profile.count_sync_ms += detail::elapsed_ms(sync_start);

        actual_edges = host_metrics[0];
        virtual_edges = host_metrics[1];
        active_pairs = host_metrics[2];
        profile.edge_count = virtual_edges;
        profile.actual_edge_count = actual_edges;
        profile.virtual_edge_count = virtual_edges;
        profile.active_pair_count = active_pairs;
        frontier_metrics_ready = true;
      };

      // Trace runs collect logical adjacency reuse for every execution mode
      // and every iteration.  Formal timing keeps this disabled so the extra
      // reduction and host synchronization cannot perturb reported runtime.
      if (options.profile_iterations) {
        compute_frontier_metrics();
      }

      bool use_pull = false;
      if (options.traversal_mode == traversal_mode_t::pull) {
        use_pull = true;
      } else if (options.traversal_mode == traversal_mode_t::hybrid) {
        if constexpr (std::is_same<Policy,
                                   algorithms::bfs_policy>::value) {
          use_pull = bfs_pull_phase;
        }
        if (has_pull_adjacency && !use_pull) {
          compute_frontier_metrics();
          if (active_pairs != 0) {
            current_count = static_cast<std::size_t>(active_pairs);
            profile.frontier_size = current_count;
          }
          use_pull = static_cast<double>(virtual_edges) >=
              iteration_pull_edge_threshold;
        }
      }
      if constexpr (std::is_same<Policy, algorithms::bfs_policy>::value) {
        bfs_pull_phase |= use_pull;
      }

      unsigned long long next_count = 0;
      unsigned long long next_unique_count = 0;
      query_mask_t next_live_slots = scheduled ? live_slots : query_mask_t{0};

      if (use_pull) {
        profile.mode = "pull";
        profile.pull_kernel_ms = detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              const value_type* pull_input =
                  thrust::raw_pointer_cast(values.data());
              if (options.pull_update_mode ==
                  pull_update_mode_t::synchronous) {
                detail::throw_if_cuda_error(
                    cudaMemcpyAsync(
                        thrust::raw_pointer_cast(pull_input_values.data()),
                        thrust::raw_pointer_cast(values.data()),
                        value_count * sizeof(value_type),
                        cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpyAsync(synchronous_pull_snapshot)");
                pull_input =
                    thrust::raw_pointer_cast(pull_input_values.data());
              }
              if (scheduled) {
                detail::launch_fused_pull_compute_scheduled<
                    Policy, graph_t, vertex_type>(
                    graph, query_count,
                    thrust::raw_pointer_cast(values.data()),
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(unique_flags.data()),
                    thrust::raw_pointer_cast(pair_counts.data()),
                    iteration_active_slots,
                    thrust::raw_pointer_cast(active_union_dev.data()), stream,
                    pull_input);
              } else {
                if (degree_partitioned_pull) {
                  const int* degree_order =
                      options.pull_degree_bucket_order.data();
                  if (options.pull_strategy ==
                      pull_strategy_t::degree_segmented) {
                    detail::launch_fused_pull_degree_segmented<
                        Policy, graph_t, vertex_type>(
                        graph, query_count,
                        thrust::raw_pointer_cast(pull_degree_vertices.data()),
                        pull_degree_offsets.data(), degree_order,
                        thrust::raw_pointer_cast(pull_segment_vertices.data()),
                        thrust::raw_pointer_cast(pull_segment_begins.data()),
                        thrust::raw_pointer_cast(pull_segment_ends.data()),
                        pull_segment_count,
                        thrust::raw_pointer_cast(values.data()),
                        thrust::raw_pointer_cast(visited_mask.data()),
                        thrust::raw_pointer_cast(next_frontier_mask.data()),
                        thrust::raw_pointer_cast(unique_flags.data()),
                        thrust::raw_pointer_cast(pair_counts.data()),
                        iteration_active_slots,
                        options.pull_high_degree_segment_threads, stream,
                        pull_input);
                  } else {
                    detail::launch_fused_pull_degree_aware<Policy, graph_t,
                                                           vertex_type>(
                        graph, query_count,
                        thrust::raw_pointer_cast(pull_degree_vertices.data()),
                        pull_degree_offsets.data(), degree_order,
                        thrust::raw_pointer_cast(values.data()),
                        thrust::raw_pointer_cast(visited_mask.data()),
                        thrust::raw_pointer_cast(next_frontier_mask.data()),
                        thrust::raw_pointer_cast(unique_flags.data()),
                        thrust::raw_pointer_cast(pair_counts.data()),
                        iteration_active_slots, stream, pull_input);
                  }
                  const bool add_reverse =
                      options.pull_sweep == pull_sweep_t::bidirectional &&
                      (options.pull_bidirectional_rounds == 0 ||
                       pull_iteration_count <
                           options.pull_bidirectional_rounds) &&
                      pull_iteration_count %
                              options.pull_bidirectional_period ==
                          0;
                  if (add_reverse) {
                    const std::size_t reverse_begin =
                        options.pull_reverse_vertex_begin;
                    const std::size_t reverse_end =
                        options.pull_reverse_vertex_end == 0
                            ? vertex_count
                            : options.pull_reverse_vertex_end;
                    detail::launch_fused_pull_range<Policy, graph_t,
                                                    vertex_type>(
                        graph, query_count,
                        thrust::raw_pointer_cast(values.data()),
                        thrust::raw_pointer_cast(visited_mask.data()),
                        thrust::raw_pointer_cast(next_frontier_mask.data()),
                        thrust::raw_pointer_cast(unique_flags.data()),
                        thrust::raw_pointer_cast(pair_counts.data()),
                        iteration_active_slots, reverse_begin, reverse_end,
                        true, stream, 0, nullptr, nullptr, nullptr, nullptr,
                        true, pull_input);
                  }
                } else {
                  detail::launch_fused_pull_range<Policy, graph_t,
                                                  vertex_type>(
                      graph, query_count,
                      thrust::raw_pointer_cast(values.data()),
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(unique_flags.data()),
                      thrust::raw_pointer_cast(pair_counts.data()),
                      iteration_active_slots, 0, vertex_count, false, stream,
                      0, nullptr, nullptr, nullptr, nullptr, false,
                      pull_input);
                  const bool add_reverse =
                      options.pull_sweep == pull_sweep_t::bidirectional &&
                      (options.pull_bidirectional_rounds == 0 ||
                       pull_iteration_count <
                           options.pull_bidirectional_rounds) &&
                      pull_iteration_count %
                              options.pull_bidirectional_period ==
                          0;
                  if (add_reverse) {
                    const std::size_t reverse_begin =
                        options.pull_reverse_vertex_begin;
                    const std::size_t reverse_end =
                        options.pull_reverse_vertex_end == 0
                            ? vertex_count
                            : options.pull_reverse_vertex_end;
                    detail::launch_fused_pull_range<Policy, graph_t,
                                                    vertex_type>(
                        graph, query_count,
                        thrust::raw_pointer_cast(values.data()),
                        thrust::raw_pointer_cast(visited_mask.data()),
                        thrust::raw_pointer_cast(next_frontier_mask.data()),
                        thrust::raw_pointer_cast(unique_flags.data()),
                        thrust::raw_pointer_cast(pair_counts.data()),
                        iteration_active_slots, reverse_begin, reverse_end,
                        true, stream, 0, nullptr, nullptr, nullptr, nullptr,
                        true, pull_input);
                  }
                }
              }
              if (options.trace_pull_updates) {
                detail::launch_trace_pull_frontier(
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    vertex_count, static_cast<unsigned int>(level),
                    thrust::raw_pointer_cast(
                        pull_update_iteration_sum.data()),
                    thrust::raw_pointer_cast(pull_update_count.data()),
                    thrust::raw_pointer_cast(pull_update_first.data()),
                    thrust::raw_pointer_cast(pull_update_last.data()), stream);
              }
            });
        detail::throw_if_cuda_error(cudaGetLastError(), "launch_fused_pull");

        profile.compact_ms += detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              thrust::inclusive_scan(thrust_policy, unique_flags.begin(),
                                     unique_flags.end(),
                                     unique_offsets.begin());
              thrust::inclusive_scan(thrust_policy, pair_counts.begin(),
                                     pair_counts.end(), pair_counts.begin());
              detail::launch_pull_frontier_compact<vertex_type>(
                  thrust::raw_pointer_cast(next_frontier_mask.data()),
                  vertex_count, thrust::raw_pointer_cast(unique_offsets.data()),
                  thrust::raw_pointer_cast(next_frontier_vertices.data()),
                  threads, stream);
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "compact_shared_pull_frontier_kernel");

        auto sync_start = std::chrono::high_resolution_clock::now();
        if (vertex_count > 0) {
          detail::throw_if_cuda_error(
              cudaMemcpyAsync(&next_unique_count,
                              thrust::raw_pointer_cast(unique_offsets.data()) +
                                  vertex_count - 1,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(pull_next_unique_count)");
          detail::throw_if_cuda_error(
              cudaMemcpyAsync(&next_count,
                              thrust::raw_pointer_cast(pair_counts.data()) +
                                  vertex_count - 1,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(pull_next_pair_count)");
        }
        if (scheduled) {
          detail::throw_if_cuda_error(
              cudaMemcpyAsync(
                  &next_live_slots,
                  thrust::raw_pointer_cast(active_union_dev.data()),
                  sizeof(query_mask_t), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(pull_active_union)");
        }
        context.synchronize();
        profile.count_sync_ms += detail::elapsed_ms(sync_start);
        ++pull_iteration_count;
      } else {
        profile.mode = "push";
        profile.shared_push_kernel_ms = detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              if (options.push_strategy ==
                  push_strategy_t::shared_node_query_parallel) {
                if (scheduled) {
                  detail::launch_shared_push_query_parallel_scheduled<
                      Policy, graph_t, vertex_type>(
                      graph,
                      thrust::raw_pointer_cast(frontier_vertices.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_vertices.data()),
                      thrust::raw_pointer_cast(next_unique_count_dev.data()),
                      thrust::raw_pointer_cast(next_pair_count_dev.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      iteration_active_slots, level,
                      thrust::raw_pointer_cast(device_start_levels.data()),
                      thrust::raw_pointer_cast(active_union_dev.data()),
                      threads, stream);
                } else {
                  detail::launch_shared_push_query_parallel<
                      Policy, graph_t, vertex_type>(
                      graph,
                      thrust::raw_pointer_cast(frontier_vertices.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_vertices.data()),
                      thrust::raw_pointer_cast(next_unique_count_dev.data()),
                      thrust::raw_pointer_cast(next_pair_count_dev.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      iteration_active_slots, level, threads, stream);
                }
              } else if (options.push_strategy == push_strategy_t::shared_node) {
                if (scheduled) {
                  detail::launch_shared_push_simple_scheduled<
                      Policy, graph_t, vertex_type>(
                      graph,
                      thrust::raw_pointer_cast(frontier_vertices.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_vertices.data()),
                      thrust::raw_pointer_cast(next_unique_count_dev.data()),
                      thrust::raw_pointer_cast(next_pair_count_dev.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      iteration_active_slots, level,
                      thrust::raw_pointer_cast(device_start_levels.data()),
                      thrust::raw_pointer_cast(active_union_dev.data()),
                      threads, stream);
                } else {
                  detail::launch_shared_push_simple<Policy, graph_t,
                                                    vertex_type>(
                      graph,
                      thrust::raw_pointer_cast(frontier_vertices.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_vertices.data()),
                      thrust::raw_pointer_cast(next_unique_count_dev.data()),
                      thrust::raw_pointer_cast(next_pair_count_dev.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      iteration_active_slots, level, threads, stream);
                }
              } else {
                if (scheduled) {
                  detail::launch_shared_push_warp_scheduled<
                      Policy, graph_t, vertex_type>(
                      graph,
                      thrust::raw_pointer_cast(frontier_vertices.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_vertices.data()),
                      thrust::raw_pointer_cast(next_unique_count_dev.data()),
                      thrust::raw_pointer_cast(next_pair_count_dev.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      iteration_active_slots, level,
                      thrust::raw_pointer_cast(device_start_levels.data()),
                      thrust::raw_pointer_cast(active_union_dev.data()),
                      threads, stream);
                } else {
                  detail::launch_shared_push_warp<Policy, graph_t, vertex_type>(
                      graph,
                      thrust::raw_pointer_cast(frontier_vertices.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count,
                      thrust::raw_pointer_cast(visited_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_mask.data()),
                      thrust::raw_pointer_cast(next_frontier_vertices.data()),
                      thrust::raw_pointer_cast(next_unique_count_dev.data()),
                      thrust::raw_pointer_cast(next_pair_count_dev.data()),
                      thrust::raw_pointer_cast(values.data()), query_count,
                      iteration_active_slots, level, threads, stream);
                }
              }
            });
        detail::throw_if_cuda_error(cudaGetLastError(),
                                    "launch_shared_push");
        profile.push_kernel_ms += profile.shared_push_kernel_ms;
        result.shared_push_kernel_ms += profile.shared_push_kernel_ms;

        auto sync_start = std::chrono::high_resolution_clock::now();
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&next_unique_count,
                            thrust::raw_pointer_cast(
                                next_unique_count_dev.data()),
                            sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync(push_next_unique_count)");
        detail::throw_if_cuda_error(
            cudaMemcpyAsync(&next_count,
                            thrust::raw_pointer_cast(next_pair_count_dev.data()),
                            sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync(push_next_pair_count)");
        context.synchronize();
        profile.count_sync_ms += detail::elapsed_ms(sync_start);
      }

      if (scheduled) {
        const query_mask_t completed_slots =
            iteration_active_slots & ~next_live_slots;
        const float completion_wall = detail::elapsed_ms(wall_start);
        for (int query = 0; query < query_count; ++query) {
          if ((completed_slots & (query_mask_t{1} << query)) != 0 &&
              query_completion_levels[static_cast<std::size_t>(query)] < 0) {
            query_completion_levels[static_cast<std::size_t>(query)] =
                static_cast<int>(level);
            query_completion_wall_ms[static_cast<std::size_t>(query)] =
                completion_wall;
          }
        }
        profile.query_convergence_mask = next_live_slots;
        live_slots = next_live_slots;
      }
      profile.iteration_wall_ms = detail::elapsed_ms(iteration_start);
      result.iteration_edge_counts.push_back(profile.edge_count);
      result.actual_iteration_edge_counts.push_back(profile.actual_edge_count);
      result.virtual_iteration_edge_counts.push_back(profile.virtual_edge_count);
      result.iteration_modes.push_back(profile.mode);
      result.iteration_wall_times_ms.push_back(profile.iteration_wall_ms);
      if (options.profile_iterations) {
        result.iteration_profiles.push_back(profile);
      }

      ++result.iterations;
      ++level;
      current_unique_count = static_cast<std::size_t>(next_unique_count);
      current_count = static_cast<std::size_t>(next_count);
      result.frontier_sizes.push_back(current_count);
      result.unique_frontier_sizes.push_back(current_unique_count);
      workspace.swap_frontiers();
    }

    result.gpu_time_ms = total_timer.end(stream);
    result.wall_time_ms = detail::elapsed_ms(wall_start);
    result.queries.reserve(queries.size());
    for (std::size_t q = 0; q < queries.size(); ++q) {
      const int completion_level = query_completion_levels[q] >= 0
          ? query_completion_levels[q]
          : std::max(0, static_cast<int>(level) - 1);
      const float completion_wall = query_completion_levels[q] >= 0
          ? query_completion_wall_ms[q]
          : result.wall_time_ms;
      result.queries.push_back(
          {static_cast<vertex_type>(q), queries[q],
           static_cast<vertex_type>(completion_level), completion_wall});
    }
    result.values = std::move(values);
    result.pull_update_iteration_sum =
        std::move(pull_update_iteration_sum);
    result.pull_update_count = std::move(pull_update_count);
    result.pull_update_first = std::move(pull_update_first);
    result.pull_update_last = std::move(pull_update_last);
    return result;
  }
};

}  // namespace puercgp
