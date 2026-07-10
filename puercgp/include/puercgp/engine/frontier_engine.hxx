#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <puercgp/algorithms/bfs.hxx>
#include <puercgp/algorithms/sssp.hxx>
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
#include <puercgp/kernels/common/pull_postprocess.hxx>
#include <puercgp/kernels/pull/fused_pull_kernels.hxx>
#include <puercgp/kernels/push/shared_push_kernels.hxx>
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
    static_assert(std::is_same<vertex_type, int>::value,
                  "puercgp frontier policies currently require int vertices");

    queries.validate(options.max_queries);
    const int query_count = static_cast<int>(queries.size());
    if (query_count > 64) {
      throw std::invalid_argument("shared frontier supports at most 64 queries");
    }

    const auto vertex_count =
        static_cast<std::size_t>(graph.get_number_of_vertices());
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
    auto& visited_mask = workspace.visited_mask_vector();
    auto& frontier_mask = workspace.frontier_mask_vector();
    auto& next_frontier_mask = workspace.next_frontier_mask_vector();
    auto& frontier_vertices = workspace.frontier_vertices_vector();
    auto& next_frontier_vertices = workspace.next_frontier_vertices_vector();
    auto& unique_count_dev = workspace.current_unique_count_vector();
    auto& next_unique_count_dev = workspace.next_unique_count_vector();
    auto& next_pair_count_dev = workspace.next_pair_count_vector();

    pull_workspace pull_state;
    pull_state.resize(vertex_count);
    auto& unique_flags = pull_state.unique_flags_vector();
    auto& pair_counts = pull_state.pair_counts_vector();
    auto& unique_offsets = pull_state.unique_offsets_vector();

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
    detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(unique_count_dev.data()));
    detail::throw_if_cuda_error(cudaGetLastError(), "reset_counter_kernel");

    if constexpr (std::is_same<Policy, algorithms::wcc_policy>::value) {
      detail::init_wcc_labels_kernel<graph_t, vertex_type>
          <<<detail::grid_for(vertex_count, threads), threads, 0, stream>>>(
              graph, query_count, thrust::raw_pointer_cast(values.data()),
              thrust::raw_pointer_cast(frontier_mask.data()),
              thrust::raw_pointer_cast(frontier_vertices.data()),
              thrust::raw_pointer_cast(unique_count_dev.data()));
      detail::throw_if_cuda_error(cudaGetLastError(), "init_wcc_labels_kernel");
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
    std::size_t current_count = queries.size();
    if constexpr (std::is_same<Policy, algorithms::wcc_policy>::value) {
      current_count =
          current_unique_count * static_cast<std::size_t>(query_count);
    }

    const double pull_frontier_threshold =
        options.pull_frontier_ratio * static_cast<double>(vertex_count);
    const double pull_edge_threshold =
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
    const query_partition_t partition =
        query_partition_t::all_slots(query_count, options.traversal_mode);

    vertex_type level = 0;
    while (current_unique_count > 0 &&
           (options.max_iterations <= 0 ||
            result.iterations < options.max_iterations)) {
      iteration_profile_t profile;
      profile.iteration = static_cast<int>(level);
      profile.frontier_size = current_count;
      profile.unique_frontier_size = current_unique_count;
      profile.pull_frontier_threshold = pull_frontier_threshold;
      profile.pull_edge_threshold = pull_edge_threshold;
      auto iteration_start = std::chrono::high_resolution_clock::now();

      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()),
                          0, mask_bytes, stream),
          "cudaMemsetAsync(next_frontier_mask)");
      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(next_unique_count_dev.data()));
      detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
          thrust::raw_pointer_cast(next_pair_count_dev.data()));
      detail::throw_if_cuda_error(cudaGetLastError(),
                                  "reset_frontier_counters");

      bool use_pull = false;
      if (options.traversal_mode == traversal_mode_t::pull) {
        use_pull = true;
      } else if (options.traversal_mode == traversal_mode_t::hybrid) {
        use_pull = has_pull_adjacency &&
                   static_cast<double>(current_unique_count) >=
                       pull_frontier_threshold;
      }

      unsigned long long next_count = 0;
      unsigned long long next_unique_count = 0;

      if (use_pull) {
        profile.mode = "pull";
        profile.pull_kernel_ms = detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              detail::launch_fused_pull_compute<Policy, graph_t, vertex_type>(
                  graph, query_count, thrust::raw_pointer_cast(values.data()),
                  thrust::raw_pointer_cast(visited_mask.data()),
                  thrust::raw_pointer_cast(next_frontier_mask.data()),
                  thrust::raw_pointer_cast(unique_flags.data()),
                  thrust::raw_pointer_cast(pair_counts.data()), stream);
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
        context.synchronize();
        profile.count_sync_ms += detail::elapsed_ms(sync_start);
      } else {
        profile.mode = "push";
        profile.shared_push_kernel_ms = detail::timed_gpu(
            stream, options.profile_iterations, [&]() {
              if (options.push_strategy ==
                  push_strategy_t::shared_node_query_parallel) {
                detail::launch_shared_push_query_parallel<Policy, graph_t,
                                                          vertex_type>(
                    graph, thrust::raw_pointer_cast(frontier_vertices.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique_count,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_vertices.data()),
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    thrust::raw_pointer_cast(next_pair_count_dev.data()),
                    thrust::raw_pointer_cast(values.data()), query_count,
                    partition.active_slots, level, threads, stream);
              } else if (options.push_strategy == push_strategy_t::shared_node) {
                detail::launch_shared_push_simple<Policy, graph_t, vertex_type>(
                    graph, thrust::raw_pointer_cast(frontier_vertices.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique_count,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_vertices.data()),
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    thrust::raw_pointer_cast(next_pair_count_dev.data()),
                    thrust::raw_pointer_cast(values.data()), query_count,
                    partition.active_slots, level, threads, stream);
              } else {
                detail::launch_shared_push_warp<Policy, graph_t, vertex_type>(
                    graph, thrust::raw_pointer_cast(frontier_vertices.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique_count,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_vertices.data()),
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    thrust::raw_pointer_cast(next_pair_count_dev.data()),
                    thrust::raw_pointer_cast(values.data()), query_count,
                    partition.active_slots, level, threads, stream);
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
      result.queries.push_back(
          {static_cast<vertex_type>(q), queries[q],
           static_cast<vertex_type>(std::max(0, static_cast<int>(level) - 1)),
           result.wall_time_ms});
    }
    result.values = std::move(values);
    return result;
  }
};

}  // namespace puercgp
