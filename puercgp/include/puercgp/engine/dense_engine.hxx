#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <puercgp/backend/graph_adapter.hxx>
#include <puercgp/backend/pull_graph_access.hxx>
#include <puercgp/core/algorithm_query_batch.hxx>
#include <puercgp/core/cuda_utils.hxx>
#include <puercgp/core/result.hxx>
#include <puercgp/engine/pull_executor.hxx>
#include <puercgp/kernels/dense/rank_kernels.hxx>
#include <puercgp/state/pull_workspace.hxx>
#include <puercgp/state/rank_workspace.hxx>

namespace puercgp {

template <typename Policy> class dense_engine {
public:
  using vertex_type = typename Policy::vertex_type;
  using value_type = typename Policy::value_type;
  using query_type = typename Policy::query_type;
  using query_batch_type = algorithm_query_batch<query_type>;
  using result_type = run_result_t<vertex_type, value_type>;

  template <typename graph_t>
  result_type run(graph_t &graph, const query_batch_type &queries,
                  execution_context &context,
                  const run_options &options) const {
    static_assert(std::is_same<vertex_type, int>::value,
                  "puercgp dense policies currently require int vertices");
    static_assert(std::is_same<value_type, float>::value,
                  "puercgp rank engine currently requires float values");

    queries.validate(options.max_queries);
    if (options.max_iterations < 0) {
      throw std::invalid_argument("max_iterations must be non-negative");
    }
    if (options.fixed_iterations && options.max_iterations <= 0) {
      throw std::invalid_argument(
          "fixed_iterations requires a positive max_iterations");
    }

    const std::size_t vertex_count =
        static_cast<std::size_t>(graph.get_number_of_vertices());
    if (vertex_count == 0) {
      throw std::invalid_argument("PageRank requires a non-empty graph");
    }
    const bool has_pull_adjacency = detail::graph_has_pull_adjacency(graph);
    if (options.traversal_mode == traversal_mode_t::pull &&
        !has_pull_adjacency) {
      throw std::invalid_argument(
          "PageRank pull requires incoming adjacency storage");
    }

    const int query_count = static_cast<int>(queries.size());
    const std::size_t value_count =
        vertex_count * static_cast<std::size_t>(query_count);
    std::vector<float> host_damping(static_cast<std::size_t>(query_count));
    std::vector<float> host_epsilon(static_cast<std::size_t>(query_count));
    std::vector<int> host_sources(static_cast<std::size_t>(query_count), -1);
    for (int query_id = 0; query_id < query_count; ++query_id) {
      const query_type &query = queries[static_cast<std::size_t>(query_id)];
      Policy::validate_query(query, vertex_count);
      host_damping[static_cast<std::size_t>(query_id)] = query.damping_factor;
      host_epsilon[static_cast<std::size_t>(query_id)] = query.epsilon;
      host_sources[static_cast<std::size_t>(query_id)] =
          Policy::query_source(query);
    }

    rank_workspace<value_type> rank_state;
    rank_state.resize(vertex_count, static_cast<std::size_t>(query_count));
    pull_workspace pull_state;
    pull_state.resize(vertex_count);
    thrust::device_vector<query_mask_t> frontier_mask(vertex_count);
    thrust::device_vector<query_mask_t> next_frontier_mask(vertex_count);
    thrust::device_vector<vertex_type> frontier_vertices(vertex_count);
    thrust::device_vector<vertex_type> next_frontier_vertices(vertex_count);

    auto &current_ranks = rank_state.current_vector();
    auto &next_ranks = rank_state.next_vector();
    auto &damping_factors = rank_state.damping_factors_vector();
    auto &epsilons = rank_state.epsilons_vector();
    auto &sources = rank_state.sources_vector();
    auto &dangling_mass = rank_state.dangling_mass_vector();
    auto &changed_queries = rank_state.changed_queries_vector();
    auto &unique_flags = pull_state.unique_flags_vector();
    auto &pair_counts = pull_state.pair_counts_vector();
    auto &unique_offsets = pull_state.unique_offsets_vector();

    cudaStream_t stream = context.stream();
    auto thrust_policy = thrust::cuda::par.on(stream);
    constexpr int threads = 256;
    const query_mask_t valid_slots =
        query_count == 64 ? ~query_mask_t{0}
                          : (query_mask_t{1} << query_count) - query_mask_t{1};

    detail::throw_if_cuda_error(
        cudaMemcpyAsync(thrust::raw_pointer_cast(damping_factors.data()),
                        host_damping.data(),
                        host_damping.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream),
        "copy PageRank damping factors");
    detail::throw_if_cuda_error(
        cudaMemcpyAsync(thrust::raw_pointer_cast(epsilons.data()),
                        host_epsilon.data(),
                        host_epsilon.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream),
        "copy PageRank epsilons");
    detail::throw_if_cuda_error(
        cudaMemcpyAsync(thrust::raw_pointer_cast(sources.data()),
                        host_sources.data(), host_sources.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream),
        "copy PageRank sources");

    auto wall_start = std::chrono::high_resolution_clock::now();
    detail::cuda_event_timer total_timer;
    total_timer.begin(stream);
    detail::init_rank_values_kernel<Policy, vertex_type>
        <<<detail::grid_for(value_count, threads), threads, 0, stream>>>(
            vertex_count, query_count, valid_slots,
            thrust::raw_pointer_cast(sources.data()),
            thrust::raw_pointer_cast(current_ranks.data()),
            thrust::raw_pointer_cast(frontier_mask.data()),
            thrust::raw_pointer_cast(frontier_vertices.data()));
    detail::throw_if_cuda_error(cudaGetLastError(),
                                "initialize PageRank values");

    result_type result;
    result.options = options;
    result.effective_query_dim = query_count;
    std::size_t current_unique_count = vertex_count;
    std::size_t current_pair_count = value_count;
    result.frontier_sizes.push_back(current_pair_count);
    result.unique_frontier_sizes.push_back(current_unique_count);
    std::vector<int> completion_iterations(
        static_cast<std::size_t>(query_count), -1);
    query_mask_t active_slots = valid_slots;

    bool sampled_push = false;
    bool sampled_pull = false;
    double push_ms_per_query = std::numeric_limits<double>::infinity();
    double pull_ms_per_query = std::numeric_limits<double>::infinity();
    const bool adaptive_hybrid =
        options.traversal_mode == traversal_mode_t::hybrid &&
        has_pull_adjacency;

    while (active_slots != 0 && (options.max_iterations <= 0 ||
                                 result.iterations < options.max_iterations)) {
      const int active_query_count = host_popcount(active_slots);
      bool use_pull = options.traversal_mode == traversal_mode_t::pull;
      if (options.traversal_mode == traversal_mode_t::hybrid) {
        if (!has_pull_adjacency) {
          use_pull = false;
        } else if (!sampled_push) {
          use_pull = false;
        } else if (!sampled_pull) {
          use_pull = true;
        } else {
          use_pull = pull_ms_per_query <= push_ms_per_query;
        }
      }

      iteration_profile_t profile;
      profile.iteration = result.iterations;
      profile.mode = use_pull ? "pull" : "push";
      profile.frontier_size = current_pair_count;
      profile.unique_frontier_size = current_unique_count;
      auto iteration_start = std::chrono::high_resolution_clock::now();

      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(dangling_mass.data()), 0,
                          static_cast<std::size_t>(query_count) * sizeof(float),
                          stream),
          "clear PageRank dangling mass");
      detail::throw_if_cuda_error(
          cudaMemsetAsync(thrust::raw_pointer_cast(changed_queries.data()), 0,
                          sizeof(query_mask_t), stream),
          "clear PageRank changed queries");
      detail::compute_rank_dangling_mass_kernel<graph_t, vertex_type>
          <<<detail::grid_for(value_count, threads), threads, 0, stream>>>(
              graph, vertex_count, query_count,
              thrust::raw_pointer_cast(current_ranks.data()), active_slots,
              thrust::raw_pointer_cast(dangling_mass.data()));
      detail::throw_if_cuda_error(cudaGetLastError(),
                                  "compute PageRank dangling mass");

      const bool measure_backend =
          options.profile_iterations || adaptive_hybrid;
      const float backend_ms =
          detail::timed_gpu(stream, measure_backend, [&]() {
            if (use_pull) {
              detail::launch_fused_sum_pull_compute<Policy, graph_t,
                                                    vertex_type>(
                  graph, query_count,
                  thrust::raw_pointer_cast(current_ranks.data()),
                  thrust::raw_pointer_cast(next_ranks.data()),
                  thrust::raw_pointer_cast(damping_factors.data()),
                  thrust::raw_pointer_cast(epsilons.data()),
                  thrust::raw_pointer_cast(sources.data()),
                  thrust::raw_pointer_cast(dangling_mass.data()),
                  thrust::raw_pointer_cast(next_frontier_mask.data()),
                  thrust::raw_pointer_cast(unique_flags.data()),
                  thrust::raw_pointer_cast(pair_counts.data()),
                  thrust::raw_pointer_cast(changed_queries.data()),
                  active_slots, stream);
            } else {
              detail::launch_fused_rank_push<Policy, graph_t, vertex_type>(
                  graph, vertex_count, query_count,
                  thrust::raw_pointer_cast(current_ranks.data()),
                  thrust::raw_pointer_cast(next_ranks.data()),
                  thrust::raw_pointer_cast(damping_factors.data()),
                  thrust::raw_pointer_cast(epsilons.data()),
                  thrust::raw_pointer_cast(sources.data()),
                  thrust::raw_pointer_cast(dangling_mass.data()),
                  thrust::raw_pointer_cast(next_frontier_mask.data()),
                  thrust::raw_pointer_cast(unique_flags.data()),
                  thrust::raw_pointer_cast(pair_counts.data()),
                  thrust::raw_pointer_cast(changed_queries.data()),
                  active_slots, stream);
            }
          });
      detail::throw_if_cuda_error(cudaGetLastError(),
                                  "launch PageRank backend");

      if (use_pull) {
        profile.pull_kernel_ms = backend_ms;
        if (measure_backend) {
          sampled_pull = true;
          pull_ms_per_query =
              static_cast<double>(backend_ms) / active_query_count;
        }
      } else {
        profile.push_kernel_ms = backend_ms;
        profile.shared_push_kernel_ms = backend_ms;
        result.shared_push_kernel_ms += backend_ms;
        if (measure_backend) {
          sampled_push = true;
          push_ms_per_query =
              static_cast<double>(backend_ms) / active_query_count;
        }
      }

      profile.compact_ms =
          detail::timed_gpu(stream, options.profile_iterations, [&]() {
            thrust::inclusive_scan(thrust_policy, unique_flags.begin(),
                                   unique_flags.end(), unique_offsets.begin());
            thrust::inclusive_scan(thrust_policy, pair_counts.begin(),
                                   pair_counts.end(), pair_counts.begin());
            detail::launch_pull_frontier_compact<vertex_type>(
                thrust::raw_pointer_cast(next_frontier_mask.data()),
                vertex_count, thrust::raw_pointer_cast(unique_offsets.data()),
                thrust::raw_pointer_cast(next_frontier_vertices.data()),
                threads, stream);
          });
      detail::throw_if_cuda_error(cudaGetLastError(),
                                  "compact PageRank changes");

      unsigned long long next_unique_raw = 0;
      unsigned long long next_pair_raw = 0;
      query_mask_t next_active_slots = 0;
      auto sync_start = std::chrono::high_resolution_clock::now();
      detail::throw_if_cuda_error(
          cudaMemcpyAsync(&next_unique_raw,
                          thrust::raw_pointer_cast(unique_offsets.data()) +
                              vertex_count - 1,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                          stream),
          "copy PageRank changed vertex count");
      detail::throw_if_cuda_error(
          cudaMemcpyAsync(
              &next_pair_raw,
              thrust::raw_pointer_cast(pair_counts.data()) + vertex_count - 1,
              sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream),
          "copy PageRank changed pair count");
      detail::throw_if_cuda_error(
          cudaMemcpyAsync(&next_active_slots,
                          thrust::raw_pointer_cast(changed_queries.data()),
                          sizeof(query_mask_t), cudaMemcpyDeviceToHost, stream),
          "copy PageRank changed queries");
      context.synchronize();
      profile.count_sync_ms = detail::elapsed_ms(sync_start);

      if (options.fixed_iterations) {
        next_active_slots = valid_slots;
      }

      profile.query_convergence_mask = next_active_slots;
      profile.iteration_wall_ms = detail::elapsed_ms(iteration_start);
      result.iteration_modes.push_back(profile.mode);
      result.iteration_wall_times_ms.push_back(profile.iteration_wall_ms);
      if (options.profile_iterations) {
        result.iteration_profiles.push_back(profile);
      }

      ++result.iterations;
      for (int query_id = 0; query_id < query_count; ++query_id) {
        if (completion_iterations[static_cast<std::size_t>(query_id)] < 0 &&
            (next_active_slots & (query_mask_t{1} << query_id)) == 0) {
          completion_iterations[static_cast<std::size_t>(query_id)] =
              result.iterations;
        }
      }

      current_unique_count = static_cast<std::size_t>(next_unique_raw);
      current_pair_count = static_cast<std::size_t>(next_pair_raw);
      result.frontier_sizes.push_back(current_pair_count);
      result.unique_frontier_sizes.push_back(current_unique_count);
      rank_state.swap_ranks();
      frontier_mask.swap(next_frontier_mask);
      frontier_vertices.swap(next_frontier_vertices);
      active_slots = next_active_slots;
    }

    result.gpu_time_ms = total_timer.end(stream);
    result.wall_time_ms = detail::elapsed_ms(wall_start);
    result.queries.reserve(queries.size());
    for (int query_id = 0; query_id < query_count; ++query_id) {
      int completion =
          completion_iterations[static_cast<std::size_t>(query_id)];
      if (completion < 0) {
        completion = result.iterations;
      }
      result.queries.push_back(
          {query_id,
           static_cast<vertex_type>(
               host_sources[static_cast<std::size_t>(query_id)]),
           static_cast<vertex_type>(completion), result.wall_time_ms});
    }
    result.values = std::move(rank_state.current_vector());
    return result;
  }

private:
  static int host_popcount(query_mask_t mask) {
    return __builtin_popcountll(static_cast<unsigned long long>(mask));
  }
};

} // namespace puercgp
