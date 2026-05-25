#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/compat/runtime_api.h>
#include <gunrock/util/timer.hxx>

#include <algorithm>
#include <chrono>
#include <limits>
#include <type_traits>
#include <vector>

namespace gunrock {
namespace cgp_bfs {

template <typename vertex_t>
struct frontier_item_t {
  vertex_t query_id;
  vertex_t vertex;
};

template <typename vertex_t>
struct query_result_t {
  vertex_t query_id;
  vertex_t source;
  vertex_t completion_level;
  float completion_wall_time_ms;
};

template <typename vertex_t>
struct result_t {
  thrust::device_vector<vertex_t> distances;
  std::vector<query_result_t<vertex_t>> queries;
  std::vector<std::size_t> frontier_sizes;
  std::vector<unsigned long long> level_edge_counts;
  float gpu_time_ms = 0;
  float wall_time_ms = 0;
  int iterations = 0;
};

namespace detail {

template <typename graph_t, typename vertex_t>
__global__ void init_kernel(graph_t G,
                            const vertex_t* sources,
                            int num_queries,
                            vertex_t* distances,
                            frontier_item_t<vertex_t>* frontier) {
  using graph_vertex_t = typename graph_t::vertex_type;
  std::size_t n_vertices = G.get_number_of_vertices();
  std::size_t total = static_cast<std::size_t>(num_queries) * n_vertices;
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  vertex_t infinity = std::numeric_limits<vertex_t>::max();

  for (std::size_t i = tid; i < total; i += stride) {
    distances[i] = infinity;
  }

  for (std::size_t q = tid; q < static_cast<std::size_t>(num_queries);
       q += stride) {
    auto source = sources[q];
    distances[q * n_vertices + static_cast<std::size_t>(source)] = 0;
    frontier[q] = {static_cast<vertex_t>(q),
                   static_cast<graph_vertex_t>(source)};
  }
}

__global__ void reset_counter_kernel(unsigned long long* counter) {
  *counter = 0ULL;
}

template <typename graph_t, typename vertex_t>
__global__ void compute_degrees_kernel(
    graph_t G,
    const frontier_item_t<vertex_t>* in_frontier,
    std::size_t in_count,
    unsigned long long* degrees) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < in_count; i += stride) {
    auto vertex = in_frontier[i].vertex;
    degrees[i] = static_cast<unsigned long long>(
        G.get_starting_edge(vertex + 1) - G.get_starting_edge(vertex));
  }
}

__device__ __forceinline__ std::size_t find_frontier_index_by_end(
    const unsigned long long* edge_ends,
    std::size_t in_count,
    unsigned long long edge_ordinal) {
  std::size_t left = 0;
  std::size_t right = in_count;
  while (left < right) {
    std::size_t mid = left + ((right - left) >> 1);
    if (edge_ends[mid] <= edge_ordinal) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

template <typename graph_t, typename vertex_t>
__global__ void expand_edge_balanced_kernel(
    graph_t G,
    const frontier_item_t<vertex_t>* in_frontier,
    std::size_t in_count,
    const unsigned long long* edge_offsets,
    const unsigned long long* edge_ends,
    unsigned long long total_edges,
    frontier_item_t<vertex_t>* out_frontier,
    unsigned long long* out_count,
    vertex_t* distances,
    vertex_t level) {
  std::size_t n_vertices = G.get_number_of_vertices();
  unsigned long long global_stride =
      static_cast<unsigned long long>(blockDim.x) * gridDim.x;

  __shared__ unsigned int block_count;
  __shared__ unsigned long long block_base;

  for (unsigned long long block_edge_base =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x;
       block_edge_base < total_edges; block_edge_base += global_stride) {
    if (threadIdx.x == 0) {
      block_count = 0;
    }
    __syncthreads();

    bool discovered = false;
    unsigned int local_position = 0;
    frontier_item_t<vertex_t> output_item{};

    unsigned long long edge_ordinal = block_edge_base + threadIdx.x;
    if (edge_ordinal < total_edges) {
      std::size_t frontier_index =
          find_frontier_index_by_end(edge_ends, in_count, edge_ordinal);
      auto item = in_frontier[frontier_index];
      auto source = item.vertex;
      auto query_id = item.query_id;
      auto query_offset = static_cast<std::size_t>(query_id) * n_vertices;
      auto local_edge = edge_ordinal - edge_offsets[frontier_index];
      auto edge = G.get_starting_edge(source) + local_edge;
      vertex_t neighbor = G.get_destination_vertex(edge);

      vertex_t* distance = distances + query_offset + neighbor;
      vertex_t old =
          atomicCAS(distance, std::numeric_limits<vertex_t>::max(), level + 1);
      if (old == std::numeric_limits<vertex_t>::max()) {
        local_position = atomicAdd(&block_count, 1U);
        output_item = {query_id, neighbor};
        discovered = true;
      }
    }

    __syncthreads();
    if (threadIdx.x == 0 && block_count > 0) {
      block_base =
          atomicAdd(out_count, static_cast<unsigned long long>(block_count));
    }
    __syncthreads();

    if (discovered) {
      out_frontier[block_base + local_position] = output_item;
    }
    __syncthreads();
  }
}

inline float elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
  auto stop = std::chrono::high_resolution_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
  return static_cast<float>(us) / 1000.0f;
}

}  // namespace detail

template <typename graph_t>
result_t<typename graph_t::vertex_type> run(
    graph_t& G,
    const std::vector<typename graph_t::vertex_type>& sources,
    std::shared_ptr<gcuda::multi_context_t> context =
        std::shared_ptr<gcuda::multi_context_t>(
            new gcuda::multi_context_t(0))) {
  using vertex_t = typename graph_t::vertex_type;
  using item_t = frontier_item_t<vertex_t>;
  static_assert(std::is_same<vertex_t, int>::value,
                "cgp_bfs currently supports int vertex ids");

  result_t<vertex_t> result;
  int num_queries = static_cast<int>(sources.size());
  auto n_vertices = static_cast<std::size_t>(G.get_number_of_vertices());
  std::size_t frontier_capacity =
      static_cast<std::size_t>(num_queries) * n_vertices;

  if (num_queries == 0) {
    return result;
  }

  auto single_context = context->get_context(0);
  auto stream = single_context->stream();
  auto policy = single_context->execution_policy();

  thrust::device_vector<vertex_t> d_sources(sources.begin(), sources.end());
  result.distances.resize(frontier_capacity);
  thrust::device_vector<item_t> frontier_a(frontier_capacity);
  thrust::device_vector<item_t> frontier_b(frontier_capacity);
  thrust::device_vector<unsigned long long> d_out_count(1);
  thrust::device_vector<unsigned long long> d_frontier_degrees(
      frontier_capacity);
  thrust::device_vector<unsigned long long> d_edge_offsets(frontier_capacity);
  thrust::device_vector<unsigned long long> d_edge_ends(frontier_capacity);

  const int threads = 256;
  int init_blocks =
      static_cast<int>((frontier_capacity + threads - 1) / threads);
  init_blocks = std::max(1, std::min(init_blocks, 65535));

  auto wall_start = std::chrono::high_resolution_clock::now();
  util::timer_t timer;
  timer.begin(stream);

  detail::init_kernel<<<init_blocks, threads, 0, stream>>>(
      G, thrust::raw_pointer_cast(d_sources.data()), num_queries,
      thrust::raw_pointer_cast(result.distances.data()),
      thrust::raw_pointer_cast(frontier_a.data()));

  std::size_t current_count = sources.size();
  vertex_t level = 0;
  result.frontier_sizes.push_back(current_count);

  while (current_count > 0) {
    int vertex_blocks =
        static_cast<int>((current_count + threads - 1) / threads);
    vertex_blocks = std::max(1, std::min(vertex_blocks, 65535));

    detail::compute_degrees_kernel<graph_t, vertex_t>
        <<<vertex_blocks, threads, 0, stream>>>(
            G, thrust::raw_pointer_cast(frontier_a.data()), current_count,
            thrust::raw_pointer_cast(d_frontier_degrees.data()));

    thrust::exclusive_scan(policy, d_frontier_degrees.begin(),
                           d_frontier_degrees.begin() + current_count,
                           d_edge_offsets.begin(), 0ULL);
    thrust::inclusive_scan(policy, d_frontier_degrees.begin(),
                           d_frontier_degrees.begin() + current_count,
                           d_edge_ends.begin());

    unsigned long long total_edges = 0;
    hipMemcpyAsync(
        &total_edges,
        thrust::raw_pointer_cast(d_edge_ends.data()) + current_count - 1,
        sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    result.level_edge_counts.push_back(total_edges);

    detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(d_out_count.data()));

    if (total_edges > 0) {
      unsigned long long edge_blocks_64 =
          (total_edges + static_cast<unsigned long long>(threads) - 1ULL) /
          static_cast<unsigned long long>(threads);
      int edge_blocks =
          static_cast<int>(std::min<unsigned long long>(edge_blocks_64, 65535));

      detail::expand_edge_balanced_kernel<graph_t, vertex_t>
          <<<edge_blocks, threads, 0, stream>>>(
              G, thrust::raw_pointer_cast(frontier_a.data()), current_count,
              thrust::raw_pointer_cast(d_edge_offsets.data()),
              thrust::raw_pointer_cast(d_edge_ends.data()), total_edges,
              thrust::raw_pointer_cast(frontier_b.data()),
              thrust::raw_pointer_cast(d_out_count.data()),
              thrust::raw_pointer_cast(result.distances.data()), level);
    }

    unsigned long long next_count = 0;
    hipMemcpyAsync(&next_count, thrust::raw_pointer_cast(d_out_count.data()),
                   sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    ++result.iterations;
    ++level;
    current_count = static_cast<std::size_t>(next_count);
    result.frontier_sizes.push_back(current_count);
    frontier_a.swap(frontier_b);
  }

  result.gpu_time_ms = timer.end(stream);
  result.wall_time_ms = detail::elapsed_ms(wall_start);
  result.queries.reserve(num_queries);
  for (int q = 0; q < num_queries; ++q) {
    result.queries.push_back({static_cast<vertex_t>(q), sources[q],
                              static_cast<vertex_t>(std::max(0, level - 1)),
                              result.wall_time_ms});
  }
  return result;
}

}  // namespace cgp_bfs
}  // namespace gunrock
