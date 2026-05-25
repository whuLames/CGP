#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/compat/runtime_api.h>
#include <gunrock/util/timer.hxx>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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

enum class traversal_mode_t { push, pull, hybrid };
enum class push_strategy_t { edge_balanced, shared_node };

using query_mask_t = std::uint32_t;

inline const char* traversal_mode_name(traversal_mode_t mode) {
  switch (mode) {
    case traversal_mode_t::push:
      return "push";
    case traversal_mode_t::pull:
      return "pull";
    case traversal_mode_t::hybrid:
      return "hybrid";
  }
  return "unknown";
}

inline const char* push_strategy_name(push_strategy_t strategy) {
  switch (strategy) {
    case push_strategy_t::edge_balanced:
      return "edge_balanced";
    case push_strategy_t::shared_node:
      return "shared_node";
  }
  return "unknown";
}

struct options_t {
  traversal_mode_t traversal_mode = traversal_mode_t::hybrid;
  push_strategy_t push_strategy = push_strategy_t::edge_balanced;
  double pull_frontier_ratio = 0.15;
  double pull_edge_ratio = 0.20;
  bool profile_levels = false;
};

struct level_profile_t {
  int level = 0;
  std::string mode;
  std::size_t frontier_size = 0;
  unsigned long long edge_count = 0;
  double pull_frontier_threshold = 0.0;
  double pull_edge_threshold = 0.0;
  float level_wall_ms = 0.0f;
  float degree_scan_ms = 0.0f;
  float push_kernel_ms = 0.0f;
  float shared_push_kernel_ms = 0.0f;
  float bitmap_build_ms = 0.0f;
  float pull_kernel_ms = 0.0f;
  float compact_ms = 0.0f;
  float count_sync_ms = 0.0f;
  std::size_t unique_frontier_size = 0;
  unsigned long long actual_edge_count = 0;
  unsigned long long virtual_edge_count = 0;
};

template <typename vertex_t>
struct result_t {
  thrust::device_vector<vertex_t> distances;
  std::vector<query_result_t<vertex_t>> queries;
  std::vector<std::size_t> frontier_sizes;
  std::vector<std::size_t> unique_frontier_sizes;
  std::vector<unsigned long long> level_edge_counts;
  std::vector<unsigned long long> actual_level_edge_counts;
  std::vector<unsigned long long> virtual_level_edge_counts;
  std::vector<std::string> level_modes;
  std::vector<float> level_wall_times_ms;
  std::vector<level_profile_t> level_profiles;
  options_t options;
  float gpu_time_ms = 0;
  float wall_time_ms = 0;
  float shared_push_kernel_ms = 0;
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

template <typename vertex_t>
__global__ void fill_distances_kernel(vertex_t* distances,
                                      std::size_t total_pairs) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  vertex_t infinity = std::numeric_limits<vertex_t>::max();

  for (std::size_t i = tid; i < total_pairs; i += stride) {
    distances[i] = infinity;
  }
}

template <typename graph_t, typename vertex_t>
__global__ void init_shared_node_sources_kernel(
    graph_t G,
    const vertex_t* sources,
    int num_queries,
    vertex_t* distances,
    query_mask_t* visited_mask,
    query_mask_t* frontier_mask,
    vertex_t* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t n_vertices = G.get_number_of_vertices();
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t q = tid; q < static_cast<std::size_t>(num_queries);
       q += stride) {
    auto source = sources[q];
    query_mask_t bit = static_cast<query_mask_t>(1u) << q;
    distances[q * n_vertices + static_cast<std::size_t>(source)] = 0;
    atomicOr(reinterpret_cast<unsigned int*>(visited_mask + source),
             static_cast<unsigned int>(bit));
    query_mask_t old = static_cast<query_mask_t>(atomicOr(
        reinterpret_cast<unsigned int*>(frontier_mask + source),
        static_cast<unsigned int>(bit)));
    if (old == 0) {
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = source;
    }
  }
}

template <typename vertex_t>
__global__ void list_to_bitmap_kernel(const frontier_item_t<vertex_t>* frontier,
                                      std::size_t count,
                                      int num_queries,
                                      unsigned char* bitmap) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < count; i += stride) {
    auto item = frontier[i];
    bitmap[static_cast<std::size_t>(item.vertex) * num_queries +
           static_cast<std::size_t>(item.query_id)] = 1;
  }
}

template <typename vertex_t>
__global__ void shared_to_bitmap_kernel(const vertex_t* frontier_vertices,
                                        const query_mask_t* frontier_mask,
                                        std::size_t unique_count,
                                        int num_queries,
                                        unsigned char* bitmap) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < unique_count; i += stride) {
    vertex_t vertex = frontier_vertices[i];
    query_mask_t mask = frontier_mask[vertex];
    while (mask != 0) {
      int query_id = __ffs(static_cast<unsigned int>(mask)) - 1;
      if (query_id < num_queries) {
        bitmap[static_cast<std::size_t>(vertex) *
                   static_cast<std::size_t>(num_queries) +
               static_cast<std::size_t>(query_id)] = 1;
      }
      mask &= (mask - 1);
    }
  }
}

template <typename vertex_t>
__global__ void bitmap_to_list_kernel(const unsigned char* bitmap,
                                      std::size_t total_pairs,
                                      int num_queries,
                                      frontier_item_t<vertex_t>* frontier,
                                      unsigned long long* out_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < total_pairs; i += stride) {
    if (bitmap[i] == 0) {
      continue;
    }
    unsigned long long position = atomicAdd(out_count, 1ULL);
    frontier[position] = {
        static_cast<vertex_t>(i % static_cast<std::size_t>(num_queries)),
        static_cast<vertex_t>(i / static_cast<std::size_t>(num_queries))};
  }
}

template <typename vertex_t>
__global__ void bitmap_to_shared_frontier_kernel(
    const unsigned char* bitmap,
    std::size_t n_vertices,
    int num_queries,
    query_mask_t* frontier_mask,
    vertex_t* frontier_vertices,
    unsigned long long* unique_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t vertex = tid; vertex < n_vertices; vertex += stride) {
    query_mask_t mask = 0;
    for (int q = 0; q < num_queries; ++q) {
      if (bitmap[vertex * static_cast<std::size_t>(num_queries) +
                 static_cast<std::size_t>(q)] != 0) {
        mask |= static_cast<query_mask_t>(1u) << q;
      }
    }
    if (mask != 0) {
      frontier_mask[vertex] = mask;
      unsigned long long position = atomicAdd(unique_count, 1ULL);
      frontier_vertices[position] = static_cast<vertex_t>(vertex);
    }
  }
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

template <typename graph_t>
__global__ void compute_bitmap_degrees_kernel(graph_t G,
                                              const unsigned char* bitmap,
                                              std::size_t total_pairs,
                                              int num_queries,
                                              unsigned long long* edge_count) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < total_pairs; i += stride) {
    if (bitmap[i] == 0) {
      continue;
    }
    auto vertex = static_cast<typename graph_t::vertex_type>(
        i / static_cast<std::size_t>(num_queries));
    auto degree = static_cast<unsigned long long>(
        G.get_starting_edge(vertex + 1) - G.get_starting_edge(vertex));
    atomicAdd(edge_count, degree);
  }
}

template <typename graph_t, typename vertex_t>
__global__ void compute_shared_node_degrees_kernel(
    graph_t G,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    unsigned long long* actual_degrees,
    unsigned long long* virtual_degrees) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < unique_count; i += stride) {
    vertex_t vertex = frontier_vertices[i];
    query_mask_t mask = frontier_mask[vertex];
    auto degree = static_cast<unsigned long long>(
        G.get_starting_edge(vertex + 1) - G.get_starting_edge(vertex));
    actual_degrees[i] = degree;
    virtual_degrees[i] =
        degree * static_cast<unsigned long long>(
                     __popc(static_cast<unsigned int>(mask)));
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

template <typename vertex_t>
__global__ void clear_shared_frontier_mask_kernel(
    const vertex_t* frontier_vertices,
    std::size_t unique_count,
    query_mask_t* frontier_mask) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = tid; i < unique_count; i += stride) {
    frontier_mask[frontier_vertices[i]] = 0;
  }
}

template <typename graph_t, typename vertex_t>
__global__ void expand_shared_node_kernel(
    graph_t G,
    const vertex_t* frontier_vertices,
    const query_mask_t* frontier_mask,
    std::size_t unique_count,
    query_mask_t* visited_mask,
    query_mask_t* next_frontier_mask,
    vertex_t* next_frontier_vertices,
    unsigned long long* next_unique_count,
    unsigned long long* next_pair_count,
    vertex_t* distances,
    int num_queries,
    vertex_t level) {
  std::size_t n_vertices = G.get_number_of_vertices();

  for (std::size_t i = blockIdx.x; i < unique_count; i += gridDim.x) {
    vertex_t source = frontier_vertices[i];
    query_mask_t active_mask = frontier_mask[source];
    auto begin = G.get_starting_edge(source);
    auto end = G.get_starting_edge(source + 1);

    for (auto edge = begin + threadIdx.x; edge < end; edge += blockDim.x) {
      vertex_t neighbor = G.get_destination_vertex(edge);
      query_mask_t old_visited = static_cast<query_mask_t>(atomicOr(
          reinterpret_cast<unsigned int*>(visited_mask + neighbor),
          static_cast<unsigned int>(active_mask)));
      query_mask_t newly = active_mask & ~old_visited;
      if (newly == 0) {
        continue;
      }

      query_mask_t bits = newly;
      while (bits != 0) {
        int query_id = __ffs(static_cast<unsigned int>(bits)) - 1;
        if (query_id < num_queries) {
          distances[static_cast<std::size_t>(query_id) * n_vertices +
                    static_cast<std::size_t>(neighbor)] = level + 1;
        }
        bits &= (bits - 1);
      }

      atomicAdd(next_pair_count,
                static_cast<unsigned long long>(
                    __popc(static_cast<unsigned int>(newly))));
      query_mask_t old_next = static_cast<query_mask_t>(atomicOr(
          reinterpret_cast<unsigned int*>(next_frontier_mask + neighbor),
          static_cast<unsigned int>(newly)));
      if (old_next == 0) {
        unsigned long long position = atomicAdd(next_unique_count, 1ULL);
        next_frontier_vertices[position] = neighbor;
      }
    }
  }
}

template <typename graph_t, typename vertex_t>
__global__ void pull_expand_kernel(graph_t G,
                                   int num_queries,
                                   const unsigned char* frontier_bitmap,
                                   unsigned char* next_frontier_bitmap,
                                   unsigned long long* out_count,
                                   query_mask_t* visited_mask,
                                   query_mask_t* next_frontier_mask,
                                   vertex_t* next_frontier_vertices,
                                   unsigned long long* next_unique_count,
                                   bool update_visited_mask,
                                   vertex_t* distances,
                                   vertex_t level) {
  constexpr unsigned int warp_size = 32;
  std::size_t n_vertices = G.get_number_of_vertices();
  std::size_t total_pairs = n_vertices * static_cast<std::size_t>(num_queries);
  unsigned int lane = threadIdx.x & (warp_size - 1);
  std::size_t warp_id =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  std::size_t warp_stride =
      (static_cast<std::size_t>(blockDim.x) * gridDim.x) >> 5;

  for (std::size_t pair = warp_id; pair < total_pairs; pair += warp_stride) {
    vertex_t query_id =
        static_cast<vertex_t>(pair % static_cast<std::size_t>(num_queries));
    vertex_t vertex =
        static_cast<vertex_t>(pair / static_cast<std::size_t>(num_queries));
    std::size_t query_offset = static_cast<std::size_t>(query_id) * n_vertices;
    vertex_t* distance = distances + query_offset + vertex;

    bool found = false;
    if (*distance == std::numeric_limits<vertex_t>::max()) {
      auto begin = G.get_starting_edge(vertex);
      auto end = G.get_starting_edge(vertex + 1);
      for (auto edge = begin + lane; edge < end; edge += warp_size) {
        vertex_t neighbor = G.get_destination_vertex(edge);
        if (frontier_bitmap[static_cast<std::size_t>(neighbor) *
                                static_cast<std::size_t>(num_queries) +
                            static_cast<std::size_t>(query_id)] != 0) {
          found = true;
          break;
        }
      }
    }

    unsigned int mask = __any_sync(0xffffffffU, found);
    if (lane == 0 && mask != 0) {
      *distance = level + 1;
      next_frontier_bitmap[static_cast<std::size_t>(vertex) *
                               static_cast<std::size_t>(num_queries) +
                           static_cast<std::size_t>(query_id)] = 1;
      if (update_visited_mask) {
        query_mask_t bit = static_cast<query_mask_t>(1u)
                           << static_cast<std::size_t>(query_id);
        atomicOr(reinterpret_cast<unsigned int*>(visited_mask + vertex),
                 static_cast<unsigned int>(bit));
        query_mask_t old_next = static_cast<query_mask_t>(atomicOr(
            reinterpret_cast<unsigned int*>(next_frontier_mask + vertex),
            static_cast<unsigned int>(bit)));
        if (old_next == 0) {
          unsigned long long position = atomicAdd(next_unique_count, 1ULL);
          next_frontier_vertices[position] = vertex;
        }
      }
      atomicAdd(out_count, 1ULL);
    }
  }
}

inline float elapsed_ms(std::chrono::high_resolution_clock::time_point start) {
  auto stop = std::chrono::high_resolution_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
  return static_cast<float>(us) / 1000.0f;
}

template <typename function_t>
float timed_gpu(hipStream_t stream, bool enabled, function_t&& function) {
  if (!enabled) {
    function();
    return 0.0f;
  }

  hipEvent_t start;
  hipEvent_t stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, stream);
  function();
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);
  float ms = 0.0f;
  hipEventElapsedTime(&ms, start, stop);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  return ms;
}

}  // namespace detail

template <typename graph_t>
result_t<typename graph_t::vertex_type> run(
    graph_t& G,
    const std::vector<typename graph_t::vertex_type>& sources,
    std::shared_ptr<gcuda::multi_context_t> context =
        std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0)),
    const options_t& options = options_t()) {
  using vertex_t = typename graph_t::vertex_type;
  using item_t = frontier_item_t<vertex_t>;
  static_assert(std::is_same<vertex_t, int>::value,
                "cgp_bfs currently supports int vertex ids");

  result_t<vertex_t> result;
  result.options = options;
  int num_queries = static_cast<int>(sources.size());
  auto n_vertices = static_cast<std::size_t>(G.get_number_of_vertices());
  auto n_edges = static_cast<std::size_t>(G.get_number_of_edges());
  std::size_t frontier_capacity =
      static_cast<std::size_t>(num_queries) * n_vertices;
  double pull_frontier_threshold =
      options.pull_frontier_ratio * static_cast<double>(frontier_capacity);
  double pull_edge_threshold = options.pull_edge_ratio *
                               static_cast<double>(num_queries) *
                               static_cast<double>(n_edges);

  if (num_queries == 0) {
    return result;
  }
  if (options.push_strategy == push_strategy_t::shared_node &&
      num_queries > 32) {
    throw std::invalid_argument(
        "shared_node push strategy supports at most 32 queries");
  }

  auto single_context = context->get_context(0);
  auto stream = single_context->stream();
  auto policy = single_context->execution_policy();

  thrust::device_vector<vertex_t> d_sources(sources.begin(), sources.end());
  result.distances.resize(frontier_capacity);
  thrust::device_vector<item_t> frontier_a(frontier_capacity);
  thrust::device_vector<item_t> frontier_b(frontier_capacity);
  thrust::device_vector<unsigned char> frontier_bitmap(frontier_capacity);
  thrust::device_vector<unsigned char> next_frontier_bitmap(frontier_capacity);
  thrust::device_vector<unsigned long long> d_out_count(1);
  thrust::device_vector<unsigned long long> d_shared_pair_count(1);
  thrust::device_vector<unsigned long long> d_frontier_degrees(
      frontier_capacity);
  thrust::device_vector<unsigned long long> d_edge_offsets(frontier_capacity);
  thrust::device_vector<unsigned long long> d_edge_ends(frontier_capacity);
  thrust::device_vector<query_mask_t> visited_mask;
  thrust::device_vector<query_mask_t> frontier_mask;
  thrust::device_vector<query_mask_t> next_frontier_mask;
  thrust::device_vector<vertex_t> shared_frontier_a;
  thrust::device_vector<vertex_t> shared_frontier_b;
  thrust::device_vector<unsigned long long> d_shared_actual_degrees;
  thrust::device_vector<unsigned long long> d_shared_virtual_degrees;
  if (options.push_strategy == push_strategy_t::shared_node) {
    visited_mask.resize(n_vertices);
    frontier_mask.resize(n_vertices);
    next_frontier_mask.resize(n_vertices);
    shared_frontier_a.resize(n_vertices);
    shared_frontier_b.resize(n_vertices);
    d_shared_actual_degrees.resize(n_vertices);
    d_shared_virtual_degrees.resize(n_vertices);
  }

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
  std::size_t current_unique_count = sources.size();
  if (options.push_strategy == push_strategy_t::shared_node) {
    detail::fill_distances_kernel<vertex_t>
        <<<init_blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(result.distances.data()),
            frontier_capacity);
    hipMemset(thrust::raw_pointer_cast(visited_mask.data()), 0,
              n_vertices * sizeof(query_mask_t));
    hipMemset(thrust::raw_pointer_cast(frontier_mask.data()), 0,
              n_vertices * sizeof(query_mask_t));
    hipMemset(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
              n_vertices * sizeof(query_mask_t));
    detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(d_out_count.data()));
    int source_blocks =
        static_cast<int>((sources.size() + threads - 1) / threads);
    source_blocks = std::max(1, std::min(source_blocks, 65535));
    detail::init_shared_node_sources_kernel<graph_t, vertex_t>
        <<<source_blocks, threads, 0, stream>>>(
            G, thrust::raw_pointer_cast(d_sources.data()), num_queries,
            thrust::raw_pointer_cast(result.distances.data()),
            thrust::raw_pointer_cast(visited_mask.data()),
            thrust::raw_pointer_cast(frontier_mask.data()),
            thrust::raw_pointer_cast(shared_frontier_a.data()),
            thrust::raw_pointer_cast(d_out_count.data()));
    unsigned long long unique_count = 0;
    hipMemcpyAsync(&unique_count, thrust::raw_pointer_cast(d_out_count.data()),
                   sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    current_unique_count = static_cast<std::size_t>(unique_count);
  }
  vertex_t level = 0;
  bool current_is_bitmap = false;
  result.frontier_sizes.push_back(current_count);
  result.unique_frontier_sizes.push_back(
      options.push_strategy == push_strategy_t::shared_node
          ? current_unique_count
          : current_count);

  while (current_count > 0) {
    unsigned long long total_edges = 0;
    unsigned long long actual_edges = 0;
    unsigned long long virtual_edges = 0;
    bool edge_count_ready = false;
    level_profile_t profile;
    profile.level = static_cast<int>(level);
    profile.frontier_size = current_count;
    profile.unique_frontier_size =
        options.push_strategy == push_strategy_t::shared_node
            ? current_unique_count
            : current_count;
    profile.pull_frontier_threshold = pull_frontier_threshold;
    profile.pull_edge_threshold = pull_edge_threshold;
    auto level_start = std::chrono::high_resolution_clock::now();

    auto compact_bitmap_to_list = [&]() {
      profile.compact_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
                thrust::raw_pointer_cast(d_out_count.data()));
            int compact_blocks = static_cast<int>(
                (frontier_capacity + static_cast<std::size_t>(threads) - 1) /
                static_cast<std::size_t>(threads));
            compact_blocks = std::max(1, std::min(compact_blocks, 65535));
            detail::bitmap_to_list_kernel<vertex_t>
                <<<compact_blocks, threads, 0, stream>>>(
                    thrust::raw_pointer_cast(frontier_bitmap.data()),
                    frontier_capacity, num_queries,
                    thrust::raw_pointer_cast(frontier_a.data()),
                    thrust::raw_pointer_cast(d_out_count.data()));
          });

      unsigned long long compact_count = 0;
      hipMemcpyAsync(&compact_count,
                     thrust::raw_pointer_cast(d_out_count.data()),
                     sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
      hipStreamSynchronize(stream);
      current_count = static_cast<std::size_t>(compact_count);
      current_is_bitmap = false;
    };

    auto compact_bitmap_to_shared = [&]() {
      hipMemset(thrust::raw_pointer_cast(frontier_mask.data()), 0,
                n_vertices * sizeof(query_mask_t));
      profile.compact_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
                thrust::raw_pointer_cast(d_out_count.data()));
            int compact_blocks = static_cast<int>(
                (n_vertices + static_cast<std::size_t>(threads) - 1) /
                static_cast<std::size_t>(threads));
            compact_blocks = std::max(1, std::min(compact_blocks, 65535));
            detail::bitmap_to_shared_frontier_kernel<vertex_t>
                <<<compact_blocks, threads, 0, stream>>>(
                    thrust::raw_pointer_cast(frontier_bitmap.data()),
                    n_vertices, num_queries,
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    thrust::raw_pointer_cast(shared_frontier_a.data()),
                    thrust::raw_pointer_cast(d_out_count.data()));
          });

      unsigned long long compact_unique_count = 0;
      hipMemcpyAsync(&compact_unique_count,
                     thrust::raw_pointer_cast(d_out_count.data()),
                     sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
      hipStreamSynchronize(stream);
      current_unique_count = static_cast<std::size_t>(compact_unique_count);
      current_is_bitmap = false;
    };

    auto compute_push_edge_count = [&]() {
      int vertex_blocks =
          static_cast<int>((current_count + threads - 1) / threads);
      vertex_blocks = std::max(1, std::min(vertex_blocks, 65535));

      profile.degree_scan_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            detail::compute_degrees_kernel<graph_t, vertex_t>
                <<<vertex_blocks, threads, 0, stream>>>(
                    G, thrust::raw_pointer_cast(frontier_a.data()),
                    current_count,
                    thrust::raw_pointer_cast(d_frontier_degrees.data()));

            thrust::exclusive_scan(policy, d_frontier_degrees.begin(),
                                   d_frontier_degrees.begin() + current_count,
                                   d_edge_offsets.begin(), 0ULL);
            thrust::inclusive_scan(policy, d_frontier_degrees.begin(),
                                   d_frontier_degrees.begin() + current_count,
                                   d_edge_ends.begin());
          });

      auto count_sync_start = std::chrono::high_resolution_clock::now();
      hipMemcpyAsync(
          &total_edges,
          thrust::raw_pointer_cast(d_edge_ends.data()) + current_count - 1,
          sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
      hipStreamSynchronize(stream);
      profile.count_sync_ms += detail::elapsed_ms(count_sync_start);
      edge_count_ready = true;
    };

    auto compute_shared_push_edge_count = [&]() {
      int vertex_blocks =
          static_cast<int>((current_unique_count + threads - 1) / threads);
      vertex_blocks = std::max(1, std::min(vertex_blocks, 65535));

      profile.degree_scan_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            detail::compute_shared_node_degrees_kernel<graph_t, vertex_t>
                <<<vertex_blocks, threads, 0, stream>>>(
                    G, thrust::raw_pointer_cast(shared_frontier_a.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique_count,
                    thrust::raw_pointer_cast(d_shared_actual_degrees.data()),
                    thrust::raw_pointer_cast(d_shared_virtual_degrees.data()));

            thrust::inclusive_scan(
                policy, d_shared_actual_degrees.begin(),
                d_shared_actual_degrees.begin() + current_unique_count,
                d_shared_actual_degrees.begin());
            thrust::inclusive_scan(
                policy, d_shared_virtual_degrees.begin(),
                d_shared_virtual_degrees.begin() + current_unique_count,
                d_shared_virtual_degrees.begin());
          });

      auto count_sync_start = std::chrono::high_resolution_clock::now();
      hipMemcpyAsync(
          &actual_edges,
          thrust::raw_pointer_cast(d_shared_actual_degrees.data()) +
              current_unique_count - 1,
          sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
      hipMemcpyAsync(
          &virtual_edges,
          thrust::raw_pointer_cast(d_shared_virtual_degrees.data()) +
              current_unique_count - 1,
          sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
      hipStreamSynchronize(stream);
      profile.count_sync_ms += detail::elapsed_ms(count_sync_start);
      total_edges = virtual_edges;
      edge_count_ready = true;
    };

    if (current_is_bitmap && options.traversal_mode == traversal_mode_t::push) {
      if (options.push_strategy == push_strategy_t::shared_node) {
        compact_bitmap_to_shared();
      } else {
        compact_bitmap_to_list();
      }
    }

    if (!current_is_bitmap &&
        options.push_strategy == push_strategy_t::shared_node) {
      compute_shared_push_edge_count();
    } else if (!current_is_bitmap) {
      compute_push_edge_count();
    } else if (options.traversal_mode == traversal_mode_t::hybrid &&
               static_cast<double>(current_count) < pull_frontier_threshold) {
      if (options.push_strategy == push_strategy_t::shared_node) {
        compact_bitmap_to_shared();
        compute_shared_push_edge_count();
      } else {
        compact_bitmap_to_list();
        compute_push_edge_count();
      }
    }

    bool use_pull = false;
    if (options.traversal_mode == traversal_mode_t::pull) {
      use_pull = true;
    } else if (options.traversal_mode == traversal_mode_t::hybrid) {
      use_pull = static_cast<double>(current_count) >= pull_frontier_threshold;
      if (!use_pull && edge_count_ready) {
        use_pull = static_cast<double>(total_edges) >= pull_edge_threshold;
      }
    }

    if (use_pull && !current_is_bitmap) {
      hipMemset(thrust::raw_pointer_cast(frontier_bitmap.data()), 0,
                frontier_capacity * sizeof(unsigned char));
      std::size_t bitmap_count =
          options.push_strategy == push_strategy_t::shared_node
              ? current_unique_count
              : current_count;
      int bitmap_blocks = static_cast<int>(
          (bitmap_count + static_cast<std::size_t>(threads) - 1) / threads);
      bitmap_blocks = std::max(1, std::min(bitmap_blocks, 65535));
      profile.bitmap_build_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            if (options.push_strategy == push_strategy_t::shared_node) {
              detail::shared_to_bitmap_kernel<vertex_t>
                  <<<bitmap_blocks, threads, 0, stream>>>(
                      thrust::raw_pointer_cast(shared_frontier_a.data()),
                      thrust::raw_pointer_cast(frontier_mask.data()),
                      current_unique_count, num_queries,
                      thrust::raw_pointer_cast(frontier_bitmap.data()));
            } else {
              detail::list_to_bitmap_kernel<vertex_t>
                  <<<bitmap_blocks, threads, 0, stream>>>(
                      thrust::raw_pointer_cast(frontier_a.data()),
                      current_count, num_queries,
                      thrust::raw_pointer_cast(frontier_bitmap.data()));
            }
          });
      current_is_bitmap = true;
    }

    if (use_pull && current_is_bitmap && !edge_count_ready &&
        options.profile_levels) {
      profile.degree_scan_ms += detail::timed_gpu(stream, true, [&]() {
        detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(d_out_count.data()));
        int degree_blocks =
            static_cast<int>((frontier_capacity + threads - 1) / threads);
        degree_blocks = std::max(1, std::min(degree_blocks, 65535));
        detail::compute_bitmap_degrees_kernel<graph_t>
            <<<degree_blocks, threads, 0, stream>>>(
                G, thrust::raw_pointer_cast(frontier_bitmap.data()),
                frontier_capacity, num_queries,
                thrust::raw_pointer_cast(d_out_count.data()));
      });
      auto count_sync_start = std::chrono::high_resolution_clock::now();
      hipMemcpyAsync(&total_edges, thrust::raw_pointer_cast(d_out_count.data()),
                     sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
      hipStreamSynchronize(stream);
      profile.count_sync_ms += detail::elapsed_ms(count_sync_start);
      edge_count_ready = true;
    }

    detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(d_out_count.data()));
    detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(d_shared_pair_count.data()));

    if (use_pull) {
      hipMemset(thrust::raw_pointer_cast(next_frontier_bitmap.data()), 0,
                frontier_capacity * sizeof(unsigned char));
      if (options.push_strategy == push_strategy_t::shared_node) {
        hipMemset(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                  n_vertices * sizeof(query_mask_t));
      }
      std::size_t warp_count = frontier_capacity;
      int pull_blocks =
          static_cast<int>(((warp_count * 32) + threads - 1) / threads);
      pull_blocks = std::max(1, std::min(pull_blocks, 65535));
      profile.pull_kernel_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            detail::pull_expand_kernel<graph_t, vertex_t>
                <<<pull_blocks, threads, 0, stream>>>(
                    G, num_queries,
                    thrust::raw_pointer_cast(frontier_bitmap.data()),
                    thrust::raw_pointer_cast(next_frontier_bitmap.data()),
                    thrust::raw_pointer_cast(d_out_count.data()),
                    options.push_strategy == push_strategy_t::shared_node
                        ? thrust::raw_pointer_cast(visited_mask.data())
                        : nullptr,
                    options.push_strategy == push_strategy_t::shared_node
                        ? thrust::raw_pointer_cast(next_frontier_mask.data())
                        : nullptr,
                    options.push_strategy == push_strategy_t::shared_node
                        ? thrust::raw_pointer_cast(shared_frontier_b.data())
                        : nullptr,
                    options.push_strategy == push_strategy_t::shared_node
                        ? thrust::raw_pointer_cast(d_shared_pair_count.data())
                        : nullptr,
                    options.push_strategy == push_strategy_t::shared_node,
                    thrust::raw_pointer_cast(result.distances.data()), level);
          });
    } else if (options.push_strategy == push_strategy_t::shared_node &&
               actual_edges > 0) {
      hipMemset(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                n_vertices * sizeof(query_mask_t));
      int vertex_blocks = static_cast<int>(
          std::min<std::size_t>(std::max<std::size_t>(current_unique_count, 1),
                                65535));
      profile.shared_push_kernel_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            detail::expand_shared_node_kernel<graph_t, vertex_t>
                <<<vertex_blocks, threads, 0, stream>>>(
                    G, thrust::raw_pointer_cast(shared_frontier_a.data()),
                    thrust::raw_pointer_cast(frontier_mask.data()),
                    current_unique_count,
                    thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(next_frontier_mask.data()),
                    thrust::raw_pointer_cast(shared_frontier_b.data()),
                    thrust::raw_pointer_cast(d_out_count.data()),
                    thrust::raw_pointer_cast(d_shared_pair_count.data()),
                    thrust::raw_pointer_cast(result.distances.data()),
                    num_queries, level);
          });
      profile.push_kernel_ms += profile.shared_push_kernel_ms;
      result.shared_push_kernel_ms += profile.shared_push_kernel_ms;
      int clear_blocks =
          static_cast<int>((current_unique_count + threads - 1) / threads);
      clear_blocks = std::max(1, std::min(clear_blocks, 65535));
      detail::clear_shared_frontier_mask_kernel<vertex_t>
          <<<clear_blocks, threads, 0, stream>>>(
              thrust::raw_pointer_cast(shared_frontier_a.data()),
              current_unique_count,
              thrust::raw_pointer_cast(frontier_mask.data()));
    } else if (total_edges > 0) {
      unsigned long long edge_blocks_64 =
          (total_edges + static_cast<unsigned long long>(threads) - 1ULL) /
          static_cast<unsigned long long>(threads);
      int edge_blocks =
          static_cast<int>(std::min<unsigned long long>(edge_blocks_64, 65535));

      profile.push_kernel_ms +=
          detail::timed_gpu(stream, options.profile_levels, [&]() {
            detail::expand_edge_balanced_kernel<graph_t, vertex_t>
                <<<edge_blocks, threads, 0, stream>>>(
                    G, thrust::raw_pointer_cast(frontier_a.data()),
                    current_count,
                    thrust::raw_pointer_cast(d_edge_offsets.data()),
                    thrust::raw_pointer_cast(d_edge_ends.data()), total_edges,
                    thrust::raw_pointer_cast(frontier_b.data()),
                    thrust::raw_pointer_cast(d_out_count.data()),
                    thrust::raw_pointer_cast(result.distances.data()), level);
          });
    }

    unsigned long long next_count = 0;
    unsigned long long next_unique_count = 0;
    auto count_sync_start = std::chrono::high_resolution_clock::now();
    if (options.push_strategy == push_strategy_t::shared_node) {
      hipMemcpyAsync(&next_unique_count,
                     thrust::raw_pointer_cast(use_pull
                                                  ? d_shared_pair_count.data()
                                                  : d_out_count.data()),
                     sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
      hipMemcpyAsync(&next_count,
                     thrust::raw_pointer_cast(use_pull
                                                  ? d_out_count.data()
                                                  : d_shared_pair_count.data()),
                     sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
    } else {
      hipMemcpyAsync(&next_count, thrust::raw_pointer_cast(d_out_count.data()),
                     sizeof(unsigned long long), hipMemcpyDeviceToHost, stream);
    }
    hipStreamSynchronize(stream);
    profile.count_sync_ms += detail::elapsed_ms(count_sync_start);
    if (options.push_strategy != push_strategy_t::shared_node) {
      next_unique_count = next_count;
    }

    profile.mode = use_pull ? "pull" : "push";
    profile.edge_count = total_edges;
    profile.actual_edge_count =
        options.push_strategy == push_strategy_t::shared_node && !use_pull
            ? actual_edges
            : total_edges;
    profile.virtual_edge_count =
        options.push_strategy == push_strategy_t::shared_node && !use_pull
            ? virtual_edges
            : total_edges;
    profile.level_wall_ms = detail::elapsed_ms(level_start);
    result.level_edge_counts.push_back(total_edges);
    result.actual_level_edge_counts.push_back(profile.actual_edge_count);
    result.virtual_level_edge_counts.push_back(profile.virtual_edge_count);
    result.level_modes.push_back(profile.mode);
    result.level_wall_times_ms.push_back(profile.level_wall_ms);
    result.level_profiles.push_back(profile);
    ++result.iterations;
    ++level;
    current_count = static_cast<std::size_t>(next_count);
    result.frontier_sizes.push_back(current_count);
    if (use_pull) {
      frontier_bitmap.swap(next_frontier_bitmap);
      current_is_bitmap = true;
      if (options.push_strategy == push_strategy_t::shared_node) {
        shared_frontier_a.swap(shared_frontier_b);
        frontier_mask.swap(next_frontier_mask);
        current_unique_count = static_cast<std::size_t>(next_unique_count);
      } else {
        current_unique_count = current_count;
      }
    } else {
      if (options.push_strategy == push_strategy_t::shared_node) {
        shared_frontier_a.swap(shared_frontier_b);
        frontier_mask.swap(next_frontier_mask);
        current_unique_count = static_cast<std::size_t>(next_unique_count);
      } else {
        frontier_a.swap(frontier_b);
      }
      current_is_bitmap = false;
    }
    result.unique_frontier_sizes.push_back(
        options.push_strategy == push_strategy_t::shared_node
            ? current_unique_count
            : current_count);
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
