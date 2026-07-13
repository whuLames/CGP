/*
 * bench_fused_pull.cu
 * Benchmarks old BFS-specific fused pull vs new algorithm-independent fused pull.
 *
 * Usage:
 *   ./bench_fused_pull <csr_dir> <sources> [--repeats=N] [--warmup=N]
 *
 * Example:
 *   ./bench_fused_pull /path/to/soc-sinaweibo 53297474,23176989,23515621,47808273
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

using Policy  = puercgp::algorithms::bfs_policy;
using value_t = Policy::value_type;
using vertex_t = int;
using graph_t  = puercgp::csr_graph_view<int, int, float>;
using query_mask_t = puercgp::query_mask_t;

/* ---- CUDA helpers ---- */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(err) << "\n";                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

/* ---- Initialize BFS sources into shared frontier ---- */
void init_bfs_state(
    graph_t graph,
    int query_count,
    const std::vector<int>& sources,
    thrust::device_vector<value_t>& values,
    thrust::device_vector<query_mask_t>& visited_mask,
    thrust::device_vector<query_mask_t>& frontier_mask,
    thrust::device_vector<vertex_t>& frontier_vertices,
    thrust::device_vector<unsigned long long>& unique_count,
    cudaStream_t stream) {

  auto vertex_count = graph.get_number_of_vertices();
  auto total_values = vertex_count * static_cast<std::size_t>(query_count);

  // Fill values with infinity
  int threads = 256;
  int blocks = puercgp::detail::grid_for(total_values, threads);
  puercgp::detail::fill_values_kernel<Policy><<<blocks, threads, 0, stream>>>(
      thrust::raw_pointer_cast(values.data()), total_values);
  CUDA_CHECK(cudaGetLastError());

  // Clear masks
  CUDA_CHECK(cudaMemsetAsync(
      thrust::raw_pointer_cast(visited_mask.data()), 0,
      vertex_count * sizeof(query_mask_t), stream));
  CUDA_CHECK(cudaMemsetAsync(
      thrust::raw_pointer_cast(frontier_mask.data()), 0,
      vertex_count * sizeof(query_mask_t), stream));

  // Reset counter
  puercgp::detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
      thrust::raw_pointer_cast(unique_count.data()));
  CUDA_CHECK(cudaGetLastError());

  // Set source vertices
  thrust::device_vector<vertex_t> d_sources(sources);
  puercgp::detail::init_shared_sources_kernel<Policy, graph_t, vertex_t>
      <<<1, query_count, 0, stream>>>(
          graph, thrust::raw_pointer_cast(d_sources.data()), query_count,
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(visited_mask.data()),
          thrust::raw_pointer_cast(frontier_mask.data()),
          thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(unique_count.data()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/* ---- Run one full BFS pull-only iteration using OLD kernel ---- */
float run_old_kernel_iteration(
    graph_t graph, int query_count, int level,
    thrust::device_vector<value_t>& values,
    thrust::device_vector<query_mask_t>& visited_mask,
    thrust::device_vector<query_mask_t>& next_frontier_mask,
    thrust::device_vector<unsigned long long>& actual_degrees,
    thrust::device_vector<unsigned long long>& virtual_degrees,
    thrust::device_vector<vertex_t>& frontier_vertices,
    thrust::device_vector<unsigned long long>& unique_count,
    thrust::device_vector<unsigned long long>& pair_count,
    cudaStream_t stream) {

  auto vertex_count = graph.get_number_of_vertices();

  // Clear next frontier
  CUDA_CHECK(cudaMemsetAsync(
      thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
      vertex_count * sizeof(query_mask_t), stream));

  // Time the kernel
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));

  puercgp::detail::launch_fused_ge_bfs_pull<graph_t, vertex_t>(
      graph, query_count,
      thrust::raw_pointer_cast(values.data()),
      level,
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(next_frontier_mask.data()),
      thrust::raw_pointer_cast(actual_degrees.data()),
      thrust::raw_pointer_cast(virtual_degrees.data()),
      puercgp::query_slots_mask(query_count),
      stream);

  CUDA_CHECK(cudaEventRecord(stop, stream));

  // Compact: inclusive scan + compact kernel
  auto par = thrust::cuda::par.on(stream);
  thrust::inclusive_scan(
      par, actual_degrees.begin(), actual_degrees.begin() + vertex_count,
      actual_degrees.begin());
  thrust::inclusive_scan(
      par, virtual_degrees.begin(), virtual_degrees.begin() + vertex_count,
      virtual_degrees.begin());

  int threads = 256;
  puercgp::detail::compact_shared_pull_frontier_kernel<vertex_t>
      <<<puercgp::detail::grid_for(vertex_count, threads), threads, 0,
         stream>>>(
          thrust::raw_pointer_cast(next_frontier_mask.data()),
          vertex_count,
          thrust::raw_pointer_cast(actual_degrees.data()),
          thrust::raw_pointer_cast(frontier_vertices.data()));

  // Read back counts
  CUDA_CHECK(cudaMemcpyAsync(
      thrust::raw_pointer_cast(pair_count.data()),
      thrust::raw_pointer_cast(actual_degrees.data()) + vertex_count - 1,
      sizeof(unsigned long long), cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      thrust::raw_pointer_cast(unique_count.data()),
      thrust::raw_pointer_cast(virtual_degrees.data()) + vertex_count - 1,
      sizeof(unsigned long long), cudaMemcpyDeviceToDevice, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return ms;
}

/* ---- Run one full BFS pull-only iteration using NEW kernel ---- */
float run_new_kernel_iteration(
    graph_t graph, int query_count,
    thrust::device_vector<value_t>& values,
    thrust::device_vector<query_mask_t>& visited_mask,
    thrust::device_vector<query_mask_t>& next_frontier_mask,
    thrust::device_vector<unsigned long long>& actual_degrees,
    thrust::device_vector<unsigned long long>& virtual_degrees,
    thrust::device_vector<vertex_t>& frontier_vertices,
    thrust::device_vector<unsigned long long>& unique_count,
    thrust::device_vector<unsigned long long>& pair_count,
    cudaStream_t stream) {

  auto vertex_count = graph.get_number_of_vertices();

  // Clear next frontier
  CUDA_CHECK(cudaMemsetAsync(
      thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
      vertex_count * sizeof(query_mask_t), stream));

  // Time the kernel
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));

  puercgp::detail::launch_fused_pull<Policy, graph_t, vertex_t>(
      graph, query_count,
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(next_frontier_mask.data()),
      thrust::raw_pointer_cast(actual_degrees.data()),
      thrust::raw_pointer_cast(virtual_degrees.data()),
      stream);

  CUDA_CHECK(cudaEventRecord(stop, stream));

  // Compact: inclusive scan + compact kernel
  auto par = thrust::cuda::par.on(stream);
  thrust::inclusive_scan(
      par, actual_degrees.begin(), actual_degrees.begin() + vertex_count,
      actual_degrees.begin());
  thrust::inclusive_scan(
      par, virtual_degrees.begin(), virtual_degrees.begin() + vertex_count,
      virtual_degrees.begin());

  int threads = 256;
  puercgp::detail::compact_shared_pull_frontier_kernel<vertex_t>
      <<<puercgp::detail::grid_for(vertex_count, threads), threads, 0,
         stream>>>(
          thrust::raw_pointer_cast(next_frontier_mask.data()),
          vertex_count,
          thrust::raw_pointer_cast(actual_degrees.data()),
          thrust::raw_pointer_cast(frontier_vertices.data()));

  // Read back counts
  CUDA_CHECK(cudaMemcpyAsync(
      thrust::raw_pointer_cast(pair_count.data()),
      thrust::raw_pointer_cast(actual_degrees.data()) + vertex_count - 1,
      sizeof(unsigned long long), cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      thrust::raw_pointer_cast(unique_count.data()),
      thrust::raw_pointer_cast(virtual_degrees.data()) + vertex_count - 1,
      sizeof(unsigned long long), cudaMemcpyDeviceToDevice, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return ms;
}

/* ---- Full BFS pull-only run with timing ---- */
struct bench_result {
  float kernel_ms;       // sum of kernel-only times
  float total_ms;        // wall time
  int iterations;
  unsigned long long final_pair_count;
  thrust::host_vector<value_t> values;
};

bench_result run_bfs_old(
    graph_t graph, int query_count, const std::vector<int>& sources,
    cudaStream_t stream) {

  auto vertex_count = graph.get_number_of_vertices();
  auto total_values = vertex_count * static_cast<std::size_t>(query_count);

  thrust::device_vector<value_t> values(total_values);
  thrust::device_vector<query_mask_t> visited_mask(vertex_count, 0);
  thrust::device_vector<query_mask_t> frontier_mask(vertex_count, 0);
  thrust::device_vector<query_mask_t> next_frontier_mask(vertex_count, 0);
  thrust::device_vector<vertex_t> frontier_vertices(vertex_count);
  thrust::device_vector<unsigned long long> actual_degrees(vertex_count, 0);
  thrust::device_vector<unsigned long long> virtual_degrees(vertex_count, 0);
  thrust::device_vector<unsigned long long> unique_count(1, 0);
  thrust::device_vector<unsigned long long> pair_count(1, 0);

  init_bfs_state(graph, query_count, sources, values, visited_mask,
                 frontier_mask, frontier_vertices, unique_count, stream);

  bench_result result{};
  result.kernel_ms = 0;
  result.iterations = 0;

  cudaEvent_t total_start, total_stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventRecord(total_start, stream));

  for (int level = 0; ; ++level) {
    float iter_ms = run_old_kernel_iteration(
        graph, query_count, level, values, visited_mask, next_frontier_mask,
        actual_degrees, virtual_degrees, frontier_vertices,
        unique_count, pair_count, stream);

    result.kernel_ms += iter_ms;
    result.iterations++;

    unsigned long long pairs = 0;
    CUDA_CHECK(cudaMemcpyAsync(&pairs,
        thrust::raw_pointer_cast(pair_count.data()),
        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (pairs == 0) break;

    // Swap frontier masks
    thrust::swap(frontier_mask, next_frontier_mask);

    // Reset actual/virtual degrees for next iteration
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(actual_degrees.data()), 0,
        vertex_count * sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(virtual_degrees.data()), 0,
        vertex_count * sizeof(unsigned long long), stream));
  }

  CUDA_CHECK(cudaEventRecord(total_stop, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventElapsedTime(&result.total_ms, total_start, total_stop));
  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));

  result.final_pair_count = 0;
  result.values = values;
  return result;
}

bench_result run_bfs_new(
    graph_t graph, int query_count, const std::vector<int>& sources,
    cudaStream_t stream) {

  auto vertex_count = graph.get_number_of_vertices();
  auto total_values = vertex_count * static_cast<std::size_t>(query_count);

  thrust::device_vector<value_t> values(total_values);
  thrust::device_vector<query_mask_t> visited_mask(vertex_count, 0);
  thrust::device_vector<query_mask_t> frontier_mask(vertex_count, 0);
  thrust::device_vector<query_mask_t> next_frontier_mask(vertex_count, 0);
  thrust::device_vector<vertex_t> frontier_vertices(vertex_count);
  thrust::device_vector<unsigned long long> actual_degrees(vertex_count, 0);
  thrust::device_vector<unsigned long long> virtual_degrees(vertex_count, 0);
  thrust::device_vector<unsigned long long> unique_count(1, 0);
  thrust::device_vector<unsigned long long> pair_count(1, 0);

  init_bfs_state(graph, query_count, sources, values, visited_mask,
                 frontier_mask, frontier_vertices, unique_count, stream);

  bench_result result{};
  result.kernel_ms = 0;
  result.iterations = 0;

  cudaEvent_t total_start, total_stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventRecord(total_start, stream));

  for (int level = 0; ; ++level) {
    float iter_ms = run_new_kernel_iteration(
        graph, query_count, values, visited_mask, next_frontier_mask,
        actual_degrees, virtual_degrees, frontier_vertices,
        unique_count, pair_count, stream);

    result.kernel_ms += iter_ms;
    result.iterations++;

    unsigned long long pairs = 0;
    CUDA_CHECK(cudaMemcpyAsync(&pairs,
        thrust::raw_pointer_cast(pair_count.data()),
        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (pairs == 0) break;

    // Swap frontier masks
    thrust::swap(frontier_mask, next_frontier_mask);

    // Reset actual/virtual degrees for next iteration
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(actual_degrees.data()), 0,
        vertex_count * sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(virtual_degrees.data()), 0,
        vertex_count * sizeof(unsigned long long), stream));
  }

  CUDA_CHECK(cudaEventRecord(total_stop, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventElapsedTime(&result.total_ms, total_start, total_stop));
  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));

  result.final_pair_count = 0;
  result.values = values;
  return result;
}

/* ---- Main ---- */
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <csr_dir> <sources_csv> [--repeats=N] [--warmup=N]\n";
    return 1;
  }

  std::string csr_dir = argv[1];
  std::string sources_csv = argv[2];
  int repeats = 5;
  int warmup = 2;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--repeats=", 0) == 0) {
      repeats = std::max(1, std::stoi(arg.substr(10)));
    } else if (arg.rfind("--warmup=", 0) == 0) {
      warmup = std::max(0, std::stoi(arg.substr(9)));
    }
  }

  auto graph = puercgp_examples::load_graph_auto(csr_dir, true);
  auto graph_view = graph.view();
  auto sources = puercgp_examples::parse_sources(sources_csv);
  int query_count = static_cast<int>(sources.size());

  std::cout << "graph=" << csr_dir << "\n";
  std::cout << "vertices=" << graph.vertices << " edges=" << graph.edges
            << "\n";
  std::cout << "queries=" << query_count << " sources=" << sources_csv
            << "\n";
  std::cout << "warmup=" << warmup << " repeats=" << repeats << "\n\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Warmup
  for (int w = 0; w < warmup; ++w) {
    auto r_old = run_bfs_old(graph_view, query_count, sources, stream);
    auto r_new = run_bfs_new(graph_view, query_count, sources, stream);
  }

  // Benchmark
  std::vector<float> old_kernel_times, old_total_times;
  std::vector<float> new_kernel_times, new_total_times;
  std::vector<int> old_iters, new_iters;

  for (int r = 0; r < repeats; ++r) {
    auto r_old = run_bfs_old(graph_view, query_count, sources, stream);
    old_kernel_times.push_back(r_old.kernel_ms);
    old_total_times.push_back(r_old.total_ms);
    old_iters.push_back(r_old.iterations);

    auto r_new = run_bfs_new(graph_view, query_count, sources, stream);
    new_kernel_times.push_back(r_new.kernel_ms);
    new_total_times.push_back(r_new.total_ms);
    new_iters.push_back(r_new.iterations);

    // Correctness check: compare last repeat
    if (r == repeats - 1) {
      std::size_t mismatches = 0;
      auto total_values = static_cast<std::size_t>(graph.vertices) *
                          static_cast<std::size_t>(query_count);
      for (std::size_t i = 0; i < total_values; ++i) {
        if (r_old.values[i] != r_new.values[i]) {
          mismatches++;
          if (mismatches <= 5) {
            std::cerr << "  MISMATCH at index " << i
                      << ": old=" << r_old.values[i]
                      << " new=" << r_new.values[i] << "\n";
          }
        }
      }
      std::cout << "correctness_mismatches=" << mismatches << "\n";
    }
  }

  // Report
  float old_k_med = puercgp_examples::median(old_kernel_times);
  float old_t_med = puercgp_examples::median(old_total_times);
  float new_k_med = puercgp_examples::median(new_kernel_times);
  float new_t_med = puercgp_examples::median(new_total_times);

  std::cout << "\n=== Results ===\n";
  std::cout << "kernel,iterations(kernel_med_ms,total_med_ms)\n";
  std::cout << "old_bfs_specific,  iters=" << old_iters[old_iters.size() / 2]
            << ", kernel_ms=" << old_k_med
            << ", total_ms=" << old_t_med << "\n";
  std::cout << "new_generic,        iters=" << new_iters[new_iters.size() / 2]
            << ", kernel_ms=" << new_k_med
            << ", total_ms=" << new_t_med << "\n";

  float ratio = new_k_med / old_k_med;
  std::cout << "kernel_ratio(new/old)=" << ratio << "\n";
  if (ratio > 1.0f) {
    std::cout << "slowdown=" << (ratio - 1.0f) * 100.0f << "%\n";
  } else {
    std::cout << "speedup=" << (1.0f / ratio - 1.0f) * 100.0f << "%\n";
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
