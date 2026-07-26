#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

namespace {

struct mode_measurement {
  float median_gpu_ms = 0.0f;
  float median_wall_ms = 0.0f;
  std::vector<float> samples;
};

bool sampled_csr_is_symmetric(const puercgp_examples::host_csr_graph &graph) {
  constexpr std::size_t target_samples = 100000;
  const std::size_t edge_count = static_cast<std::size_t>(graph.edges);
  const std::size_t stride = std::max<std::size_t>(1, edge_count / target_samples);
  std::size_t next_sample = 0;
  std::size_t checked = 0;
  for (int source = 0; source < graph.vertices && next_sample < edge_count;
       ++source) {
    const std::size_t begin =
        static_cast<std::size_t>(graph.row_offsets[source]);
    const std::size_t end =
        static_cast<std::size_t>(graph.row_offsets[source + 1]);
    while (next_sample < begin) {
      next_sample += stride;
    }
    for (; next_sample < end; next_sample += stride) {
      const int destination = graph.column_indices[next_sample];
      const auto reverse_begin = graph.column_indices.begin() +
          graph.row_offsets[static_cast<std::size_t>(destination)];
      const auto reverse_end = graph.column_indices.begin() +
          graph.row_offsets[static_cast<std::size_t>(destination) + 1];
      if (!std::binary_search(reverse_begin, reverse_end, source)) {
        return false;
      }
      ++checked;
    }
  }
  return checked != 0;
}

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::vector<float>
sample_ranks(const thrust::device_vector<float> &ranks,
             std::size_t vertex_count, int query_count,
             cudaStream_t stream) {
  constexpr std::size_t sample_count = 16;
  std::vector<float> samples(sample_count *
                             static_cast<std::size_t>(query_count));
  const float *device_ranks = thrust::raw_pointer_cast(ranks.data());
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const std::size_t vertex =
        sample * (vertex_count - 1) / (sample_count - 1);
    check_cuda(cudaMemcpyAsync(
                   samples.data() + sample * query_count,
                   device_ranks + vertex * query_count,
                   static_cast<std::size_t>(query_count) * sizeof(float),
                   cudaMemcpyDeviceToHost, stream),
               "copy sampled PageRank values");
  }
  check_cuda(cudaStreamSynchronize(stream), "synchronize rank samples");
  return samples;
}

template <typename Policy, typename graph_t, typename query_batch_t>
mode_measurement run_mode(graph_t &graph, const query_batch_t &queries,
                          puercgp::traversal_mode_t mode, int iterations,
                          int repeats) {
  puercgp::execution_context context;
  puercgp::run_options options;
  options.traversal_mode = mode;
  options.profile_iterations = false;
  options.max_iterations = iterations;
  options.max_queries = 64;

  std::vector<float> gpu_times;
  std::vector<float> wall_times;
  mode_measurement measurement;
  gpu_times.reserve(static_cast<std::size_t>(repeats));
  wall_times.reserve(static_cast<std::size_t>(repeats));

  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto result = puercgp::run<Policy>(graph, queries, context, options);
    if (result.iterations != iterations) {
      throw std::runtime_error(
          std::string(puercgp::traversal_mode_name(mode)) + " stopped after " +
          std::to_string(result.iterations) + " iterations; expected " +
          std::to_string(iterations));
    }
    gpu_times.push_back(result.gpu_time_ms);
    wall_times.push_back(result.wall_time_ms);
    if (repeat + 1 == repeats) {
      measurement.samples = sample_ranks(
          result.values, static_cast<std::size_t>(graph.get_number_of_vertices()),
          static_cast<int>(queries.size()), context.stream());
    }
    std::cerr << "mode=" << puercgp::traversal_mode_name(mode)
              << " repeat=" << (repeat + 1) << "/" << repeats
              << " iterations=" << result.iterations
              << " gpu_ms=" << result.gpu_time_ms
              << " wall_ms=" << result.wall_time_ms << '\n';
  }

  measurement.median_gpu_ms = puercgp_examples::median(std::move(gpu_times));
  measurement.median_wall_ms = puercgp_examples::median(std::move(wall_times));
  return measurement;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: " << argv[0]
                << " GRAPH_PATH [Q=32] [ITERATIONS=20] [REPEATS=3] "
                   "[ALGORITHM=pagerank]\n";
      return 2;
    }

    const std::string graph_path = argv[1];
    const int query_count = argc > 2 ? std::stoi(argv[2]) : 32;
    const int iterations = argc > 3 ? std::stoi(argv[3]) : 20;
    const int repeats = argc > 4 ? std::stoi(argv[4]) : 3;
    const std::string algorithm = argc > 5 ? argv[5] : "pagerank";
    if (query_count <= 0 || query_count > 64 || iterations <= 0 ||
        repeats <= 0) {
      throw std::invalid_argument(
          "Q must be in [1,64], and iterations/repeats must be positive");
    }

    const bool binary_csr = std::filesystem::is_directory(graph_path);
    std::cerr << "loading graph=" << graph_path << '\n';
    auto storage =
        puercgp_examples::load_graph_auto(graph_path, !binary_csr);
    auto graph = storage.view();

    if (binary_csr) {
      if (!sampled_csr_is_symmetric(storage)) {
        throw std::runtime_error(
            "binary CSR failed the sampled symmetry check; refusing to alias "
            "outgoing adjacency as incoming adjacency");
      }
      // The experiment datasets are stored as symmetric CSR. Alias the
      // outgoing CSR as incoming CSR to avoid materializing an identical
      // transpose.
      graph.pull_row_offsets = graph.row_offsets;
      graph.pull_column_indices = graph.column_indices;
      graph.pull_edge_weights = graph.edge_weights;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query GPU memory");
    std::cerr << "loaded vertices=" << graph.get_number_of_vertices()
              << " edges=" << graph.get_number_of_edges()
              << " gpu_free_gib="
              << static_cast<double>(free_bytes) / (1ULL << 30) << '\n';

    mode_measurement push;
    mode_measurement pull;
    if (algorithm == "pagerank") {
      std::vector<puercgp::algorithms::pagerank_query> descriptors(
          static_cast<std::size_t>(query_count));
      for (auto &query : descriptors) {
        query.damping_factor = 0.85f;
        query.epsilon = 1.0e-30f;
      }
      puercgp::algorithms::pagerank_query_batch queries(
          std::move(descriptors));
      push = run_mode<puercgp::algorithms::pagerank_policy>(
          graph, queries, puercgp::traversal_mode_t::push, iterations,
          repeats);
      pull = run_mode<puercgp::algorithms::pagerank_policy>(
          graph, queries, puercgp::traversal_mode_t::pull, iterations,
          repeats);
    } else if (algorithm == "ppr") {
      std::vector<puercgp::algorithms::ppr_query> descriptors;
      descriptors.reserve(static_cast<std::size_t>(query_count));
      for (int query_id = 0; query_id < query_count; ++query_id) {
        const int source = static_cast<int>(
            static_cast<unsigned long long>(query_id) *
            static_cast<unsigned long long>(graph.get_number_of_vertices()) /
            static_cast<unsigned long long>(query_count));
        descriptors.push_back({source, 0.85f, 1.0e-30f});
      }
      puercgp::algorithms::ppr_query_batch queries(std::move(descriptors));
      push = run_mode<puercgp::algorithms::ppr_policy>(
          graph, queries, puercgp::traversal_mode_t::push, iterations,
          repeats);
      pull = run_mode<puercgp::algorithms::ppr_policy>(
          graph, queries, puercgp::traversal_mode_t::pull, iterations,
          repeats);
    } else {
      throw std::invalid_argument("algorithm must be pagerank or ppr");
    }

    float max_sample_abs_delta = 0.0f;
    float max_sample_rel_delta = 0.0f;
    for (std::size_t i = 0; i < push.samples.size(); ++i) {
      const float absolute = std::fabs(push.samples[i] - pull.samples[i]);
      const float scale =
          std::max({std::fabs(push.samples[i]), std::fabs(pull.samples[i]),
                    1.0e-30f});
      max_sample_abs_delta = std::max(max_sample_abs_delta, absolute);
      max_sample_rel_delta =
          std::max(max_sample_rel_delta, absolute / scale);
    }

    const double speedup = static_cast<double>(push.median_gpu_ms) /
                           static_cast<double>(pull.median_gpu_ms);
    const std::string graph_name =
        std::filesystem::path(graph_path).filename().string();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "algorithm,graph,vertices,edges,Q,iterations,repeats,push_gpu_ms,"
                 "pull_gpu_ms,pull_speedup,push_wall_ms,pull_wall_ms,"
                 "max_sample_abs_delta,max_sample_rel_delta\n";
    std::cout << algorithm << ',' << graph_name << ','
              << graph.get_number_of_vertices() << ','
              << graph.get_number_of_edges() << ',' << query_count << ','
              << iterations << ',' << repeats << ',' << push.median_gpu_ms
              << ',' << pull.median_gpu_ms << ',' << speedup << ','
              << push.median_wall_ms << ',' << pull.median_wall_ms << ','
              << std::scientific << std::setprecision(9)
              << max_sample_abs_delta << ',' << max_sample_rel_delta << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
