#define main bench_phase_schedule_q32_main
#include "bench_phase_schedule.cu"
#undef main

#include <puercgp/core/green_context.hxx>
#include <puercgp/kernels/pull/fused_pull_kernels.hxx>

namespace {

float median_pull(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <typename graph_t>
float prepare_and_measure(graph_t graph, const std::vector<int>& sources,
                          int target_step, int warmup, int repeats,
                          cudaStream_t stream) {
  const std::size_t V = graph.get_number_of_vertices();
  const std::size_t value_count = V * kQueryCount;
  const std::size_t mask_bytes = V * sizeof(mask_t);
  thrust::device_vector<int> device_sources(sources);
  thrust::device_vector<int> values(value_count);
  thrust::device_vector<int> values_backup(value_count);
  thrust::device_vector<mask_t> visited_mask(V);
  thrust::device_vector<mask_t> visited_backup(V);
  thrust::device_vector<mask_t> frontier_mask(V);
  thrust::device_vector<mask_t> next_frontier_mask(V);
  thrust::device_vector<int> frontier_vertices(V);
  thrust::device_vector<int> next_frontier_vertices(V);
  thrust::device_vector<unsigned long long> unique_count_dev(1);
  thrust::device_vector<unsigned long long> next_unique_count_dev(1);
  thrust::device_vector<unsigned long long> next_pair_count_dev(1);
  thrust::device_vector<unsigned long long> unique_flags(V);
  thrust::device_vector<unsigned long long> pair_counts(V);
  thrust::device_vector<int> next_levels_dev(kQueryCount);

  puercgp::detail::fill_values_kernel<policy_t>
      <<<puercgp::detail::grid_for(value_count, kThreads), kThreads, 0,
         stream>>>(thrust::raw_pointer_cast(values.data()), value_count);
  cudaMemsetAsync(thrust::raw_pointer_cast(visited_mask.data()), 0, mask_bytes,
                  stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(frontier_mask.data()), 0, mask_bytes,
                  stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                  mask_bytes, stream);
  puercgp::detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
      thrust::raw_pointer_cast(unique_count_dev.data()));
  puercgp::detail::init_shared_sources_kernel<policy_t>
      <<<1, 64, 0, stream>>>(
          graph, thrust::raw_pointer_cast(device_sources.data()), kQueryCount,
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(visited_mask.data()),
          thrust::raw_pointer_cast(frontier_mask.data()),
          thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(unique_count_dev.data()));

  unsigned long long unique_raw = 0;
  cudaMemcpyAsync(&unique_raw, thrust::raw_pointer_cast(unique_count_dev.data()),
                  sizeof(unique_raw), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::size_t current_unique = unique_raw;

  for (int step = 0; step < target_step; ++step) {
    cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                    mask_bytes, stream);
    puercgp::detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(next_unique_count_dev.data()));
    puercgp::detail::reset_counter_kernel<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(next_pair_count_dev.data()));
    std::array<int, kQueryCount> levels{};
    levels.fill(step + 1);
    cudaMemcpyAsync(thrust::raw_pointer_cast(next_levels_dev.data()),
                    levels.data(), kQueryCount * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
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
            thrust::raw_pointer_cast(next_levels_dev.data()), valid_mask());
    cudaMemcpyAsync(&unique_raw,
                    thrust::raw_pointer_cast(next_unique_count_dev.data()),
                    sizeof(unique_raw), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    current_unique = unique_raw;
    thrust::swap(frontier_mask, next_frontier_mask);
    thrust::swap(frontier_vertices, next_frontier_vertices);
  }

  cudaMemcpyAsync(thrust::raw_pointer_cast(values_backup.data()),
                  thrust::raw_pointer_cast(values.data()),
                  value_count * sizeof(int), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(visited_backup.data()),
                  thrust::raw_pointer_cast(visited_mask.data()), mask_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> times;
  for (int iteration = 0; iteration < warmup + repeats; ++iteration) {
    cudaMemcpyAsync(thrust::raw_pointer_cast(values.data()),
                    thrust::raw_pointer_cast(values_backup.data()),
                    value_count * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(visited_mask.data()),
                    thrust::raw_pointer_cast(visited_backup.data()), mask_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(next_frontier_mask.data()), 0,
                    mask_bytes, stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(unique_flags.data()), 0,
                    V * sizeof(unsigned long long), stream);
    cudaMemsetAsync(thrust::raw_pointer_cast(pair_counts.data()), 0,
                    V * sizeof(unsigned long long), stream);
    cudaStreamSynchronize(stream);
    puercgp::detail::cuda_event_timer timer;
    timer.begin(stream);
    puercgp::detail::launch_fused_pull<policy_t, graph_t, int>(
        graph, kQueryCount, thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(visited_mask.data()),
        thrust::raw_pointer_cast(next_frontier_mask.data()),
        thrust::raw_pointer_cast(unique_flags.data()),
        thrust::raw_pointer_cast(pair_counts.data()), stream);
    float elapsed = timer.end(stream);
    if (iteration >= warmup) times.push_back(elapsed);
  }
  return median_pull(times);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: bench_pull_sm_scaling <graph> <target-step> "
                 "<output.csv> [--seed=42] [--repeats=5] [--warmup=2]\n";
    return 2;
  }
  std::string graph_path = argv[1];
  int target_step = std::stoi(argv[2]);
  std::filesystem::path output = argv[3];
  unsigned int seed = 42;
  int repeats = 5;
  int warmup = 2;
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--seed=", 0) == 0)
      seed = static_cast<unsigned int>(std::stoul(arg.substr(7)));
    else if (arg.rfind("--repeats=", 0) == 0)
      repeats = std::stoi(arg.substr(10));
    else if (arg.rfind("--warmup=", 0) == 0)
      warmup = std::stoi(arg.substr(9));
  }

  auto graph_storage = puercgp_examples::load_graph_auto(graph_path, true);
  auto graph = graph_storage.view();
  auto sources = make_unique_sources(graph_storage.vertices, seed);
  const std::array<unsigned int, 8> sm_counts = {2, 4, 8, 16, 32, 48, 64, 80};
  std::filesystem::create_directories(output.parent_path());
  std::ofstream csv(output);
  csv << "requested_sms,actual_sms,target_step,pull_ms\n";
  for (unsigned int requested : sm_counts) {
    puercgp::detail::green_context context(0, requested);
    float elapsed = prepare_and_measure(graph, sources, target_step, warmup,
                                         repeats, context.stream());
    csv << requested << ',' << context.actual_sm_count() << ',' << target_step
        << ',' << elapsed << '\n';
    std::cout << "requested_sms=" << requested
              << " actual_sms=" << context.actual_sm_count()
              << " pull_ms=" << elapsed << '\n';
  }
  return 0;
}
