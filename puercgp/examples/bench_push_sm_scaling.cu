#define main bench_phase_schedule_q32_main
#include "bench_phase_schedule.cu"
#undef main

#include <puercgp/core/green_context.hxx>

namespace {

float median_value(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: bench_push_sm_scaling <graph> <output.csv> "
                 "[--seed=42] [--repeats=7]\n";
    return 2;
  }
  std::string graph_path = argv[1];
  std::filesystem::path output = argv[2];
  unsigned int seed = 42;
  int repeats = 7;
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--seed=", 0) == 0)
      seed = static_cast<unsigned int>(std::stoul(arg.substr(7)));
    else if (arg.rfind("--repeats=", 0) == 0)
      repeats = std::stoi(arg.substr(10));
  }

  auto graph_storage = puercgp_examples::load_graph_auto(graph_path, false);
  auto graph = graph_storage.view();
  auto sources = make_unique_sources(graph_storage.vertices, seed);
  unsigned int sample_mod = choose_sample_mod(graph_storage.vertices);
  const std::array<unsigned int, 8> sm_counts = {2, 4, 8, 16, 32, 48, 64, 80};

  std::filesystem::create_directories(output.parent_path());
  std::ofstream csv(output);
  csv << "requested_sms,actual_sms,step,unique_vertices,union_edges,"
         "virtual_edges,sharing,active_queries,push_ms\n";

  for (unsigned int requested : sm_counts) {
    puercgp::detail::green_context context(0, requested);
    std::vector<replay_result> runs;
    runs.reserve(repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
      runs.push_back(run_schedule(graph, sources, nullptr, true, false,
                                  sample_mod, context.stream(), true));
    }
    std::size_t steps = runs.front().steps.size();
    for (const auto& run : runs) {
      if (run.steps.size() != steps)
        throw std::runtime_error("BFS step count changed across repeats");
    }
    for (std::size_t step = 0; step < steps; ++step) {
      const auto& trace = runs.front().steps[step];
      std::vector<float> times;
      for (const auto& run : runs) times.push_back(run.steps[step].push_ms);
      double sharing = trace.scheduled_union_edges == 0 ? 1.0 :
          static_cast<double>(trace.scheduled_virtual_edges) /
              trace.scheduled_union_edges;
      csv << requested << ',' << context.actual_sm_count() << ',' << step
          << ',' << trace.unique_vertices << ','
          << trace.scheduled_union_edges << ','
          << trace.scheduled_virtual_edges << ',' << sharing << ','
          << __builtin_popcountll(trace.scheduled_mask) << ','
          << median_value(times) << '\n';
    }
    std::cout << "requested_sms=" << requested
              << " actual_sms=" << context.actual_sm_count()
              << " steps=" << steps << '\n';
  }
  return 0;
}
