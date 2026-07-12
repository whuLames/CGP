#define main bench_phase_schedule_q32_main
#include "bench_phase_schedule.cu"
#undef main

#include <map>

namespace {

constexpr int kTotalQueries = 128;
constexpr int kBatchCount = kTotalQueries / kQueryCount;

struct global_query_trace {
  int source = -1;
  int levels = 0;
  std::unordered_map<int, std::pair<short, unsigned int>> vertices;
};

struct aggregate_result {
  double gpu_ms = 0.0;
  double push_ms = 0.0;
  unsigned long long union_edges = 0;
  unsigned long long virtual_edges = 0;
  int steps = 0;
  bool correct = true;
};

std::vector<int> make_global_sources(int vertex_count, unsigned int seed) {
  std::mt19937 rng(seed);
  std::unordered_set<int> used;
  std::vector<int> sources;
  sources.reserve(kTotalQueries);
  while (sources.size() < kTotalQueries) {
    int source = static_cast<int>(rng() % vertex_count);
    if (used.insert(source).second) sources.push_back(source);
  }
  return sources;
}

std::vector<int> batch_sources(const std::vector<int>& sources,
                               const std::vector<int>& query_ids) {
  std::vector<int> result;
  result.reserve(kQueryCount);
  for (int id : query_ids) result.push_back(sources[id]);
  return result;
}

std::vector<std::vector<int>> sequential_batches() {
  std::vector<std::vector<int>> batches(kBatchCount);
  for (int q = 0; q < kTotalQueries; ++q)
    batches[q / kQueryCount].push_back(q);
  return batches;
}

std::vector<std::vector<unsigned long long>> build_affinity(
    const std::vector<global_query_trace>& traces, int max_delta) {
  using occurrence = std::tuple<int, short, unsigned int>;
  std::unordered_map<int, std::vector<occurrence>> inverted;
  for (int q = 0; q < kTotalQueries; ++q) {
    for (const auto& [vertex, state] : traces[q].vertices)
      inverted[vertex].emplace_back(q, state.first, state.second);
  }

  const int width = 2 * max_delta + 1;
  std::vector<std::vector<unsigned long long>> buckets(
      kTotalQueries * kTotalQueries,
      std::vector<unsigned long long>(width, 0));
  for (const auto& [vertex, occurrences] : inverted) {
    (void)vertex;
    for (std::size_t i = 0; i < occurrences.size(); ++i) {
      auto [q, lq, dq] = occurrences[i];
      for (std::size_t j = i + 1; j < occurrences.size(); ++j) {
        auto [r, lr, dr] = occurrences[j];
        int delta = static_cast<int>(lq) - static_cast<int>(lr);
        if (std::abs(delta) > max_delta) continue;
        unsigned long long weight = std::min(dq, dr);
        buckets[q * kTotalQueries + r][delta + max_delta] += weight;
        buckets[r * kTotalQueries + q][-delta + max_delta] += weight;
      }
    }
  }

  std::vector<std::vector<unsigned long long>> affinity(
      kTotalQueries, std::vector<unsigned long long>(kTotalQueries, 0));
  for (int q = 0; q < kTotalQueries; ++q) {
    for (int r = q + 1; r < kTotalQueries; ++r) {
      const auto& values = buckets[q * kTotalQueries + r];
      affinity[q][r] = affinity[r][q] =
          *std::max_element(values.begin(), values.end());
    }
  }
  return affinity;
}

std::vector<std::vector<int>> greedy_batches(
    const std::vector<std::vector<unsigned long long>>& affinity) {
  std::vector<char> assigned(kTotalQueries, 0);
  std::vector<std::vector<int>> batches;
  for (int batch_id = 0; batch_id < kBatchCount; ++batch_id) {
    int seed = -1;
    unsigned long long seed_score = 0;
    for (int q = 0; q < kTotalQueries; ++q) {
      if (assigned[q]) continue;
      unsigned long long score = 0;
      for (int r = 0; r < kTotalQueries; ++r)
        if (!assigned[r]) score += affinity[q][r];
      if (seed < 0 || score > seed_score) {
        seed = q;
        seed_score = score;
      }
    }
    std::vector<int> batch{seed};
    assigned[seed] = 1;
    while (batch.size() < kQueryCount) {
      int best = -1;
      unsigned long long best_score = 0;
      for (int q = 0; q < kTotalQueries; ++q) {
        if (assigned[q]) continue;
        unsigned long long score = 0;
        for (int member : batch) score += affinity[q][member];
        if (best < 0 || score > best_score) {
          best = q;
          best_score = score;
        }
      }
      batch.push_back(best);
      assigned[best] = 1;
    }
    batches.push_back(std::move(batch));
  }
  return batches;
}

void add_work(aggregate_result& aggregate, const replay_result& result) {
  aggregate.gpu_ms += result.gpu_ms;
  aggregate.push_ms += result.push_ms;
  aggregate.steps += result.global_steps;
  for (const auto& step : result.steps) {
    aggregate.union_edges += step.scheduled_union_edges;
    aggregate.virtual_edges += step.scheduled_virtual_edges;
  }
}

void write_batches(const std::filesystem::path& output,
                   const std::string& name,
                   const std::vector<std::vector<int>>& batches,
                   const std::vector<int>& sources) {
  std::ofstream file(output / (name + "_batches.csv"));
  file << "batch,slot,query_id,source\n";
  for (std::size_t b = 0; b < batches.size(); ++b)
    for (std::size_t slot = 0; slot < batches[b].size(); ++slot)
      file << b << ',' << slot << ',' << batches[b][slot] << ','
           << sources[batches[b][slot]] << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: bench_phase_schedule_n128 <graph> <output-dir> "
                 "[--seed=42] [--repeats=5]\n";
    return 2;
  }
  std::string graph_path = argv[1];
  std::filesystem::path output = argv[2];
  unsigned int seed = 42;
  int repeats = 5;
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
  auto sources = make_global_sources(graph_storage.vertices, seed);
  auto baseline_batches = sequential_batches();
  unsigned int sample_mod = choose_sample_mod(graph_storage.vertices);

  std::ofstream source_file(output / "sources.csv");
  source_file << "query_id,source\n";
  for (int q = 0; q < kTotalQueries; ++q)
    source_file << q << ',' << sources[q] << '\n';

  std::vector<global_query_trace> global_traces(kTotalQueries);
  std::array<unsigned long long, kTotalQueries> reference_counts{};
  std::array<unsigned long long, kTotalQueries> reference_sums{};
  std::cout << "profiling N=128 Q=32 sample_mod=" << sample_mod << '\n';
  for (int b = 0; b < kBatchCount; ++b) {
    auto local_sources = batch_sources(sources, baseline_batches[b]);
    auto trace = run_schedule(graph, local_sources, nullptr, true, true,
                              sample_mod);
    for (int slot = 0; slot < kQueryCount; ++slot) {
      int q = baseline_batches[b][slot];
      global_traces[q].source = sources[q];
      reference_counts[q] = trace.visited_counts[slot];
      reference_sums[q] = trace.distance_sums[slot];
    }
    for (const auto& step : trace.steps) {
      for (const auto& record : step.samples) {
        mask_t bits = record.mask;
        while (bits != 0) {
          int slot = __builtin_ffsll(static_cast<long long>(bits)) - 1;
          int q = baseline_batches[b][slot];
          global_traces[q].vertices[record.vertex] =
              {static_cast<short>(step.step), record.degree};
          global_traces[q].levels =
              std::max(global_traces[q].levels, step.step + 1);
          bits &= bits - 1;
        }
      }
    }
  }

  auto affinity = build_affinity(global_traces, 16);
  auto selected_batches = greedy_batches(affinity);
  write_batches(output, "baseline", baseline_batches, sources);
  write_batches(output, "affinity", selected_batches, sources);

  std::ofstream affinity_file(output / "affinity.csv");
  affinity_file << "query_a,query_b,score\n";
  for (int q = 0; q < kTotalQueries; ++q)
    for (int r = q + 1; r < kTotalQueries; ++r)
      affinity_file << q << ',' << r << ',' << affinity[q][r] << '\n';

  const std::array<std::string, 4> methods = {
      "sequential_baseline", "affinity_baseline", "affinity_offset",
      "affinity_pause"};
  std::map<std::string, std::vector<aggregate_result>> measured;
  std::map<std::string, aggregate_result> trace_metrics;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (const auto& method : methods) {
      const auto& batches = method == "sequential_baseline"
          ? baseline_batches : selected_batches;
      aggregate_result total;
      for (int b = 0; b < kBatchCount; ++b) {
        auto local_sources = batch_sources(sources, batches[b]);
        auto base_trace = run_schedule(graph, local_sources, nullptr, true,
                                       true, sample_mod);
        trace_model model = build_trace_model(base_trace);
        schedule_t schedule = make_baseline_schedule(model);
        if (method == "affinity_offset") {
          auto heavy = make_heavy_schedule(model);
          schedule = make_offset_search_schedule(model, heavy, seed + b + 1);
        } else if (method == "affinity_pause") {
          schedule = make_greedy_pause_schedule(model);
        }
        auto result = run_schedule(graph, local_sources, &schedule,
                                   repeat == 0, false, sample_mod);
        add_work(total, result);
        for (int slot = 0; slot < kQueryCount; ++slot) {
          int q = batches[b][slot];
          total.correct &= result.visited_counts[slot] == reference_counts[q];
          total.correct &= result.distance_sums[slot] == reference_sums[q];
        }
        if (repeat == 0) {
          std::filesystem::path batch_dir = output / method /
              ("batch_" + std::to_string(b));
          std::filesystem::create_directories(batch_dir);
          write_schedule(batch_dir, schedule);
          write_step_trace(batch_dir, result);
        }
      }
      measured[method].push_back(total);
      if (repeat == 0) trace_metrics[method] = total;
      std::cout << "repeat=" << repeat << ' ' << method
                << " gpu_ms=" << total.gpu_ms
                << " correct=" << (total.correct ? "yes" : "no") << '\n';
      if (!total.correct) return 1;
    }
  }

  std::ofstream summary(output / "summary.csv");
  summary << "method,gpu_ms,push_ms,union_edges,virtual_edges,sharing,steps,correct\n";
  for (const auto& method : methods) {
    auto results = measured[method];
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
      return a.gpu_ms < b.gpu_ms;
    });
    const auto& value = results[results.size() / 2];
    const auto& trace_value = trace_metrics[method];
    double sharing = trace_value.union_edges == 0 ? 1.0 :
        static_cast<double>(trace_value.virtual_edges) /
            trace_value.union_edges;
    summary << method << ',' << value.gpu_ms << ',' << value.push_ms << ','
            << trace_value.union_edges << ',' << trace_value.virtual_edges
            << ',' << sharing << ',' << trace_value.steps << ','
            << (value.correct ? 1 : 0) << '\n';
  }
  return 0;
}
