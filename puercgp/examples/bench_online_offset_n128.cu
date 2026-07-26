#define main bench_phase_schedule_q32_main
#include "bench_phase_schedule.cu"
#undef main

#include <map>

#include <puercgp/scheduling/online_offset_evaluator.hxx>

namespace {

constexpr int kOnlineTotalQueries = 128;
constexpr int kOnlineBatchCount = kOnlineTotalQueries / kQueryCount;

struct online_aggregate {
  double gpu_ms = 0.0;
  double evaluator_ms = 0.0;
  unsigned long long union_edges = 0;
  unsigned long long virtual_edges = 0;
  int steps = 0;
  bool correct = true;
};

std::vector<int> make_online_sources(int vertex_count, unsigned int seed) {
  std::mt19937 rng(seed);
  std::unordered_set<int> used;
  std::vector<int> sources;
  sources.reserve(kOnlineTotalQueries);
  while (sources.size() < kOnlineTotalQueries) {
    int source = static_cast<int>(rng() % vertex_count);
    if (used.insert(source).second) sources.push_back(source);
  }
  return sources;
}

std::vector<std::vector<int>> make_sequential_online_batches() {
  std::vector<std::vector<int>> batches(kOnlineBatchCount);
  for (int query = 0; query < kOnlineTotalQueries; ++query) {
    batches[query / kQueryCount].push_back(query);
  }
  return batches;
}

std::vector<std::vector<int>> load_online_batches(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open batch file: " + path.string());
  std::string line;
  std::getline(input, line);
  std::vector<std::vector<int>> batches(kOnlineBatchCount);
  std::vector<char> seen(kOnlineTotalQueries, 0);
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::stringstream row(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(row, field, ',')) fields.push_back(field);
    if (fields.size() < 3) throw std::runtime_error("invalid batch row");
    int batch = std::stoi(fields[0]);
    int slot = std::stoi(fields[1]);
    int query = std::stoi(fields[2]);
    if (batch < 0 || batch >= kOnlineBatchCount || slot < 0 ||
        slot >= kQueryCount || query < 0 || query >= kOnlineTotalQueries ||
        seen[query]) {
      throw std::runtime_error("invalid batch assignment");
    }
    if (static_cast<int>(batches[batch].size()) != slot) {
      throw std::runtime_error("batch slots must be contiguous");
    }
    batches[batch].push_back(query);
    seen[query] = 1;
  }
  for (const auto& batch : batches) {
    if (batch.size() != kQueryCount) {
      throw std::runtime_error("batch file must assign N=128 and Q=32");
    }
  }
  return batches;
}

std::vector<int> gather_batch_sources(const std::vector<int>& sources,
                                      const std::vector<int>& query_ids) {
  std::vector<int> result;
  result.reserve(query_ids.size());
  for (int query : query_ids) result.push_back(sources[query]);
  return result;
}

schedule_t make_runtime_offset_schedule(const std::vector<int>& offsets,
                                        const std::string& name) {
  if (offsets.size() != kQueryCount) {
    throw std::invalid_argument("online offset plan must contain Q offsets");
  }
  schedule_t schedule;
  schedule.name = name;
  for (int query = 0; query < kQueryCount; ++query) {
    schedule.offsets[query] = offsets[query];
  }
  std::array<int, kQueryCount> local{};
  for (int step = 0; step < 4096; ++step) {
    mask_t scheduled = 0;
    std::array<int, kQueryCount> before{};
    for (int query = 0; query < kQueryCount; ++query) {
      before[query] = local[query];
      if (step >= offsets[query]) {
        scheduled |= mask_t{1} << query;
        ++local[query];
      }
    }
    schedule.local_before.push_back(before);
    schedule.alive_masks.push_back(valid_mask());
    schedule.masks.push_back(scheduled);
  }
  return schedule;
}

void accumulate_trace(online_aggregate& aggregate,
                      const replay_result& result) {
  aggregate.gpu_ms += result.gpu_ms;
  aggregate.steps += result.global_steps;
  for (const auto& step : result.steps) {
    aggregate.union_edges += step.scheduled_union_edges;
    aggregate.virtual_edges += step.scheduled_virtual_edges;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: bench_online_offset_n128 <graph> <phase-index> "
                 "<output-dir> [--seed=42] [--repeats=3] [--landmarks=32] "
                 "[--batch-swaps=16] [--batch-file=path] "
                 "[--rebuild-index]\n";
    return 2;
  }
  std::string graph_path = argv[1];
  std::filesystem::path index_path = argv[2];
  std::filesystem::path output = argv[3];
  unsigned int seed = 42;
  int repeats = 3;
  int landmark_count = 32;
  int batch_swaps = 16;
  bool rebuild_index = false;
  std::filesystem::path provided_batch_path;
  for (int i = 4; i < argc; ++i) {
    std::string argument = argv[i];
    if (argument.rfind("--seed=", 0) == 0)
      seed = static_cast<unsigned int>(std::stoul(argument.substr(7)));
    else if (argument.rfind("--repeats=", 0) == 0)
      repeats = std::stoi(argument.substr(10));
    else if (argument.rfind("--landmarks=", 0) == 0)
      landmark_count = std::stoi(argument.substr(12));
    else if (argument.rfind("--batch-swaps=", 0) == 0)
      batch_swaps = std::stoi(argument.substr(14));
    else if (argument.rfind("--batch-file=", 0) == 0)
      provided_batch_path = argument.substr(13);
    else if (argument == "--rebuild-index")
      rebuild_index = true;
  }
  if (repeats <= 0 || landmark_count <= 0 || batch_swaps < 0) {
    throw std::invalid_argument("invalid benchmark configuration");
  }
  std::filesystem::create_directories(output);

  auto graph_storage = puercgp_examples::load_graph_auto(graph_path, false);
  auto graph = graph_storage.view();
  auto preprocessing_start = std::chrono::steady_clock::now();
  bool built_index = rebuild_index || !std::filesystem::is_regular_file(index_path);
  puercgp::scheduling::landmark_phase_index phase_index;
  if (!rebuild_index && std::filesystem::is_regular_file(index_path)) {
    phase_index =
        puercgp::scheduling::landmark_phase_index::load(index_path);
    if (phase_index.vertex_count() != graph_storage.vertices) {
      throw std::runtime_error("phase index does not match graph");
    }
  } else {
    phase_index = puercgp::scheduling::landmark_phase_index::build(
        graph_storage.row_offsets, graph_storage.column_indices,
        landmark_count);
    phase_index.save(index_path);
  }
  double preprocessing_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - preprocessing_start).count();

  auto sources = make_online_sources(graph_storage.vertices, seed);
  auto sequential_batches = make_sequential_online_batches();
  puercgp::scheduling::online_offset_evaluator evaluator(phase_index);
  auto batch_plan =
      evaluator.plan_batches(sources, kQueryCount, batch_swaps);
  const auto& landmark_batches = batch_plan.query_ids;
  double grouping_ms = batch_plan.evaluator_ms;
  std::vector<std::vector<int>> provided_batches;
  if (!provided_batch_path.empty()) {
    provided_batches = load_online_batches(provided_batch_path);
  }

  std::array<unsigned long long, kOnlineTotalQueries> reference_counts{};
  std::array<unsigned long long, kOnlineTotalQueries> reference_sums{};
  std::map<std::string, std::vector<online_aggregate>> measured;
  std::map<std::string, online_aggregate> traced;
  std::vector<std::string> methods = {
      "sequential_baseline", "sequential_online_offset",
      "landmark_batch_online_offset"};
  if (!provided_batches.empty()) {
    methods.push_back("provided_batch_online_offset");
  }
  unsigned int sample_mod = choose_sample_mod(graph_storage.vertices);

  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (const auto& method : methods) {
      const auto& batches = method == "landmark_batch_online_offset"
          ? landmark_batches
          : (method == "provided_batch_online_offset" ? provided_batches
                                                        : sequential_batches);
      online_aggregate total;
      if (method == "landmark_batch_online_offset") {
        total.evaluator_ms += grouping_ms;
      }
      for (int batch_id = 0; batch_id < kOnlineBatchCount; ++batch_id) {
        auto batch_sources = gather_batch_sources(sources, batches[batch_id]);
        std::vector<int> offsets(kQueryCount, 0);
        if (method != "sequential_baseline") {
          auto plan = evaluator.evaluate(batch_sources);
          offsets = std::move(plan.offsets);
          total.evaluator_ms += plan.evaluator_ms;
        }
        auto schedule = make_runtime_offset_schedule(offsets, method);
        auto result = run_schedule(graph, batch_sources, &schedule,
                                   repeat == 0, false, sample_mod);
        accumulate_trace(total, result);
        for (int slot = 0; slot < kQueryCount; ++slot) {
          int query = batches[batch_id][slot];
          if (method == "sequential_baseline" && repeat == 0) {
            reference_counts[query] = result.visited_counts[slot];
            reference_sums[query] = result.distance_sums[slot];
          } else {
            total.correct &=
                result.visited_counts[slot] == reference_counts[query];
            total.correct &= result.distance_sums[slot] == reference_sums[query];
          }
        }
      }
      measured[method].push_back(total);
      if (repeat == 0) traced[method] = total;
      std::cout << "repeat=" << repeat << ' ' << method
                << " gpu_ms=" << total.gpu_ms
                << " evaluator_ms=" << total.evaluator_ms
                << " correct=" << (total.correct ? "yes" : "no") << '\n';
      if (!total.correct) return 1;
    }
  }

  std::ofstream source_file(output / "sources.csv");
  source_file << "query_id,source,predicted_length\n";
  for (int query = 0; query < kOnlineTotalQueries; ++query) {
    source_file << query << ',' << sources[query] << ','
                << phase_index.estimate_phase_length(sources[query]) << '\n';
  }
  std::ofstream batch_file(output / "landmark_batches.csv");
  batch_file << "batch,slot,query_id,source\n";
  for (int batch = 0; batch < kOnlineBatchCount; ++batch) {
    for (int slot = 0; slot < kQueryCount; ++slot) {
      int query = landmark_batches[batch][slot];
      batch_file << batch << ',' << slot << ',' << query << ','
                 << sources[query] << '\n';
    }
  }

  std::ofstream plan_file(output / "online_plans.csv");
  plan_file << "batch,slot,query_id,source,predicted_length,offset,"
               "alignment_score,evaluator_ms\n";
  for (int batch = 0; batch < kOnlineBatchCount; ++batch) {
    auto batch_sources = gather_batch_sources(sources, landmark_batches[batch]);
    auto plan = evaluator.evaluate(batch_sources);
    for (int slot = 0; slot < kQueryCount; ++slot) {
      int query = landmark_batches[batch][slot];
      plan_file << batch << ',' << slot << ',' << query << ','
                << sources[query] << ',' << plan.predicted_lengths[slot] << ','
                << plan.offsets[slot] << ',' << plan.alignment_score << ','
                << plan.evaluator_ms << '\n';
    }
  }

  std::ofstream metadata(output / "metadata.csv");
  metadata << "graph,vertices,edges,index_path,index_built,landmarks,"
              "preprocessing_ms,index_bytes,seed,repeats,batch_swaps\n";
  metadata << graph_path << ',' << graph_storage.vertices << ','
           << graph_storage.edges << ',' << index_path.string() << ','
           << (built_index ? 1 : 0) << ',' << phase_index.landmark_count() << ','
           << preprocessing_ms << ',' << std::filesystem::file_size(index_path)
           << ',' << seed << ',' << repeats << ',' << batch_swaps << '\n';

  std::ofstream summary(output / "summary.csv");
  summary << "method,gpu_ms,evaluator_ms,end_to_end_ms,union_edges,"
             "virtual_edges,sharing,steps,correct\n";
  for (const auto& method : methods) {
    auto values = measured[method];
    std::sort(values.begin(), values.end(), [](const auto& first,
                                               const auto& second) {
      return first.gpu_ms + first.evaluator_ms <
             second.gpu_ms + second.evaluator_ms;
    });
    const auto& value = values[values.size() / 2];
    const auto& trace = traced[method];
    double sharing = trace.union_edges == 0
        ? 1.0
        : static_cast<double>(trace.virtual_edges) / trace.union_edges;
    summary << method << ',' << value.gpu_ms << ',' << value.evaluator_ms
            << ',' << value.gpu_ms + value.evaluator_ms << ','
            << trace.union_edges << ',' << trace.virtual_edges << ','
            << sharing << ',' << trace.steps << ',' << (value.correct ? 1 : 0)
            << '\n';
  }
  std::cout << "preprocessing_ms=" << preprocessing_ms
            << " landmarks=" << phase_index.landmark_count()
            << " index=" << index_path << '\n';
  return 0;
}
