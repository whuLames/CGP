#pragma once
#include "graphweft/kernels.hpp"
#include <functional>
#include <map>
#include <string>
#include <vector>
namespace graphweft {
struct Query { uint64_t id; uint32_t source; double score = 0; uint32_t offset = 0; uint64_t feature_key = 0; };
struct Options {
  Algorithm algorithm = Algorithm::BFS;
  Layout layout = Layout::VertexMajor;
  FrontierMode frontier = FrontierMode::Unordered;
  uint32_t capacity = 32;
  uint32_t group_width = 8;
  double memory_fraction = .8;
  double pull_threshold = .2;
  bool sort_by_score = false;
  enum class Predictor { Imported, ImportedKey, CoreDistance, WeightedBoundary } predictor = Predictor::Imported;
  bool use_offsets = false;
  bool phase_offsets = false;
  uint32_t landmarks = 16;
  uint32_t max_offset = 16;
  bool copy_results_to_cpu = false;
  bool profile_compare = false;
  // Optional CUDA-event timing of the production graph kernel only.
  bool profile_kernel = false;
  // One CSV row per executed round. Setting this also enables profile_kernel.
  std::string round_metrics_path;
  enum class Selector { Threshold, Push, Pull, Replay } selector = Selector::Threshold;
  std::vector<KernelId> replay;
  std::string checkpoint_path;
  uint32_t checkpoint_round = 0;
  std::string plan_output_path;
};
struct AllocationPlan {
  std::map<std::string, uint64_t> bytes;
  uint64_t total() const;
};
AllocationPlan allocation_plan(const HostGraph&, const Options&, uint32_t capacity);
uint32_t max_capacity(const HostGraph&, const Options&, uint64_t allowed_bytes, uint32_t limit);
struct RoundFeatures { uint64_t edge_pairs = 0, active_queries = 0; double density = 0; bool density_valid = false; };
struct QueryResult { uint64_t id; uint32_t source; uint32_t completion_local_round; std::vector<float> values; };
struct RunStats { uint64_t batches = 0, rounds = 0, push_rounds = 0, pull_rounds = 0; double planning_ms = 0, prediction_ms = 0, initialization_ms = 0, copy_ms = 0, kernel_ms = 0, kernel_gpu_ms = 0, frontier_ms = 0, compare_ms = 0, feature_ms = 0, selector_ms = 0, transfer_ms = 0, round_ms = 0, execution_ms = 0, task_wall_ms = 0, total_ms = 0; };
using ResultCallback = std::function<void(const QueryResult&)>;
struct RoundSnapshot {
  uint64_t batch_index;
  uint32_t global_round, used_slots, physical_slots, words;
  Layout layout;
  std::vector<uint64_t> query_ids;
  std::vector<float> values;
  std::vector<uint64_t> mask;
  std::vector<uint32_t> frontier;
};
using RoundCallback = std::function<void(const RoundSnapshot&)>;
// Diagnostic hook before the production round. Only new_values/error_flag are scratch.
// The executor restores scratch after the probe; input state must remain unchanged.
using DeviceRoundProbe = std::function<void(const Context&, uint64_t batch, uint32_t round)>;
RunStats run(const HostGraph&, std::vector<Query>, const Options&, const ResultCallback& = {}, const RoundCallback& = {}, const DeviceRoundProbe& = {});
std::vector<Query> load_queries(const std::string& path, uint32_t vertices, const std::string& graph_identity = {}, bool require_identity = false, uint32_t expected_capacity = 0);
}
