#pragma once
#include "graphweft/kernels.hpp"
#include <functional>
#include <map>
#include <string>
#include <vector>
namespace graphweft {
struct Query {
  uint64_t id; uint32_t source; double score = 0; uint32_t offset = 0; uint64_t feature_key = 0;
  // -1 inherits Options::algorithm.  Keeping this field last preserves the
  // source compatibility of existing aggregate initializers and frozen plans.
  int algorithm = -1;
  // Optional reference-only length used by the oracle diagnostic planner.
  uint32_t reference_rounds = 0;
};
struct Options {
  Algorithm algorithm = Algorithm::BFS;
  Layout layout = Layout::VertexMajor;
  FrontierMode frontier = FrontierMode::Unordered;
  FrontierBuildMode frontier_build = FrontierBuildMode::Fused;
  bool frontier_mask64 = true;
  uint32_t capacity = 32;
  uint32_t group_width = 8;
  // Reclaim a physical group as soon as all its members finish.  Queries are
  // grouped by algorithm and groups from algorithm queues are interleaved.
  bool group_refill = false;
  // Experimental refill ablation: allow homogeneous SSSP groups to be
  // reclaimed independently instead of waiting for a synchronized wave.
  bool eager_sssp_refill = false;
  // Admit a replacement group only when the current resident set is below a
  // bounded concurrency level and the last round remained Push.  This turns
  // refill into slack stealing instead of unconditional slot replacement.
  bool interference_aware_refill = false;
  uint32_t refill_max_active_groups = 2;
  // Start at most one group from the next cohort while the current cohort has
  // a lone Push straggler, then restore a full cohort when that straggler
  // retires.  This preserves batch efficiency after crossing the barrier.
  bool interference_bridge_refill = false;
  // Experimental Iteration ablation: predict one Push mapping per G-sized
  // group and merge groups which select the same partition kernel.
  bool group_iteration_mapping = false;
  bool same_algorithm_groups = false;
  bool oracle_order = false;
  double memory_fraction = .8;
  double pull_threshold = .2;
  enum class PushMapping { Shared, Static, Degree, Density, Adaptive, Iteration } push_mapping = PushMapping::Shared;
  uint32_t push_query_lanes = 8;
  uint32_t push_grain = 0; // 0/1/2: 1/2/4 warps; 3/4: 2/4 blocks.
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
  // Used only after a non-Replay selector has chosen Pull. DensePull means
  // auto, preserving the historical slot-aware selection policy.
  KernelId pull_kernel = KernelId::DensePull;
  std::string checkpoint_path;
  uint32_t checkpoint_round = 0;
  std::string plan_output_path;
  std::string completion_output_path;
  std::string schedule_events_path;
};
struct AllocationPlan {
  std::map<std::string, uint64_t> bytes;
  uint64_t total() const;
};
AllocationPlan allocation_plan(const HostGraph&, const Options&, uint32_t capacity);
uint32_t max_capacity(const HostGraph&, const Options&, uint64_t allowed_bytes, uint32_t limit);
struct RoundFeatures {
  uint64_t edge_pairs = 0, active_queries = 0;
  double density = 0;
  bool density_valid = false;
  uint64_t vertex_pairs = 0;
  uint32_t frontier_vertices = 0, graph_vertices = 0;
  uint64_t graph_edges = 0;
};
struct QueryResult { uint64_t id; uint32_t source; uint32_t completion_local_round; std::vector<float> values; };
struct QueryFingerprint {
  uint64_t id = 0; uint32_t source = 0, completion_local_round = 0, vertices = 0;
  uint64_t sum = 0, xor_value = 0;
};
struct CompletionRecord {
  uint64_t id = 0; uint32_t source = 0; Algorithm algorithm = Algorithm::BFS;
  uint32_t slot = 0, group = 0, activation_round = 0, completion_round = 0, service_rounds = 0;
  // Host wall-clock timestamps relative to workload submission.  They are
  // sampled at synchronization points that already exist in the executor, so
  // enabling completion metadata does not add a GPU synchronization.
  double activation_ms = 0, completion_ms = 0, waiting_ms = 0,
         service_ms = 0, submit_to_completion_ms = 0;
};
struct RunStats {
  uint64_t batches = 0, rounds = 0, push_rounds = 0, pull_rounds = 0;
  uint64_t group_refills = 0, completed_slot_rounds = 0, active_slot_rounds = 0, capacity_slot_rounds = 0;
  uint64_t refill_admitted_groups = 0, refill_deferred_groups = 0;
  uint64_t refill_deferred_pull_groups = 0, refill_deferred_capacity_groups = 0;
  uint64_t refill_deferred_incompatible_groups = 0;
  uint64_t group_mapping_rounds = 0, group_mapping_divergent_rounds = 0, group_mapping_launches = 0;
  uint64_t final_drain_rounds = 0;
  double planning_ms = 0, prediction_ms = 0, initialization_ms = 0, recycle_ms = 0, copy_ms = 0, kernel_ms = 0, kernel_gpu_ms = 0, frontier_ms = 0, compare_ms = 0, feature_ms = 0, adaptive_preparation_ms = 0, selector_ms = 0, transfer_ms = 0, round_ms = 0, execution_ms = 0, task_wall_ms = 0, total_ms = 0;
  double workload_ms = 0;
  std::vector<CompletionRecord> completions;
};
using ResultCallback = std::function<void(const QueryResult&)>;
using FingerprintCallback = std::function<void(const QueryFingerprint&)>;
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
RunStats run(const HostGraph&, std::vector<Query>, const Options&, const ResultCallback& = {}, const RoundCallback& = {}, const DeviceRoundProbe& = {}, const FingerprintCallback& = {});
std::vector<Query> load_queries(const std::string& path, uint32_t vertices, const std::string& graph_identity = {}, bool require_identity = false, uint32_t expected_capacity = 0);
}
