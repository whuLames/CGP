#pragma once
#include "graphweft/graph.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
namespace graphweft {
enum class Algorithm : int { BFS, SSSP, SSWP };
enum class Layout : int { VertexMajor, Grouped };
enum class KernelId : int { SharedPush, DensePull, AdaptivePush, PushPartitionBase = 100,
  PullCheckFreeBase = 200, PullCheckBase = 300,
  GroupedG8Edge4Warp4Pull = 400,
  VmFusedSerialSmemQ32 = 401,
  VmFusedSerialSmemQ16 = 402,
  VmFusedSerialSmemQ8 = 403,
  VmParallelSmemShuffleQ32 = 404,
  VmParallelSmemShuffleQ16 = 405,
  VmParallelSmemShuffleQ8 = 406,
  // Dense Pull additions deliberately start in a new range.  Values 0--406
  // are checkpoint/replay ABI and must not be renumbered.
  DenseFusedSerialGlobalQ32 = 500,
  DenseFusedSerialGlobalQ16,
  DenseFusedSerialGlobalQ8,
  DenseSerialSharedQ16,
  DenseSerialSharedQ8,
  DenseSerialShuffleQ16,
  DenseSerialShuffleQ8,
  DenseSerialSmemQ32,
  DenseSerialSmemShuffleQ16,
  DenseSerialSmemShuffleQ8,
  DenseParallelGlobalQ32,
  DenseParallelSharedQ16,
  DenseParallelSharedQ8,
  DenseParallelShuffleQ16,
  DenseParallelShuffleQ8 };
static_assert(int(KernelId::SharedPush)==0 && int(KernelId::DensePull)==1 &&
              int(KernelId::AdaptivePush)==2 && int(KernelId::PushPartitionBase)==100 &&
              int(KernelId::PullCheckFreeBase)==200 && int(KernelId::PullCheckBase)==300 &&
              int(KernelId::GroupedG8Edge4Warp4Pull)==400 &&
              int(KernelId::VmFusedSerialSmemQ32)==401 &&
              int(KernelId::VmParallelSmemShuffleQ8)==406,
              "legacy replay KernelId ABI changed");
constexpr int push_partition_count = 30;
constexpr int pull_partition_count = 30;
constexpr int adaptive_push_bucket_count = 5;
inline constexpr KernelId dense_pull_candidates[] = {
  KernelId::DenseFusedSerialGlobalQ32,KernelId::DenseFusedSerialGlobalQ16,
  KernelId::DenseFusedSerialGlobalQ8,KernelId::VmFusedSerialSmemQ32,
  KernelId::VmFusedSerialSmemQ16,KernelId::VmFusedSerialSmemQ8,
  KernelId::DenseSerialSharedQ16,KernelId::DenseSerialSharedQ8,
  KernelId::DenseSerialShuffleQ16,KernelId::DenseSerialShuffleQ8,
  KernelId::DenseSerialSmemQ32,KernelId::DenseSerialSmemShuffleQ16,
  KernelId::DenseSerialSmemShuffleQ8,KernelId::DenseParallelGlobalQ32,
  KernelId::DenseParallelSharedQ16,KernelId::DenseParallelSharedQ8,
  KernelId::DenseParallelShuffleQ16,KernelId::DenseParallelShuffleQ8,
  KernelId::VmParallelSmemShuffleQ32,KernelId::VmParallelSmemShuffleQ16,
  KernelId::VmParallelSmemShuffleQ8};
inline constexpr int dense_pull_candidate_count =
  int(sizeof(dense_pull_candidates)/sizeof(dense_pull_candidates[0]));
// Returns -1 for zero work, then W1/W2/W4/B2/B4 for the fixed load bands.
__host__ __device__ constexpr int adaptive_push_bucket(uint64_t load) {
  return load==0?-1:load<=256?0:load<=1024?1:load<=4096?2:load<=16384?3:4;
}
struct PushPartition { uint32_t group_size, warps_per_block, blocks_per_vertex; };
KernelId push_partition_id(int index);
PushPartition push_partition(int index);
KernelId pull_partition_id(int index, bool check);
PushPartition pull_partition(int index);
KernelId vm_pull_id(bool parallel, uint32_t query_width);
KernelId default_pull_kernel(uint32_t physical_slots, uint32_t vertices, uint64_t edges);
bool is_pull_kernel(KernelId);
bool is_dense_pull_kernel(KernelId);
// Canonical replay/override spelling. Returns nullptr for non-dense kernels.
const char* dense_pull_token(KernelId);
// Accepts the canonical pull-dense-* tokens and the six legacy pull-vm-*
// aliases. Throws std::invalid_argument for an unknown token.
KernelId parse_dense_pull_token(const std::string&);
enum class FrontierMode : int { Unordered, Stable };
enum class FrontierBuildMode : int { Scan, Fused, Direct };
struct FrontierOutput {
  uint64_t* mask = nullptr;
  uint32_t* flags = nullptr;
  uint32_t* list = nullptr;
  uint32_t* count = nullptr;
  uint64_t* pair_count = nullptr;
  uint32_t* slot_count = nullptr;
  bool direct_append = false;
  bool mask64 = true;
};
struct ValueView {
  float* data;
  uint32_t vertices, slots, group_width;
  Layout layout;
  __host__ __device__ size_t index(uint32_t vertex, uint32_t slot) const {
    return layout == Layout::VertexMajor ? size_t(vertex) * slots + slot
      : (size_t(slot / group_width) * vertices + vertex) * group_width + slot % group_width;
  }
};
struct Context {
  GraphView graph;
  ValueView old_values, new_values;
  const uint64_t* frontier_mask;
  const uint32_t* frontier;
  const uint32_t* frontier_count;
  const uint8_t* live_slots;
  uint32_t slots, words;
  Algorithm algorithm;
  cudaStream_t stream;
  int* error_flag;
  // Optional host-known frontier size for launch sizing; kernels still read the device count.
  uint32_t frontier_size_hint = UINT32_MAX;
  // Optional per-slot algorithms.  A null pointer preserves the legacy
  // homogeneous-batch contract and uses algorithm for every slot.
  const Algorithm* slot_algorithms = nullptr;
  // Optional update-driven frontier output.  Null mask preserves the
  // standalone old/new comparison path.
  FrontierOutput frontier_output = {};
};
void shared_push(const Context&);
void dense_pull(const Context&);
void adaptive_push(const Context&, const uint32_t* buckets,
                   const uint32_t counts[adaptive_push_bucket_count]);
void launch(KernelId, const Context&);
struct FrontierContext {
  ValueView old_values, new_values;
  uint64_t* mask;
  uint32_t* flags;
  uint32_t* list;
  uint32_t* count;
  uint64_t* pair_count;
  uint32_t* slot_count;
  const uint8_t* live_slots;
  uint32_t slots, words;
  Algorithm algorithm;
  FrontierMode mode;
  void* stable_temp;
  size_t stable_temp_bytes;
  cudaStream_t stream;
  const Algorithm* slot_algorithms = nullptr;
};
size_t stable_temp_bytes(uint32_t vertices);
void build_frontier(const FrontierContext&, cudaEvent_t compare_start = nullptr, cudaEvent_t compare_end = nullptr);
void prepare_fused_frontier(const FrontierContext&);
void finish_fused_frontier(const FrontierContext&);
void finish_direct_frontier(const FrontierContext&);
void rebuild_frontier(const FrontierContext&);
void count_edge_pairs(GraphView, const uint64_t* mask, const uint32_t* list, const uint32_t* count, uint32_t words, uint64_t* output, cudaStream_t);
// Aggregate the Iteration model's work features independently for aligned
// query groups. Outputs contain group_count entries and must be zeroed by the
// caller. This diagnostic path currently requires group_width=32.
void count_group_features(GraphView, const uint64_t* mask, const uint32_t* list,
                          const uint32_t* count, uint32_t words,
                          uint32_t group_count, uint32_t* frontier_vertices,
                          uint64_t* vertex_pairs, uint64_t* edge_pairs,
                          cudaStream_t);
void classify_edge_pairs(GraphView, const uint64_t* mask, const uint32_t* list,
                         const uint32_t* count, uint32_t words, uint64_t* output,
                         uint8_t* categories, uint32_t* category_counts,
                         cudaStream_t);
void scatter_adaptive_buckets(const uint32_t* list, const uint32_t* count,
                              const uint8_t* categories, uint32_t* buckets,
                              uint32_t* cursors, uint32_t maximum_count,
                              const uint32_t counts[adaptive_push_bucket_count],
                              cudaStream_t);
void initialize_values(ValueView, Algorithm, cudaStream_t);
void initialize_values(ValueView, Algorithm, const Algorithm*, cudaStream_t);
void reset_slots(ValueView, const uint8_t*, Algorithm, const Algorithm*, cudaStream_t);
void reset_slot_range(ValueView, uint32_t first_slot, uint32_t slot_count,
                      Algorithm, const Algorithm*, cudaStream_t);
void activate(ValueView, uint64_t*, const uint32_t*, const uint8_t*, uint32_t slots, uint32_t words, Algorithm, cudaStream_t);
void activate(ValueView, uint64_t*, const uint32_t*, const uint8_t*, uint32_t slots, uint32_t words, Algorithm, const Algorithm*, cudaStream_t);
void clear_slot_mask(uint64_t*, uint32_t vertices, uint32_t words, const uint8_t*, uint32_t slots, cudaStream_t);
void clear_mask(uint64_t*, uint32_t vertices, uint32_t words, cudaStream_t);
// Order-independent, full-vector validation digest.  Each output entry covers
// every (vertex,value-bit-pattern) pair for one physical query slot.
void fingerprint_values(ValueView, uint32_t first_slot, uint32_t slot_count,
                        uint64_t* sums, uint64_t* xors, cudaStream_t);
}
