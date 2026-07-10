#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include <puercgp/core/types.hxx>

namespace puercgp {

template <typename vertex_t>
struct query_result_t {
  vertex_t query_id = 0;
  vertex_t source = 0;
  vertex_t completion_level = 0;
  float completion_wall_time_ms = 0.0f;
};

struct iteration_profile_t {
  int iteration = 0;
  std::string mode;
  std::size_t frontier_size = 0;
  std::size_t unique_frontier_size = 0;
  unsigned long long edge_count = 0;
  unsigned long long actual_edge_count = 0;
  unsigned long long virtual_edge_count = 0;
  double pull_frontier_threshold = 0.0;
  double pull_edge_threshold = 0.0;
  float iteration_wall_ms = 0.0f;
  float degree_scan_ms = 0.0f;
  float push_kernel_ms = 0.0f;
  float shared_push_kernel_ms = 0.0f;
  float pull_kernel_ms = 0.0f;
  float compact_ms = 0.0f;
  float count_sync_ms = 0.0f;
  // 本轮结束时各 slot 的活跃 frontier 位掩码：bit s=1 表示 slot s 本轮产生了 frontier 写入
  // （未收敛）；bit s=0 表示 slot s 本轮无写入（已收敛）。replenishment 调度依据。
  query_mask_t query_convergence_mask = 0;
};

template <typename vertex_t, typename value_t>
struct run_result_t {
  thrust::device_vector<value_t> values;
  std::vector<query_result_t<vertex_t>> queries;
  std::vector<std::size_t> frontier_sizes;
  std::vector<std::size_t> unique_frontier_sizes;
  std::vector<unsigned long long> iteration_edge_counts;
  std::vector<unsigned long long> actual_iteration_edge_counts;
  std::vector<unsigned long long> virtual_iteration_edge_counts;
  std::vector<std::string> iteration_modes;
  std::vector<float> iteration_wall_times_ms;
  std::vector<iteration_profile_t> iteration_profiles;
  run_options options;
  int effective_query_dim = 0;
  float gpu_time_ms = 0.0f;
  float wall_time_ms = 0.0f;
  float shared_push_kernel_ms = 0.0f;
  int iterations = 0;
};

}  // namespace puercgp
