#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ppbench {

using mask_t = std::uint64_t;

enum class gen_mode_t { density, overlap, mixed };
enum class exec_mode_t {
  all_push,
  all_pull,
  mixed_serial,
  mixed_concurrent
};
enum class update_mode_t { spmm_sum };

struct bench_config_t {
  std::string graph_path;
  std::string csv_path;
  std::string out_dir;
  gen_mode_t gen_mode = gen_mode_t::density;
  exec_mode_t exec_mode = exec_mode_t::all_push;
  update_mode_t update_mode = update_mode_t::spmm_sum;
  int Q = 32;
  int Q_push = 16;
  int Q_pull = 16;
  int repeat = 7;
  int warmup = 2;
  int seed = 42;
  int threads = 256;
  double rho_v = 0.01;
  double rho_q = 0.25;
  double rho_push = 0.005;
  double rho_pull = 0.10;
  double overlap = 0.50;
  double overlap_push = 0.25;
  double overlap_pull = 0.75;
  double cross_overlap = 0.0;
  int frontier_size_per_query = 0;
  bool self_test = false;
  bool density_pull_sweep = false;
};

struct host_graph_t {
  int V = 0;
  int E = 0;
  std::vector<int> row_offsets;
  std::vector<int> column_indices;
  std::vector<float> weights;
  std::vector<int> in_row_offsets;
  std::vector<int> in_column_indices;
  std::vector<float> in_weights;
  std::string name;
};

struct device_graph_t {
  int V = 0;
  int E = 0;
  const int* row_offsets = nullptr;
  const int* column_indices = nullptr;
  const float* weights = nullptr;
  const int* in_row_offsets = nullptr;
  const int* in_column_indices = nullptr;
  const float* in_weights = nullptr;
};

struct device_graph_storage_t {
  int* row_offsets = nullptr;
  int* column_indices = nullptr;
  float* weights = nullptr;
  int* in_row_offsets = nullptr;
  int* in_column_indices = nullptr;
  float* in_weights = nullptr;
  device_graph_t view{};
};

struct frontier_host_t {
  int Q = 0;
  std::vector<mask_t> frontier_mask;
  std::vector<int> frontier_vertices;
  std::size_t sum_per_query_frontier_size = 0;
  std::size_t active_query_pairs = 0;
  double real_overlap = 0.0;
  double avg_popcount_per_active_vertex = 0.0;
};

struct device_state_t {
  mask_t* frontier_mask = nullptr;
  float* spmm_input = nullptr;
  float* spmm_output = nullptr;
  float* spmm_pull_input = nullptr;
  float* spmm_pull_output = nullptr;
  int* frontier_vertices = nullptr;
};

struct bench_stats_t {
  unsigned long long vertices_scanned = 0;
  unsigned long long edges_scanned = 0;
  unsigned long long active_query_pairs = 0;
  unsigned long long src_value_loads = 0;
  unsigned long long dst_value_loads = 0;
  unsigned long long weight_loads = 0;
  unsigned long long relax_ops = 0;
  unsigned long long atomic_ops = 0;
  unsigned long long successful_updates = 0;
};

struct bench_result_t {
  float push_ms = 0.0f;
  float pull_ms = 0.0f;
  float total_ms = 0.0f;
  bench_stats_t stats{};
};

inline const char* to_string(gen_mode_t mode) {
  switch (mode) {
    case gen_mode_t::density:
      return "density";
    case gen_mode_t::overlap:
      return "overlap";
    case gen_mode_t::mixed:
      return "mixed";
  }
  return "unknown";
}

inline const char* to_string(exec_mode_t mode) {
  switch (mode) {
    case exec_mode_t::all_push:
      return "all_push";
    case exec_mode_t::all_pull:
      return "all_pull";
    case exec_mode_t::mixed_serial:
      return "mixed_serial";
    case exec_mode_t::mixed_concurrent:
      return "mixed_concurrent";
  }
  return "unknown";
}

inline const char* to_string(update_mode_t mode) {
  switch (mode) {
    case update_mode_t::spmm_sum:
      return "spmm_sum";
  }
  return "unknown";
}

}  // namespace ppbench
