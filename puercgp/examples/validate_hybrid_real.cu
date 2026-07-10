/*
 * validate_hybrid_real.cu
 * 在真实大图上验证 run_heterogeneous 的 BFS+SSSP+WCC 混合 batch 正确性。
 *
 * 用法：
 *   validate_hybrid_real <graph> <bfs_srcs_csv> <sssp_srcs_csv> <wcc_count>
 *                        [repeats] [max_iters]
 *
 *   graph       : CSR 目录（含 csr_vlist.bin + csr_elist.bin [+] csr_weightlist.bin）
 *                 或 Matrix Market .mtx 文件
 *   bfs_srcs    : 逗号分隔的 BFS 源点，如 0,997,1994
 *   sssp_srcs   : 逗号分隔的 SSSP 源点（可与 bfs_srcs 不同）
 *   wcc_count   : WCC query 数（每个 WCC query 是全顶点 label propagation，
 *                 source 填 0 不使用）
 *   repeats     : 重复次数，默认 7（取中位数）
 *   max_iters   : 主循环最大迭代轮次（WCC 收敛兜底），默认 1000
 *
 * 示例：
 *   ./build/validate_hybrid_real /home/zyl/data/csr_data/cit-Patents \
 *       0,100,1000 0,1,2 1 7 1000
 *
 * 比对策略：
 *   BFS slot  : 取整比对，INT_MAX 与 +inf 双向视为 INF（等价 cpu_bfs 结果）
 *   SSSP slot : 浮点 epsilon = 1e-3f（unified float 在大图路径累加误差放宽）
 *   WCC slot  : 直接 == 比对（GPU 与 CPU 都是 min-reduce，收敛值 = CC 内 min vertex id）
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

using puercgp::algorithms::algo_kind_t;
using puercgp::algorithms::unified_infinity;
using puercgp::algorithms::unified_value_t;
using puercgp::csr_graph_view;
using puercgp::execution_context;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::run_heterogeneous;
using puercgp::run_options;
using puercgp::traversal_mode_t;

// 视 INT_MAX 与 +inf 均为"不可达"，返回是否双方都不可达
inline bool both_unreachable_int(float gpu, int cpu) {
  const int int_inf = std::numeric_limits<int>::max();
  return std::isinf(gpu) && cpu == int_inf;
}

inline puercgp::traversal_mode_t parse_traversal_mode(const std::string& value) {
  if (value == "push") return puercgp::traversal_mode_t::push;
  if (value == "pull") return puercgp::traversal_mode_t::pull;
  if (value == "hybrid") return puercgp::traversal_mode_t::hybrid;
  throw std::invalid_argument("traversal_mode must be push, pull, or hybrid");
}

inline puercgp::push_strategy_t parse_push_strategy(const std::string& value) {
  if (value == "shared_node") return puercgp::push_strategy_t::shared_node;
  if (value == "shared_node_warp") return puercgp::push_strategy_t::shared_node_warp;
  throw std::invalid_argument(
      "push_strategy must be shared_node or shared_node_warp");
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <graph> <bfs_srcs_csv> <sssp_srcs_csv> <wcc_count>"
              << " [repeats] [max_iters] [mode=push|pull|hybrid]"
              << " [push_strategy=shared_node|shared_node_warp]\n";
    return 2;
  }

  const std::string matrix = argv[1];
  const std::string bfs_text = argv[2];
  const std::string sssp_text = argv[3];
  const int wcc_count = std::max(0, std::stoi(argv[4]));
  const int repeats = (argc > 5) ? std::max(1, std::stoi(argv[5])) : 7;
  const int max_iters = (argc > 6) ? std::max(1, std::stoi(argv[6])) : 1000;
  const std::string mode_text = (argc > 7) ? argv[7] : "push";
  const std::string push_text = (argc > 8) ? argv[8] : "shared_node";

  // 加载图（自动识别 CSR 目录 vs mtx）
  const bool build_pull_adjacency = mode_text != "push";
  auto graph =
      puercgp_examples::load_graph_auto(matrix, build_pull_adjacency);
  auto graph_view = graph.view();
  const int V = graph.vertices;
  const long long E = static_cast<long long>(graph.edges);
  std::cout << "graph=" << matrix << "\n";
  std::cout << "vertices=" << V << " edges=" << E << "\n";

  // 解析 sources
  auto bfs_srcs = puercgp_examples::parse_sources(bfs_text);
  auto sssp_srcs = puercgp_examples::parse_sources(sssp_text);
  if (bfs_srcs.empty() && sssp_srcs.empty() && wcc_count == 0) {
    std::cerr << "validate_hybrid_real requires at least one query\n";
    return 2;
  }
  const int Q_total =
      static_cast<int>(bfs_srcs.size() + sssp_srcs.size()) + wcc_count;
  if (Q_total > 64) {
    std::cerr << "validate_hybrid_real: total queries (" << Q_total
              << ") exceed 64 (query_mask_t hard limit)\n";
    return 2;
  }

  // 构造 hybrid_query_batch：BFS slot + SSSP slot + WCC slot 顺序排列
  std::vector<query_descriptor_t> descs;
  descs.reserve(static_cast<std::size_t>(Q_total));
  for (int s : bfs_srcs) {
    query_descriptor_t d;
    d.source = s;
    d.kind = algo_kind_t::bfs;
    d.source_value = unified_value_t(0);
    descs.push_back(d);
  }
  for (int s : sssp_srcs) {
    query_descriptor_t d;
    d.source = s;
    d.kind = algo_kind_t::sssp;
    d.source_value = unified_value_t(0);
    descs.push_back(d);
  }
  for (int i = 0; i < wcc_count; ++i) {
    query_descriptor_t d;
    d.source = 0;  // WCC 不使用 source（per-vertex init）
    d.kind = algo_kind_t::wcc;
    d.source_value = unified_value_t(0);
    descs.push_back(d);
  }
  hybrid_query_batch batch(descs);
  batch.validate();

  const int Q = static_cast<int>(batch.size());
  std::cout << "queries: bfs=" << bfs_srcs.size() << " sssp=" << sssp_srcs.size()
            << " wcc=" << wcc_count << " total=" << Q
            << " repeats=" << repeats << " max_iters=" << max_iters
            << " mode=" << mode_text << " push=" << push_text << "\n";

  // run options：mode 由 CLI 指定（push/pull/hybrid）
  // 注意：pull/hybrid 模式要求 CSR 双向（对称）。mtx symmetric 已自动双向；
  // CSR bin 假定已双向存储（如 cit-Patents 默认 CSR 含双向边）。
  execution_context context;
  run_options options;
  options.max_iterations = max_iters;
  options.traversal_mode = parse_traversal_mode(mode_text);
  options.push_strategy = parse_push_strategy(push_text);
  options.profile_iterations = false;

  // ===== CPU references =====
  // 大图上 cpu_bfs/cpu_sssp/cpu_wcc 较慢，仅在 V <= 阈值时跑（cit-Patents 3.77M 可接受）
  const bool run_cpu_check = V <= 5'000'000;
  std::cout << "cpu_reference_check=" << (run_cpu_check ? "yes" : "no") << "\n";

  std::vector<std::vector<int>> bfs_refs;
  std::vector<std::vector<float>> sssp_refs;
  std::vector<float> wcc_ref;
  if (run_cpu_check) {
    bfs_refs.reserve(bfs_srcs.size());
    for (int s : bfs_srcs) {
      std::cout << "computing cpu_bfs from source " << s << "...\n";
      bfs_refs.push_back(puercgp_examples::cpu_bfs(graph, s));
    }
    sssp_refs.reserve(sssp_srcs.size());
    for (int s : sssp_srcs) {
      std::cout << "computing cpu_sssp from source " << s << "...\n";
      sssp_refs.push_back(puercgp_examples::cpu_sssp(graph, s));
    }
    if (wcc_count > 0) {
      std::cout << "computing cpu_wcc (label propagation)...\n";
      wcc_ref = puercgp_examples::cpu_wcc(graph, max_iters);
    }
  }

  // ===== Warmup =====
  std::cout << "warmup...\n";
  {
    auto warmup = run_heterogeneous(graph_view, batch, context, options);
    (void)warmup;
  }

  // ===== Repeats =====
  std::vector<float> wall_times;
  std::vector<float> gpu_times;
  wall_times.reserve(static_cast<std::size_t>(repeats));
  gpu_times.reserve(static_cast<std::size_t>(repeats));

  std::size_t bfs_mismatches = 0;
  std::size_t sssp_mismatches = 0;
  std::size_t wcc_mismatches = 0;
  float max_sssp_abs_delta = 0.0f;
  int last_iterations = 0;

  for (int r = 0; r < repeats; ++r) {
    auto result = run_heterogeneous(graph_view, batch, context, options);
    wall_times.push_back(result.wall_time_ms);
    gpu_times.push_back(result.gpu_time_ms);
    last_iterations = result.iterations;

    // 仅末轮比对
    if (r == repeats - 1 && run_cpu_check) {
      std::vector<unified_value_t> h_values(result.values.size());
      thrust::copy(result.values.begin(), result.values.end(),
                   h_values.begin());

      const float sssp_eps = 1e-3f;
      // BFS slot 比对（slot 索引按 descs 顺序）
      for (std::size_t qi = 0; qi < bfs_srcs.size(); ++qi) {
        const auto& ref = bfs_refs[qi];
        for (int v = 0; v < V; ++v) {
          auto idx = static_cast<std::size_t>(v) * static_cast<std::size_t>(Q) +
                     qi;
          float gpu = h_values[idx];
          int cpu = ref[static_cast<std::size_t>(v)];
          if (both_unreachable_int(gpu, cpu)) continue;
          // GPU BFS slot 是 float(level)；cpu 是 int level。取整比对
          int gpu_int = std::isinf(gpu) ? std::numeric_limits<int>::max()
                                        : static_cast<int>(gpu + 0.5f);
          if (gpu_int != cpu) ++bfs_mismatches;
        }
      }
      // SSSP slot 比对
      const std::size_t sssp_base = bfs_srcs.size();
      for (std::size_t qi = 0; qi < sssp_srcs.size(); ++qi) {
        const auto& ref = sssp_refs[qi];
        for (int v = 0; v < V; ++v) {
          auto idx = static_cast<std::size_t>(v) * static_cast<std::size_t>(Q) +
                     (sssp_base + qi);
          float gpu = h_values[idx];
          float cpu = ref[static_cast<std::size_t>(v)];
          bool both_inf = std::isinf(gpu) && std::isinf(cpu);
          float delta = both_inf ? 0.0f : std::fabs(gpu - cpu);
          max_sssp_abs_delta = std::max(max_sssp_abs_delta, delta);
          if (!both_inf && delta > sssp_eps) ++sssp_mismatches;
        }
      }
      // WCC slot 比对（所有 WCC query 期望值相同）
      if (wcc_count > 0) {
        const std::size_t wcc_base = bfs_srcs.size() + sssp_srcs.size();
        for (int qi = 0; qi < wcc_count; ++qi) {
          for (int v = 0; v < V; ++v) {
            auto idx =
                static_cast<std::size_t>(v) * static_cast<std::size_t>(Q) +
                (wcc_base + static_cast<std::size_t>(qi));
            float gpu = h_values[idx];
            float cpu = wcc_ref[static_cast<std::size_t>(v)];
            if (gpu != cpu) ++wcc_mismatches;
          }
        }
      }
    }
  }

  // ===== Report =====
  std::cout << "\n=== validate_hybrid_real results ===\n";
  std::cout << "iterations=" << last_iterations << "\n";
  std::cout << "bfs_mismatches=" << bfs_mismatches << "\n";
  std::cout << "sssp_mismatches=" << sssp_mismatches
            << " max_abs_delta=" << max_sssp_abs_delta << "\n";
  std::cout << "wcc_mismatches=" << wcc_mismatches << "\n";
  std::cout << "wall_ms_median=" << puercgp_examples::median(wall_times) << "\n";
  std::cout << "gpu_ms_median=" << puercgp_examples::median(gpu_times) << "\n";

  const std::size_t total_mismatches =
      bfs_mismatches + sssp_mismatches + wcc_mismatches;
  std::cout << "\nvalidate_hybrid_real: "
            << (total_mismatches == 0 ? "ALL PASS" : "FAIL") << "\n";
  return total_mismatches == 0 ? 0 : 1;
}
