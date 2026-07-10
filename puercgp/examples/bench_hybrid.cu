/*
 * bench_hybrid.cu
 * 性能对比：异构 hybrid batch vs 同质 sequential vs 理想上界。
 * 服务 ICDE 论文 Experiment 4（异构 query 分析）。
 *
 * 用法：
 *   bench_hybrid <graph> <bfs_srcs_csv> <sssp_srcs_csv> <wcc_count>
 *                [--repeats=7] [--warmup=2] [--mode=hybrid|push|pull]
 *
 * 三组对比：
 *   sequential  : run<bfs_policy> + run<sssp_policy> + run<wcc_policy> 串行
 *   hybrid      : run_heterogeneous 一次混合 batch
 *   ideal_upper : N× 同质 BFS（理论共享遍历上界）
 *
 * 输出 CSV 格式，可直接喂 matplotlib。
 */
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

using puercgp::algorithms::algo_kind_t;
using puercgp::algorithms::bfs_policy;
using puercgp::algorithms::sssp_policy;
using puercgp::algorithms::wcc_policy;
using puercgp::execution_context;
using puercgp::hybrid_query_batch;
using puercgp::query_batch;
using puercgp::query_descriptor_t;
using puercgp::run;
using puercgp::run_heterogeneous;
using puercgp::run_options;
using puercgp::traversal_mode_t;

inline traversal_mode_t parse_mode(const std::string& v) {
  if (v == "push") return traversal_mode_t::push;
  if (v == "pull") return traversal_mode_t::pull;
  if (v == "hybrid") return traversal_mode_t::hybrid;
  throw std::invalid_argument("mode must be push/pull/hybrid");
}

inline puercgp::push_strategy_t parse_push(const std::string& v) {
  if (v == "block" || v == "shared_node")
    return puercgp::push_strategy_t::shared_node;
  if (v == "warp" || v == "shared_node_warp")
    return puercgp::push_strategy_t::shared_node_warp;
  throw std::invalid_argument("push must be shared_node/shared_node_warp");
}

inline float medianf(std::vector<float> v) {
  if (v.empty()) return 0.0f;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

inline int mediani(std::vector<int> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <graph> <bfs_srcs_csv> <sssp_srcs_csv> <wcc_count>"
              << " [--repeats=7] [--warmup=2] [--mode=hybrid]"
              << " [--push=block|warp]\n";
    return 2;
  }

  const std::string matrix = argv[1];
  auto bfs_srcs = puercgp_examples::parse_sources(argv[2]);
  auto sssp_srcs = puercgp_examples::parse_sources(argv[3]);
  const int wcc_count = std::max(0, std::stoi(argv[4]));

  int repeats = 7;
  int warmup = 2;
  std::string mode_text = "hybrid";
  std::string push_text = "block";
  bool run_ideal = true;
  for (int i = 5; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--repeats=", 0) == 0)
      repeats = std::max(1, std::stoi(arg.substr(10)));
    else if (arg.rfind("--warmup=", 0) == 0)
      warmup = std::max(0, std::stoi(arg.substr(9)));
    else if (arg.rfind("--mode=", 0) == 0)
      mode_text = arg.substr(7);
    else if (arg.rfind("--push=", 0) == 0)
      push_text = arg.substr(7);
    else if (arg == "--no-ideal")
      run_ideal = false;
  }

  const bool build_pull_adjacency = mode_text != "push";
  auto graph =
      puercgp_examples::load_graph_auto(matrix, build_pull_adjacency);
  auto graph_view = graph.view();
  const int total_q = static_cast<int>(bfs_srcs.size() + sssp_srcs.size()) +
                      wcc_count;
  if (total_q > 64) {
    std::cerr << "total queries " << total_q << " > 64\n";
    return 2;
  }

  std::cout << "graph=" << matrix << "\n";
  std::cout << "vertices=" << graph.vertices << " edges=" << graph.edges << "\n";
  std::cout << "queries: bfs=" << bfs_srcs.size()
            << " sssp=" << sssp_srcs.size() << " wcc=" << wcc_count
            << " total=" << total_q << "\n";
  std::cout << "repeats=" << repeats << " warmup=" << warmup
            << " mode=" << mode_text << " push=" << push_text << "\n\n";

  execution_context ctx;
  // 内存安全的对比设置：
  //   - sequential / ideal_upper（同质引擎）统一用 shared_node_warp
  //     （历史 list 路径已删除，避免 SSSP 在密图/大图上 OOM）
  //   - hybrid（异构引擎）用 CLI 指定的 push（block/warp）
  //   - 所有对比路径均使用 shared_node 系列，避免 edge-level list 分配
  run_options opt_seq;
  opt_seq.max_iterations = 1000;
  opt_seq.traversal_mode = parse_mode(mode_text);
  opt_seq.push_strategy = puercgp::push_strategy_t::shared_node_warp;
  opt_seq.profile_iterations = false;

  run_options opt_hyb;
  opt_hyb.max_iterations = 1000;
  opt_hyb.traversal_mode = parse_mode(mode_text);
  opt_hyb.push_strategy = parse_push(push_text);
  opt_hyb.profile_iterations = false;

  // ===== Group 1: Sequential（N 个同质 batch 串行）=====
  auto run_sequential_once = [&]() {
    float wall_sum = 0.0f, gpu_sum = 0.0f;
    int iter_sum = 0;
    if (!bfs_srcs.empty()) {
      query_batch<int> bq(bfs_srcs);
      auto r = run<bfs_policy>(graph_view, bq, ctx, opt_seq);
      wall_sum += r.wall_time_ms;
      gpu_sum += r.gpu_time_ms;
      iter_sum += r.iterations;
    }
    if (!sssp_srcs.empty()) {
      query_batch<int> sq(sssp_srcs);
      auto r = run<sssp_policy>(graph_view, sq, ctx, opt_seq);
      wall_sum += r.wall_time_ms;
      gpu_sum += r.gpu_time_ms;
      iter_sum += r.iterations;
    }
    if (wcc_count > 0) {
      std::vector<int> wq_srcs(wcc_count, 0);
      query_batch<int> wq(wq_srcs);
      auto r = run<wcc_policy>(graph_view, wq, ctx, opt_seq);
      wall_sum += r.wall_time_ms;
      gpu_sum += r.gpu_time_ms;
      iter_sum += r.iterations;
    }
    return std::make_tuple(wall_sum, gpu_sum, iter_sum);
  };

  for (int w = 0; w < warmup; ++w) (void)run_sequential_once();
  std::vector<float> seq_walls, seq_gpus;
  std::vector<int> seq_iters;
  for (int r = 0; r < repeats; ++r) {
    auto [wall, gpu, iter] = run_sequential_once();
    seq_walls.push_back(wall);
    seq_gpus.push_back(gpu);
    seq_iters.push_back(iter);
  }

  // ===== Group 2: Hybrid（1 次混合 batch）=====
  std::vector<query_descriptor_t> descs;
  descs.reserve(static_cast<std::size_t>(total_q));
  for (int s : bfs_srcs)
    descs.push_back({s, algo_kind_t::bfs, 0.0f});
  for (int s : sssp_srcs)
    descs.push_back({s, algo_kind_t::sssp, 0.0f});
  for (int i = 0; i < wcc_count; ++i)
    descs.push_back({0, algo_kind_t::wcc, 0.0f});
  hybrid_query_batch batch(descs);
  batch.validate();

  std::vector<float> hyb_walls, hyb_gpus;
  std::vector<int> hyb_iters;
  for (int w = 0; w < warmup; ++w) (void)run_heterogeneous(graph_view, batch, ctx, opt_hyb);
  for (int r = 0; r < repeats; ++r) {
    auto res = run_heterogeneous(graph_view, batch, ctx, opt_hyb);
    hyb_walls.push_back(res.wall_time_ms);
    hyb_gpus.push_back(res.gpu_time_ms);
    hyb_iters.push_back(res.iterations);
  }

  // ===== Group 3: Ideal upper bound（N× 同质 BFS，可选跳过避免 OOM）=====
  std::vector<float> ub_walls, ub_gpus;
  std::vector<int> ub_iters;
  if (run_ideal) {
    std::vector<int> ub_srcs;
    for (int s : bfs_srcs) ub_srcs.push_back(s);
    for (int s : sssp_srcs) ub_srcs.push_back(s);
    while (static_cast<int>(ub_srcs.size()) < total_q) ub_srcs.push_back(0);
    if (static_cast<int>(ub_srcs.size()) > 64) ub_srcs.resize(64);
    query_batch<int> ubq(ub_srcs);

    for (int w = 0; w < warmup; ++w) (void)run<bfs_policy>(graph_view, ubq, ctx, opt_seq);
    for (int r = 0; r < repeats; ++r) {
      auto res = run<bfs_policy>(graph_view, ubq, ctx, opt_seq);
      ub_walls.push_back(res.wall_time_ms);
      ub_gpus.push_back(res.gpu_time_ms);
      ub_iters.push_back(res.iterations);
    }
  }

  // ===== Report =====
  const float seq_w = medianf(seq_walls), seq_g = medianf(seq_gpus);
  const int seq_i = mediani(seq_iters);
  const float hyb_w = medianf(hyb_walls), hyb_g = medianf(hyb_gpus);
  const int hyb_i = mediani(hyb_iters);

  std::cout << "=== Results (median over " << repeats << ") ===\n";
  std::cout << "group,wall_ms,gpu_ms,iterations,speedup_vs_seq\n";
  std::cout << "sequential," << seq_w << "," << seq_g << "," << seq_i << ",1.00x\n";
  std::cout << "hybrid," << hyb_w << "," << hyb_g << "," << hyb_i << ","
            << (hyb_w > 0 ? seq_w / hyb_w : 0.0f) << "x\n";
  if (run_ideal) {
    const float ub_w_m = medianf(ub_walls), ub_g_m = medianf(ub_gpus);
    const int ub_i_m = mediani(ub_iters);
    std::cout << "ideal_upper," << ub_w_m << "," << ub_g_m << "," << ub_i_m
              << "," << (ub_w_m > 0 ? seq_w / ub_w_m : 0.0f) << "x\n";
  }

  std::cout << "\n=== Verdict ===\n";
  if (hyb_w < seq_w) {
    std::cout << "GO: hybrid is " << (seq_w / hyb_w) << "x faster than sequential\n";
  } else {
    std::cout << "NO-GO: hybrid is " << (hyb_w / seq_w)
              << "x slower than sequential (consider P3 warp optimization)\n";
  }
  return 0;
}
