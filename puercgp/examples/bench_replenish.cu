/*
 * bench_replenish.cu
 * replenishment latency 基准：对比 sequential 分批 vs replenish 的 per-query latency 分布
 *
 * 用法：
 *   bench_replenish <graph> [--total=200] [--bfs=100] [--sssp=100] [--seed=42]
 *                   [--batch-size=32] [--discard-results] [--repeats=5] [--warmup=1]
 *                   [--mode=push] [--push=warp]
 *
 * latency 定义：query 结果计算完成的时刻（相对各自计时起点）
 *   sequential: 批次累积 wall（含每批 init，run_heterogeneous 每批重新分配 buffer 的悲观估计）
 *   replenish:  completion_wall（不含 init，从第一轮 push 起算）
 * source 固定 seed 随机生成，两组共用同一组（公平）
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

#include "mtx_loader.hxx"

using puercgp::algorithms::algo_kind_t;
using puercgp::execution_context;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::run_heterogeneous;
using puercgp::run_options;
using puercgp::run_replenish_pipeline;
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
inline float percentilef(std::vector<float> v, float p) {
  if (v.empty()) return 0.0f;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1) + 0.5f)];
}
inline float minf(std::vector<float> v) {
  if (v.empty()) return 0.0f;
  return *std::min_element(v.begin(), v.end());
}
inline float maxf(std::vector<float> v) {
  if (v.empty()) return 0.0f;
  return *std::max_element(v.begin(), v.end());
}
inline float meanf(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  float s = 0.0f;
  for (float x : v) s += x;
  return s / static_cast<float>(v.size());
}

// 一次 sequential 分批跑：返回 per-query latency（累积 wall）+ 总 wall
static std::pair<std::vector<float>, float> run_sequential(
    const puercgp_examples::host_csr_graph& graph, int N,
    const std::vector<query_descriptor_t>& all_descs, int batch_size,
    execution_context& ctx, const run_options& opt) {
  auto graph_view = graph.view();
  std::vector<float> latency(static_cast<std::size_t>(N), 0.0f);
  float accum = 0.0f;
  for (int start = 0; start < N; start += batch_size) {
    int end = std::min(start + batch_size, N);
    std::vector<query_descriptor_t> batch(all_descs.begin() + start,
                                          all_descs.begin() + end);
    hybrid_query_batch b(batch);
    b.validate();
    auto r = run_heterogeneous(graph_view, b, ctx, opt);
    accum += r.wall_time_ms;
    for (int i = start; i < end; ++i)
      latency[static_cast<std::size_t>(i)] = accum;
  }
  return {latency, accum};
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <graph> [--bfs=100] [--sssp=96] [--wcc=0] [--seed=42]"
              << " [--chain-length=0] [--batch-size=32] [--discard-results]"
              << " [--repeats=5] [--warmup=1] [--mode=push] [--push=warp]\n";
    return 2;
  }
  const std::string matrix = argv[1];
  int bfs_count = 100, sssp_count = 96, wcc_count = 0, seed = 42;
  int chain_length = 0, batch_size = 32, repeats = 5, warmup = 1;
  std::string mode_text = "push", push_text = "warp";
  bool discard = true;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto getv = [&](const std::string& pre) -> int {
      return std::atoi(a.c_str() + pre.size());
    };
    if (a.rfind("--bfs=", 0) == 0) bfs_count = getv("--bfs=");
    else if (a.rfind("--sssp=", 0) == 0) sssp_count = getv("--sssp=");
    else if (a.rfind("--wcc=", 0) == 0) wcc_count = getv("--wcc=");
    else if (a.rfind("--seed=", 0) == 0) seed = getv("--seed=");
    else if (a.rfind("--chain-length=", 0) == 0) chain_length = getv("--chain-length=");
    else if (a.rfind("--batch-size=", 0) == 0) batch_size = getv("--batch-size=");
    else if (a.rfind("--repeats=", 0) == 0) repeats = getv("--repeats=");
    else if (a.rfind("--warmup=", 0) == 0) warmup = getv("--warmup=");
    else if (a.rfind("--mode=", 0) == 0) mode_text = a.substr(7);
    else if (a.rfind("--push=", 0) == 0) push_text = a.substr(7);
    else if (a == "--discard-results") discard = true;
    else if (a == "--no-discard") discard = false;
  }
  const int total = bfs_count + sssp_count + wcc_count;

  const bool build_pull_adjacency = mode_text != "push";
  auto graph =
      puercgp_examples::load_graph_auto(matrix, build_pull_adjacency);
  const int V_main = graph.vertices;  // 主图顶点数（attach_chain 前）
  if (chain_length > 1) {
    puercgp_examples::attach_chain(graph, chain_length, build_pull_adjacency);
  }
  const int V = graph.vertices;  // attach 后（主图 + 长链）
  auto graph_view = graph.view();

  // 固定 seed 随机生成 source（两组共用同一组）
  // BFS/SSSP source 限制在主图 [0,V_main)（不进入长链，收敛快）
  // WCC 覆盖全图（含长链，label 沿链逐跳传播 → O(L) 轮长尾）
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::vector<query_descriptor_t> all_descs;
  all_descs.reserve(static_cast<std::size_t>(total));
  for (int i = 0; i < bfs_count; ++i)
    all_descs.push_back({static_cast<int>(rng() % V_main), algo_kind_t::bfs, 0.0f});
  for (int i = 0; i < sssp_count; ++i)
    all_descs.push_back({static_cast<int>(rng() % V_main), algo_kind_t::sssp, 0.0f});
  for (int i = 0; i < wcc_count; ++i)
    all_descs.push_back({0, algo_kind_t::wcc, 0.0f});

  std::cout << "graph=" << matrix << " V_main=" << V_main << " V=" << V
            << " E=" << graph.edges << " chain_length=" << chain_length << "\n";
  std::cout << "total=" << total << " (bfs=" << bfs_count << " sssp=" << sssp_count
            << " wcc=" << wcc_count << ") batch_size=" << batch_size
            << " seed=" << seed << " discard=" << (discard ? "on" : "off")
            << " mode=" << mode_text << " push=" << push_text << "\n";
  std::cout << "sequential batches: " << (total + batch_size - 1) / batch_size
            << "\n\n";

  execution_context ctx;
  run_options opt;
  opt.max_iterations = (chain_length > 1) ? (chain_length + 200) : 1000;
  opt.traversal_mode = parse_mode(mode_text);
  opt.push_strategy = parse_push(push_text);
  opt.profile_iterations = false;
  run_options opt_repl = opt;
  opt_repl.enable_replenishment = true;
  opt_repl.discard_results = discard;

  // warmup
  for (int w = 0; w < warmup; ++w) {
    (void)run_sequential(graph, total, all_descs, batch_size, ctx, opt);
    (void)run_replenish_pipeline(graph_view, all_descs, ctx, opt_repl);
  }

  // repeats：记录每次 wall + 最后一次 latency 分布 + WCC 收敛轮次
  std::vector<float> seq_walls, repl_walls;
  std::vector<float> last_seq_lat, last_repl_lat;
  int last_repl_iterations = 0;
  std::vector<int> last_wcc_levels;
  for (int r = 0; r < repeats; ++r) {
    auto [slat, sw] = run_sequential(graph, total, all_descs, batch_size, ctx, opt);
    seq_walls.push_back(sw);
    last_seq_lat = std::move(slat);
    auto rr = run_replenish_pipeline(graph_view, all_descs, ctx, opt_repl);
    repl_walls.push_back(rr.wall_time_ms);
    last_repl_iterations = rr.iterations;
    last_repl_lat.clear();
    for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i)
      last_repl_lat.push_back(rr.queries[i].completion_wall_time_ms);
    last_wcc_levels.clear();
    for (int i = 0; i < wcc_count; ++i) {
      std::size_t idx = static_cast<std::size_t>(bfs_count + sssp_count + i);
      last_wcc_levels.push_back(rr.queries[idx].completion_level);
    }
  }

  // ===== Throughput =====
  const float seq_tw = medianf(seq_walls), repl_tw = medianf(repl_walls);
  std::cout << "=== Throughput (median over " << repeats << ") ===\n";
  std::cout << "sequential_total_ms=" << seq_tw << " replenish_total_ms=" << repl_tw
            << " speedup=" << (repl_tw > 0 ? seq_tw / repl_tw : 0.0f) << "x\n\n";

  // ===== Latency distribution（最后一次 run）=====
  std::cout << "=== Latency distribution (ms, last run) ===\n";
  std::cout << "group,    min,    p25,  median,    p75,    p90,    p99,    max,   mean\n";
  auto row = [&](const char* name, const std::vector<float>& v) {
    std::cout << name << ","
              << minf(v) << "," << percentilef(v, 0.25f) << ","
              << percentilef(v, 0.50f) << "," << percentilef(v, 0.75f) << ","
              << percentilef(v, 0.90f) << "," << percentilef(v, 0.99f) << ","
              << maxf(v) << "," << meanf(v) << "\n";
  };
  row("sequential", last_seq_lat);
  row("replenish ", last_repl_lat);
  std::cout << "\n注：sequential latency 含每批 init（每批重新分配 buffer 的悲观估计）；"
            << "replenish latency 不含 init（从第一轮 push 起算）\n";
  if (wcc_count > 0) {
    std::cout << "\n=== WCC convergence levels (long-tail confirmation) ===\n";
    std::cout << "replenish_total_iterations=" << last_repl_iterations
              << " (WCC 主导，expect ≈ chain_length+主图轮次)\n";
    for (int i = 0; i < wcc_count; ++i) {
      std::cout << "  wcc_q" << i << ": completion_level=" << last_wcc_levels[i]
                << " (chain_length=" << chain_length << ")\n";
    }
  }
  return 0;
}
