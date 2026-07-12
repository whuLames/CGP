/*
 * validate_replenish.cu
 * 阶段4 端到端验证：replenish_frontier_engine slot 复用主循环
 *
 * toy graph: 0->1(w=2), 0->2(w=5), 1->3(w=1)（单向 CSR，push 模式）
 * N=5 query, Q=2 slot：覆盖两次 slot 复用和最终 Q=1 尾批压缩。
 *
 * 期望（pipeline 结果 row-major values[q*V+v]）：
 *   q0 BFS  src=0: [0,1,1,2]
 *   q1 SSSP src=0: [0,2,5,3]
 *   q2 BFS  src=1: [INF,0,INF,1]   (1->3)
 *   q3 SSSP src=1: [INF,0,INF,1]   (1->3, w=1)
 *
 * 验证：5 个 query 结果正确 + 全部 completed + slot 复用确实发生。
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

using puercgp::algorithms::algo_kind_t;
using puercgp::algorithms::unified_infinity;
using puercgp::algorithms::unified_value_t;
using puercgp::csr_graph_view;
using puercgp::execution_context;
using puercgp::query_descriptor_t;
using puercgp::query_mask_t;
using puercgp::run_heterogeneous;
using puercgp::hybrid_query_batch;
using puercgp::run_options;
using puercgp::run_replenish_pipeline;
using puercgp::traversal_mode_t;

static int failures = 0;
static void check(const char* name, bool cond) {
  printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) ++failures;
}

int main() {
  const int V = 4, E = 3;
  std::vector<int> h_row = {0, 2, 3, 3, 3};
  std::vector<int> h_col = {1, 2, 3};
  std::vector<float> h_w = {2.0f, 5.0f, 3.0f};  // 1->3 w=3，使 BFS1(=1)/SSSP1(=3) 可分辨
  thrust::device_vector<int> d_row(h_row);
  thrust::device_vector<int> d_col(h_col);
  thrust::device_vector<float> d_w(h_w);

  csr_graph_view<int, int, float> graph;
  graph.number_of_vertices = V;
  graph.number_of_edges = E;
  graph.row_offsets = thrust::raw_pointer_cast(d_row.data());
  graph.column_indices = thrust::raw_pointer_cast(d_col.data());
  graph.edge_weights = thrust::raw_pointer_cast(d_w.data());

  // N=4 query，Q=2：前 2 进 slot，后 2 进 pending
  std::vector<query_descriptor_t> all_queries;
  all_queries.push_back({0, algo_kind_t::bfs, unified_value_t(0)});
  all_queries.push_back({0, algo_kind_t::sssp, unified_value_t(0)});
  all_queries.push_back({1, algo_kind_t::bfs, unified_value_t(0)});
  all_queries.push_back({1, algo_kind_t::sssp, unified_value_t(0)});
  all_queries.push_back({2, algo_kind_t::bfs, unified_value_t(0)});

  execution_context context;
  run_options options;
  options.enable_replenishment = true;
  options.traversal_mode = traversal_mode_t::push;  // toy 单向 CSR
  options.max_iterations = 100;
  options.max_queries = 2;

  auto result = run_replenish_pipeline(graph, all_queries, context, options);

  const auto INF = unified_infinity();
  printf("replenish pipeline: iterations=%d, N=%d, V=%d\n",
         result.iterations, static_cast<int>(all_queries.size()), V);
  check("effective_query_dim == N (5)", result.effective_query_dim == 5);
  check("queries.size() == 5",
        result.queries.size() == 5);
  check("values.size() == N*V (20)", result.values.size() == 20);
  // slot 复用由 q2/q3 结果正确证明（它们必须经 pending 注入），iterations 值依赖图结构
  check("iterations >= 3 (BFS0/SSSP0 convergence rounds)", result.iterations >= 3);

  std::vector<unified_value_t> h_values(result.values.size());
  thrust::copy(result.values.begin(), result.values.end(), h_values.begin());

  const int N = 5;
  printf("q0 BFS src=0 (expect [0,1,1,2]):\n");
  check("values[0*V+0]=0", h_values[0 * V + 0] == 0.0f);
  check("values[0*V+1]=1", h_values[0 * V + 1] == 1.0f);
  check("values[0*V+2]=1", h_values[0 * V + 2] == 1.0f);
  check("values[0*V+3]=2", h_values[0 * V + 3] == 2.0f);

  printf("q1 SSSP src=0 (expect [0,2,5,5], 0->1->3 w2+w3):\n");
  check("values[1*V+0]=0", h_values[1 * V + 0] == 0.0f);
  check("values[1*V+1]=2", h_values[1 * V + 1] == 2.0f);
  check("values[1*V+2]=5", h_values[1 * V + 2] == 5.0f);
  check("values[1*V+3]=5 (0->1->3)", h_values[1 * V + 3] == 5.0f);

  printf("q2 BFS src=1 (expect [INF,0,INF,1]):\n");
  check("values[2*V+0]=INF", h_values[2 * V + 0] == INF);
  check("values[2*V+1]=0", h_values[2 * V + 1] == 0.0f);
  check("values[2*V+2]=INF", h_values[2 * V + 2] == INF);
  check("values[2*V+3]=1", h_values[2 * V + 3] == 1.0f);

  printf("q3 SSSP src=1 (expect [INF,0,INF,3], w=3):\n");
  check("values[3*V+0]=INF", h_values[3 * V + 0] == INF);
  check("values[3*V+1]=0", h_values[3 * V + 1] == 0.0f);
  check("values[3*V+2]=INF", h_values[3 * V + 2] == INF);
  check("values[3*V+3]=3 (1->3 w3)", h_values[3 * V + 3] == 3.0f);

  printf("q4 BFS src=2 tail cohort (expect [INF,INF,0,INF]):\n");
  check("values[4*V+0]=INF", h_values[4 * V + 0] == INF);
  check("values[4*V+1]=INF", h_values[4 * V + 1] == INF);
  check("values[4*V+2]=0", h_values[4 * V + 2] == 0.0f);
  check("values[4*V+3]=INF", h_values[4 * V + 3] == INF);

  printf("completion status:\n");
  bool all_completed = true;
  for (int q = 0; q < N; ++q) {
    if (result.queries[q].completion_level <= 0) all_completed = false;
  }
  check("all 5 queries completed (completion_level > 0)", all_completed);
  (void)N;

  if (failures == 0) {
    printf("\nvalidate_replenish: ALL PASS\n");
    return EXIT_SUCCESS;
  }
  printf("\nvalidate_replenish: %d FAILURES\n", failures);
  return EXIT_FAILURE;
}
