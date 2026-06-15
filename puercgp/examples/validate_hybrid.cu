/*
 * validate_hybrid.cu
 * 端到端验证 run_heterogeneous：BFS + SSSP + WCC 三方混合 batch
 *
 * toy graph: 0->1(w=2), 0->2(w=5), 1->3(w=1)
 * batch     : [BFS src=0, SSSP src=0, WCC(全顶点)]
 *
 * 期望：
 *   BFS slot(0) : [0,1,1,2]   (层级)
 *   SSSP slot(1): [0,2,5,3]   (0->1->3 = 2+1=3)
 *   WCC slot(2) : [0,0,0,0]   (全连通，label 收敛到 min=0)
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

using puercgp::algorithms::algo_kind_t;
using puercgp::algorithms::unified_infinity;
using puercgp::algorithms::unified_value_t;
using puercgp::csr_graph_view;
using puercgp::execution_context;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::query_mask_t;
using puercgp::run_heterogeneous;
using puercgp::run_options;

static int failures = 0;
static void check(const char* name, bool cond) {
  printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) ++failures;
}

int main() {
  const int V = 4, E = 3, Q = 3;
  std::vector<int> h_row = {0, 2, 3, 3, 3};
  std::vector<int> h_col = {1, 2, 3};
  std::vector<float> h_w = {2.0f, 5.0f, 1.0f};
  thrust::device_vector<int> d_row(h_row);
  thrust::device_vector<int> d_col(h_col);
  thrust::device_vector<float> d_w(h_w);

  csr_graph_view<int, int, float> graph;
  graph.number_of_vertices = V;
  graph.number_of_edges = E;
  graph.row_offsets = thrust::raw_pointer_cast(d_row.data());
  graph.column_indices = thrust::raw_pointer_cast(d_col.data());
  graph.edge_weights = thrust::raw_pointer_cast(d_w.data());

  std::vector<query_descriptor_t> descs;
  query_descriptor_t d_bfs;
  d_bfs.source = 0;
  d_bfs.kind = algo_kind_t::bfs;
  d_bfs.source_value = unified_value_t(0);
  query_descriptor_t d_sssp;
  d_sssp.source = 0;
  d_sssp.kind = algo_kind_t::sssp;
  d_sssp.source_value = unified_value_t(0);
  query_descriptor_t d_wcc;
  d_wcc.source = 0;
  d_wcc.kind = algo_kind_t::wcc;
  d_wcc.source_value = unified_value_t(0);
  descs.push_back(d_bfs);
  descs.push_back(d_sssp);
  descs.push_back(d_wcc);

  hybrid_query_batch batch(descs);
  batch.validate();
  execution_context context;
  run_options options;
  options.max_iterations = 100;  // WCC 收敛兜底
  // toy 图是单向 CSR（0→1, 0→2, 1→3），pull kernel 假设双向 CSR，
  // 故 toy smoke 路径固定走 push。真实图（双向 CSR）的 pull 验证
  // 由 validate_hybrid_real 覆盖。
  options.traversal_mode = puercgp::traversal_mode_t::push;

  auto result = run_heterogeneous(graph, batch, context, options);

  std::vector<unified_value_t> h_values(result.values.size());
  thrust::copy(result.values.begin(), result.values.end(), h_values.begin());

  const unified_value_t INF = unified_infinity();
  printf("iterations: %d\n", result.iterations);
  printf("BFS slot(0) [层级]:\n");
  check("values[0,0]=0", h_values[0 * Q + 0] == 0.0f);
  check("values[1,0]=1", h_values[1 * Q + 0] == 1.0f);
  check("values[2,0]=1", h_values[2 * Q + 0] == 1.0f);
  check("values[3,0]=2", h_values[3 * Q + 0] == 2.0f);

  printf("SSSP slot(1) [距离]:\n");
  check("values[0,1]=0", h_values[0 * Q + 1] == 0.0f);
  check("values[1,1]=2", h_values[1 * Q + 1] == 2.0f);
  check("values[2,1]=5", h_values[2 * Q + 1] == 5.0f);
  check("values[3,1]=3 (2+1)", h_values[3 * Q + 1] == 3.0f);

  printf("WCC slot(2) [label, 全连通→0]:\n");
  check("values[0,2]=0", h_values[0 * Q + 2] == 0.0f);
  check("values[1,2]=0", h_values[1 * Q + 2] == 0.0f);
  check("values[2,2]=0", h_values[2 * Q + 2] == 0.0f);
  check("values[3,2]=0", h_values[3 * Q + 2] == 0.0f);

  if (failures == 0) {
    printf("\nvalidate_hybrid: ALL PASS\n");
    return EXIT_SUCCESS;
  }
  printf("\nvalidate_hybrid: %d FAILURES\n", failures);
  return EXIT_FAILURE;
}
