/*
 * validate_wcc.cu
 * 同质 WCC（Label Propagation）验证：单算法基线
 *
 * toy graph: 0->1, 0->2, 1->3（有向，但 LP 沿出边传播可达全顶点）
 * 期望：label 收敛到 0（所有可达顶点的 min vertex id）
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <puercgp/puercgp.hxx>

using puercgp::algorithms::wcc_policy;
using puercgp::csr_graph_view;
using puercgp::execution_context;
using puercgp::query_batch;
using puercgp::run;
using puercgp::run_options;
using puercgp::traversal_mode_t;
using puercgp::push_strategy_t;

static int failures = 0;
static void check(const char* name, bool cond) {
  printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) ++failures;
}

int main() {
  const int V = 4, E = 3;
  std::vector<int> h_row = {0, 2, 3, 3, 3};
  std::vector<int> h_col = {1, 2, 3};
  thrust::device_vector<int> d_row(h_row);
  thrust::device_vector<int> d_col(h_col);

  csr_graph_view<int, int, float> graph;
  graph.number_of_vertices = V;
  graph.number_of_edges = E;
  graph.row_offsets = thrust::raw_pointer_cast(d_row.data());
  graph.column_indices = thrust::raw_pointer_cast(d_col.data());
  graph.edge_weights = nullptr;  // WCC 不用 weight

  query_batch<int> queries({0});  // source 0（占位，WCC 是全顶点 init）
  execution_context context;
  run_options options;
  options.max_iterations = 100;
  options.traversal_mode = traversal_mode_t::push;
  options.push_strategy = push_strategy_t::shared_node;

  auto result = run<wcc_policy>(graph, queries, context, options);

  std::vector<float> h_values(result.values.size());
  thrust::copy(result.values.begin(), result.values.end(), h_values.begin());

  printf("iterations: %d\n", result.iterations);
  printf("WCC labels [全连通→0]:\n");
  check("label[0]=0", h_values[0] == 0.0f);
  check("label[1]=0", h_values[1] == 0.0f);
  check("label[2]=0", h_values[2] == 0.0f);
  check("label[3]=0", h_values[3] == 0.0f);

  if (failures == 0) {
    printf("\nvalidate_wcc: ALL PASS\n");
    return EXIT_SUCCESS;
  }
  printf("\nvalidate_wcc: %d FAILURES\n", failures);
  return EXIT_FAILURE;
}
