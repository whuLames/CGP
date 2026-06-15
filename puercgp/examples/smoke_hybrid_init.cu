/*
 * smoke_hybrid_init.cu
 * 验证 hybrid_engine 的三个 init kernel 在混合 batch 下的初始化状态
 *
 * toy graph: 0->1->2->3 线性链
 * batch     : [BFS src=0, SSSP src=1, WCC]
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
using puercgp::hybrid_detail::fill_unified_kernel;
using puercgp::hybrid_detail::grid_for;
using puercgp::hybrid_detail::init_hybrid_sources_kernel;
using puercgp::hybrid_detail::init_wcc_all_vertices_kernel;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::query_mask_t;

#define CUDA_CHECK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); return EXIT_FAILURE; } } while (0)

static int failures = 0;
static void check(const char* name, bool cond) {
  printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) ++failures;
}

int main() {
  const int V = 4, E = 3, Q = 3;
  std::vector<int> h_row = {0, 1, 2, 3, 3};
  std::vector<int> h_col = {1, 2, 3};
  thrust::device_vector<int> d_row(h_row);
  thrust::device_vector<int> d_col(h_col);

  csr_graph_view<int, int, float> graph;
  graph.number_of_vertices = V;
  graph.number_of_edges = E;
  graph.row_offsets = thrust::raw_pointer_cast(d_row.data());
  graph.column_indices = thrust::raw_pointer_cast(d_col.data());
  graph.edge_weights = nullptr;

  // batch: [BFS src=0, SSSP src=1, WCC]
  std::vector<query_descriptor_t> descs;
  query_descriptor_t d_bfs;
  d_bfs.source = 0;
  d_bfs.kind = algo_kind_t::bfs;
  d_bfs.source_value = unified_value_t(0);
  query_descriptor_t d_sssp;
  d_sssp.source = 1;
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
  auto views = batch.upload_to_device();

  thrust::device_vector<unified_value_t> values(
      static_cast<std::size_t>(V) * Q);
  thrust::device_vector<query_mask_t> visited_mask(V, 0);
  thrust::device_vector<query_mask_t> frontier_mask(V, 0);
  thrust::device_vector<int> frontier_vertices(V, -1);
  thrust::device_vector<unsigned long long> unique_count(1, 0);

  constexpr int threads = 256;

  fill_unified_kernel<<<grid_for(V * Q, threads), threads>>>(
      thrust::raw_pointer_cast(values.data()),
      static_cast<std::size_t>(V) * Q);
  CUDA_CHECK(cudaGetLastError());

  init_hybrid_sources_kernel<<<grid_for(Q, threads), threads>>>(
      views.sources, views.kinds, views.source_values, Q,
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(frontier_mask.data()),
      thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(unique_count.data()));
  CUDA_CHECK(cudaGetLastError());

  init_wcc_all_vertices_kernel<<<grid_for(V * Q, threads), threads>>>(
      graph, views.kinds, Q,
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(frontier_mask.data()),
      thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(unique_count.data()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unified_value_t> h_values(static_cast<std::size_t>(V) * Q);
  std::vector<query_mask_t> h_visited(V), h_frontier(V);
  thrust::copy(values.begin(), values.end(), h_values.begin());
  thrust::copy(visited_mask.begin(), visited_mask.end(), h_visited.begin());
  thrust::copy(frontier_mask.begin(), frontier_mask.end(), h_frontier.begin());
  unsigned long long h_uc = 0;
  thrust::copy(unique_count.begin(), unique_count.end(), &h_uc);

  const unified_value_t INF = unified_infinity();
  printf("values matrix [v*Q+q]:\n");
  check("BFS values[0,0]=0", h_values[0 * Q + 0] == 0.0f);
  check("BFS values[1,0]=INF", h_values[1 * Q + 0] == INF);
  check("BFS values[2,0]=INF", h_values[2 * Q + 0] == INF);
  check("SSSP values[1,1]=0", h_values[1 * Q + 1] == 0.0f);
  check("SSSP values[0,1]=INF", h_values[0 * Q + 1] == INF);
  check("SSSP values[3,1]=INF", h_values[3 * Q + 1] == INF);
  check("WCC values[0,2]=0", h_values[0 * Q + 2] == 0.0f);
  check("WCC values[1,2]=1", h_values[1 * Q + 2] == 1.0f);
  check("WCC values[2,2]=2", h_values[2 * Q + 2] == 2.0f);
  check("WCC values[3,2]=3", h_values[3 * Q + 2] == 3.0f);

  printf("visited_mask (BFS only):\n");
  check("visited[0] bit0 set", (h_visited[0] & query_mask_t(1)) != 0);
  check("visited[1] bit0 clear", (h_visited[1] & query_mask_t(1)) == 0);

  printf("frontier_mask:\n");
  check("frontier[0] bit0 (BFS src)", (h_frontier[0] & query_mask_t(1)) != 0);
  check("frontier[1] bit1 (SSSP src)", (h_frontier[1] & query_mask_t(2)) != 0);
  check("frontier[0] bit2 (WCC)", (h_frontier[0] & query_mask_t(4)) != 0);
  check("frontier[1] bit2 (WCC)", (h_frontier[1] & query_mask_t(4)) != 0);
  check("frontier[2] bit2 (WCC)", (h_frontier[2] & query_mask_t(4)) != 0);
  check("frontier[3] bit2 (WCC)", (h_frontier[3] & query_mask_t(4)) != 0);

  printf("unique_count: %llu (expect 4)\n", h_uc);
  check("unique_count==4", h_uc == 4);

  if (failures == 0) {
    printf("\nsmoke_hybrid_init: ALL PASS\n");
    return EXIT_SUCCESS;
  }
  printf("\nsmoke_hybrid_init: %d FAILURES\n", failures);
  return EXIT_FAILURE;
}
