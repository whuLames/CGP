/*
 * smoke_hybrid_push.cu
 * 验证 expand_shared_node_hybrid_kernel 在混合 BFS+SSSP batch 下单轮 push 的正确性
 *
 * toy graph: 0->1(w=2), 0->2(w=5), 1->3(w=1)
 * batch     : [BFS src=0, SSSP src=0]
 *
 * 期望一轮 push（level 0→1）后：
 *   BFS slot(0) : values[1,0]=1, values[2,0]=1, values[3,0]=INF
 *   SSSP slot(1): values[1,1]=2, values[2,1]=5, values[3,1]=INF
 *   next_frontier: vertex 1, 2 (bit0 | bit1)
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
using puercgp::hybrid_detail::expand_shared_node_hybrid_kernel;
using puercgp::hybrid_detail::fill_unified_kernel;
using puercgp::hybrid_detail::grid_for;
using puercgp::hybrid_detail::init_hybrid_sources_kernel;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::query_mask_t;
using puercgp::detail::reset_counter_kernel;

#define CUDA_CHECK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); return EXIT_FAILURE; } } while (0)

static int failures = 0;
static void check(const char* name, bool cond) {
  printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) ++failures;
}

int main() {
  // toy CSR: 0->{1,2}, 1->{3}
  const int V = 4, E = 3, Q = 2;
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

  // batch: [BFS src=0, SSSP src=0]
  std::vector<query_descriptor_t> descs;
  query_descriptor_t d_bfs;
  d_bfs.source = 0;
  d_bfs.kind = algo_kind_t::bfs;
  d_bfs.source_value = unified_value_t(0);
  query_descriptor_t d_sssp;
  d_sssp.source = 0;
  d_sssp.kind = algo_kind_t::sssp;
  d_sssp.source_value = unified_value_t(0);
  descs.push_back(d_bfs);
  descs.push_back(d_sssp);

  hybrid_query_batch batch(descs);
  batch.validate();
  auto views = batch.upload_to_device();
  query_mask_t bfs_mask = batch.bfs_slot_mask();      // bit0
  query_mask_t nonbfs_mask = batch.nonbfs_slot_mask(); // bit1

  thrust::device_vector<unified_value_t> values(
      static_cast<std::size_t>(V) * Q);
  thrust::device_vector<query_mask_t> visited_mask(V, 0);
  thrust::device_vector<query_mask_t> frontier_mask(V, 0);
  thrust::device_vector<query_mask_t> next_frontier_mask(V, 0);
  thrust::device_vector<int> frontier_vertices(V, -1);
  thrust::device_vector<int> next_frontier_vertices(V, -1);
  thrust::device_vector<unsigned long long> unique_count(1, 0);
  thrust::device_vector<unsigned long long> next_unique_count(1, 0);
  thrust::device_vector<unsigned long long> next_pair_count(1, 0);
  thrust::device_vector<query_mask_t> active_union_dev(1, 0);

  constexpr int threads = 256;

  // 1. fill INF
  fill_unified_kernel<<<grid_for(V * Q, threads), threads>>>(
      thrust::raw_pointer_cast(values.data()),
      static_cast<std::size_t>(V) * Q);
  CUDA_CHECK(cudaGetLastError());

  // 2. init sources（BFS+SSSP src=0）
  reset_counter_kernel<<<1, 1>>>(
      thrust::raw_pointer_cast(unique_count.data()));
  init_hybrid_sources_kernel<<<grid_for(Q, threads), threads>>>(
      views.sources, views.kinds, views.source_values, Q,
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(frontier_mask.data()),
      thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(unique_count.data()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long h_uc = 0;
  thrust::copy(unique_count.begin(), unique_count.end(), &h_uc);
  check("init unique_count==1 (vertex 0)", h_uc == 1);

  // 3. 单轮 push（level 0→1）
  expand_shared_node_hybrid_kernel<csr_graph_view<int, int, float>>
      <<<grid_for(static_cast<std::size_t>(h_uc), threads), threads>>>(
          graph,
          thrust::raw_pointer_cast(frontier_vertices.data()),
          thrust::raw_pointer_cast(frontier_mask.data()),
          static_cast<std::size_t>(h_uc),
          thrust::raw_pointer_cast(visited_mask.data()),
          thrust::raw_pointer_cast(next_frontier_mask.data()),
          thrust::raw_pointer_cast(next_frontier_vertices.data()),
          thrust::raw_pointer_cast(next_unique_count.data()),
          thrust::raw_pointer_cast(next_pair_count.data()),
          thrust::raw_pointer_cast(values.data()),
          views.kinds, Q, /*level=*/0, bfs_mask, nonbfs_mask,
          thrust::raw_pointer_cast(active_union_dev.data()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // active_union 收敛信号验证：BFS slot(bit0) + SSSP slot(bit1) 都改进了 vertex 1,2
  query_mask_t h_union = 0;
  thrust::copy(active_union_dev.begin(), active_union_dev.end(), &h_union);
  check("active_union == 0b11 (BFS|SSSP both improved)", h_union == 0b11);

  // verify
  std::vector<unified_value_t> h_values(static_cast<std::size_t>(V) * Q);
  std::vector<query_mask_t> h_next_frontier(V);
  thrust::copy(values.begin(), values.end(), h_values.begin());
  thrust::copy(next_frontier_mask.begin(), next_frontier_mask.end(),
               h_next_frontier.begin());

  const unified_value_t INF = unified_infinity();
  printf("BFS slot(0) after push:\n");
  check("values[0,0]=0 (src)", h_values[0 * Q + 0] == 0.0f);
  check("values[1,0]=1", h_values[1 * Q + 0] == 1.0f);
  check("values[2,0]=1", h_values[2 * Q + 0] == 1.0f);
  check("values[3,0]=INF", h_values[3 * Q + 0] == INF);

  printf("SSSP slot(1) after push:\n");
  check("values[0,1]=0 (src)", h_values[0 * Q + 1] == 0.0f);
  check("values[1,1]=2 (0+2)", h_values[1 * Q + 1] == 2.0f);
  check("values[2,1]=5 (0+5)", h_values[2 * Q + 1] == 5.0f);
  check("values[3,1]=INF", h_values[3 * Q + 1] == INF);

  printf("next_frontier_mask:\n");
  check("next_frontier[0] clear (src done)",
        h_next_frontier[0] == 0);
  check("next_frontier[1] bit0",
        (h_next_frontier[1] & query_mask_t(1)) != 0);
  check("next_frontier[1] bit1",
        (h_next_frontier[1] & query_mask_t(2)) != 0);
  check("next_frontier[2] bit0",
        (h_next_frontier[2] & query_mask_t(1)) != 0);
  check("next_frontier[2] bit1",
        (h_next_frontier[2] & query_mask_t(2)) != 0);
  check("next_frontier[3] clear",
        h_next_frontier[3] == 0);

  if (failures == 0) {
    printf("\nsmoke_hybrid_push: ALL PASS\n");
    return EXIT_SUCCESS;
  }
  printf("\nsmoke_hybrid_push: %d FAILURES\n", failures);
  return EXIT_FAILURE;
}
