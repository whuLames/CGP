/*
 * smoke_replenish.cu
 * replenishment 阶段1 + 阶段2 验证
 *
 * 阶段1（active_union 收敛检测）：
 *   run_heterogeneous + profile_iterations，验证每轮 query_convergence_mask
 *   正确反映 slot 活跃/收敛状态。toy 图 0->1,0->2,1->3，batch=[BFS,SSSP,WCC]。
 *
 * 阶段2（reinit_single_slot）：
 *   init 2 BFS(src=0,1) → launch_reinit_single_slot(slot=1, new src=2)
 *   验证 slot 1 被清理（values 全 INF 除新 source、mask bit1 清除）
 *   且新 source=2 正确设置、unique_count 增量。
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
using puercgp::detail::reset_counter_kernel;
using puercgp::execution_context;
using puercgp::hybrid_detail::fill_unified_kernel;
using puercgp::hybrid_detail::grid_for;
using puercgp::hybrid_detail::init_hybrid_sources_kernel;
using puercgp::hybrid_detail::launch_reinit_single_slot;
using puercgp::hybrid_detail::launch_snapshot_slot_values;
using puercgp::hybrid_query_batch;
using puercgp::query_descriptor_t;
using puercgp::query_mask_t;
using puercgp::run_heterogeneous;
using puercgp::run_options;
using puercgp::traversal_mode_t;

static int failures = 0;
static void check(const char* name, bool cond) {
  printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) ++failures;
}

// ===================== 阶段1：active_union 收敛检测 =====================
static void run_stage1() {
  printf("=== Stage 1: active_union convergence detection ===\n");
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
  query_descriptor_t d_bfs{0, algo_kind_t::bfs, unified_value_t(0)};
  query_descriptor_t d_sssp{0, algo_kind_t::sssp, unified_value_t(0)};
  query_descriptor_t d_wcc{0, algo_kind_t::wcc, unified_value_t(0)};
  descs.push_back(d_bfs);
  descs.push_back(d_sssp);
  descs.push_back(d_wcc);

  hybrid_query_batch batch(descs);
  batch.validate();
  execution_context context;
  run_options options;
  options.max_iterations = 100;
  options.traversal_mode = traversal_mode_t::push;  // toy 单向 CSR
  options.profile_iterations = true;

  auto result = run_heterogeneous(graph, batch, context, options);

  printf("iterations: %d\n", result.iterations);
  check("iterations == 3", result.iterations == 3);
  check("iteration_profiles.size() == iterations",
        result.iteration_profiles.size() ==
            static_cast<std::size_t>(result.iterations));

  if (result.iteration_profiles.size() == 3) {
    auto m0 = result.iteration_profiles[0].query_convergence_mask;
    auto m1 = result.iteration_profiles[1].query_convergence_mask;
    auto m2 = result.iteration_profiles[2].query_convergence_mask;
    printf("per-iteration query_convergence_mask (bit0=BFS,bit1=SSSP,bit2=WCC):\n");
    printf("  lv0 mask = 0x%llx\n", static_cast<unsigned long long>(m0));
    printf("  lv1 mask = 0x%llx (subset of lv0; WCC may converge early)\n",
           static_cast<unsigned long long>(m1));
    printf("  lv2 mask = 0x%llx\n", static_cast<unsigned long long>(m2));
    check("lv0 mask == 0b111 (all active at start)", m0 == 0b111);
    // 单调性：收敛单向，每轮 active mask 是前一轮子集
    check("lv1 mask monotone (subset of lv0)", (m1 & ~m0) == 0);
    check("lv2 mask monotone (subset of lv1)", (m2 & ~m1) == 0);
    check("lv2 mask == 0 (all converged at end)", m2 == 0);
    if ((m0 & 0b100) != 0 && (m1 & 0b100) == 0) {
      printf("  observed: WCC(bit2) converged at lv1 earlier than BFS/SSSP "
             "(per-slot differential convergence detected)\n");
    }
  }

  // 最终结果正确性（确保 active_union 不破坏语义）
  std::vector<unified_value_t> h_values(result.values.size());
  thrust::copy(result.values.begin(), result.values.end(), h_values.begin());
  check("BFS values[3,0]=2", h_values[3 * Q + 0] == 2.0f);
  check("SSSP values[3,1]=3", h_values[3 * Q + 1] == 3.0f);
  check("WCC values[3,2]=0", h_values[3 * Q + 2] == 0.0f);
}

// ===================== 阶段2：reinit_single_slot =====================
static void run_stage2() {
  printf("\n=== Stage 2: reinit_single_slot ===\n");
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

  // batch: [BFS src=0, BFS src=1]
  std::vector<query_descriptor_t> descs;
  descs.push_back({0, algo_kind_t::bfs, unified_value_t(0)});
  descs.push_back({1, algo_kind_t::bfs, unified_value_t(0)});
  hybrid_query_batch batch(descs);
  batch.validate();
  auto views = batch.upload_to_device();

  thrust::device_vector<unified_value_t> values(static_cast<std::size_t>(V) * Q);
  thrust::device_vector<query_mask_t> visited_mask(V, 0);
  thrust::device_vector<query_mask_t> frontier_mask(V, 0);
  thrust::device_vector<query_mask_t> next_frontier_mask(V, 0);
  thrust::device_vector<int> frontier_vertices(V, -1);
  thrust::device_vector<unsigned long long> unique_count(1, 0);
  constexpr int threads = 256;

  // fill INF + init sources（slot0 src=0, slot1 src=1）
  fill_unified_kernel<<<grid_for(static_cast<std::size_t>(V) * Q, threads),
                        threads>>>(
      thrust::raw_pointer_cast(values.data()),
      static_cast<std::size_t>(V) * Q);
  reset_counter_kernel<<<1, 1>>>(
      thrust::raw_pointer_cast(unique_count.data()));
  init_hybrid_sources_kernel<<<grid_for(Q, threads), threads>>>(
      views.sources, views.kinds, views.source_values, Q,
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(frontier_mask.data()),
      thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(unique_count.data()));
  cudaDeviceSynchronize();

  // reinit slot=1，换新 BFS source=2
  query_descriptor_t new_d{2, algo_kind_t::bfs, unified_value_t(0)};
  launch_reinit_single_slot(graph, /*slot=*/1, new_d, Q,
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(frontier_mask.data()),
      thrust::raw_pointer_cast(next_frontier_mask.data()),
      thrust::raw_pointer_cast(frontier_vertices.data()),
      thrust::raw_pointer_cast(unique_count.data()),
      /*stream=*/0);
  cudaDeviceSynchronize();

  // 验证
  std::vector<unified_value_t> h_values(static_cast<std::size_t>(V) * Q);
  std::vector<query_mask_t> h_visited(V), h_frontier(V);
  thrust::copy(values.begin(), values.end(), h_values.begin());
  thrust::copy(visited_mask.begin(), visited_mask.end(), h_visited.begin());
  thrust::copy(frontier_mask.begin(), frontier_mask.end(), h_frontier.begin());
  unsigned long long h_uc = 0;
  thrust::copy(unique_count.begin(), unique_count.end(), &h_uc);

  const auto INF = unified_infinity();
  printf("slot 1 values after reinit (expect only v=2 = 0):\n");
  check("slot1 values[0]=INF (cleared)", h_values[0 * Q + 1] == INF);
  check("slot1 values[1]=INF (old src=1 cleared)", h_values[1 * Q + 1] == INF);
  check("slot1 values[2]=0 (new src=2)", h_values[2 * Q + 1] == 0.0f);
  check("slot1 values[3]=INF", h_values[3 * Q + 1] == INF);
  printf("slot 0 untouched (src=0 value=0):\n");
  check("slot0 values[0]=0 (untouched)", h_values[0 * Q + 0] == 0.0f);
  check("slot0 values[1]=INF (untouched)", h_values[1 * Q + 0] == INF);
  printf("visited_mask bit1 (expect only v=2):\n");
  check("visited[0] bit1 cleared", (h_visited[0] & query_mask_t(2)) == 0);
  check("visited[1] bit1 cleared (old src)", (h_visited[1] & query_mask_t(2)) == 0);
  check("visited[2] bit1 set (new src)", (h_visited[2] & query_mask_t(2)) != 0);
  printf("frontier_mask bit1 (expect only v=2):\n");
  check("frontier[2] bit1 set", (h_frontier[2] & query_mask_t(2)) != 0);
  check("frontier[0] bit1 cleared", (h_frontier[0] & query_mask_t(2)) == 0);
  printf("unique_count (init 2 + reinit 1 = 3):\n");
  check("unique_count == 3", h_uc == 3);
}

// ===================== 阶段3：snapshot_slot_values =====================
static void run_stage3() {
  printf("\n=== Stage 3: snapshot_slot_values ===\n");
  const int V = 4, Q = 2;
  // 模拟 BFS slot0=[0,1,1,2]，slot1=INF；values 是 vertex-major [V*Q]
  const auto INF = unified_infinity();
  std::vector<unified_value_t> h_init = {
      0.0f, INF,  // v0
      1.0f, INF,  // v1
      1.0f, INF,  // v2
      2.0f, INF   // v3
  };
  thrust::device_vector<unified_value_t> values(h_init);
  thrust::device_vector<query_mask_t> visited_mask(
      V, query_mask_t{1});
  thrust::device_vector<unified_value_t> final_row(V, INF);

  launch_snapshot_slot_values(/*slot=*/0, algo_kind_t::bfs, Q,
      static_cast<std::size_t>(V),
      thrust::raw_pointer_cast(values.data()),
      thrust::raw_pointer_cast(visited_mask.data()),
      thrust::raw_pointer_cast(final_row.data()), /*stream=*/0);
  cudaDeviceSynchronize();

  std::vector<unified_value_t> h_final(V);
  std::vector<unified_value_t> h_values_after(h_init.size());
  thrust::copy(final_row.begin(), final_row.end(), h_final.begin());
  thrust::copy(values.begin(), values.end(), h_values_after.begin());

  printf("slot0 snapshot (expect [0,1,1,2]):\n");
  check("final[0]=0", h_final[0] == 0.0f);
  check("final[1]=1", h_final[1] == 1.0f);
  check("final[2]=1", h_final[2] == 1.0f);
  check("final[3]=2", h_final[3] == 2.0f);
  check("values unchanged (snapshot is read-only)", h_values_after == h_init);
}

int main() {
  run_stage1();
  run_stage2();
  run_stage3();

  if (failures == 0) {
    printf("\nsmoke_replenish: ALL PASS\n");
    return EXIT_SUCCESS;
  }
  printf("\nsmoke_replenish: %d FAILURES\n", failures);
  return EXIT_FAILURE;
}
