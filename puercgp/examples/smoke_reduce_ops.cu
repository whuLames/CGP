/*
 * smoke_reduce_ops.cu
 * 验证 hybrid_detail 的 device 原语行为：
 *   - apply_first_write（BFS 路径 CAS）
 *   - apply_min_reduce（SSSP/WCC 路径 CAS 循环）
 *   - compute_candidate / compute_candidate_pull（三种 algo_kind 分支）
 *
 * 不依赖真实图，仅用 device 标量数组做单元级验证。
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <puercgp/puercgp.hxx>

using puercgp::algorithms::algo_kind_t;
using puercgp::algorithms::unified_infinity;
using puercgp::algorithms::unified_value_t;
using puercgp::hybrid_detail::apply_min_reduce;
using puercgp::hybrid_detail::apply_first_write;
using puercgp::hybrid_detail::apply_result_t;
using puercgp::hybrid_detail::compute_candidate;
using puercgp::hybrid_detail::compute_candidate_pull;

#define CUDA_CHECK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return EXIT_FAILURE; } } while (0)

// 测试 apply_first_write：INF→5（首次成功），5→3（再次失败）
__global__ void test_first_write(unified_value_t* slot, apply_result_t* out) {
  if (threadIdx.x == 0) {
    out[0] = apply_first_write(slot, unified_value_t{5});  // INF→5, 期望 improved=true
    out[1] = apply_first_write(slot, unified_value_t{3});  // 5≠INF, 期望 improved=false
  }
}

// 测试 apply_min_reduce：10→5（改进），5→8（不改进），5→3（改进）
__global__ void test_min_reduce(unified_value_t* slot, apply_result_t* out) {
  if (threadIdx.x == 0) {
    out[0] = apply_min_reduce(slot, unified_value_t{5});  // 10→5, improved=true
    out[1] = apply_min_reduce(slot, unified_value_t{8});  // 5<8,  improved=false
    out[2] = apply_min_reduce(slot, unified_value_t{3});  // 5→3,  improved=true
  }
}

// 测试 compute_candidate（push + pull 三种 algo_kind）
__global__ void test_candidate(unified_value_t* out_push,
                               unified_value_t* out_pull) {
  if (threadIdx.x == 0) {
    // push 模式
    out_push[0] = compute_candidate(algo_kind_t::bfs, 0.0f, 1.0f, 3);   // level+1=4
    out_push[1] = compute_candidate(algo_kind_t::sssp, 2.0f, 3.5f, 0);  // 2+3.5=5.5
    out_push[2] = compute_candidate(algo_kind_t::wcc, 7.0f, 1.0f, 0);   // identity=7
    // pull 模式
    out_pull[0] = compute_candidate_pull(algo_kind_t::bfs, 2.0f, 1.0f);   // 2+1=3
    out_pull[1] = compute_candidate_pull(algo_kind_t::sssp, 2.0f, 3.5f);  // 5.5
    out_pull[2] = compute_candidate_pull(algo_kind_t::wcc, 7.0f, 1.0f);   // 7
  }
}

static int failures = 0;
static void check(const char* name, bool cond, const char* detail) {
  if (cond) {
    printf("  [PASS] %s %s\n", name, detail);
  } else {
    printf("  [FAIL] %s %s\n", name, detail);
    ++failures;
  }
}

int main() {
  // ============ apply_first_write ============
  printf("apply_first_write:\n");
  {
    unified_value_t* d_slot;
    apply_result_t* d_res;
    CUDA_CHECK(cudaMalloc(&d_slot, sizeof(unified_value_t)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(apply_result_t) * 2));
    unified_value_t init = unified_infinity();
    CUDA_CHECK(cudaMemcpy(d_slot, &init, sizeof(unified_value_t),
                          cudaMemcpyHostToDevice));
    test_first_write<<<1, 32>>>(d_slot, d_res);
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_result_t h_res[2];
    CUDA_CHECK(cudaMemcpy(h_res, d_res, sizeof(apply_result_t) * 2,
                          cudaMemcpyDeviceToHost));
    check("first_write #1 INF→5 improved",
          h_res[0].improved == true, "");
    check("first_write #1 old==INF",
          h_res[0].old_value == unified_infinity(), "");
    check("first_write #2 5→3 not improved",
          h_res[1].improved == false, "");
    check("first_write #2 old==5",
          h_res[1].old_value == unified_value_t{5}, "");
    cudaFree(d_slot);
    cudaFree(d_res);
  }

  // ============ apply_min_reduce ============
  printf("apply_min_reduce:\n");
  {
    unified_value_t* d_slot;
    apply_result_t* d_res;
    CUDA_CHECK(cudaMalloc(&d_slot, sizeof(unified_value_t)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(apply_result_t) * 3));
    unified_value_t init = unified_value_t{10};
    CUDA_CHECK(cudaMemcpy(d_slot, &init, sizeof(unified_value_t),
                          cudaMemcpyHostToDevice));
    test_min_reduce<<<1, 32>>>(d_slot, d_res);
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_result_t h_res[3];
    CUDA_CHECK(cudaMemcpy(h_res, d_res, sizeof(apply_result_t) * 3,
                          cudaMemcpyDeviceToHost));
    check("min_reduce #1 10→5 improved", h_res[0].improved == true, "");
    check("min_reduce #1 old==10",
          h_res[0].old_value == unified_value_t{10}, "");
    check("min_reduce #2 5→8 not improved", h_res[1].improved == false, "");
    check("min_reduce #3 5→3 improved", h_res[2].improved == true, "");
    check("min_reduce #3 old==5",
          h_res[2].old_value == unified_value_t{5}, "");
    cudaFree(d_slot);
    cudaFree(d_res);
  }

  // ============ compute_candidate ============
  printf("compute_candidate:\n");
  {
    unified_value_t* d_push;
    unified_value_t* d_pull;
    CUDA_CHECK(cudaMalloc(&d_push, sizeof(unified_value_t) * 3));
    CUDA_CHECK(cudaMalloc(&d_pull, sizeof(unified_value_t) * 3));
    test_candidate<<<1, 32>>>(d_push, d_pull);
    CUDA_CHECK(cudaDeviceSynchronize());
    unified_value_t h_push[3], h_pull[3];
    CUDA_CHECK(cudaMemcpy(h_push, d_push, sizeof(unified_value_t) * 3,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pull, d_pull, sizeof(unified_value_t) * 3,
                          cudaMemcpyDeviceToHost));
    check("push bfs level+1=4", h_push[0] == unified_value_t{4}, "");
    check("push sssp 2+3.5=5.5", h_push[1] == unified_value_t{5.5f}, "");
    check("push wcc identity=7", h_push[2] == unified_value_t{7}, "");
    check("pull bfs nb+1=3", h_pull[0] == unified_value_t{3}, "");
    check("pull sssp 2+3.5=5.5", h_pull[1] == unified_value_t{5.5f}, "");
    check("pull wcc identity=7", h_pull[2] == unified_value_t{7}, "");
    cudaFree(d_push);
    cudaFree(d_pull);
  }

  if (failures == 0) {
    printf("\nsmoke_reduce_ops: ALL PASS\n");
    return EXIT_SUCCESS;
  }
  printf("\nsmoke_reduce_ops: %d FAILURES\n", failures);
  return EXIT_FAILURE;
}
