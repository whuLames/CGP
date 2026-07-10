#pragma once

#include <cuda_runtime.h>

#include <puercgp/algorithms/hybrid.hxx>

namespace puercgp {
namespace hybrid_detail {

// apply_reduce 的统一返回类型
//   improved    : 本次写是否产生有效更新（用于决定是否进 next_frontier）
//   old_value   : atomic 操作发生前的旧值
struct apply_result_t {
  bool improved;
  algorithms::unified_value_t old_value;
};

// ============================================================
// apply_first_write —— BFS 路径
// 语义：仅当 slot 仍为 INF 时写入 candidate（first-write wins）
// 复用现有 expand_shared_node_kernel:637-653 的 atomicCAS(INF→level+1) 模式
// 与 BFS 的 visited_mask 批量 atomicOr 互补：此函数用于无法批量化的场景
// ============================================================
__device__ __forceinline__ apply_result_t apply_first_write(
    algorithms::unified_value_t* slot,
    algorithms::unified_value_t candidate) {
  int* addr = reinterpret_cast<int*>(slot);
  int expected = __float_as_int(algorithms::unified_infinity());
  int old = atomicCAS(addr, expected, __float_as_int(candidate));
  // 仅当旧值==INF（CAS 成功）时算改进
  return {old == expected, __int_as_float(old)};
}

// ============================================================
// apply_min_reduce —— SSSP / WCC 路径
// 语义：atomicMin，若 candidate 比当前值小则替换
// CAS 循环 inline 自现有 atomic_min_value(frontier_engine.hxx:140-157)，
// 但返回 improved 标志（旧 API 只返回旧值，无法判定 frontier）
// ============================================================
__device__ __forceinline__ apply_result_t apply_min_reduce(
    algorithms::unified_value_t* slot,
    algorithms::unified_value_t candidate) {
  int* addr = reinterpret_cast<int*>(slot);
  int old = *addr;
  while (candidate < __int_as_float(old)) {
    int assumed = old;
    old = atomicCAS(addr, assumed, __float_as_int(candidate));
    if (assumed == old) {
      // CAS 成功：我们写入了 candidate，产生改进
      return {true, __int_as_float(assumed)};
    }
    // CAS 失败：别人改了，old 已更新，重新比较
  }
  // candidate >= 当前值，无改进
  return {false, __int_as_float(old)};
}

// ============================================================
// compute_candidate —— push 模式下根据算法 tag 计算 candidate value
//   BFS : level + 1            （weight 不参与，BFS 走 visited_mask 快路）
//   SSSP: source_value + weight
//   WCC : source_value          （identity，label propagation 的 relax 退化）
// 注意：kind 应在 edge loop 外层读入寄存器（slot-invariant），
//       使编译器能做 loop-invariant code motion，避免 inner loop 内 switch
// ============================================================
__device__ __forceinline__ algorithms::unified_value_t compute_candidate(
    algorithms::algo_kind_t kind,
    algorithms::unified_value_t source_value,
    float edge_weight,
    int level) {
  (void)level;  // BFS 改用相对层级（source_value+1），不再依赖全局 level
  switch (kind) {
    case algorithms::algo_kind_t::bfs:
      // BFS 层级 = source_value + 1（相对，与 pull 的 nb+1 一致）
      // 同质路径下 source_value 即当前层级，等价于 level+1；
      // replenishment 下新注入 BFS query 从相对 0 开始正确累计
      return source_value + algorithms::unified_value_t{1};
    case algorithms::algo_kind_t::sssp:
      return source_value + edge_weight;
    case algorithms::algo_kind_t::wcc:
      return source_value;  // identity：直接 propagate source 的 label
  }
  return algorithms::unified_infinity();
}

// ============================================================
// compute_candidate_pull —— pull 模式下根据算法 tag 计算 candidate value
// 关键洞察：pull 模式下 BFS 退化为 unweighted SSSP（neighbor_value + 1），
//           所以 fused_pull_hybrid 不需要双路，
//           全部 slot 走此函数即可
//   BFS : neighbor_value + 1   （等价 unweighted SSSP）
//   SSSP: neighbor_value + weight
//   WCC : neighbor_value        （identity）
// ============================================================

// 这里其实是把relax操作做松弛
__device__ __forceinline__ algorithms::unified_value_t compute_candidate_pull(
    algorithms::algo_kind_t kind,
    algorithms::unified_value_t neighbor_value,
    float edge_weight) {
  switch (kind) {
    case algorithms::algo_kind_t::bfs:
      return neighbor_value + algorithms::unified_value_t{1};
    case algorithms::algo_kind_t::sssp:
      return neighbor_value + edge_weight;
    case algorithms::algo_kind_t::wcc:
      return neighbor_value;  // identity
  }
  return algorithms::unified_infinity();
}

}  // namespace hybrid_detail
}  // namespace puercgp
