#pragma once

#include <cuda_runtime.h>

#include <puercgp/core/algorithm_id.hxx>

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

__device__ __forceinline__ apply_result_t apply_max_reduce(
    algorithms::unified_value_t* slot,
    algorithms::unified_value_t candidate) {
  int* addr = reinterpret_cast<int*>(slot);
  int old = *addr;
  while (candidate > __int_as_float(old)) {
    int assumed = old;
    old = atomicCAS(addr, assumed, __float_as_int(candidate));
    if (assumed == old) {
      return {true, __int_as_float(assumed)};
    }
  }
  return {false, __int_as_float(old)};
}

__device__ __forceinline__ apply_result_t apply_reduce(
    algorithms::algo_kind_t kind,
    algorithms::unified_value_t* slot,
    algorithms::unified_value_t candidate) {
  if (kind == algorithms::algo_kind_t::sswp) {
    return apply_max_reduce(slot, candidate);
  }
  return apply_min_reduce(slot, candidate);
}

}  // namespace hybrid_detail
}  // namespace puercgp
