#pragma once

#include <cstdint>
#include <limits>

#include <puercgp/core/types.hxx>

namespace puercgp {
namespace algorithms {

// 异构 batch 的统一 value 类型，所有算法都用 float 表示
//   - BFS 层级：vertex 数 < 2^24，层级 < 2^24，float 尾数 23 位无损
//   - SSSP 距离：原生 float
//   - WCC label：vertex id cast 为 float（同样 < 2^24 无损）
// 选用 float 而非 int，使一个 batch 能同时持有 BFS(int 层级) 与
// SSSP(float 距离) 两类 slot，避免 Policy::value_type 单态约束。
using unified_value_t = float;

// 异构 batch 内每条 query 的算法 tag（运行期分派依据）
// 取代编译期 Policy，让同一 kernel 实例同时服务多种算法
enum class algo_kind_t : std::uint8_t {
  bfs = 0,   // Breadth-First Search，first-write 语义（atomicCAS INF→level+1）
  sssp = 1,  // Single-Source Shortest Path，min-reduce 语义（atomicMin）
  wcc = 2    // Weakly Connected Components via Label Propagation，min-reduce
             // 语义，relax 退化为 identity，与 sssp 同构
};

// unified 无穷大，与 sssp_policy::infinity() 一致（float +inf）
// 用于 fill_values 初始化，以及 first_write 的 CAS 期望值
__host__ __device__ __forceinline__ constexpr unified_value_t
unified_infinity() {
  return std::numeric_limits<unified_value_t>::infinity();
}

// 统一 source value（BFS/SSSP 的起点值均为 0）
// WCC 的初始 label 是 per-vertex 的 vertex id，由专门 init kernel 处理，
// 不使用此函数
__host__ __device__ __forceinline__ constexpr unified_value_t
unified_source_value() {
  return unified_value_t{0};
}

// 编译期类型大小校验：确保 unified_value_t 是 4 字节（float），
// 与 atomicCAS(float→int reinterpret) 的前提一致
static_assert(sizeof(unified_value_t) == 4,
              "unified_value_t must be 4 bytes for atomicCAS reinterpret");
// algo_kind_t 必须是 1 字节，保证 slot_kinds[Q] 数组紧凑（Q≤64 → 64 字节）
static_assert(sizeof(algo_kind_t) == 1,
              "algo_kind_t must be 1 byte for compact slot_kinds array");

}  // namespace algorithms
}  // namespace puercgp
