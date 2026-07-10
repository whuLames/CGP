#pragma once

#include <puercgp/core/algorithm_id.hxx>

namespace puercgp {
namespace algorithms {

// 异构 batch 的统一 value 类型，所有算法都用 float 表示
//   - BFS 层级：vertex 数 < 2^24，层级 < 2^24，float 尾数 23 位无损
//   - SSSP 距离：原生 float
//   - WCC label：vertex id cast 为 float（同样 < 2^24 无损）
// 选用 float 而非 int，使一个 batch 能同时持有 BFS(int 层级) 与
// SSSP(float 距离) 两类 slot，避免 Policy::value_type 单态约束。
// 异构 batch 内每条 query 的算法 tag（运行期分派依据）
// 取代编译期 Policy，让同一 kernel 实例同时服务多种算法
// 具体类型定义位于 <puercgp/core/algorithm_id.hxx>，这里保留语义说明和
// 兼容 include 路径。

// unified 无穷大，与 sssp_policy::infinity() 一致（float +inf）
// 用于 fill_values 初始化，以及 first_write 的 CAS 期望值
// unified_infinity() 定义位于 <puercgp/core/algorithm_id.hxx>。

// 统一 source value（BFS/SSSP 的起点值均为 0）
// WCC 的初始 label 是 per-vertex 的 vertex id，由专门 init kernel 处理，
// 不使用此函数
// unified_source_value() 定义位于 <puercgp/core/algorithm_id.hxx>。

// 编译期类型大小校验：确保 unified_value_t 是 4 字节（float），
// 与 atomicCAS(float→int reinterpret) 的前提一致
// algo_kind_t 必须是 1 字节，保证 slot_kinds[Q] 数组紧凑（Q≤64 → 64 字节）

}  // namespace algorithms
}  // namespace puercgp
