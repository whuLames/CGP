# Hybrid Batch 已完成工作

> **最后更新：2026-06-14**
>
> 对应实施计划 `/home/zyl/.claude/plans/sunny-dancing-fog.md` 的 Step 1-7 核心部分。
> 本文档记录截至 2026-06-14 已完成并验证的内容。

## 总体目标

为 puercgp 实现 **hybrid batch**：一个 batch 内同时执行 BFS + SSSP + WCC
混合算法，使论文能反驳"这只是简单 batching"的 reviewer 质疑
（`docs/FRAMING_SINGLE_GPU_CONCURRENT.md` Contribution 1/2/3）。

核心障碍：`frontier_engine.hxx` 以 `template <typename Policy>` 单态贯穿，
`Policy::value_type` 在 kernel 签名固化（BFS=int, SSSP=float），无法单 batch
多算法。本工作把"编译期单 Policy"改为"运行期 per-query algo tag"，
异构走新建路径，同质路径零改动保留作回归保护。

## 已完成的 7 步交付

### Step 1 — 基础类型（`algorithms/hybrid.hxx`）

- `using unified_value_t = float`（BFS 层级 / SSSP 距离 / WCC label 统一表示）
- `enum class algo_kind_t : uint8_t { bfs, sssp, wcc }`
- `unified_infinity()` / `unified_source_value()` device 函数
- `static_assert(sizeof(unified_value_t)==4)` 校验（atomicCAS reinterpret 前提）

### Step 2 — device 原语（`core/reduce_ops.hxx`）

四个 `__device__ __forceinline__` 函数，是 hybrid kernel 的 update 段基石：

| 函数 | 语义 | 复用来源 |
|------|------|---------|
| `apply_first_write(slot, cand)` | BFS：CAS(INF→cand) | `expand_shared_node_kernel:637-653` 的 atomicCAS 模式 |
| `apply_min_reduce(slot, cand)` | SSSP/WCC：CAS 循环 atomicMin | `atomic_min_value:140-157`（加 improved 标志）|
| `compute_candidate(kind, src, w, level)` | push 模式：BFS=`level+1`/SSSP=`src+w`/WCC=`src` | Policy::relax 的运行期泛化 |
| `compute_candidate_pull(kind, nb, w)` | pull 模式：BFS=`nb+1`(unweighted SSSP)/SSSP=`nb+w`/WCC=`nb` | 关键洞察：pull 下 BFS 与 SSSP 同构 |

### Step 3 — query 描述符（`core/query_descriptor.hxx`）

- `struct query_descriptor_t { source, kind, source_value }`
- `class hybrid_query_batch`：host 容器 + `upload_to_device()`（const，device 缓存 mutable）
- `bfs_slot_mask()` / `nonbfs_slot_mask()`：host 端预算的 Q-bit mask，
  作为 kernel 参数传入（O(1) 寄存器常量，避免 inner loop 查表）
- `has_bfs()` / `has_wcc()`：init 分支判定
- `validate()`：非空、≤64 query、source 非负

### Step 4 — 异构 init kernels（`engine/hybrid_engine.hxx`）

三个 init kernel（`hybrid_detail` namespace）：

- `fill_unified_kernel`：填 `unified_infinity()`
- `init_hybrid_sources_kernel`：BFS slot（source_value + visited_mask + frontier）
  / SSSP slot（source_value + frontier）/ WCC slot 跳过
- `init_wcc_all_vertices_kernel`：WCC per-vertex label=vertex_id，全顶点入 frontier

### Step 5 — heterogeneous push kernel（核心）

`expand_shared_node_hybrid_kernel`：双路设计

- **路径 A（BFS slot）**：`atomic_or_query_mask(visited+nbr, active & bfs_slot_mask)`
  批量标记，与现有 `expand_shared_node_kernel:637-653` 完全一致，保留 BFS 批量优化
- **路径 B（SSSP/WCC slot）**：遍历 `active & nonbfs_slot_mask` 的 set bits，
  逐 query 调 `apply_min_reduce` + `compute_candidate`
- 合并 `improved`，调 `mark_next_shared_frontier`
- launch：block=256，grid=`grid_for(unique_count, 256)`，无 shared mem

### Step 6 — run 主循环 + 端到端入口

- `fused_pull_hybrid_simple_kernel` + `launch_fused_pull_hybrid`（pull 路径 kernel，
  **已就位但未接入 run 主循环**，见 remaining_plan）
- `class hybrid_frontier_engine`：`run()` 方法，push-only 主循环
  （init → while unique_count>0 → expand → swap → 计数同步）
- `run_heterogeneous(graph, batch, ctx, options)`：顶层入口

### Step 7 — WCC 同质基线

- `algorithms/wcc.hxx`：`wcc_policy`（value_type=float，relax=identity，should_update=`a<c`）
- `frontier_engine.hxx` 集成：新增 `detail::init_wcc_labels_kernel`，
  init 段加 `if constexpr (is_same<Policy, wcc_policy>)` 分支（最小侵入，不触及 BFS/SSSP）

## 文件清单

### 新建文件（9 个）

| 路径 | 类型 | 内容 |
|------|------|------|
| `include/puercgp/algorithms/hybrid.hxx` | header | unified_value_t / algo_kind_t |
| `include/puercgp/core/reduce_ops.hxx` | header | apply_reduce / compute_candidate |
| `include/puercgp/core/query_descriptor.hxx` | header | hybrid_query_batch |
| `include/puercgp/engine/hybrid_engine.hxx` | header | 异构 kernels + hybrid_frontier_engine |
| `include/puercgp/algorithms/wcc.hxx` | header | wcc_policy |
| `examples/smoke_reduce_ops.cu` | 验证 | device 原语单元测试 |
| `examples/smoke_hybrid_init.cu` | 验证 | init kernel 状态验证 |
| `examples/smoke_hybrid_push.cu` | 验证 | 单轮 push 正确性 |
| `examples/validate_hybrid.cu` | 验证 | 端到端 BFS+SSSP+WCC 混合 |
| `examples/validate_wcc.cu` | 验证 | WCC 同质基线 |

### 修改文件

| 路径 | 改动 |
|------|------|
| `include/puercgp/puercgp.hxx` | 加 5 个新 header 的 include |
| `include/puercgp/engine/frontier_engine.hxx` | 加 wcc.hxx include + init_wcc_labels_kernel + init 段 WCC 分支 |
| `CMakeLists.txt` | foreach 加 5 个新 example target |

## 验证结果汇总

全部测试在 RTX/GPU 上实跑通过：

| 测试程序 | 检查数 | 结果 |
|---------|--------|------|
| `smoke_reduce_ops` | 13 | **ALL PASS**（apply_first_write / apply_min_reduce / compute_candidate 行为正确）|
| `smoke_hybrid_init` | 19 | **ALL PASS**（BFS/SSSP source init + WCC per-vertex init + mask 状态）|
| `smoke_hybrid_push` | 13 | **ALL PASS**（单轮 push BFS 层级 + SSSP 距离 + next_frontier 标记）|
| `validate_hybrid` | 12 | **ALL PASS**（BFS+SSSP+WCC 三方混合，3 轮收敛，全值正确）|
| `validate_wcc` | 4 | **ALL PASS**（WCC 同质 label 收敛）|
| `validate_bfs` | — | 回归正常（distance_mismatches=0）|
| `validate_sssp` | — | 回归正常（编译通过）|

**核心证据**：`validate_hybrid` 在 toy 加权图（0→1(w2), 0→2(w5), 1→3(w1)）上，
batch = [BFS src=0, SSSP src=0, WCC]，3 轮迭代后：
- BFS slot 层级 `[0,1,1,2]` ✓
- SSSP slot 距离 `[0,2,5,3]` ✓
- WCC slot label `[0,0,0,0]` ✓（全连通收敛到 min vertex id）

## 关键设计决策（回顾）

| 决策 | 选择 | 理由 |
|------|------|------|
| value 统一 | `float` | BFS 层级 < 2^24 无损；复用 SSSP 的 float CAS 路径 |
| 同质路径保护 | 异构 kernel 全新建 | 现有 20+ 处 `if constexpr(bfs_policy)` 零改动，回归保护 |
| BFS/非BFS 双路 | host 端预算 bfs_slot_mask/nonbfs_slot_mask 作 kernel 参数 | O(1) 寄存器常量，避免 kernel 内 shared memory 拆分 |
| WCC 同构 | relax=identity，复用 min-reduce | 与 SSSP 完全同构，差异只在 init（per-vertex label）|
| slot_kinds 传递 | global memory pointer | header-only 友好，L1 命中，inner loop 外层读一次进寄存器 |
| pull 模式无双路 | BFS pull = `nb+1`，与 unweighted SSSP 同构 | 简化 fused_pull_hybrid |

## 如何运行验证

```bash
cd /home/zyl/Projects/ocgp/puercgp
cmake -S . -B build && cmake --build build
./build/smoke_reduce_ops      # device 原语
./build/smoke_hybrid_init     # init kernel
./build/smoke_hybrid_push     # 单轮 push
./build/validate_hybrid       # 端到端混合 batch（核心）
./build/validate_wcc          # WCC 同质基线
```
