# Hybrid Batch 待完成工作

> **最后更新：2026-06-14**
>
> 对应实施计划 `/home/zyl/.claude/plans/sunny-dancing-fog.md` 中未完成部分。
> 已完成内容见 `../completed_plan/hybrid_batch_implementation.md`。
> 本文档按优先级排序，每项给出现状、目标、实施要点。

## 现状概述

Hybrid batch 的**功能核心已完成并端到端验证**：`run_heterogeneous` 能在
push 模式下跑通 BFS+SSSP+WCC 混合 batch（`validate_hybrid` ALL PASS）。

未完成项集中在三方面：**pull 路径接入**、**性能优化**、**真实图验证**。
均不影响 hybrid 功能正确性，是完整度与性能的提升。

---

## P0：真实图验证（最高优先级）

### 现状
所有验证（`validate_hybrid` 等）用 toy 4 顶点图。未在真实图（cit-Patents、
soc-orkut 等）上验证正确性。

### 目标
把 `validate_hybrid` 接入现有 CSR bin / mtx loader，在真实图上跑混合 batch，
确认大图、多 query 场景正确且不崩溃。

### 实施要点
1. 复用 `examples/validate_bfs.cu` 的 `load_graph_auto` 或 `load_matrix_market`
2. CLI：`validate_hybrid <graph> <bfs_srcs> <sssp_srcs> <wcc_count> [repeats]`
3. CPU reference：
   - `cpu_bfs`（queue-based，复用 validate_bfs 的）
   - `cpu_sssp`（Bellman-Ford/Dijkstra，复用 validate_sssp 的）
   - `cpu_wcc_lp`（LP 迭代 + label 归一化到 min vertex id，**需新写**）
4. 比对方式按 algo_kind：BFS 取整、SSSP float epsilon（建议 1e-3，因 unified float）、
   WCC label 归一化后比对
5. 数据集优先级：cit-Patents（小）→ soc-LiveJournal1（中）→ soc-orkut（大）

### 风险
- WCC(LP) 在大图收敛轮次可能爆炸（path graph O(V) 轮），强制 `max_iterations` 上限
- unified float 的 SSSP 距离在长路径累加误差，需放宽 epsilon 或记录误差分布

### 工作量
3-4 天（含 cpu_wcc_lp 实现与多数据集调试）

---

## P1：fused_pull_hybrid 接入 run 主循环

### 现状
`fused_pull_hybrid_simple_kernel` + `launch_fused_pull_hybrid` 已实现并编译通过
（`engine/hybrid_engine.hxx`），但 `hybrid_frontier_engine::run()` 是 **push-only**，
没有调用 pull kernel。

### 目标
让 `run_heterogeneous` 支持 `traversal_mode = pull / hybrid`，复用现有 frontier
密度判定逻辑（`pull_frontier_ratio` / `pull_edge_ratio`）自动切换 push/pull。

### 实施要点
1. **frontier 表示转换**：push 用 shared（frontier_vertices + frontier_mask），
   pull 输出 next_frontier_mask + per-vertex unique_flags/pair_counts。需要：
   - `compact_shared_pull_frontier_kernel` 的 hybrid 版（从 per-vertex mask 重建
     frontier_vertices list + next_unique_count）
   - inclusive_scan 聚合 unique_flags（复用现有 `thrust::inclusive_scan` 模式）
2. **主循环决策**：每轮根据 `current_unique_count` vs `pull_frontier_threshold`
   决定走 push 还是 pull；pull 后做 frontier 表示转换回 shared
3. **visited_mask 一致性**：pull kernel 设 `visited_mask |= improved`，
   push BFS 用 visited 判重——两者共享同一 visited_mask（与现有 frontier_engine 一致）
4. **smem pull 变体**（query_count 33-64）：当前只实现 simple 版（≤32），
   需补 `fused_pull_hybrid_smem_kernel`（参照 `fused_pull_smem_kernel:1747-1848`）

### 参考代码
- 现有 pull 分派与 compact：`frontier_engine.hxx:2646-2820`
- `launch_fused_pull`：`frontier_engine.hxx:1850-1878`
- `compact_shared_pull_frontier_kernel`：`frontier_engine.hxx:1880+`

### 工作量
5-6 天（frontier 表示转换是主循环最复杂的部分）

---

## P2：性能对比 benchmark（`bench_hybrid.cu`）

### 现状
无性能数据。不知道 hybrid batch 相比"3 个同质 batch 串行执行"是赢是输，
也不知道相比 iBFS baseline 表现如何。

### 目标
产出 ICDE 论文 Experiment 4（异构 query 分析）的核心数据：
- hybrid vs N× sequential 同质执行的吞吐/时间对比
- 不同算法组合（BFS+SSSP / BFS+WCC / 三方混合）的加速比
- shared traversal ratio（hybrid 共享了多少图遍历）

### 实施要点
1. CLI：`bench_hybrid <graph> <config>`，config 描述混合 batch 组成
2. 三组对比：
   - sequential：`run<bfs_policy>` + `run<sssp_policy>` + `run<wcc_policy>` 串行
   - hybrid：`run_heterogeneous` 一次混合 batch
   - 理想上界：单算法 N× 同质 batch（如 N× BFS）
3. 指标：wall_ms / gpu_ms / kernel_ms / iterations，取 7 次中位数
4. 输出 CSV，便于后续画图

### 工作量
2-3 天

### 风险
- 如果 hybrid 输给 sequential（因双路开销 + WCC 长尾），需要回头优化
  （见 P3 warp hybrid）。这是论文 Go/No-Go 的关键数据点
  （`FRAMING_SINGLE_GPU_CONCURRENT.md:701-713`）

---

## P3：expand_shared_node_warp_hybrid_kernel（性能优化）

### 现状
异构 push 只有 block 版（`expand_shared_node_hybrid_kernel`）。
现有同质 SSSP 默认 fallback 走 warp 版（`expand_shared_node_warp_kernel:787-874`），
性能更好。hybrid 缺对应版本。

### 目标
提供 warp 级异构 push kernel，作为 block 版的性能优化补充。

### 实施要点
1. 参照 `expand_shared_node_warp_kernel:787-874` 结构
2. update 段套用 Step 5 的双路设计（BFS 批量 + 非 BFS apply_min_reduce）
3. launch：warp-per-vertex，`grid_for(unique_count * 32, threads)`
4. 在 `hybrid_frontier_engine::run` 加 push_strategy 分派
   （shared_node / shared_node_warp）

### 工作量
2-3 天

---

## P4：完整验证矩阵

### 现状
验证矩阵（plan 定义）只覆盖了 BFS+SSSP+WCC 三方混合这一个点。

### 目标
覆盖 plan 验证矩阵的全部组合 × 多 Q × 多数据集。

### 待补组合
| 组合 | 状态 |
|------|------|
| {BFS} × {SSSP} × {WCC} 单算法回归 | WCC✓（validate_wcc）；BFS/SSSP 同质已有 validate_bfs/sssp |
| BFS+SSSP | validate_hybrid 含（但 toy 图）|
| BFS+WCC | validate_hybrid 含 |
| SSSP+WCC | validate_hybrid 含（验证 apply_min_reduce 通用性）|
| BFS+SSSP+WCC | validate_hybrid 含 |
| Q ∈ {4, 16, 32, 48, 64} 扩展 | ❌ 当前 toy 图 Q=3 |

### 实施要点
1. 扩展 validate_hybrid 支持参数化 Q 和算法组合
2. 在真实图上跑全矩阵
3. 与同质 validate_bfs/sssp/wcc 交叉验证（同算法同 source 结果一致）

### 工作量
2 天（依赖 P0 真实图 loader）

---

## P5：代码清理与小改进

### 5.1 `validate_hybrid.cu` 未使用变量
`const unified_value_t INF = unified_infinity();` 声明未用（编译 warning）。
删除或用于不可达顶点的 INF 比对。

### 5.2 pull kernel 的 smem 变体
`launch_fused_pull_hybrid` 对 query_count > 32 抛异常（"smem variant TODO"）。
需补 `fused_pull_hybrid_smem_kernel`（P1 的一部分）。

### 5.3 unified float 精度分析
论文需报告 SSSP 距离的 float 精度。建议：
- 在真实图上测 max relative error 分布
- 若超阈值，提供 `unified_value_t = double` 的 typedef 开关（一行改动 + 寄存器压力评估）

### 5.4 hybrid_query_batch 的 device 缓存 mutable
当前 `d_sources_` 等用 mutable 支持 const `upload_to_device`。
可考虑改为每次返回临时 device_vector（RAII），但会增加拷贝。当前方案可接受。

---

## 后续 Roadmap（ICDE 投稿视角）

按 `FRAMING_SINGLE_GPU_CONCURRENT.md` 的 Minimum Bar（第 601-613 行），
当前满足情况：

| Bar | 状态 |
|-----|------|
| 1. 端到端吞吐 vs 串行/naive streams | ❌ 需 P2 bench |
| 2. 强 baseline（blind batching + SpMM/GNN）| ❌ 未对比 |
| 3. 异构 workload 增益 | ⚠️ 功能就位（P0 真实图验证），性能数据缺（P2）|
| 4. 兼容性分组/fallback | ❌ 未来 Feature（不在本 plan）|
| 5. push/pull 差异处理 | ⚠️ push✓ pull 待 P1 |
| 6. latency/fairness | ❌ 未来 Feature（replenishment）|
| 7. break-even | ❌ 未来 |

**建议执行顺序**：P0（真实图正确性）→ P2（性能数据，决定 Go/No-Go）→ P1（pull 接入）→
P3（warp 优化）→ P4（完整矩阵）。

P0 和 P2 是 ICDE 投稿的前置：P0 证明大图正确，P2 证明有性能收益。
两者完成后可判断 hybrid batch 这个 contribution 是否足够支撑论文。
