# PuerCGP 论文详细叙事与章节设计

更新时间：2026-07-18

## 1. 论文定位

### 1.1 推荐的一句话主张

> PuerCGP treats concurrent graph traversals as a multi-query execution plan and reduces their effective GPU memory-access cost through topology sharing, query-dimensional coalescing, and online cost-aware coordination.

中文含义：PuerCGP 不把并发 query 当成若干独立 GPU 任务，而是构造一个多查询执行计划，通过共享 topology access、沿 query 维合并状态访问，以及在线代价驱动的调度，减少总的有效 GPU 访存代价。

这里建议使用 **effective memory-access cost** 或 **exposed memory-stall cost**，不要简单写成“降低 DRAM latency”。shared push 减少访问次数，并没有改变一次 DRAM 请求的物理延迟；query-parallel pull 通过 coalescing 减少事务并摊销延迟。最终应证明的是总 memory stall、transaction 数量或单位有效工作对应的访存代价下降。

### 1.2 推荐标题候选

1. `PuerCGP: Data-Movement-Aware Concurrent Graph Traversal on GPUs`
2. `PuerCGP: Cost-Aware Shared Execution for Concurrent GPU Graph Queries`
3. `Sharing, Coalescing, and Aligning Concurrent Graph Queries on GPUs`

第 2 个标题最接近数据库的 multi-query optimization 语境；第 1 个更偏系统；第 3 个直接但容易与 Glign 的 alignment 叙事发生正面重叠。

### 1.3 建议冻结的 scope

主文建议聚焦：

- 静态、单 GPU、in-memory graph。
- 同一图上的 concurrent source-based iterative traversal queries。
- BFS 和 SSSP 至少都进入主实验。
- batch capacity `Q <= 64`，总 query 数 `N` 可以大于 `Q`。
- closed-window workload 为主：一个 scheduling window 内已有 `N` 个 query，系统在线构造 batch 和 offset；“在线”表示不运行 query trace oracle，而非任意到达流。
- 如果要声称 online service，再增加 open-loop arrivals、queueing latency 和 fairness 实验。

建议不进入主文：

- replenish：当前没有稳定吞吐或 latency 收益。
- pause/resume：多数图回退。
- query-level push/pull concurrent SM partition：当前最佳结果仍劣于 all-pull。
- `Q*V` layout：pull 慢 7.0x-32.8x，可作为设计选择消融。
- WCC：不是 source-specific query，且会使 offset 叙事复杂化；可作为 framework extensibility 的 appendix。
- PageRank：当前 dense engine 未完成，除非后续实现和验证，否则不要列为支持算法。

### 1.4 论文贡献组织

推荐四项贡献，其中前两项是技术核心：

1. **Characterization and formulation**：揭示 GPU concurrent traversal 的两个互补 memory opportunity，并建立统一的 multi-query cost formulation。
2. **Multi-query physical operators**：设计 shared push 和 query-dimensional coalesced pull，在相同 query-state layout 上执行并共享 frontier semantics。
3. **Online multi-query optimizer**：用 graph summary 和轻量运行时统计做 batch formation、fixed-offset alignment 与 per-iteration push/pull selection。
4. **System and evaluation**：实现可扩展的 GPU engine，在 BFS/SSSP、不同图结构、Q/N 和 source distribution 下证明吞吐、latency、memory behavior 和 optimizer robustness。

如果最终 SSSP 和 heterogeneous workload 结果不足，应将第 4 项改成 BFS-specific system，避免“general CGAQ engine”的过度声明。

## 2. 统一问题模型

### 2.1 基本定义

给定静态图 `G=(V,E)` 和 query 集合 `B={q_0,...,q_{Q-1}}`。query `q` 在本地迭代 `l` 的 frontier 为 `F_q^l`。调度 offset `o_q` 将本地迭代映射到 global step：

```text
l(q,t) = t - o_q
```

global step `t` 的 union frontier：

```text
U_t = union_q F_q^{l(q,t)}
```

每个 `v in U_t` 带 query mask：

```text
M_t(v) = { q | v in F_q^{l(q,t)} }
```

### 2.2 区分三类工作量

1. **Independent edge work**：

```text
E_ind(t) = sum_q sum_{v in F_q} degree(v)
```

2. **Shared topology work**：

```text
E_union(t) = sum_{v in U_t} degree(v)
```

3. **Query-state pair work**：

```text
E_pair(t) = sum_{v in U_t} degree(v) * popcount(M_t(v))
```

`E_union` 表示 adjacency list 至少需要被读取多少次，`E_pair` 表示仍需执行多少 per-query candidate/update。只报告 frontier vertex overlap 不够，因为共享高出度 vertex 比共享低出度 vertex 更重要。

### 2.3 两个物理算子的统一 cost

可将每轮 measured/predicted cost 写成：

```text
C_push(t) = a_u * E_union(t)
          + a_p * E_pair(t)
          + a_f * |U_t|
          + a_a * atomic_updates(t)

C_pull(t) = b_e * neighbor_probes(t, Q_active)
          + b_s * state_transactions(t, Q_active, layout)
          + b_v * vertex_scan(t)
          + b_c * compact/postprocess(t)
```

含义：

- shared push 的主要优势是 `E_union << E_ind` 时只展开一次 adjacency list。
- pull 可能增加逻辑 probe/scan，但 `V*Q` layout 和 query-level thread mapping 使同一 neighbor 的多个 query state 连续访问，降低 transaction cost。
- hybrid 不是“稀疏用 push、稠密用 pull”两个独立故事，而是在每轮选择 `min(C_push,C_pull)` 的物理执行计划。
- batch formation 和 offset 改变 `U_t`、`M_t(v)` 以及 active query density，因此会改变上述 cost。

第一版模型可以使用线性系数，系数由 microbenchmark/online calibration 得到。论文必须比较：传统 frontier-size threshold、当前 virtual-edge threshold、拟合模型和 per-step oracle。

### 2.4 全局优化目标

对于 `N > Q`，需要同时决定 batch partition `P`、query offsets `O` 和每轮 operator `D_t`：

```text
minimize  sum_{B in P} sum_t C_{D_t}(B, O, t) + C_plan + C_post
subject to |B| <= Q, 0 <= o_q <= O_max
```

不需要声称求全局最优。论文应说明精确 frontier 在执行前未知，PuerCGP 用 graph summary 估计 affinity，再用运行时 frontier metrics 做 operator selection。

## 3. Abstract 需要表达的内容

建议按五句话组织：

1. 现实问题：许多图分析应用会在同一图上产生批量或并发 source-based queries。
2. 关键问题：独立 GPU 执行重复发起不规则 topology/state access；已有联合执行没有统一处理共享数量、transaction efficiency 和 traversal alignment。
3. 核心 insight：query dimension 既能共享 adjacency access，也能形成 coalesced state access，但两者的最佳物理算子随 batch 和迭代变化。
4. 系统：PuerCGP 提供 shared-push、query-parallel pull 和 online cost-aware optimizer。
5. 结果：最后只写完整实验后的 geometric mean、最大收益、planner overhead 和适用范围，不能使用当前局部实验拼接出的数字。

Abstract 不应出现：

- “GPU utilization is low”作为唯一动机。
- “first concurrent GPU graph processing system”。
- 只报告相对旧 scheduler/refactor baseline 的 2.47x。

## 4. Introduction

### 4.1 Paragraph 1：从真实的 multi-query graph workload 开场

重点内容：

- concurrent query 不是为了人为填满 GPU，而是上层任务本身会生成多个 source query。
- 例子优先使用完整应用：betweenness/closeness centrality 中的大量 BFS、landmark labeling 中的批量 BFS/SSSP、route/path analysis 中的多源查询。
- 引用 ForkGraph/KGraph/iBFS 的应用和 trace；若能获得自己的真实或公开 query trace，放一张 arrival/source distribution 图。

需要的证据：

- 至少一个真实应用产生的 query batch，而非仅随机 source。
- workload 规模：每轮/每分钟 query 数、batch size、query 类型。
- 如果没有生产 trace，明确写成 application-generated batched workload，不要伪装成在线服务 trace。

### 4.2 Paragraph 2：指出 GPU 上的核心瓶颈

重点内容：

- graph traversal 的不规则 adjacency/state access 造成较高 long-latency memory stall。
- query-at-a-time 重复扫描相同 topology；naive multi-stream concurrency 让独立 kernel 竞争 cache/HBM，不能直接共享访问。
- concurrency 的价值不是额外 parallelism 本身，而是暴露了 query dimension，可让执行计划跨 query 合并访问。

需要的实验：Motivation Figure 1。

| Panel | 对比 | 指标 |
|---|---|---|
| (a) | sequential single-query vs naive multi-stream vs fused batch | batch time、query/s |
| (b) | 三者 memory behavior | long-scoreboard stall、DRAM/L2 read sectors、global transactions |
| (c) | 相同 query 的重复 topology work | `E_ind`、`E_union`、sharing factor |

### 4.3 Paragraph 3：统一两个 memory opportunity

重点内容：

- **Spatial/topology sharing**：多个 query 在同一 global step 激活同一 vertex，可只读取一次 adjacency list，并用 query mask 传播多个状态。
- **Query-dimensional coalescing**：当 push 的 query-state/random update 代价上升时，按 vertex 固定、沿 query 维映射 thread，使 warp 访问 `values[v,q:q+31]` 的连续地址。
- 这两个机会都减少 effective memory cost：一个减少请求数量，一个提高每个 transaction 的有效载荷。

需要的实验：Motivation Figure 2。

- push：`E_ind/E_union` 与 measured speedup 的相关性。
- pull：`V*Q` vs `Q*V`、不同 thread mapping 的 sectors/request 和 kernel time。
- 不能只放 latency；要用 counter 证明 mechanism。

### 4.4 Paragraph 4：为什么单一算子仍不够

重点内容：

- shared push 的收益依赖 degree-weighted frontier overlap；query-state pair work 和 atomic contention 仍然存在。
- query-parallel pull 具有规则/coalesced access，但可能扫描更多 vertex/edge。
- query 的 source、graph structure 和迭代位置共同决定两者相对成本。
- 因而需要一个 multi-query physical-plan selector，而不是固定 frontier-size threshold。

需要的实验：

- 选 3-4 个真实 run，逐 iteration 画 `C_push measured`、`C_pull measured`、传统阈值选择和 oracle 选择。
- 展示同一 frontier size 但 degree/mask density 不同导致最佳 operator 不同的 case。

### 4.5 Paragraph 5：为什么还需要 query coordination

重点内容：

- 即使 physical operator 支持 sharing，FIFO grouping 和全部 query 同时启动也可能让相似 frontier 出现在不同 global steps。
- batch formation 决定“谁共享”，fixed offset 决定“何时共享”。
- 精确 frontier trace 才能得到 oracle，但运行前不可知；PuerCGP 使用 graph preprocessing summary 和 source lookup 在线估计。

需要的实验：

- zero-offset、offline oracle、Glign-style heuristic、PuerCGP online estimator 的 union edges 和 runtime。
- evaluator time 必须包含在端到端时间中。
- 索引构建时间和空间单独报告。

### 4.6 Paragraph 6：系统和贡献

贡献条目建议：

1. 首次系统化刻画/建模 GPU concurrent graph traversal 中 topology sharing 与 query-dimensional coalescing 的联合 trade-off。这里“首次”必须在完成 iBFS/LCCG 对照后再决定是否保留。
2. 设计可组合的 shared-push 和 coalesced-pull physical operators，并保持统一 frontier/value representation。
3. 设计 online multi-query optimizer，联合 batch/offset planning 与 runtime direction selection。
4. 在多算法、多图、不同 query 数和 source workload 上进行完整评估。

## 5. Preliminary and Motivation

推荐章节名：`Background and Motivation`，而不是只写 `Preliminaries`。

### 5.1 Concurrent source-based graph queries

表达内容：

- 定义 graph、query、batch window、`N` 与 capacity `Q`。
- 区分 homogeneous query batch 和 mixed BFS/SSSP batch。
- 说明每个 query 语义独立，系统优化不能改变 query 结果，只改变共享执行顺序。
- 说明 closed-window 和真正 streaming arrival 的区别。

建议图表：应用到 query batch 的示意图；一张 workload table。

### 5.2 GPU graph traversal and data layout

表达内容：

- CSR/incoming CSR。
- push 和 pull 的语义，不按“两个独立阶段”展开，而是两种等价 physical operators。
- `values[V][Q]`、`visited_mask[V]`、shared frontier list、per-vertex query mask。
- `Q<=64` 来自一个 `uint64_t` mask；这既是性能设计也是当前限制。

### 5.3 Why independent execution wastes memory work

表达内容：

- 定义 `E_ind`、`E_union`、`E_pair`。
- 证明/说明 shared push 对 adjacency read 的上界。
- 给出多图、多 source seed 的 overlap distribution，不能只挑 twitter。

需要的实验：

- `Q=8/16/32/64` 的 sharing factor 分布。
- uniform/random、degree-stratified、localized/hotspot source。
- 每个 graph 至少 5 个 source seeds。

### 5.4 Why memory transaction efficiency matters

表达内容：

- 同一个 neighbor 下 query value 连续，因此 warp 沿 Q 维访问可 coalesce。
- `Q*V` 虽便于单 slot save/reset，却破坏 pull 的连续访问；当前 microbenchmark 7.0x-32.8x 回退是初步证据。
- 编译器不会自动把跨 query 的随机访问变成该 mapping；它来自显式 thread/data layout 设计。

需要的实验：

- `V*Q` vs `Q*V`。
- vertex-parallel/query-serial vs vertex-query 2D mapping。
- Q 不满 warp 时的效率：Q=4/8/16/32/48/64。
- transaction/sector、L2 hit、DRAM bytes、stall reason。

### 5.5 Why scheduling changes sharing

表达内容：

- frontier overlap 是 `(query, local iteration)` 的函数。
- lockstep 只对齐 iteration id，不保证访问相同 graph region。
- fixed offset 可以改变 union frontier，而不改变任何 query 的内部迭代顺序和结果。
- pause/resume 搜索空间大且当前效果差，因此本系统只采用 bounded fixed offset。

需要的实验：

- N=Q=32 的 zero/heavy/offline-offset 对比作为 opportunity study。
- N>Q 的 FIFO/selective batch/offline oracle。
- 这些 oracle 实验只能证明 headroom，不能作为最终在线收益。

### 5.6 Design goals

列出四个目标：

1. 减少 topology memory accesses。
2. 提高 query-state transaction efficiency。
3. 在线选择低 cost plan，planner overhead 可控。
4. 保持 exact per-query semantics 和可扩展算法接口。

同时列出 non-goals：dynamic graph、multi-GPU、general graph pattern query、dense PageRank、immediate replenishment。

## 6. System Overview

### 6.1 Architecture figure

建议画四层：

```text
Graph preprocessing
  CSR + reverse CSR -> graph summary / landmark phase index

Query admission
  source + algorithm descriptor -> scheduling window

Multi-query optimizer
  batch formation -> fixed offsets -> per-step push/pull selector

GPU execution engine
  shared frontier/value matrix -> shared push | coalesced pull -> postprocess
```

图中明确 offline 与 online 边界，以及计入 end-to-end latency 的组件。

### 6.2 Query lifecycle

按以下顺序描述一次运行：

1. 加载 graph/index。
2. 收集一个 `N`-query window。
3. planner 生成容量 `Q` 的 batches 和 per-query offset。
4. 初始化 `V*Q` value matrix、visited masks 和 shared frontier。
5. 每个 global step 获取 frontier metrics，选择 push/pull。
6. operator 生成 `next_frontier_mask`；pull postprocess 生成 compact shared frontier。
7. 所有 query 完成后返回每 query result 和 profile。

### 6.3 Programming interface

表达内容：

- `algorithm_traits` 提供 infinity、source initialization、candidate computation 和 update semantics。
- homogeneous fast path 与 heterogeneous descriptor path 的关系。
- 支持范围必须与实验一致；PageRank placeholder 不应出现在支持列表。

### 6.4 Correctness invariants

需要写清楚：

- per-query visited/value state 独立。
- query mask 只合并 topology traversal，不合并 query result。
- fixed offset 只延迟启动，不改变 local iteration order。
- push/pull 在同一算法下实现相同 relaxation/first-visit 语义。
- atomic OR 返回旧 mask，保证同一 query 对 neighbor 的首次发现唯一。

## 7. Method 1：Multi-Query GPU Physical Operators

推荐章节名：`Shared and Coalesced GPU Execution`。

### 7.1 Shared frontier representation

表达内容：

- `frontier_vertices[0:unique_count]` 保存 union frontier。
- `frontier_mask[v]` 或 compact mask 保存激活该 vertex 的 query。
- 同一 vertex 只进入 shared frontier 一次，mask 表示多个 query。
- 用 `next_unique_count` 和 `next_pair_count` 区分 topology size 与 query-pair size。

建议图：4 个 query 的 frontier 合并示例，标出 `E_ind` 与 `E_union`。

### 7.2 Shared push operator

表达内容：

- 一个 warp/CTA 取得一个 shared frontier vertex。
- 邻接表只遍历一次；active mask 驱动不同 query 的 candidate/update。
- BFS 用 mask-wide atomic first-visit；SSSP 用 per-query min reduction。
- warp variant 与 query-parallel experimental variant 的选择依据。
- 原子操作、mask popcount 和高 degree load balance 的成本。

必须提供：

- 伪代码，不直接贴 CUDA 实现。
- work/traffic 分析：topology `O(E_union)`，state work `O(E_pair)`。
- 相对 iBFS joint traversal 的差异表。

### 7.3 Query-dimensional coalesced pull operator

表达内容：

- 2D block/warp mapping：固定 vertex，lane/thread 对应 query slot。
- 所有 lane 沿相同 neighbor 序列前进，访问 `values[neighbor * Q + q]`。
- adjacency index 可由 warp broadcast/cache 共享；query values 在 Q 维连续。
- early exit、active slot mask 和 query count 尾部处理。
- `simple` 与 `smem` kernel 的选择；当前结果显示 Q<=64 下 simple 更快，但应在完整矩阵后决定是否保留 smem 描述。

必须提供：

- thread-to-data mapping 图。
- memory address 示例。
- 与 iBFS bottom-up 的逐项差异：状态编码、thread mapping、early termination、transaction 行为。

### 7.4 Frontier postprocessing

表达内容：

- pull 为每 vertex 直接得到 `next_frontier_mask` 和 `unique_flag`。
- scan/compact 将 flags 转成下一轮 shared frontier list。
- 为什么 push 可在更新时 append，而 pull 更适合统一 compact。
- postprocess 的成本必须进入 pull cost 和 runtime breakdown。

### 7.5 Algorithm abstraction

表达内容：

- `algorithm_traits` 的静态信息和 device candidate/update。
- BFS first-write、SSSP min-reduce 的差异。
- mixed algorithm 如果进入主文，需要说明 query tag 分支和统一 value type 的成本。

### 7.6 Operator correctness

建议给一个简短定理/引理：

- 合并 adjacency traversal 不改变各 query 的可见 candidate 集合。
- atomic update 保证每 query 的 update semantics 与独立执行一致。
- push/pull 产生相同的 next frontier/value fixed point。

不必做复杂理论，但必须有清晰不变量和 CPU reference 验证。

## 8. Method 2：Online Cost-Aware Multi-Query Optimizer

推荐章节名：`Online Multi-Query Planning`。

### 8.1 Optimizer inputs and outputs

输入：

- 静态 graph summary/index。
- query source 和 algorithm kind。
- batch capacity `Q`、最大 offset。
- runtime frontier metrics：`unique vertices`、`actual edges`、`virtual edges`、`active pairs`。

输出：

- query-to-batch assignment。
- per-query fixed offset。
- 每轮 push/pull physical operator。

### 8.2 Graph preprocessing

当前实现：

- 最多 4 个 farthest-point landmarks 估计 phase length。
- 其余 landmark 通过随机 edge source 采样，近似 degree-proportional vertex sampling。
- 从 landmark 做 BFS，使用 landmark-major `uint16_t` 距离矩阵。
- 时间 `O(L(V+E))`，空间约 `2LV` bytes。

论文需要补充：

- 与 Glign top-high-degree BFS index 的区别。
- landmark count/精度/空间 trade-off。
- directed graph 使用原图还是 reverse graph，并与 query traversal 方向严格对应。
- `uint16_t` overflow/unreachable 语义。
- static graph index 的复用和更新限制。

### 8.3 Online affinity estimation and batching

当前思路：

- 对 source pair 构造 landmark relative-level difference histogram。
- 直方图最大 bucket 近似两 query 通过 offset 可达到的 alignment affinity。
- 贪心形成 capacity-Q batches，再做有限 swap。

论文增强建议：

- affinity 应预测 degree-weighted `E_union` reduction，而不是只计 landmark 数。
- score 应区分 topology bytes 和 query-state cost。
- 加 admission-order/fairness constraint，避免低 affinity query 无限等待。
- 与 FIFO、random、iBFS GroupBy、Glign closest-HV、offline trace oracle 比较。

### 8.4 Fixed-offset planning

表达内容：

- completion/heavy-phase estimate 给出初值。
- pairwise relative-level histogram 给出 score。
- bounded coordinate ascent 搜索 `[0,O_max]`。
- query 启动后连续执行，避免 pause/resume 的 frontier carry 和额外 global steps。

需要报告：

- 复杂度与 planner latency。
- predicted score 与实际 union-edge saving 的 Pearson/Spearman correlation。
- online schedule 到 offline oracle 的 gap。
- offset 对 per-query p50/p95 latency 的影响。

### 8.5 Per-iteration operator selection

当前已实现的 homogeneous heuristic：

```text
virtual_edge_count >= pull_edge_ratio * E * Q
```

论文版需要从 threshold 升级为可解释 cost model：

- push features：`E_union`、`E_pair`、mask density、degree skew、estimated atomics。
- pull features：remaining/unvisited pairs、active slots、expected probes、Q occupancy、postprocess cost。
- 模型可以是离线校准的线性模型或极小 lookup table；不必强行使用 ML predictor。
- 每次决策开销必须低于 iteration time 的小比例。

必须比较：

- all-push。
- all-pull。
- Beamer/传统 frontier threshold。
- virtual-edge threshold。
- proposed cost model。
- measured per-iteration oracle。

### 8.6 End-to-end planning

强调顺序：

```text
graph summary
 -> online batch formation
 -> fixed offsets
 -> runtime push/pull selection
 -> shared/coalesced execution
```

当前 online offset benchmark 仍是 all-push wrapper；在论文中声称 end-to-end optimizer 前，必须真正接入 hybrid engine 并将 planner、初始化、postprocess 和 result handling 都计时。

## 9. Implementation

建议只写影响设计的实现事实：

- CUDA/C++ header-only 或 library 结构。
- query mask 为 64-bit，因此最大 Q=64。
- `V*Q` value matrix 与 per-vertex mask。
- shared frontier double buffering。
- incoming/outgoing CSR。
- pull scan/compact implementation。
- stream 和 execution context。
- profile counters 和 planner index format。

不要在正文大量列文件名或类结构；这些放 artifact 文档。

需要新增的实现信息：

- 代码行数、CUDA 版本、编译 flags。
- device memory footprint 公式。
- index memory 和 engine workspace memory。
- Q=64 时最大可处理 graph 规模。

## 10. Experiments

### 10.1 Research questions

| RQ | 问题 | 主要结果 |
|---|---|---|
| RQ1 | PuerCGP 是否提高端到端吞吐并控制 query latency？ | external baseline comparison |
| RQ2 | 收益是否真的来自更少/更高效的 memory access？ | Nsight counters + traffic model |
| RQ3 | shared push、coalesced pull 和 hybrid 各贡献多少？ | operator ablation |
| RQ4 | online batch/offset planner 是否准确且代价可控？ | oracle gap、overhead、correlation |
| RQ5 | 系统对 Q、N、graph structure 和 source distribution 是否稳健？ | scalability/sensitivity |
| RQ6 | 设计是否能泛化到 BFS 之外？ | SSSP 和 mixed workload |
| RQ7 | preprocessing、memory footprint 和调度延迟是否可接受？ | system overhead |

### 10.2 Platform

最低要求：

- 当前 V100-SXM2-32GB 作为主平台。
- 再增加至少一张更新架构 GPU，例如 A100/H100/RTX 4090 中可获得的一张。
- 记录 CPU、内存、GPU、SM 数、HBM、CUDA、driver、clock policy。
- 锁定 GPU clocks 或至少报告温度/频率，避免误差。

### 10.3 Datasets

建议至少 8 张图，覆盖：

- citation/web：cit-Patents、indochina。
- social：LiveJournal、Orkut、Twitter、SinaWeibo。
- long-diameter/low-degree：roadNet-CA，作为 robustness/negative case。
- synthetic RMAT/Kronecker：控制 degree skew、diameter 和 scale。

每张图报告 `|V|`、`|E|`、directed/undirected、average/max degree、有效直径、连通分量、CSR bytes。

### 10.4 Query workloads

至少包含：

- `Q = 8,16,32,64`。
- `N = Q, 4Q, 16Q`；资源允许时增加 512/1024 queries。
- 每个配置至少 5 组不同 source seed。
- source distributions：uniform、degree-stratified、localized/hotspot、application-derived。
- BFS 和 weighted SSSP；权重生成/来源必须固定和公开。
- 如果 mixed workload 进入主文，测试 BFS:SSSP 为 1:3、1:1、3:1。

### 10.5 Baselines

GPU 必选：

1. 单 query 高性能 GPU baseline 顺序执行，例如 Gunrock。
2. naive multi-stream/independent kernels，证明并发本身不等于共享。
3. [iBFS](https://doi.org/10.1145/2882903.2882959)，最重要的 direct baseline。
4. 当前 GE-SpMM/TCRGraph pull baseline，隔离 query-parallel pull 的来源。
5. PuerCGP all-push、all-pull、hybrid、hybrid+online planning。

CPU context baselines：

- ForkGraph、Glign；Krill/KGraph 若 artifact 可复现则加入。
- CPU/GPU 跨硬件比较不能作为唯一 speedup。报告硬件、线程数和功耗/成本背景，并以同设备 GPU baseline 为主结论。

Scheduler baselines：

- FIFO/random batches + zero offset。
- iBFS GroupBy。
- Glign closest-high-degree batching/delayed start。
- offline trace oracle。
- PuerCGP online planner。

### 10.6 Metrics

用户级指标：

- total makespan、queries/s。
- per-query p50/p95/p99 latency；offset 等待计入 latency。
- speedup geometric mean、min/max 和置信区间。
- correctness mismatches。

机制指标：

- `E_ind`、`E_union`、`E_pair`、sharing factor。
- actual neighbor probes、atomic attempts/successes。
- DRAM bytes、L2 sectors/hit rate、global sectors/request。
- long-scoreboard memory stall、achieved bandwidth。
- pull postprocess、planner、init/result handling breakdown。

系统开销：

- graph index build time/size。
- online evaluator latency。
- GPU peak memory。
- cost-model decision overhead。

### 10.7 RQ1：端到端性能

图表：

- 主图：每个 dataset/algorithm 的 throughput speedup，所有数均包含 planner 和 GPU execution。
- 表：PuerCGP 相对 iBFS/Gunrock/naive stream 的 geometric mean、最差和最好结果。
- latency CDF 或 p50/p95 table。

通过标准：

- 不是只在 twitter/indochina 有收益。
- 对主要 GPU baseline 的 geometric mean 有明显提升，最差 case 回退可解释且受控。
- 至少 BFS 和 SSSP 中一个强、另一个不明显退化；否则缩小论文 scope。

### 10.8 RQ2：memory mechanism

图表：

- push：`E_union` reduction 对 kernel speedup scatter plot。
- pull：transactions/edge、sectors/request、stall reduction。
- end-to-end：总 DRAM/L2 traffic 和 long-scoreboard stall。

关键要求：用同一 query set 比较，不能只比较不同 source selection。

### 10.9 RQ3：operator 和 optimizer 消融

逐步开启：

```text
independent execution
 -> shared frontier push
 -> query-dimensional pull
 -> runtime hybrid selector
 -> selective batching
 -> fixed offset
```

另做：

- `V*Q` vs `Q*V`。
- simple vs smem pull。
- shared-node vs warp push。
- scan/compact postprocess 占比。
- threshold selector vs cost model vs oracle。

### 10.10 RQ4：online planner

必须报告：

- online end-to-end speedup，不能排除 evaluator time。
- offline index 成本单独报告。
- online score 与 actual union-edge saving/runtime 的相关性。
- online 到 offline oracle 的 gap。
- landmark count `L=8/16/32/64/128` sensitivity。
- `O_max`、batch swap 次数、coordinate ascent 初值数量 sensitivity。
- Glign/iBFS scheduler direct comparison。

### 10.11 RQ5：scalability and robustness

- Q scaling：8 到 64。
- N scaling：Q 到 16Q。
- graph size scaling：synthetic scale。
- degree skew/diameter sensitivity。
- no-sharing adversarial workload：证明 planner 不会产生严重回退。
- long-diameter road graph 可以展示适用边界，不要删除负结果。

### 10.12 RQ6：algorithm generality

- BFS exact distance/visited count。
- SSSP exact或 epsilon reference，报告权重类型。
- mixed BFS/SSSP 的 correctness 和 throughput。
- runtime algo tag 分支开销消融。
- 若 WCC 仅为功能演示，放 appendix，不纳入主要平均 speedup。

### 10.13 统计规范

- warmup 2 次，至少 5 次正式重复，报告 median；关键结论给 95% CI。
- query source 至少 5 个 seeds；图中可报告跨 seed 分布。
- 所有策略使用完全相同的 source、graph orientation 和 preprocessing。
- 明确计时边界：graph loading/index build 是否包含，planner/init/result copy 是否包含。
- 原始 CSV、命令、commit hash 和环境信息进入 artifact。

## 11. 当前已有证据及其可用范围

| 已有结果 | 当前结论 | 论文中可否直接使用 |
|---|---|---|
| hybrid scheduler 4 图、Q=16/32/64，12/12 通过 3% gate | 重构后不劣于旧实现，部分 case 明显改善 | 只能作为 regression/engineering 证据，不是 external speedup |
| online batching+offset，N=128/Q=32 all-push BFS | 6 图均正确，3 图超过 1.4x，范围 1.152x-2.309x | 可作为 preliminary；尚未接入 hybrid，只有一个 seed |
| offline batching+offset | 4 图 1.318x-2.124x | 证明 opportunity；不能当 online system 收益 |
| V*Q vs Q*V | Q*V 慢 7.0x-32.8x | 可作为 layout 消融，需用最终 kernel 重跑和 counters |
| SM scaling | push 8-16 SM 饱和，pull 随 SM 增长 | 可作为 characterization/appendix；当前并发 partition 无收益 |
| replenish | 标准图近似持平，roadNet 下降 | 不作为贡献，可用于解释固定 cohort/window 设计 |
| 旧 external baseline 表 | 某些 Q 上胜过 iBFS，某些 Q 上更慢 | 必须用最终代码、统一 query、完整计时边界重跑 |
| heterogeneous BFS/SSSP/WCC | 功能路径存在 | 缺完整 correctness/performance matrix，不能直接声称论文级支持 |

## 12. Related Work

正文建议分为三组：

1. Concurrent graph shared execution：CGraph、GraphM、ForkGraph、Krill、Glign、KGraph。
2. GPU concurrent graph traversal：iBFS 为核心，EGraph/PMGraph 为 dynamic/streaming 邻接工作。
3. Multi-query optimization：shared scans/global query plan，把 PuerCGP 定位为 graph-specific GPU physical planning。

不要将 related work 写成“CPU 工作都不适合 GPU”。应逐项承认它们已解决的 sharing/alignment 问题，再明确 PuerCGP 的物理算子和 cost domain 不同。

## 13. Conclusion

只总结三点：

1. concurrent query dimension 是降低图遍历 memory cost 的执行机会，不只是额外并行度。
2. topology sharing 与 query-dimensional coalescing 是互补 physical plans。
3. online multi-query planning 能在不依赖完整 trace 的情况下组织和选择这些 plans。

不要在 conclusion 引入未验证的 future feature，例如 replenish、Green Context 或 CPU-GPU 协同。

## 14. 审稿人最可能的质疑

1. **“这是 iBFS + Glign 在新 GPU 上的组合。”**
   必须用 mechanism delta、GPU-specific cost model、query-parallel pull counters 和 direct baselines 回答。
2. **“只有 BFS，不是 general concurrent graph processing。”**
   主文至少补 SSSP，或把 title/scope 缩成 concurrent BFS traversal。
3. **“随机挑 source 制造 sharing。”**
   增加真实应用、多个 seeds、source distribution 和 adversarial no-sharing workload。
4. **“收益来自 offline oracle。”**
   所有主结果包含 online evaluator，并报告 oracle gap。
5. **“memory latency 只是口号。”**
   报告 stall、transactions、DRAM/L2 traffic，并将模型预测与 measured runtime 对齐。
6. **“预处理索引太大。”**
   报告 O(VL) 空间、构建时间、landmark sensitivity，并提供压缩/小 L 配置。
7. **“CPU/GPU baseline 不公平。”**
   主要结论建立在同 GPU 的 iBFS/Gunrock/naive baseline 上，CPU 只作 context。
8. **“offset 提高吞吐但牺牲 tail latency。”**
   offset 等待必须计入 per-query latency，并加入 bounded window/fairness。
9. **“hybrid selector 只是手工阈值。”**
   比较 threshold、cost model 和 per-iteration oracle，报告误选率。
10. **“结果不可复现。”**
    固化 source lists、raw CSV、commands、commit、Docker/环境和 correctness references。

## 15. 推荐的主文图表清单

| 编号 | 内容 | 支撑 claim |
|---|---|---|
| Fig. 1 | real/application query workload + naive GPU memory profile | 问题真实且 memory-stall dominated |
| Fig. 2 | `E_ind`、`E_union` 与 query-dimensional access 示意 | 两个互补机会 |
| Fig. 3 | PuerCGP architecture | offline/online/operator 边界 |
| Fig. 4 | shared push dataflow | topology sharing |
| Fig. 5 | pull thread/data mapping | query-dimensional coalescing |
| Fig. 6 | online planner and cost model | multi-query optimization |
| Fig. 7 | overall throughput | 端到端收益 |
| Fig. 8 | latency and Q/N scaling | 系统行为 |
| Fig. 9 | memory counters/traffic | 机制证明 |
| Fig. 10 | ablation + oracle gap | 每项设计必要性 |
| Table 1 | dataset/workload | 实验覆盖 |
| Table 2 | iBFS/Glign/Krill/PuerCGP capability delta | novelty 边界 |
| Table 3 | preprocessing/planner/memory overhead | practical cost |
