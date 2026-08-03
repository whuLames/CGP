# GraphWeft：面向 GPU 并发图查询的多查询执行引擎——论文设计与实验蓝图

> 文档用途：面向 SIGMOD 2027 / ICDE 2027 的论文写作、系统补全和实验排期。
>
> 更新时间：2026-08-03。本文中的性能数值均来自当前项目已有记录，不能在完成统一协议复测前直接作为投稿终稿数字。

## 0. 状态标记

- **[已实现]**：当前代码中已有相应机制。
- **[已有数据—初步]**：已有可定位的数据，但实验协议、覆盖范围或统计强度尚不足以直接投稿。
- **[已有数据—需重跑]**：历史实现或旧实验能说明趋势，但必须在最终代码和统一协议下重跑。
- **[缺失—P0]**：支撑核心 claim 的必做实验；缺失时不建议投稿。
- **[缺失—P1]**：显著增强完整性或回应审稿人的实验；时间不足时可裁剪。
- **[负结果/非主线]**：已有实验显示收益不稳定，适合作为设计取舍或附录，而不应成为核心贡献。
- **[历史数据—不可直接引用]**：存在语义、实现版本或公平性风险，只能用于制定新实验。

## 1. 投稿定位与写作约束

### 1.1 目标会议

当前优先目标为 SIGMOD 2027，ICDE 2027 可作为同层次备选。两者的 Research Paper 均允许正文最多 12 页（参考文献不计入正文页数），但 SIGMOD 使用 ACM 双栏模板并采用双盲评审，ICDE 使用 IEEE 模板并采用单盲评审。SIGMOD 要求论文在数据管理领域有明确关联，ICDE 的主题直接包含 graph data management、query processing/optimization 和 modern hardware。

- SIGMOD 2027 投稿要求：[Research Papers Call](https://2027.sigmod.org/calls_papers_sigmod_research.shtml)。Round 4 摘要截止 2026-10-10，全文截止 2026-10-17。
- ICDE 2027 投稿要求：[Submission Guidelines](https://icde2027.github.io/submission-guidelines.html) 与 [Research Papers Call](https://icde2027.github.io/cf-research-papers.html)。第二轮全文截止 2026-11-11。
- SIGMOD 可提交一个简短 appendix，但正文必须自洽；ICDE 不允许 appendix。因此所有核心设计、实验协议和主结论都应放入 12 页正文。
- 两个会议都重视 artifact。实验脚本、query sets、原始结果、正确性校验和绘图脚本应在匿名仓库中形成可复现链路。

### 1.2 数据库论文视角

论文不应被写成“为三个 CUDA kernel 做优化”，而应写成一个面向并发图查询的 **GPU multi-query execution engine**：

1. **逻辑工作负载层**：输入一批独立但结构相同的 source-based graph queries，例如 multi-source BFS、SSSP 或 SSWP。
2. **多查询计划层**：选择哪些查询组成一个 batch，并为其选择相对启动偏移，使同一时刻的活跃顶点更可能重合。
3. **物理执行层**：在每轮迭代中选择 topology-sharing expansion 或 query-dimensional coalesced gather。
4. **运行时状态层**：使用精确的 vertex–query membership mask 保留查询语义，并收集真实工作量和代价特征。

这一表述把系统贡献连接到 SIGMOD/ICDE 熟悉的 multi-query optimization、physical operator selection、cost-based optimization 和 modern hardware execution，而不是只强调 GPU 微架构技巧。

### 1.3 建议页数预算

| 章节 | 页数目标 | 内容 |
|---|---:|---|
| Introduction | 1.25–1.5 | 问题、现有方法缺口、核心 insight、贡献 |
| Background & Motivation | 1.25 | CGQ、GPU memory-stall 分析、机会量化 |
| System Overview | 0.75 | 查询生命周期、组件和统一目标 |
| System Design | 3.0–3.5 | 三个核心设计及正确性/复杂度 |
| Implementation | 0.5–0.75 | 数据布局、算法接口、kernel/运行时细节 |
| Evaluation | 3.25–3.75 | 端到端、机制、消融、扩展性、开销 |
| Related Work | 0.75–1.0 | iBFS、Glign、ForkGraph、KGraph、GPU systems |
| Conclusion | 0.2–0.3 | 一句话重申结果和适用范围 |

## 2. 命名、范围与核心 Thesis

### 2.1 系统命名

建议全文使用 **GraphWeft** 作为 working name，推荐标题为 *GraphWeft: Cost-Aware Multi-Query Execution for Concurrent Graph Processing on GPUs*。`weft` 指与经线（warp）交织的纬线：它既表示将多个查询编织进共享执行，也呼应 GPU warp，但不会把系统限制为某一种 kernel。初步检索未发现同名的图处理或数据库系统；正式投稿前仍应再次检查论文索引、代码仓库和商标。

不再使用 `CGraph`。`CGraph` 已是 USENIX ATC 2018 论文 *CGraph: A Correlations-aware Approach for Efficient Concurrent Iterative Graph Processing* 的系统名称，继续使用会造成明显的 related-work 和检索混淆：[CGraph 官方页面](https://www.usenix.org/conference/atc18/presentation/zhang-yu)。

### 2.2 明确的问题范围

本文当前最稳健的 scope 是：

- 静态、单机、单 GPU、图完全驻留显存；
- 一个 closed window 中有总计 `N` 个 source-based iterative graph queries，GPU 同时容纳 `Q` 个 active slots，当前 `Q <= 64`；
- 核心算法为 BFS 和 SSSP；SSWP 可作为额外通用性结果，但不必写入最强 claim；
- 支持总查询数 `N > Q` 的分批执行；
- “online scheduling” 表示不依赖完整查询执行轨迹的 oracle，而不是任意到达流、持续服务或严格 QoS；
- 不把动态图、多 GPU、PageRank/WCC、抢占、SM partitioning 和 replenish 作为当前主线。

若最终没有 arrival-stream 实验，不要使用 “online service” 或 “continuous query service” 一类会触发吞吐、排队延迟、饥饿和 SLA 追问的表述。

### 2.3 核心术语修正

“Memory Access Latency (MAL)” 可以作为 motivation 中的硬件现象，但不宜作为所有设计的统一技术名词：共享邻接表减少的是访问次数/字节数，而内存合并减少的是事务数量和单位有效数据的访问代价，两者通常不会改变 DRAM 单次访问的物理延迟。

推荐统一使用：

- **effective GPU memory-access cost**；
- **data-movement cost**；
- **memory-stall cost**；
- 分别量化 **topology-access volume** 与 **state-access transaction efficiency**。

### 2.4 推荐的核心 Thesis

> GraphWeft treats concurrent graph traversals as a multi-query execution plan and reduces their effective GPU memory-access cost through topology sharing, query-dimensional coalescing, and online cost-aware coordination.

中文表述：

> GraphWeft 将并发图遍历视为一个多查询执行计划，通过共享图拓扑扫描、合并查询维度的状态访问，并在线协调查询和物理执行方式，降低 GPU 上的有效数据移动代价。

这个 thesis 包含三个必须分别被实验验证的动作：

1. **减少访问量**：相同活跃顶点的邻接表只扫描一次。
2. **降低单位有效工作的事务代价**：让 warp lanes 访问同一邻居的连续 query states。
3. **提高前两类机会被实际利用的概率**：选择可共享的查询、启动时机和物理算子。

## 3. 问题定义与代价域

设查询 `q` 在其局部迭代 `l` 的 frontier 为 `F_q^l`，查询启动偏移为 `o_q`。在全局迭代 `t`，查询局部进度为 `l_q(t) = t - o_q`，所有当前 frontier 的并集为：

`U_t = union_q F_q^{l_q(t)}`。

对每个 `v in U_t`，维护一个精确查询掩码：

`M_t(v) = { q | v in F_q^{l_q(t)} }`。

当前实现以一个 64-bit mask 表示 `M_t(v)`，因此每批最多 64 个 active slots。三个基础工作量指标为：

- 独立执行的邻接访问量：`E_ind(t) = sum_q sum_{v in F_q} degree(v)`；
- 共享扫描的唯一邻接访问量：`E_union(t) = sum_{v in U_t} degree(v)`；
- 必须执行的 vertex-query 逻辑工作：`E_pair(t) = sum_{v in U_t} degree(v) * popcount(M_t(v))`。

可定义拓扑共享率：

`Share(t) = 1 - E_union(t) / E_ind(t)`。

它只度量可以避免的邻接扫描，不能代替总运行时间，因为系统仍需执行 query-specific state update、原子操作、frontier materialization 和调度逻辑。

统一代价模型可以写成：

`C_total(P, O, D) = C_plan(P, O) + sum_t C_{D_t}(X_t)`，

其中 `P` 是 batch partition，`O` 是查询偏移，`D_t` 是第 `t` 轮的物理算子，`X_t` 是运行时特征。

共享扩展的候选模型：

`C_expand(t) = a_u E_union + a_p E_pair + a_f |U_t| + a_a A_t`，

其中 `A_t` 为算法相关的原子/归约工作量。Gather 的候选模型：

`C_gather(t) = b_e E_probe + b_s T_state + b_v |V_scan| + b_c C_compact`，

其中 `T_state` 是状态数组的实际内存事务数。最终论文不要求公式能精确预测微秒数，但必须证明这些特征比单一 frontier-size threshold 更能解释和选择执行计划。

## 4. 主要贡献：建议写成可证伪的 Claims

1. **问题与机会识别。** 我们刻画了 GPU concurrent graph queries 的两个互补数据移动瓶颈：重复的 topology accesses，以及分散的 per-query state accesses；并说明仅对查询并发或单一 push/pull kernel 优化不足以同时解决两者。
2. **精确的多查询物理执行。** 我们设计了基于 vertex–query mask 的共享状态和两种互补算子：topology-sharing expansion 复用邻接扫描，query-dimensional coalesced gather 提高状态访问的内存事务效率，同时保持每个查询的独立语义。
3. **在线查询协调。** 我们使用轻量图索引估计查询 frontier 的阶段相关性，联合选择 batch composition 与启动偏移，以提高共享扩展能够捕获的重叠。
4. **代价感知的运行时选择。** 我们根据实时唯一边、vertex-query pairs、frontier 和访问特征在两种算子之间选择，并将 planning overhead 纳入端到端代价。
5. **系统与评测。** 我们在 GraphWeft 中实现上述设计，并在统一查询集合、方向语义和计时协议下，与独立/多流 GPU 执行、iBFS、Glign/ForkGraph 等系统比较；通过 end-to-end、消融、硬件计数器、oracle gap 和多查询规模实验验证各项 claim。

注意：当前不能声称“第一个 GPU concurrent graph processing system”。iBFS 已于 SIGMOD 2016 在 GPU 上联合执行并发 BFS。也不能把 joint frontier、共享邻接扫描、简单 GroupBy、delayed start 或 `V x Q` 布局单独声明为首创；这些分别与 iBFS 和 Glign 有明显先例。本文的区别必须落在 **typed exact state + 两类 GPU 物理算子 + 同一代价域下的在线协调与选择** 的组合及其定量收益上。

## 5. 系统概览

### 5.1 查询生命周期

1. 主机接收一组查询及其算法类型、source 和可选权重。
2. Online Planner 使用轻量 landmark index 为查询建立相位相关性近似，将查询分为不超过 `Q` 的 batches，并选择相对启动偏移。
3. Workspace 初始化 `V x Q` typed states、visited/frontier masks 和 union frontier。
4. 每一轮收集 `E_union`、`E_pair`、active vertices/pairs 等特征。
5. Runtime Selector 选择 topology-sharing expansion 或 query-dimensional coalesced gather。
6. 算子只处理 mask 指定的查询，生成 exact next-frontier masks，并做 compact/termination check。
7. batch 完成后返回 per-query results，继续下一批。

### 5.2 建议的系统架构图

正文 Figure 2 应同时表现“查询层”和“GPU 物理执行层”：

```text
Query window
    |
    v
Landmark index --> affinity estimator --> batch + start-offset plan
                                             |
                                             v
                     exact union frontier + vertex-query masks
                                             |
                         runtime cost-feature collector
                              /                  \
                             v                    v
             topology-sharing expansion   coalesced gather
                              \                  /
                               exact next frontier
                                      |
                              per-query results
```

图中应用醒目标注三类决策：**who shares、when to share、how to execute**。

### 5.3 当前实现映射

- `[已实现]` `V x Q` 布局：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/core/layout.hxx`。
- `[已实现]` typed values、exact masks、union frontiers 与计数器：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/state/engine_workspace.hxx`。
- `[已实现]` 算法抽象：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/algorithms/algorithm_traits.hxx`。
- `[已实现]` 共享扩展：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/push/shared_push_kernels.hxx`。
- `[已实现]` query-dimensional gather：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/pull/fused_pull_kernels.hxx`。
- `[已实现]` frontier/work metrics：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/common/frontier_metrics.hxx`。
- `[已实现]` 当前运行时选择器：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/engine/frontier_engine.hxx`。
- `[已实现]` online offset evaluator：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/scheduling/online_offset_evaluator.hxx`。
- `[已实现]` online runner 与启动计划：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/scheduling/online_runner.hxx`、`/home/zyl/Projects/ocgp/puercgp/include/puercgp/scheduling/slot_start_schedule.hxx`。

## 6. 系统设计

### 6.1 基于 Vertex-Sharing 的执行范式

#### 6.1.1 设计目标与挑战

不同查询经常在相近阶段访问相同顶点。若逐查询执行或简单使用 CUDA streams，每个查询会独立读取相同 CSR row；传统 query-level parallelism 增加了并发，却没有消除重复拓扑访问。另一方面，直接把 frontiers 做无标签并集会丢失查询身份，使 visited、distance 和 next frontier 相互污染。

因此执行层需要同时满足：

1. 邻接表按 unique active vertex 读取一次；
2. 对每条边只执行 mask 中实际活跃的 query updates；
3. 输出仍是每个查询精确可分离的 frontier 和 result；
4. BFS 的 bitwise state 与 SSSP 的 typed value/reduction 使用同一执行接口；
5. 在重叠较低时不会因 query-pair expansion 产生不可控额外工作。

#### 6.1.2 统一状态表示

GraphWeft 将 query identity 嵌入顶点状态，而不是为每个查询复制一套完全独立的执行队列。每个 union-frontier vertex `v` 关联 64-bit exact mask `M(v)`；第 `q` 位仅在查询 `q` 当前确实包含 `v` 时置位。状态数组采用 `value[v * Q + q]`，即 vertex-major、query-contiguous 的 `V x Q` 布局。

该布局有两方面作用：

- expansion 可通过 mask 跳过不活跃 query，避免把 union frontier 误当成所有查询共同 frontier；
- gather 中一个 warp 的 lanes 映射到查询维，固定 neighbor `u` 时访问连续的 `value[u, q]`，为合并事务创造条件。

需要在论文中明确区分三个对象：union vertex、active vertex-query pair 和逻辑 edge-query relaxation。否则审稿人会质疑“扫描一次邻接表”是否真的等于“所有计算只做一次”。

#### 6.1.3 Topology-Sharing Expansion：减少 Memory Access Count

当前 warp-level kernel 将一个 warp 分配给一个 union-frontier vertex。warp 首先读取 `M(v)` 和 CSR row boundaries，随后协作遍历 `N(v)`；相同邻接表不再为每个 query 重读。对 BFS，query bits 可用 mask/atomic-OR 更新；对 SSSP 等 typed algorithms，则对 mask 中的查询执行独立 relaxation 和 reduction。

其收益上界由 `E_ind / E_union` 决定，但实际收益还受以下因素影响：

- `popcount(M(v))` 决定 per-query update 数量；
- 高度数顶点的共享比低度数顶点更有价值；
- 原子冲突和 next-frontier materialization 可能抵消邻接复用；
- `|U|` 很小或重叠很低时，union/mask 管理可能成为额外开销。

论文应展示如下伪代码，而不是展开 CUDA 细节：

```text
for v in union_frontier in parallel:
    active_queries = exact_mask[v]
    for u in neighbors(v) cooperatively:
        for q in set_bits(active_queries):
            if Algorithm.relax(value[v,q], edge(v,u), value[u,q]):
                next_mask[u] |= bit(q)
```

**正确性不变量**：`bit q in next_mask[u]` 当且仅当查询 `q` 的独立执行会在该轮激活 `u`。证明可对迭代轮数归纳：base case 是每个 source 的独立 bit；induction step 中 kernel 只遍历 `M(v)` 的 set bits，并调用与独立执行一致的 algorithm-specific transition。

**当前代码状态**：`[已实现]`，默认使用 warp variant；simple 和 query-parallel variants 可用作实现消融。路径：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/push/shared_push_kernels.hxx`。

#### 6.1.4 Query-Dimensional Coalesced Gather：降低 Memory Access Cost

当 frontier 变稠密时，从活跃源点向外扩展会产生大量离散状态写入。GraphWeft 的另一物理算子从目标顶点读取 incoming neighbors 的状态。关键映射不是“一个线程完成一个顶点的所有查询”，而是让 `threadIdx.x` 对应 query lane、`threadIdx.y` 对应一个或一组目标顶点。这样，同一时刻 warp lanes 对固定 neighbor `u` 访问 `value[u, q:q+W]`，与 `V x Q` 连续布局匹配。

这类算子的核心 claim 应写成“减少完成同样逻辑 probes 所需的 memory transactions/bytes per useful update”，而非“降低 DRAM latency”。必须用 Nsight Compute 的 sector/request、global load transactions、DRAM bytes、long-scoreboard stalls、L2 hit rate 和 achieved bandwidth 证明机制。

Gather 之后仍需生成 exact next-frontier mask 并 compact 活跃顶点。正文应把 scan/compact 计入总算子代价，不能只报告 kernel 内层的 neighbor loop。

**当前代码状态**：`[已实现]` simple query-dimensional mapping 为默认版本；共享内存版本保留用于实验。路径：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/pull/fused_pull_kernels.hxx`。

**现有机制迹象**：`[已有数据—初步]` 当前 NCU 结果显示 pull 仍有显著 long-scoreboard stall，但仅能作为瓶颈线索，不能直接证明新 mapping 优于合理对照。分析：`/home/zyl/Projects/ocgp/puercgp/experiments/20260727_slot_group_validation/FINAL_ANALYSIS.md`；原始 NCU：`/home/zyl/Projects/ocgp/puercgp/experiments/20260727_slot_group_validation/ncu/`。

#### 6.1.5 两个算子为什么必须并存

Topology-sharing expansion 的收益来自 **跨查询复用 CSR rows**，最适合稀疏/中等 frontier 且相同顶点重叠较高的阶段；coalesced gather 的收益来自 **查询维度连续访问 state**，适合 frontier 较稠密、push 写入/原子开销高的阶段。二者分别优化 topology volume 和 state transaction efficiency，不是同一 kernel 的两个命名版本。

建议在系统概览中把它们命名为：

- `SharedExpand`：topology-sharing sparse operator；
- `CoalescedGather`：query-coalesced dense operator。

可以在 background 中用 push/pull 对齐既有文献，但标题和贡献列表使用上述物理含义更明确的名称。

#### 6.1.6 设计备选与边界

- **Per-query execution / CUDA streams**：没有跨查询拓扑复用，是必须比较的基础方案。
- **Query-major `Q x V` 布局**：有利于单查询连续遍历，但不利于固定 vertex/neighbor 下的 query-lane 合并；需要直接布局消融。
- **无精确 mask 的 union frontier**：可能做无效 query-pair work，甚至污染语义；只能作为反例。
- **2x32 slot grouping**：已有结果不稳定，不应升级为核心设计。`[负结果/非主线]` `/home/zyl/Projects/ocgp/experiments/20260731-153439_ge_spmm_pull_group_slots/README.md`。
- **Replenish/持续补充查询**：当前整体负收益，应作为被拒绝设计或 future work。`[负结果/非主线]` `/home/zyl/Projects/ocgp/puercgp/experiments/refactor_reports/stage10_replenish_benchmark.md`。

### 6.2 基于最大化 Query-Sharing 的调度策略

#### 6.2.1 调度问题

共享执行只有在查询的 frontiers 同时到达相同顶点时才有效。随机把查询装入同一 batch 只能利用偶然重叠；即使两个查询访问相似区域，如果其图距离进度错位，瞬时 frontiers 仍可能几乎不重合。因此 planner 需要共同回答：

- **Who shares?** 哪些查询放在同一个 batch；
- **When do they share?** 每个查询相对何时启动；
- **What is optimized?** 不仅最大化顶点交集，更应优先重合高度数顶点，从而最小化 `sum_t E_union(t)` 或预测的执行代价。

形式上，在 slot limit `Q` 下，planner 对查询集合划分 `P = {B_1, ...}` 并选择 offsets `O`，近似最小化：

`min_{P,O} C_plan(P,O) + sum_B sum_t C_expand(B, O, t)`。

精确计算需要知道每个查询的完整 frontier trace，在线场景不可得，因此系统使用轻量图索引构造相位相关性代理。

#### 6.2.2 轻量 Landmark Index

当前 evaluator 首先选择最多四个 farthest-point phase landmarks，以覆盖图中距离阶段；其余 landmarks 通过随机边源点抽样，形成 degree-biased 代表。对每个 landmark 执行 BFS，并以 `uint16` 存储顶点到 landmark 的距离。

若 landmark 数量为 `L`，建索引时间为 `O(L(|V|+|E|))`，主要距离存储约为 `2L|V|` bytes。论文必须报告：

- `L`、采样策略和不可达值处理；
- 一次性 index construction time 与峰值内存；
- index 是否可跨 query windows 复用；
- 大图上相对于 CSR 的额外内存比例；
- landmark 数和计划质量/开销的敏感性。

**当前代码状态**：`[已实现]` `/home/zyl/Projects/ocgp/puercgp/include/puercgp/scheduling/online_offset_evaluator.hxx`。

需要准确描述其目标：非 phase landmarks 的权重当前为 1，但它们通过 degree-biased source sampling 得到。因此当前 score 是 **degree-biased overlap proxy**，不是对 degree-weighted union-edge objective 的精确计算。论文不能把代理分数写成精确最优解。

#### 6.2.3 Pairwise Affinity 与 Batch Formation

对每对候选 source queries，planner 根据其到 landmarks 的相对 level 差构造直方图，估计两个 traversal 在某个相对偏移下同时经过相同图区域的可能性。planner 再以 greedy batching 建立初始组，并通过局部 swap 改善组内总 affinity。

这一设计相较只比较 source embedding 距离更适合论文主线，因为它显式考虑了 traversal phase 与 relative offset。但必须通过以下实验把“score”连接到真正的 GPU 收益：

1. pairwise score 与实际 `E_union/E_ind` 的 Spearman/Pearson 相关性；
2. score 与 end-to-end time saving 的相关性；
3. 预测的最佳 offset 与 offline trace oracle 的差距；
4. 在低重叠、road-like 或 disconnected graph 上的失败模式。

#### 6.2.4 Relative Start-Offset Search

给定 batch 后，planner 从 completion-aligned、zero-offset 和 random starts 初始化，在 `[0, O_max]` 上做 coordinate ascent。offset 使不同 source 的 traversal 阶段在全局时间上对齐，增加高价值 frontier overlap。

论文中必须把 offset 与 “delayed execution” 区分开：offset 是计划的一部分，不能忽略等待导致的批处理 makespan、per-query latency 和 GPU 空转。总时间必须从 batch 第一条查询可启动时计到最后一条查询完成，并包含 evaluator time。

**当前代码状态**：`[已实现]` `/home/zyl/Projects/ocgp/puercgp/include/puercgp/scheduling/online_offset_evaluator.hxx`、`/home/zyl/Projects/ocgp/puercgp/include/puercgp/scheduling/online_runner.hxx`。

#### 6.2.5 当前证据与解释边界

- `[已有数据—初步]` all-push BFS、`N=128, Q=32`、单 seed 的 online grouping+offset 在六张图上均正确，记录的 speedup 为 `1.152x–2.309x`；同时记录 union-edge reduction 和约 `10.7–14.5 ms` evaluator overhead。总览：`/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/README.md`；原始数据：`/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/`。
- `[已有数据—初步]` offline/sample opportunity，`N=Q=32`：`/home/zyl/Projects/ocgp/puercgp/experiments/20260711-152345_phase_schedule_q32/README.md`。
- `[已有数据—初步]` offline/sample opportunity，`N=128,Q=32`：`/home/zyl/Projects/ocgp/puercgp/experiments/20260711_phase_schedule_n128/README.md`。
- `[已有数据—初步]` 当前 hybrid engine 的 `N=256,Q=64` 结果中，planner 仅在部分 BFS case 启用，BFS GPU+evaluator 几何平均收益约 `1.064x`；SSSP/SSWP 当前安全绕过 planner，不能用来证明跨算法调度收益。总览与原始结果：`/home/zyl/Projects/ocgp/puercgp/experiments/20260719_online_runner_n256_q64/README.md`、`/home/zyl/Projects/ocgp/puercgp/experiments/20260719_online_runner_n256_q64/summary.csv`。

现有结果说明 planner 对 all-push 的收益可能比 hybrid 更明显，这是合理现象：grouping/offset 直接提高 SharedExpand 的 frontier overlap，而 hybrid 可能在密集阶段切换到 Gather。论文必须将“调度机会”和“最终联合系统收益”分开报告，不能只选 all-push 的最好数字代表最终系统。

#### 6.2.6 必须比较的调度基线

- FIFO / input order + zero offset；
- random batching + zero offset；
- random batching + planner offset；
- affinity batching + zero offset；
- iBFS-style GroupBy；
- Glign-style source affinity/closestHV + delayed start；
- GraphWeft joint grouping+offset；
- offline full-trace oracle 或受限 exhaustive oracle（只用于小图/小 batch）。

最有说服力的比较是在同一 GraphWeft execution engine 中仅替换 planner，从而排除 CPU/GPU、kernel、数据布局和计时差异。外部系统端到端对比仍需保留，但不能代替这个同引擎消融。

### 6.3 基于最小化内存访问代价的计算模式切换

#### 6.3.1 选择目标

每一轮的执行方式不是固定的：稀疏且重叠高的 frontier 更适合 SharedExpand；稠密或状态访问更适合 query-coalesced 的阶段更可能适合 CoalescedGather。selector 的目标是最小化 **总迭代时间**，而不是仅最小化逻辑边数。

输入特征至少应包括：

- `E_union`：共享后需扫描的唯一边；
- `E_pair`：必须执行的 edge-query work；
- `|U|` 与 active vertex-query pair count；
- active query slots、图的 `|V|/|E|`、平均/分位 degree；
- 前一轮 operator time 和可选的 sampled transaction/stall proxy；
- 算法类型，因为 BFS bit operations 与 SSSP atomics/reductions 代价不同。

#### 6.3.2 当前启发式

当前 `frontier_engine.hxx` 使用近似条件：

`virtual_edges >= pull_edge_ratio * active_slot_count * |E|`，

其中 `virtual_edges` 对应独立等价的 edge-query work。BFS 一旦切换到 Gather 采用 sticky policy。该机制已能稳定运行，但它仍是一个手工 threshold，不应在论文中直接称为“cost model”或“cost-optimal selector”。

- `[已实现]` 选择逻辑：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/engine/frontier_engine.hxx`。
- `[已实现]` 特征统计：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/common/frontier_metrics.hxx`。
- `[已有数据—需重跑]` 当前重构回归表明旧策略与新实现的 12 个组合均通过回归门槛，但它没有与 per-iteration oracle 或外部 direction optimizer 比较：`/home/zyl/Projects/ocgp/puercgp/experiments/refactor_reports/stage9_hybrid_scheduler.md`。

#### 6.3.3 论文级代价选择器

推荐实现一个低参数、可解释的 calibrated model，而不是复杂黑盒：

`T_expand_hat = a0 + a1 E_union + a2 E_pair + a3 |U| + a4 A_proxy`；

`T_gather_hat = b0 + b1 E_probe + b2 T_state_proxy + b3 |V_scan| + b4 C_compact_proxy`。

系数可在少量与 evaluation 分离的 training graphs/query sets 上离线校准，或启动时通过 microbenchmark 校准。论文必须说明训练/测试隔离，避免在每张测试图上调到最优 threshold。若时间不足，可采用分段线性模型，但至少要证明：

1. 选择准确率和 oracle regret 优于单一 frontier/edge threshold；
2. selector 自身统计与决策开销远小于节省时间；
3. end-to-end 性能不是由为每个 case 人工选最佳策略得到；
4. 对 Q、算法和图结构变化具有稳健性。

#### 6.3.4 Oracle Replay 设计

为了评估 selector，而不改变查询语义，建议记录一遍执行中的每轮 exact frontier/mask 或可重放源集合；分别运行 SharedExpand 与 CoalescedGather 得到该轮时间，然后构造：

- `AlwaysExpand`；
- `AlwaysGather`；
- `CurrentThreshold`；
- `CalibratedCostModel`；
- `PerIterationOracle = min(T_expand, T_gather)`。

报告 selector accuracy、总 regret、错误方向（错误选 Gather 或错误选 Expand）、切换次数、决策开销和 end-to-end speedup。oracle 不是部署方案，只提供上界及剩余优化空间。

#### 6.3.5 语义与计时注意事项

- Directed graph 的 expansion 使用 outgoing CSR，gather 必须使用 incoming CSR；两者结果应与独立 reference 完全一致。
- 旧 `results/bfs_q64_*` 目录包含单 seed 和方向语义风险，只能做诊断，不能作为论文证据。`[历史数据—不可直接引用]` `/home/zyl/Projects/ocgp/puercgp/results/bfs_q64_hybrid_strategy_eval/`。
- 必须将 frontier scan、compact、metrics collection、operator switch 和同步开销计入相应 operator 或 end-to-end 时间。
- 如果 BFS 使用 sticky switching，需与允许来回切换以及 oracle 对比，解释避免振荡与减少统计开销的权衡。

## 7. 实现章节应覆盖的内容

实现章节建议只写影响可复现性和性能模型的细节：

1. **图存储**：CSR/CSC 是否同时驻留、directed/undirected 处理、边权类型和预处理时间。
2. **Workspace**：`V x Q` typed values、64-bit visited/frontier masks、union frontier lists、actual/virtual degree counters。
3. **Warp 映射**：SharedExpand 的 warp-per-union-vertex；Gather 的 query-lane x vertex-row mapping。
4. **Algorithm interface**：BFS 的 bit mask 更新、SSSP/SSWP 的 typed relaxation/reduction；终止条件。
5. **批次管理**：`N > Q` 时如何切 batch、offset 如何注入、planner 何时运行、是否与 GPU 执行重叠。
6. **限制**：`Q <= 64` 来源于 mask 宽度；图和状态必须容纳于单 GPU；landmark index 目前为 BFS-distance-based，对 weighted scheduling 尚未完成。

避免在实现章节枚举所有模板和 helper；把最关键的 kernel mapping 画成一张 8–16 lane 的简图会比代码片段更清楚。

## 8. 实验评测总体协议

### 8.1 核心研究问题

评测应按 claim 组织，而不是按脚本或数据集组织：

- **RQ1：** 并发图查询的主要 GPU 数据移动瓶颈是什么，跨查询复用机会有多大？
- **RQ2：** GraphWeft 相比现有系统和合理 GPU 基线，端到端性能如何？
- **RQ3：** SharedExpand 是否确实减少 topology accesses，并把减少量转化为时间收益？
- **RQ4：** CoalescedGather 是否确实减少 state transactions/stalls？
- **RQ5：** online planner 是否在包含开销后提高共享度和端到端性能？
- **RQ6：** cost-aware selector 是否接近 per-iteration oracle 并优于固定策略/threshold？
- **RQ7：** 系统对 batch size、总查询数、图结构、query locality 和算法是否稳健？
- **RQ8：** 计划、索引和状态的时间/空间开销是否可接受，结果是否正确？

### 8.2 公平性与统计协议

- 所有系统使用相同图方向、权重、source query files 和输出语义。
- 主结果至少覆盖 `Q = 8, 16, 32, 64`；总查询数覆盖 `N = Q` 和 `N = 4Q`，若时间允许增加 `N = 16Q`。
- 每个 graph/algorithm/Q 至少使用 5 个独立 source seeds；每个配置 2 次 warmup + 5 次 measured runs，主文报告 median，误差条使用跨 seeds 的 95% bootstrap CI 或清晰报告分布。
- GPU 使用固定时钟/持久模式（若权限允许），记录型号、显存、driver、CUDA、编译器和功耗/温度；各系统在同一机器上运行。
- end-to-end 从查询 batch 已准备好开始，包含 planner、offset idle、GPU kernel、frontier materialization 和必要同步；graph load/index build 单独报告，但不能混在某些系统而从另一些系统中排除。
- 另外报告纯 execution time，帮助分解 planning 与 execution；两个数字必须明确命名。
- OOM 不是性能点：标注容量上限，并在所有系统都能运行的严格相同 Q 上给出公平主表。
- 不使用 “Best GraphWeft” 作为默认系统结果。投稿主结果必须来自一个固定、可部署的 planner+selector policy；oracle/best-of 只作为上界。

### 8.3 数据集与工作负载矩阵

至少包含：

- 高度 skewed social/web graphs：Twitter、Sina Weibo、Orkut、LiveJournal、Indochina；
- citation graph：cit-Patents；
- 低度、长直径 road graph：至少一张 roadNet，作为预期低共享/负结果；
- 若显存允许，增加一张更大规模图以证明容量和扩展性。

算法主线：BFS、SSSP；SSWP 为可选第三个。query distribution 至少包含：

- uniform random sources；
- degree-stratified sources（low/medium/high degree）；
- localized/hotspot sources（模拟相关查询）；
- application-derived queries，例如 betweenness/closeness 的多源 BFS，或路径/可达性服务的批量 source queries。

若没有真实工作负载，SIGMOD/ICDE 审稿人可能认为“并发 query overlap”由人工 source 选择制造。应用派生或真实 query trace 是 `[缺失—P0]`。

## 9. 分章节实验设计与现有数据映射

### 9.1 RQ1：Motivation——数据移动瓶颈与共享机会

**假设。** 独立/多流 GPU 执行对同一 CSR rows 和 per-query states 产生重复或低效事务；查询重叠在真实图中显著，但随图结构、query phase 和 source locality 变化。

**实验。** 对 independent sequential、CUDA multi-stream 和 GraphWeft fixed operators，在 BFS/SSSP、Q=8/16/32/64 上收集：kernel breakdown、DRAM bytes、global load sectors/requests、L2 hit rate、long-scoreboard stall、achieved bandwidth、`E_ind/E_union/E_pair`。将 union-edge reduction 与时间 speedup 作散点图。

**正文图。** Figure 1(a) 时间/内存 stall breakdown；Figure 1(b) 可避免 topology accesses 的分布；Figure 1(c) overlap 与实际 speedup 的相关性。

**现有证据。**

- `[已有数据—初步]` pull NCU 与 long-scoreboard 线索：`/home/zyl/Projects/ocgp/puercgp/experiments/20260727_slot_group_validation/FINAL_ANALYSIS.md`。
- `[已有数据—初步]` online all-push 中已记录 union-edge reduction 与运行时间：`/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/`。
- `[已有数据—需重跑]` SM-scaling 可辅助区分 occupancy 与 memory bottleneck：`/home/zyl/Projects/ocgp/puercgp/experiments/20260711_green_context_sm_scaling/README.md`。

**缺失。**

- `[缺失—P0]` 在最终代码、统一 source sets 上进行 independent/multistream/SharedExpand/Gather 的同轮 NCU 对比。
- `[缺失—P0]` BFS 与 SSSP 都需覆盖；当前 NCU 不能支持系统级“所有算法均 memory-bound”的表述。
- `[缺失—P0]` 应用派生 workload 的 overlap distribution。
- `[缺失—P1]` roofline 或 bytes-per-effective-edge 归一化分析。

### 9.2 RQ2：端到端性能

**假设。** 固定、可部署的 GraphWeft policy 在多种图、算法与 query concurrency 下优于独立 GPU 执行、GPU streams 和现有并发图处理系统；收益不依赖某一个 Q 或单一 seed。

**基线。**

- 单查询 GPU library 顺序执行；
- 同一 library 的 CUDA multi-stream；
- iBFS（BFS，GPU）；
- Gunrock sequential/multi-stream（BFS/SSSP）；
- Glign 与 ForkGraph（CPU context，用于回答并发处理系统位置，硬件平台需单独披露）；
- 可运行时加入 KGraph；若无法开源复现，则只作 related work，不能伪造数字。

**指标。** batch makespan、queries/s、相对每个 baseline 的 speedup、per-query median/P95 latency、peak GPU memory。主文报告固定 GraphWeft policy；Oracle-GraphWeft 只画虚线上界。

**正文图。** 每算法一张 normalized throughput 图；一张 query concurrency scaling；一张 latency/throughput 权衡或表格。

**现有证据。**

- `[已有数据—初步]` 外部 baseline、`N=256,Q=64`：`/home/zyl/Projects/ocgp/experiments/20260719-233131_external_baseline_n256/RESULTS.md`，汇总数据：`/home/zyl/Projects/ocgp/experiments/20260719-233131_external_baseline_n256/result.csv`。历史汇总中 best configuration/iBFS 的 BFS 几何平均约 `0.694x`，意味着 Q=64 聚合性能仍弱于 iBFS；这个负结果必须正面解决，不能隐藏。
- `[已有数据—初步]` Gunrock multistream：`/home/zyl/Projects/ocgp/experiments/20260725_gunrock_multistream_n256/RESULTS.md`，汇总：`/home/zyl/Projects/ocgp/experiments/20260725_gunrock_multistream_n256/gunrock_puercgp_comparison.csv`。
- `[已有数据—初步]` Twitter OOM 复查：`/home/zyl/Projects/ocgp/experiments/20260728_gunrock_twitter_oom_recheck/README.md`。
- `[历史数据—不可直接引用]` Q=2–64 的 iBFS/Glign/ForkGraph/GraphWeft 旧版本比较：`/home/zyl/Projects/ocgp/experiment/experiments/20260609_forkgraph_glign_ibfs_bfs/COMBINED_SUMMARY_WITH_PUERCGP_HYBRID_GE_SPMM.md`、`/home/zyl/Projects/ocgp/experiment/experiments/20260609_forkgraph_glign_ibfs_bfs/combined_results_with_puercgp_hybrid_ge_spmm.csv`。

**缺失。**

- `[缺失—P0]` 最终代码下 Q=8/16/32/64、多个 seeds、同源同方向的完整重跑；尤其验证此前 Q=16/32 相对 iBFS 的优势是否仍成立。
- `[缺失—P0]` iBFS 与 GraphWeft 的 kernel-only 和 full-batch 两种公平时间；确认数据转换、query grouping 和 offset 等开销边界。
- `[缺失—P0]` Gunrock 在所有系统共同可容纳的严格相同 Q 上的主结果；目前不同图的有效 Q 不一致。
- `[缺失—P0]` 固定 policy 结果，禁止按 case 选择 “Best GraphWeft”。
- `[缺失—P0]` application-derived workload。
- `[缺失—P1]` 第二种 GPU 架构；至少一张图/两个 Q 即可验证可移植性。

### 9.3 RQ3：SharedExpand 的访问量与收益

**假设。** exact vertex sharing 将 topology reads 从 `E_ind` 降到接近 `E_union`，且收益随 degree-weighted overlap 上升；mask/query update overhead 在有足够共享时被摊销。

**消融。**

1. Per-query sequential；
2. CUDA multi-stream；
3. Union frontier without scheduling；
4. Exact SharedExpand + random batch/zero offset；
5. Exact SharedExpand + full planner；
6. simple / warp / query-parallel SharedExpand variants。

**指标。** CSR row loads、DRAM bytes、global transactions、`E_ind/E_union/E_pair`、atomics、frontier construction time、TEPS 和 total time。按 overlap deciles 画 speedup，展示 break-even point。

**现有证据。**

- `[已有数据—初步]` all-push online grouping+offset，含 union-edge statistics：`/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/`。
- `[已有数据—初步]` Q32 phase schedule opportunity：`/home/zyl/Projects/ocgp/puercgp/experiments/20260711-152345_phase_schedule_q32/`。
- `[已有数据—初步]` N128 phase schedule opportunity：`/home/zyl/Projects/ocgp/puercgp/experiments/20260711_phase_schedule_n128/`。

**缺失。**

- `[缺失—P0]` 同一 frontier replay 下 independent 与 SharedExpand 的硬件计数器直接对照。
- `[缺失—P0]` 分开报告 topology reduction、query-pair work、mask/atomic overhead，避免把逻辑工作量减少与内存优化混为一谈。
- `[缺失—P0]` 无调度、仅 grouping、仅 offset、joint planner 的端到端消融。
- `[缺失—P1]` 按顶点 degree bucket 展示高价值共享来自哪些顶点。

### 9.4 RQ4：CoalescedGather 的内存事务效率

**假设。** 对固定 neighbor 把 warp lanes 映射到连续 query slots，比 query-major 或 query-serial mapping 使用更少的 state load sectors/transactions，并在 dense traversal stages 降低 long-scoreboard stall 和运行时间。

**布局/映射消融。**

1. `Q x V` query-major；
2. `V x Q` 但 thread-per-vertex、query-serial；
3. `V x Q` query-lane coalescing（当前设计）；
4. 当前保留的 shared-memory variant；
5. iBFS bottom-up（BFS 外部参考）。

**协议。** 使用完全相同的 incoming CSR、frontiers 和 results；Q=8/16/32/48/64，覆盖 warp 边界；分别报告 kernel inner loop 与 scan+compact 后的完整 operator time。

**指标。** global load sectors/request、DRAM bytes、L1/L2 hit、achieved bandwidth、long-scoreboard cycles、registers、occupancy、useful state probes/s。

**现有证据。**

- `[已有数据—初步]` 当前 pull NCU：`/home/zyl/Projects/ocgp/puercgp/experiments/20260727_slot_group_validation/ncu/`。
- `[已有数据—初步]` 报告曾观察 `Q x V` 明显慢于 `V x Q`，但未找到可直接复核的完整原始数据，只能作为重新实验的线索：`/home/zyl/Projects/ocgp/puercgp/docs/recent_optimization_progress_20260713.md`。
- `[负结果/非主线]` 2x32 grouping 只在部分图有效：`/home/zyl/Projects/ocgp/experiments/20260731-153439_ge_spmm_pull_group_slots/result.csv`。

**缺失。**

- `[缺失—P0]` 最终代码下完整 layout/mapping ablation 和原始 NCU CSV。
- `[缺失—P0]` 将 transaction reduction 与 operator/end-to-end speedup 关联。
- `[缺失—P0]` Q<32 时 lane utilization 的处理和结果；否则设计可能只在 Q=32/64 成立。
- `[缺失—P1]` 不同 value size（bit/int32/float/64-bit）的敏感性。

### 9.5 RQ5：Online Planner 的计划质量

**假设。** multi-landmark pairwise phase proxy 比随机/FIFO、仅 source-distance 或简单 GroupBy 更准确地识别可共享查询；joint grouping+offset 在计入 evaluator time 与延迟后仍获益，并接近 offline oracle。

**同引擎对照。** FIFO、random、iBFS GroupBy、Glign-compatible closestHV/delayed-start、GraphWeft grouping only、GraphWeft offset only、GraphWeft joint、offline trace oracle。

**指标。** `E_union/E_ind`、degree-weighted overlap、makespan、throughput、P50/P95 latency、plan time、index time/memory、oracle gap、score/real saving correlation。

**现有证据。**

- `[已有数据—初步]` all-push BFS `N=128,Q=32`：`/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/README.md`；每图原始 `summary.csv`、`online_plans.csv`、`landmark_batches.csv` 和 `sources.csv` 位于同目录的 `final-*` 子目录。
- `[已有数据—初步]` hybrid `N=256,Q=64`：`/home/zyl/Projects/ocgp/puercgp/experiments/20260719_online_runner_n256_q64/summary.csv`。

**缺失。**

- `[缺失—P0]` iBFS GroupBy 与 Glign-compatible scheduler 在同一 GraphWeft engine 中的实现和直接对比。
- `[缺失—P0]` 至少 5 个 source seeds，Q=8/16/32/64，包含 road/低共享图。
- `[缺失—P0]` 小规模 full-trace oracle、offset 预测误差和 plan-quality gap。
- `[缺失—P0]` index construction、index memory、per-window plan time、amortization break-even。
- `[缺失—P0]` planner score 与实际 union-edge/time saving 的相关性。
- `[缺失—P1]` landmark 数 `L`、`O_max`、初始化个数、swap 次数的敏感性。
- `[缺失—P1]` SSSP-aware affinity；当前 BFS-distance index 对 weighted execution 的适用性尚未建立。

### 9.6 RQ6：Cost-Aware Operator Selection

**假设。** 同时考虑 unique topology work、query-pair work 和 state transaction proxy 的模型，比 Beamer-style frontier threshold 或当前 virtual-edge threshold 更接近 oracle；联合 selector 比 always-expand/always-gather 稳健。

**对照。** AlwaysExpand、AlwaysGather、frontier-size threshold、current virtual-edge threshold、calibrated model、per-iteration oracle。

**指标。** end-to-end time、per-iteration prediction error、operator selection accuracy、regret、false-expand/false-gather 次数、switch count、metrics+decision overhead。

**现有证据。**

- `[已有数据—需重跑]` 当前 threshold 的回归验证：`/home/zyl/Projects/ocgp/puercgp/experiments/refactor_reports/stage9_hybrid_scheduler.md`、`/home/zyl/Projects/ocgp/puercgp/experiments/refactor_reports/stage9_hybrid_scheduler_summary.csv`。
- `[历史数据—不可直接引用]` 旧 Q64 all-push/all-pull/hybrid 诊断：`/home/zyl/Projects/ocgp/puercgp/results/bfs_q64_hybrid_strategy_eval/`。

**缺失。**

- `[缺失—P0]` calibrated model 的实现、训练/测试划分和固定参数。
- `[缺失—P0]` exact frontier replay 下的 per-iteration oracle。
- `[缺失—P0]` selector accuracy/regret/overhead，以及与 end-to-end 的连接。
- `[缺失—P0]` BFS/SSSP × Q × graph structure 覆盖。
- `[缺失—P1]` sticky、bidirectional switching 和 hysteresis 的消融。

### 9.7 RQ7：扩展性、稳健性和适用边界

**假设。** GraphWeft 的收益随并发度和相关性增加，但不会在低重叠图上灾难性退化；planner 和 state 的空间开销随 `V、Q、L` 可解释地增长。

**实验维度。**

- `Q=1/2/4/8/16/32/64`；
- `N=Q/4Q/16Q`；
- graph size、degree skew、diameter；
- source locality/degree；
- overlap 分位；
- memory footprint 随 Q 和 L；
- 若有第二块 GPU，跨架构抽样验证。

**现有证据。**

- `[历史数据—不可直接引用]` 旧 Q=2–64 扩展曲线：`/home/zyl/Projects/ocgp/experiment/experiments/20260609_forkgraph_glign_ibfs_bfs/combined_results_with_puercgp_hybrid_ge_spmm.csv`。
- `[已有数据—初步]` N256/Q64 外部 baseline：`/home/zyl/Projects/ocgp/experiments/20260719-233131_external_baseline_n256/result.csv`。
- `[负结果/非主线]` replenish throughput/latency：`/home/zyl/Projects/ocgp/puercgp/experiments/replenish_e2e_throughput.md`、`/home/zyl/Projects/ocgp/puercgp/experiments/replenish_latency_benchmark.md`。

**缺失。**

- `[缺失—P0]` 最终代码下完整 Q-scaling 和 N-scaling，多 seeds。
- `[缺失—P0]` road/低重叠负结果与最坏 slowdown，证明 selector 能回退。
- `[缺失—P0]` peak memory 公式与实测、最大可运行 Q/N/graph。
- `[缺失—P1]` 第二 GPU 架构和更大图。
- `[缺失—P1]` 若要声称 online service，必须增加 open-loop arrivals、吞吐-延迟曲线和 starvation；否则删除该 claim。

### 9.8 RQ8：算法通用性与正确性

**假设。** exact masks 和 typed value interface 保持独立执行语义；系统机制不仅适用于 BFS bit operations，也适用于 SSSP typed relaxations。

**正确性协议。** 与 CPU reference/可靠单查询 GPU 输出逐 query 比较；BFS 比较每个 vertex distance，SSSP 比较距离（定义浮点容差），SSWP 比较 widest-path value。覆盖 directed/undirected、weighted/unweighted、不可达顶点、重复 source、零边图/小图、高度数顶点，以及 mask 边界 Q=1/31/32/33/63/64。

**现有证据。**

- `[已有数据—初步]` N256/Q64 online runs 的 fingerprints：`/home/zyl/Projects/ocgp/puercgp/experiments/20260719_online_runner_n256_q64/README.md`。
- `[已有数据—初步]` 外部 weighted cross-system correctness artifacts：`/home/zyl/Projects/ocgp/experiments/20260719-233131_external_baseline_n256/artifacts/correctness/`。
- `[已有数据—初步]` 重构最终验证：`/home/zyl/Projects/ocgp/puercgp/experiments/refactor_reports/stage8_final_validation.md`。

**缺失。**

- `[缺失—P0]` 最终代码、所有主数据集和所有主配置的逐 query correctness，而非仅 aggregate fingerprint。
- `[缺失—P0]` directed graph 上 outgoing expansion 与 incoming gather 的语义一致性回归。
- `[缺失—P0]` Q 跨 warp/mask 边界的 sanitizer/unit tests。
- `[缺失—P1]` 随机小图 differential testing 和 race/check 工具验证。

### 9.9 RQ9：规划与系统开销

**假设。** planner/index/frontier metrics 的开销可被 batch execution 摊销，且显存开销不抵消可处理规模。

**分解项。** graph conversion、CSR/CSC、landmark index build、query planning、workspace initialization、per-iteration metrics、scan/compact、kernel execution、host-device synchronization、result copy。

**现有证据。**

- `[已有数据—初步]` online planner overhead：`/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/README.md`。
- `[已有数据—初步]` external experiment 的环境与流程 artifacts：`/home/zyl/Projects/ocgp/experiments/20260719-233131_external_baseline_n256/artifacts/environment/`。

**缺失。**

- `[缺失—P0]` 统一 end-to-end 分解及 index amortization curve。
- `[缺失—P0]` workspace/index/CSR+CSC 的理论和实测 peak memory。
- `[缺失—P0]` metrics collection on/off 消融。
- `[缺失—P1]` 能耗或 GPU energy/query；不是核心，但可增强 modern-hardware story。

### 9.10 负结果与被拒绝设计

SIGMOD/ICDE 正文不必展示全部负结果，但应利用它们解释选择：

- `[负结果/非主线]` replenish 在已有测试中吞吐与尾延迟不稳定，说明动态填槽会破坏同步共享机会；当前保留 closed-window batching。路径：`/home/zyl/Projects/ocgp/puercgp/experiments/refactor_reports/stage10_replenish_benchmark.md`。
- `[负结果/非主线]` 2x32 pull grouping 仅在 Twitter 有收益，在多数图退化，说明 occupancy/SM partition 不是普适解。路径：`/home/zyl/Projects/ocgp/experiments/20260731-153439_ge_spmm_pull_group_slots/README.md`。
- `[历史数据—不可直接引用]` 旧 hybrid direction 结果有 directed semantics 风险，不能用漂亮数字替代最终正确实验。

正文可以用一个 “Design alternatives” 段落概括，完整表格放 artifact 或补充材料；这会向审稿人表明设计由证据驱动，而不是只报告成功尝试。

## 10. Claim–Evidence Matrix

| 论文 Claim | 必需证据 | 当前状态 | 投稿前判定 |
|---|---|---|---|
| CGQ 受重复 topology access 和低效 state transaction 限制 | NCU + bytes/transactions + overlap | 有零散 NCU/union-edge 数据 | `[缺失—P0]` 统一复测 |
| SharedExpand 减少 topology accesses | `E_ind` vs `E_union`、DRAM/CSR reads、时间 | all-push 初步数据 | `[缺失—P0]` 直接机制消融 |
| CoalescedGather 改善 query-state accesses | layout/mapping NCU 消融 | 只有当前 mapping NCU 与文字总结 | `[缺失—P0]` 原始数据重跑 |
| Planner 最大化有价值共享 | 多基线、oracle gap、相关性、含开销 | 单 seed all-push 较强，hybrid 收益较小 | `[缺失—P0]` 多 seed/同引擎基线 |
| Selector 最小化每轮代价 | model vs threshold vs oracle | 只有 threshold 回归 | `[缺失—P0]` cost model/replay |
| 端到端优于 SOTA | Q/算法/图/seed 完整矩阵 | Q64 对 iBFS 聚合仍弱 | `[缺失—P0]` 最终公平复测与优化 |
| 跨算法且语义精确 | BFS/SSSP correctness + performance | 有 fingerprints/weighted checks | `[缺失—P0]` 完整 differential tests |
| 在线开销可接受 | index、plan、metrics、memory、amortization | 部分 planner ms | `[缺失—P0]` 全链路分解 |

任何摘要或 introduction 中的定量句都应能指向该矩阵的一行和一张最终 figure/table。若某个 P0 证据没有完成，应收缩相应 claim，而不是用相关但不同协议的旧实验填充。

## 11. 建议的正文 Figures 与 Tables

1. **Figure 1：Motivation。** independent/multistream 的 memory-stall breakdown；同一 CSR row 重复读取示意；真实 workload 的 overlap 分布。
2. **Figure 2：System overview。** planner、exact shared state、selector 和两个物理算子。
3. **Figure 3：Vertex-sharing execution。** union frontier、per-vertex exact masks、SharedExpand 数据流。
4. **Figure 4：CoalescedGather mapping。** `V x Q` 布局与 warp lane 地址；对照 `Q x V`。
5. **Figure 5：Query planner。** landmark phase profiles、pairwise offset histogram、batch+offset 输出。
6. **Figure 6：Cost selector。** 特征、两条预测代价曲线和切换点。
7. **Table 1：Datasets/workloads。** V/E、directed、weighted、degree、diameter proxy、query source distribution。
8. **Table 2：End-to-end。** GraphWeft 与外部 systems；若柱状图更清楚则表格放 appendix/artifact。
9. **Figure 7：End-to-end speedup across Q。** BFS/SSSP 分面。
10. **Figure 8：Mechanism ablations。** SharedExpand、Gather、Planner、Selector 的增量收益。
11. **Figure 9：Planner/oracle 与 selector/oracle gap。** 展示剩余优化空间。
12. **Table 3：Overhead/memory/correctness。** index、plan、metrics、peak memory、validation cases。

每张性能图的 caption 必须写清：硬件、算法、Q/N、统计量、误差条、计时边界和越高/越低越好。不要让读者从正文猜实验协议。

## 12. 投稿前 P0 实验执行顺序

为避免在不稳定实现上浪费大量 GPU 时间，建议按下列依赖顺序推进：

1. **冻结语义与协议。** 确定 directed/undirected、CSR/CSC、query files、BFS/SSSP reference、计时边界、固定 GraphWeft policy。
2. **正确性门槛。** 完成 Q mask 边界、directed gather、随机小图 differential tests；失败则停止性能实验。
3. **单算子机制验证。** SharedExpand 的 topology counters；Gather 的 layout/mapping NCU；先证明两个设计分别兑现其 claim。
4. **Selector oracle replay。** 收集 per-iteration 双算子时间，完成 calibrated model 与 threshold/oracle 对比。
5. **Planner 同引擎基线。** FIFO/random/iBFS/Glign-compatible/GraphWeft/oracle，多 seeds；同时记录 planning overhead。
6. **端到端主矩阵。** BFS/SSSP × datasets × Q=8/16/32/64 × seeds，比较 iBFS/Gunrock/streams/CPU context systems。
7. **应用 workload 与负例。** application-derived queries、road graph、source locality，验证边界和回退。
8. **扩展性与开销。** N-scaling、memory、index amortization；再决定第二 GPU、能耗等 P1。
9. **一键复现。** 为每张主图保留 config、raw CSV、日志、git commit、环境、汇总和绘图脚本。

前五步完成前，不建议锁定摘要中的 speedup 数字。第六步若仍显示 Q64 对 iBFS 聚合落后，应将论文结论限定为具有相关查询或 Q=16/32 的工作负载，或者继续优化，而不能笼统声称全面 SOTA。

## 13. 与相关工作的区分及其对实验设计的启示

### 13.1 iBFS（SIGMOD 2016）

[iBFS: Concurrent Breadth-First Search on GPUs](https://www2.seas.gwu.edu/~howie/publications/iBFS-SIGMOD16.pdf) 已提出在 GPU 上联合执行多条 BFS，包括 joint frontier/status、bitwise 表示、GroupBy 以及 top-down/bottom-up 方向选择。GraphWeft 不能把“GPU 并发 BFS”“共享邻接表”或“方向切换”本身作为 novelty。

差异应落在：typed multi-algorithm exact states、以 query 维内存合并为目标的 Gather 物理算子、显式区分 topology volume 与 state transactions 的代价域，以及 joint grouping+offset+runtime selection。实验必须包含与 iBFS 的 Q-scaling、全批次端到端、访问事务和 component breakdown；否则审稿人会认为只是 iBFS 的实现变体。

### 13.2 Glign（ASPLOS 2023）

[Glign: Taming Misaligned Graph Traversals in Concurrent Graph Processing](https://par.nsf.gov/servlets/purl/10390494) 已强调 misaligned traversals、query-oblivious union frontier、affinity batching、delayed start 和 `V x Q` layout。GraphWeft 不能把“对齐查询”“按 affinity 分组”或该布局单独作为 novelty。

需要证明 GraphWeft 的 planner 是面向 GPU topology-access cost 的 phase/offset proxy，而非简单复现 closest-source embedding；同时 GraphWeft 的 query-dimensional Gather 和在线 per-iteration operator selection 是 Glign 的 CPU cache-oriented执行所不覆盖的。最重要的证据是同引擎 Glign-compatible scheduler 对照、planner/oracle gap，以及 GPU transactions—not 只做跨硬件 wall-clock comparison。

### 13.3 ForkGraph（SIGMOD 2021）与 Krill（SC 2021）

[ForkGraph](https://arxiv.org/abs/2103.14915) 通过 LLC-sized partitions 和 fork-processing 改善 CPU cache locality；[Krill](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap214.html) 通过编译器/运行时、property buffer 和 graph-kernel fusion 减少 CPU 并发图查询的内存访问。它们说明论文必须同时报告 work efficiency、cache/memory counters、end-to-end 和 memory overhead。

GraphWeft 的区别不是“也减少内存访问”，而是针对 GPU SIMT/coalescing/atomics 的物理算子和 warp mapping，并使用 exact query masks 在一条 GPU execution path 中共享 topology。

### 13.4 KGraph（ICDE 2025）

[KGraph: An Efficient Memoization Engine for Concurrent Graph Query Processing](https://ink.library.smu.edu.sg/sis_research/10407/) 通过 memoization 复用 concurrent graph queries 的中间结果，并使用真实 Grab taxi workload。它提示本文需要应用派生 workload，并清楚区分 **同轮 topology scan sharing** 与 **跨查询/跨时间 materialized-result reuse**。若 GraphWeft 不持久化和复用查询结果，就不应使用 memoization 术语。

### 13.5 Gunrock

[Gunrock](https://escholarship.org/uc/item/6xz7z9k0) 是数据中心化、frontier-based 的 GPU graph processing library，适合作为成熟单查询及多流执行基础。GraphWeft 的定位是跨查询的 execution planning 和共享状态，而非替代所有单查询 GPU primitives。

## 14. Introduction 的建议叙事

建议用五段完成，而不是从“GPU 很快、图很重要”开始：

1. **场景与需求。** 图分析服务经常同时收到许多结构相同、source 不同的遍历查询；把它们逐个或用 streams 并行执行会重复搬运 topology/state。
2. **关键观察。** GPU 上的瓶颈不是并行度不足，而是有效数据移动代价：相同 CSR rows 被多次扫描，query states 又产生低效事务；查询 frontier overlap 具有阶段性，不能由静态 batch 自动利用。
3. **现有缺口。** iBFS 主要面向 BFS 并已共享 frontier；Glign/ForkGraph/KGraph 主要面向 CPU cache、alignment 或 memoization。它们没有在统一 GPU 代价域内共同处理 topology sharing、query-state coalescing 和在线物理计划选择。
4. **系统 insight。** 把并发图遍历视为 multi-query plan：exact mask 允许安全共享，两个互补物理算子分别优化访问量与事务效率，planner/selector 决定谁、何时、如何共享。
5. **贡献与结果。** 用 3–4 条可度量贡献收束；最终数字只从统一 P0 主矩阵中填写。

避免以下高风险句式：

- “We are the first GPU concurrent graph processing system.”
- “GraphWeft reduces memory latency by reducing accesses.”
- “Our scheduler maximizes sharing.”（除非有精确最优化证明；当前应写 approximates/increases。）
- “GraphWeft consistently outperforms all baselines.”（当前 Q64/iBFS 数据不支持。）
- “The planner supports online weighted graph workloads.”（当前 SSSP/SSWP planner 路径不支持。）

## 15. 最终投稿检查表

### 15.1 系统与实验完整性

- [ ] 主系统名统一为 GraphWeft，消除 CGraph 冲突。
- [ ] scope 明确到 static/single-GPU/source-based/Q<=64。
- [ ] 所有核心 claims 都在 Claim–Evidence Matrix 中有 P0 证据。
- [ ] 最终 policy 固定，不按 case 选择 best implementation。
- [ ] iBFS 与 Glign 的重叠贡献不被误报为 novelty。
- [ ] BFS/SSSP 的 directed semantics 和逐查询结果通过验证。
- [ ] end-to-end 包含 planning/offset/scan/compact/metrics 必要开销。
- [ ] Q、N、seeds、graphs、source distributions 足够覆盖。
- [ ] 报告低重叠图和负结果，不只选择有利数据集。
- [ ] artifact 能从 raw data 生成每张主图。

### 15.2 SIGMOD/ICDE 写作检查

- [ ] 标题和摘要体现 data/query processing，而非仅 CUDA optimization。
- [ ] Introduction 形成 Problem → Gap → Insight → Contributions。
- [ ] 每个设计小节说明 challenge、mechanism、correctness、trade-off 和 evidence。
- [ ] 每个实验小节以 research question/hypothesis 开始，以结论和限制结束。
- [ ] 每张图 caption 自包含，坐标和统计协议完整。
- [ ] Related Work 按问题维度综合比较，不逐篇罗列。
- [ ] 12 页内保留端到端、机制、消融、扩展性和开销；不把核心证据放 appendix。
- [ ] SIGMOD 双盲材料和匿名 artifact 去除作者/机构信息。
- [ ] 若投 ICDE，按要求披露生成式 AI 使用并确保作者对内容负责。

## 16. 当前总体判断

当前项目已经具备一条可以成立的系统主线：exact shared state、SharedExpand、CoalescedGather、online grouping/offset 和 hybrid selector 均有代码基础；all-push scheduling、hybrid correctness、外部 baseline 和 NCU 也已有初步数据。真正的短板不是“缺少更多故事”，而是四项核心证据尚未闭环：

1. CoalescedGather 相对合理布局/映射对照的直接 transaction-level 证明；
2. GraphWeft planner 相对 iBFS/Glign-compatible planner 与 offline oracle 的同引擎比较；
3. 当前 threshold 到论文级 cost-aware selector 的实现和 oracle regret；
4. 最终代码、统一协议、多 Q、多 seed 下对 iBFS/Gunrock 的端到端结果。

完成这四项后，叙事逻辑已经足够支撑一篇完整的 SIGMOD/ICDE 系统论文；在此之前，最稳健的表述应是“设计蓝图和 preliminary evidence”，而不是已经验证的全面 SOTA 系统。
