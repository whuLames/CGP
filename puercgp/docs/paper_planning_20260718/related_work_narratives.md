# Concurrent Graph Processing 相关工作与叙事分析

更新时间：2026-07-18

## 1. 检索范围与结论先行

本文从 TKDE 2024 survey [A Survey on Concurrent Processing of Graph Analytical Queries: Systems and Algorithms](https://doi.org/10.1109/TKDE.2024.3393936) 出发，优先分析 CCF 2022 目录中的 A 类会议或期刊，并补充少量直接相关的 B/C 类工作。CCF 等级依据 [CCF 推荐目录 2022 第六版（2024-06-28 修订）](https://ccf.atom.im/v6.1/)。

最重要的结论如下。

1. 顶会论文通常不是仅用“现实中存在并发查询”完成动机，而是用真实工作负载或 profiling 证明并发查询之间存在可利用的共享，同时证明朴素并发无法兑现这种共享。
2. 成功叙事普遍遵循“真实 workload -> 可量化机会 -> naive 方案失败 -> 针对性机制 -> 机制指标和端到端收益”的结构。
3. “减少数据访问代价”是该领域已经被验证过的顶会叙事。CGraph、GraphM、ForkGraph、Krill、LCCG、Glign 和 iBFS 都以不同方式围绕重复访问、缓存失配或访存不规则性展开。
4. PuerCGP 不能声称首次在 GPU 上执行并发 BFS、首次使用 shared frontier、首次做 sharing-aware batching、首次采用 push/pull，或首次通过 delayed start 对齐迭代。这些内容分别已被 iBFS 和 Glign 明确提出。
5. 更可信的差异化方向是：把 GPU 上的 shared push、query-dimensional coalesced pull 和代价驱动的物理执行计划选择统一为一个多查询执行引擎，再用在线调度增加这些算子的共享机会。但该差异仍需通过与 iBFS/Glign 的直接机制和实验对比来确认。

## 2. Survey 给出的分析框架

Survey 将 Concurrent Graph Analytical Queries（CGAQ）工作归纳为三个问题：

1. 系统利用了什么 sharing opportunity？
2. 系统采用什么 scheduling 技术放大 sharing？
3. 系统使用什么 execution/system optimization 将 sharing 转化为性能？

这个框架适合直接用来组织 PuerCGP 的 related work：

| 维度 | 代表性已有工作 | PuerCGP 应回答的问题 |
|---|---|---|
| Sharing opportunity | iBFS/Krill 的联合遍历，CGraph/GraphM 的图结构访问共享，KGraph 的结果复用 | 同一顶点的邻接表访问能否跨 query 共享？同一邻居的 query state 访问能否合并事务？ |
| Scheduling | iBFS GroupBy，Glign affinity batching/delayed start，ForkGraph partition scheduling | 哪些 query 应进入同一 batch？何时启动？每轮应执行 push 还是 pull？ |
| Execution optimization | bit mask、kernel fusion、graph partitioning、hardware coalescing | 如何在 GPU 上以低开销维护 per-query 状态并把共享转化为更少或更高效的内存事务？ |

PuerCGP 的相关工作不应只是按年份罗列，而应沿这三个维度指出：已有工作利用了哪一种共享，它在哪种硬件和算法范围内成立，以及 PuerCGP 补充了什么尚未覆盖的执行选择。

## 3. 核心论文概览

| 工作 | 发表 venue | CCF | 核心问题 | 主要方法 | 与 PuerCGP 的关系 |
|---|---|---:|---|---|---|
| [iBFS](https://doi.org/10.1145/2882903.2882959) | SIGMOD 2016 | A | GPU 上多个 BFS 独立执行无法充分共享 frontier 和访存 | joint traversal、joint frontier/status、GroupBy、bitwise、top-down/bottom-up | 最直接先行工作，覆盖 shared frontier、batching 和 push/pull |
| [CGraph](https://www.usenix.org/conference/atc18/presentation/zhang-yu) | USENIX ATC 2018 | A | 并发迭代作业重复访问共享图并产生缓存干扰 | LTP 数据中心执行、统一 partition 访问顺序、core-subgraph scheduling | 支撑“减少访问次数和数据访问成本”主线 |
| [GraphM](https://doi.org/10.1145/3295500.3356143) | SC 2019 | A | 单作业 storage engine 导致多作业重复存储与访问 | graph/job state 解耦、Share-Synchronize、partition scheduling | 数据访问共享和存储层解耦的代表 |
| [ForkGraph](https://doi.org/10.1145/3448016.3457253) | SIGMOD 2021 | A | FPP query 的不协调随机访问造成严重 LLC miss | LLC-sized partition、buffered execution、yield/priority scheduling | 很好的数据库式 problem-to-system 叙事范例 |
| [Krill](https://doi.org/10.1145/3458817.3476159) | SC 2021 | A | 多作业访存互相干扰且 property 管理低效 | property buffer、graph kernel fusion | 与 query mask/fused execution 接近 |
| [LCCG](https://doi.org/10.1145/3458817.3480854) | SC 2021 | A | 并发遍历不规则且资源竞争严重 | topology-aware regularization、图访问合并、state access coalescing | 几乎直接支持“访问次数 + 单次访问效率”的统一叙事，但属于硬件方案 |
| [Glign](https://doi.org/10.1145/3567955.3567963) | ASPLOS 2023 | A | query traversal 在迭代上失配，破坏共享甚至增加 cache miss | query-oblivious frontier、delayed start、affinity batching | 与 shared frontier、offset 和 batching 高度重叠 |
| [EGraph](https://doi.org/10.1109/TKDE.2022.3171588) | TKDE 2023 | A | 动态图多 snapshot 作业的 CPU-GPU 传输重复 | LPS 模型、利用 snapshot 间数据访问相似性 | GPU 并发图系统，但问题是动态 snapshot/传输，不是静态多源遍历 |
| [PMGraph](https://doi.org/10.1145/3689337) | TACO 2024 | A | streaming graph 上 CGQ 有重复计算和昂贵访存 | 合并相同顶点、规则化更新顺序、prefetch | 说明时序调度、合并与隐藏延迟可以形成统一设计 |
| [KGraph](https://doi.org/10.1109/ICDE65448.2025.00081) | ICDE 2025 | A | CGQ 结果存在大量重复计算，但 naive memoization 代价高 | partition memoization、pivot selection、cost model | 数据管理叙事最完整的近期范例 |
| [MultiQx-GPU](https://www.vldb.org/pvldb/vol7/p1011-wang.pdf) | PVLDB 2014 | A | 单 SQL query 不能占满 GPU，朴素并发会发生资源冲突 | query scheduling、device-memory swapping | 可作为 utilization-first 叙事对照，不是图查询直接先行工作 |
| [Sharing Data and Work](https://www.vldb.org/pvldb/vol6/p637-psaroudakis.pdf) | PVLDB 2013 | A | query-at-a-time 丢失并发分析查询间的共享 | simultaneous pipelining 与 global query plan | 为“多查询物理执行计划”提供数据库理论定位 |

补充的直接相关工作包括 ICPP 2018（CCF B）的 [C-Graph](https://doi.org/10.1145/3225058.3225136)、HiPC 2020（CCF C）的 [SimGQ](https://doi.org/10.1109/HiPC50609.2020.00014)，以及 GRADES-NDA 2018 workshop 的 [Q-Graph](https://doi.org/10.1145/3210259.3210265)。它们不作为主要顶会叙事样本，但需要在 related work 中覆盖。

## 4. 逐篇叙事逻辑

### 4.1 iBFS：从专门应用到共享执行

论文：[iBFS: Concurrent Breadth-First Search on GPUs](https://www2.seas.gwu.edu/~howie/publications/iBFS-SIGMOD16.pdf)

叙事链条：

1. BFS 是大量图应用的基础操作；一些应用需要从不同源点同时运行大量 BFS。
2. 顺序执行或让多个 BFS kernel 独立并发，没有利用多个 traversal 的 shared frontier，也不能稳定获得高 GPU 并行度。
3. 实测 frontier 可以在 query 间显著重叠，因此可将多个 BFS 变为一次 joint traversal。
4. joint frontier queue 和 joint status array 将多个实例放进一个 kernel；bitwise 状态让一个机器字表示多个 BFS。
5. GPU 容量限制使所有 query 无法一次执行，因此用基于源点出度和公共高出度邻居的 GroupBy 构造高 sharing batch。
6. top-down 通过共享邻接访问减少工作，bottom-up 通过共享和 early termination 改善执行。
7. 最后用消融、访存事务、单 GPU 性能和 112 GPU 扩展性闭环。

对 PuerCGP 的约束：

- `shared frontier + query bit mask` 不是新概念。
- sharing-aware batching 不是新概念。
- GPU concurrent BFS 的 top-down/bottom-up 切换不是新概念。
- 必须直接比较 PuerCGP 与 iBFS 的 pull thread mapping、内存 layout、切换策略和 GroupBy。

可能的差异：iBFS 主要是 BFS 特化系统；PuerCGP 试图提供可扩展算法语义、query-parallel pull、统一 cost model 和在线跨 iteration 对齐。但这些差异必须表现为明确机制与端到端收益，不能只写成工程泛化。

### 4.2 CGraph：先证明空间/时间相关性

论文：[CGraph: A Correlations-aware Approach for Efficient Concurrent Iterative Graph Processing](https://www.usenix.org/system/files/conference/atc18/atc18-zhang-yu.pdf)

叙事链条：

1. 用真实社交网络 trace 证明同一平台会同时运行多个 iterative graph jobs。
2. 指出 graph structure 占主要数据量，而各 job 沿不同路径独立访问它，产生重复加载、缓存干扰和有限带宽竞争。
3. 将机会定义为 spatial correlation 和 temporal correlation，而非泛泛的“并发性”。
4. 提出 data-centric LTP 模型，将 graph structure 与 job-specific state 解耦，统一 partition 的访问次序。
5. 再用 core-subgraph scheduling 决定加载顺序，放大 cache reuse。
6. 以 data-access time、吞吐和不同作业数证明相关性被兑现。

可借鉴点：PuerCGP 也应先定义可测量的两类共享，而不是先介绍 kernel。适合定义为：

- topology sharing：同一 frontier vertex 的 adjacency list 跨 query 只访问一次；
- query-dimensional locality：同一 neighbor 的多个 query value 通过连续 layout 合并访问。

### 4.3 GraphM：从 storage engine 不匹配切入

论文：[GraphM: An Efficient Storage System for High Throughput of Concurrent Graph Processing](https://www.comp.nus.edu.sg/~hebs/pub/SC19-GraphM.pdf)

叙事链条：

1. Facebook/Huawei 类平台会在同一图上运行大量迭代任务。
2. 现有 storage engine 与单个 graph system 高耦合，导致公共图的多份拷贝和重复访问。
3. 真实 trace 和 profiling 分别证明并发度、共享图比例、重复访问次数和 LLC miss。
4. 将解决方案定位为可插入现有系统的通用 storage layer，而不是另一个孤立算法。
5. Share-Synchronize 统一 partition 流式顺序；scheduler 提高已加载 partition 的利用率。
6. 在多个已有 graph systems 上集成，证明通用性。

可借鉴点：如果 PuerCGP 强调 framework，需要在多个算法和 workload 上证明统一物理执行层确实通用。只有 BFS 结果不足以支持 framework 级声明。

### 4.4 ForkGraph：定义新 workload pattern

论文：[Cache-Efficient Fork-Processing Patterns on Large Graphs](https://arxiv.org/abs/2103.14915)

叙事链条：

1. 先定义 Fork-Processing Pattern（FPP）：同一图上从大量不同源点发起独立 query。
2. 用 BC、NCP、landmark labeling 等完整应用证明 FPP 不是人为构造的 microbenchmark。
3. 对 Ligra、Gemini、GraphIt 做 profiling，说明 inter-query parallelism 虽更快，却显著增加 LLC miss。
4. 核心 insight 是把 graph 划成 LLC-sized partitions，将 query operation 按 partition 缓冲执行。
5. intra-partition 采用 work-efficient sequential algorithm 和 atomic-free consolidation；inter-partition 用 yielding 与 priority scheduling 减少冗余工作。
6. 同时提供理论 work-efficiency 和应用级性能。

可借鉴点：Introduction 里应给出至少两个由并发 source query 组成的上层应用，而不是只说多用户同时提交 BFS。算法/系统工作最好同时有机制证明和应用级意义。

### 4.5 Krill：把并发作业变成一个整体执行计划

论文：[Krill: A Compiler and Runtime System for Concurrent Graph Processing](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap214.html)

叙事链条：

1. 单作业 graph systems 面对 massive concurrent jobs 时出现访存干扰和 property 管理问题。
2. compiler 用 property buffer 统一生成和管理每个 query 的状态。
3. runtime 用 graph kernel fusion 将多个 jobs 作为整体处理，直接减少 graph memory accesses。
4. 用 memory-access reduction、throughput 和 response latency 同时证明机制与用户收益。

可借鉴点：PuerCGP 的 query mask 和 `V*Q` value matrix 可以被描述为 multi-query physical state，而不是 CUDA 实现细节；但必须解释它相对 Krill property buffer/kernel fusion 的 GPU 特有收益。

### 4.6 LCCG：访问次数与访问效率可以统一

论文：[LCCG: A Locality-Centric Hardware Accelerator for High Throughput of Concurrent Graph Processing](https://wrap.warwick.ac.uk/id/eprint/159862/)

叙事链条：

1. 并发图处理同时受到 irregular traversal 和 resource contention 限制。
2. topology-aware execution 在线规则化多个 traversal。
3. 对 graph topology access 做 consolidation/reuse，同时对多个 job 的 vertex-state access 做 coalescing。
4. 通过定制硬件将这两类 locality 变为高吞吐，并报告面积代价。

可借鉴点：这篇论文说明用户当前的统一叙事是成立的：shared push 从访问数量角度降低 topology traffic，query-parallel pull 从 transaction/coalescing 角度降低有效访问代价。不过 PuerCGP 必须说明软件 GPU 上如何完成这一点，并与 LCCG 的硬件假设区分。

### 4.7 Glign：最需要正面处理的重叠

论文：[Glign: Taming Misaligned Graph Traversals in Concurrent Graph Processing](https://par.nsf.gov/servlets/purl/10390494)

叙事链条：

1. 并发查询理论上可共享图访问，但实测 Ligra-C/Krill 的 LLC miss 降幅有限，有时甚至比顺序执行更多。
2. 原因不是 sharing 不存在，而是不同 source 的 traversal 在空间和迭代时间线上 misaligned。
3. 将失配分成三个层次：intra-iteration、inter-iteration 和 batch-level。
4. query-oblivious frontier 合并同一 global iteration 的 frontier；即使增加部分计算，也减少状态和图访问开销。
5. heavy iteration 通常主导成本且更容易重叠，因此通过 delayed start 对齐 heavy iteration。
6. 用到高出度顶点的最短 hop 估计 heavy-iteration arrival，并按该估计做 affinity-oriented batching。
7. 以 LLC misses、分层消融和端到端性能证明三层 alignment。

与当前 online evaluator 的直接重叠：

- Glign 已经离线选择高出度顶点并从反图执行 BFS，保存每个 vertex 到这些顶点的距离。
- Glign 已经用距离预测 heavy iteration arrival。
- Glign 已经用 delayed start 和 affinity batching。

当前 PuerCGP evaluator 使用更多 degree-proportional landmarks、相对 level-difference histogram、batch swap 和 coordinate ascent，算法细节不同，但问题定义高度相同。因此：

1. offset evaluator 不宜单独列为第一贡献。
2. 必须实现或复现 Glign-style closest-high-degree baseline。
3. 必须证明 GPU-aware score 比 Glign heuristic 更能预测 GPU union-edge reduction 或执行时间。
4. 最合理的位置是作为 GPU physical operators 上方的 optimizer 组件，而不是宣称新的 traversal alignment 概念。

### 4.8 PMGraph：动态数据中的合并、排序和预取

论文：[PMGraph: Accelerating Concurrent Graph Queries over Streaming Graphs](https://doi.org/10.1145/3689337)

叙事链条：

1. streaming graph 上同时存在更新和大量 CGQ，现有 static-CGQ 或 single-query streaming 系统无法直接适用。
2. 重复计算和昂贵内存访问是主要问题。
3. 合并不同 query 处理的相同 vertex，规则化更新顺序以增加 overlap。
4. 根据全局 active vertex set 预取共享 vertex data，隐藏访存延迟。
5. 同时比较软件系统和硬件 accelerator。

可借鉴点：论文可以讨论“减少请求数”和“隐藏/降低单次请求延迟”是互补维度，但 PuerCGP 的静态图 scope 必须写清楚，避免与 streaming/dynamic graph 混淆。

### 4.9 KGraph：数据管理论文最值得模仿的叙事

论文：[An Efficient Memoization Engine for Concurrent Graph Query Processing](https://gaosen.org/resources/papers/KGraph.pdf)

叙事链条：

1. 用具体应用和真实 trace 开场：Grab Jakarta 峰值每分钟超过 1.5K 个 SSSP query。
2. 用完整应用说明 CGQ 是上层算法瓶颈，例如 landmark labeling 和 random walk workload。
3. 量化 repeated computation，而不是只依赖直觉。
4. 引入自然方案 memoization，但立即用实验说明 naive memoization 在大图上可能回退约 20%。
5. 把问题转化为 benefit/overhead trade-off：哪些结果值得保存、保存到什么粒度。
6. 用 partition-level memoization 和 pivot query selection 控制代价，再用 cost model 配置。
7. 以五类应用、真实 Grab case study、消融和内存开销完成闭环。

可借鉴点：PuerCGP 应同样展示 naive GPU concurrency 或单一物理算子为什么不足，并将 hybrid choice 写成 cost-based plan selection。负面结果也可以用于缩小设计空间，但不要把 replenish、SM partition 等所有探索都塞进主文。

### 4.10 MultiQx-GPU：可用但不建议作为主切入点

论文：[Concurrent Analytical Query Processing with GPUs](https://www.vldb.org/pvldb/vol7/p1011-wang.pdf)

叙事链条：

1. 单个分析 SQL query 对 GPU copy/compute/memory 的利用率不超过约 25%。
2. 数据仓库天然存在多用户并发，因此 co-running 可填充空闲资源。
3. 朴素 co-running 会带来 device-memory 冲突、thrashing 和 query failure。
4. 用 query scheduler 与 memory swapping 控制并发。
5. 以 throughput、资源利用率和 contention 验证。

对 PuerCGP 的启示：utilization-first 是被 VLDB 接受过的叙事，但它与当前系统的核心机制不完全一致。PuerCGP 的主要收益来自跨 query 共享和合并访存，不是简单地同时启动多个独立 kernel。将 utilization 作为结果指标可以，作为唯一动机会弱化真正贡献。

### 4.11 KGraph、ForkGraph 与通用 MQO 的共同点

[Sharing Data and Work Across Concurrent Analytical Queries](https://www.vldb.org/pvldb/vol6/p637-psaroudakis.pdf) 指出，query-centric 独立优化会丢失 concurrent query 之间的数据和算子共享，系统应形成 shared physical execution。这个数据库传统可以帮助 PuerCGP 避免被写成单纯的 CUDA kernel paper：

- shared push 是一个共享 topology scan/expand operator；
- query-parallel pull 是一个按 query dimension vectorize/coalesce 的 gather operator；
- hybrid scheduler 是 physical operator selection；
- batching/offset 是 multi-query plan construction；
- graph summary/index 相当于 optimizer statistics。

## 5. 优秀论文的共同叙事模板

### 模板 A：共享机会驱动

```text
真实并发 workload
  -> 多 query 访问同一图/重复子计算
  -> 独立执行重复加载或计算
  -> 定义并量化 sharing opportunity
  -> 合并执行或结果复用
  -> 访问量/计算量下降
  -> throughput/latency 提升
```

代表：CGraph、GraphM、ForkGraph、Krill、KGraph、iBFS。

### 模板 B：已有共享机制失效

```text
并发理论上应产生共享
  -> profiling 发现 cache miss/traffic 未下降
  -> 找到 traversal misalignment 或调度失配
  -> 分层定义失配
  -> 调整 batch/时间/访问顺序
  -> locality 指标和端到端性能同步改善
```

代表：Glign、PMGraph。

### 模板 C：硬件资源管理

```text
单 query 资源利用率低
  -> 并发可以填充资源
  -> 朴素并发产生 contention
  -> scheduler/resource isolation
  -> throughput 与利用率提升
```

代表：MultiQx-GPU。该模板不是 PuerCGP 当前最强的选择。

## 6. 对 PuerCGP 写作定位的直接建议

### 6.1 推荐的核心叙事

推荐陈述：

> PuerCGP 将 GPU 上的并发图查询视为一个多查询执行计划，通过 topology sharing 降低不规则邻接访问次数，通过 query-dimensional coalescing 降低 query-state 访问的事务代价，并通过在线代价驱动的调度选择和组织这两类访问方式。

这比“通过并发提高 GPU 利用率”更贴合当前机制，也更接近数据库的 multi-query optimization。

### 6.2 必须避免的 novelty claim

- 不能声称 first GPU concurrent graph/BFS system：iBFS 已在 SIGMOD 2016 完成。
- 不能声称 first shared/joint frontier：iBFS、Krill、Glign 均有相关设计。
- 不能声称 first query bit mask：iBFS 已使用 bitwise status。
- 不能声称 first sharing-aware batching：iBFS GroupBy 和 Glign affinity batching 已存在。
- 不能声称 first delayed start/offset alignment：Glign 已提出 delayed start。
- 不能声称 first concurrent push/pull BFS：iBFS 已支持 top-down/bottom-up。
- 不能把“memory-bound”直接等价为“带宽已饱和”；必须用 long-scoreboard stall、事务效率、cache/DRAM traffic 等指标证明 effective memory-access cost。

### 6.3 有机会成立但尚待证明的 claim

- 一个面向通用 source-based graph queries 的 GPU multi-query engine，而不是 BFS 特化实现。
- 在同一系统中统一 shared push 与 query-dimensional coalesced pull，并用统一的 memory-cost model 选择物理算子。
- 一个同时考虑 degree-weighted topology sharing 与 GPU transaction efficiency 的 optimizer。
- 相比 iBFS GroupBy 和 Glign closest-high-degree heuristic，更准确且在线代价可控的 batch/offset planning。
- 混合 BFS/SSSP query 的统一执行，但前提是补齐正确性、性能和调度证据。

## 7. Related Work 章节建议结构

1. **Concurrent graph query systems**：CGraph、GraphM、ForkGraph、Krill、Glign、KGraph，按 access sharing、computation sharing、alignment 分类。
2. **GPU concurrent graph traversal**：以 iBFS 为核心，补充 EGraph/PMGraph，明确 static query 与 dynamic snapshot/streaming graph 的差别。
3. **Multi-query optimization and shared execution**：用 PVLDB shared execution 工作把 PuerCGP 定位为 graph-specific multi-query physical execution。
4. **GPU graph direction optimization**：在正式写作时补齐 Enterprise、Gunrock、GE-SpMM 等单 query/operator 论文，说明它们优化单 query 或单方向，而非跨 query 的统一执行计划。

## 8. 当前文献结论对项目路线的影响

| 项目已有方向 | 文献判断 | 论文中的建议位置 |
|---|---|---|
| shared-frontier push | 有效但已有强先行工作 | 核心 operator，但 novelty 必须落在泛化、实现差异和 cost model |
| query-parallel pull + `V*Q` | 可能是重要差异 | 核心 operator；必须与 iBFS bottom-up 做 memory-transaction 对比 |
| hybrid switching | 可以统一两类 memory plan | 核心 optimizer；需要预测准确性和 oracle gap |
| selective batching | iBFS/Glign 已覆盖概念 | optimizer 子组件，必须做 direct baseline |
| fixed offset | Glign 已覆盖概念 | 调度增强，不单列首要贡献 |
| online landmark evaluator | 与 Glign 索引高度接近 | 需要 GPU-aware score 和端到端集成后再决定贡献级别 |
| heterogeneous algorithms | 可能有差异化价值 | 至少 BFS+SSSP；WCC 可放 appendix |
| replenish | 实验无稳定收益 | 不进主设计，只作为设计探索或 appendix negative result |
| SM partition / query-level push-pull concurrency | 当前为负结果 | 不进主文，除非后续改变模型并得到稳定收益 |

## 9. 主要参考链接

- [TKDE 2024 CGAQ Survey](https://ink.library.smu.edu.sg/sis_research/9913/)
- [CCF 2022 推荐目录](https://ccf.atom.im/v6.1/)
- [iBFS, SIGMOD 2016](https://www2.seas.gwu.edu/~howie/publications/iBFS-SIGMOD16.pdf)
- [CGraph, USENIX ATC 2018](https://www.usenix.org/conference/atc18/presentation/zhang-yu)
- [GraphM, SC 2019](https://wrap.warwick.ac.uk/id/eprint/130159/)
- [ForkGraph, SIGMOD 2021](https://arxiv.org/abs/2103.14915)
- [Krill, SC 2021](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap214.html)
- [LCCG, SC 2021](https://wrap.warwick.ac.uk/id/eprint/159862/)
- [Glign, ASPLOS 2023](https://par.nsf.gov/servlets/purl/10390494)
- [EGraph, TKDE 2023](https://research.usq.edu.au/item/z274q/egraph-efficient-concurrent-gpu-based-dynamic-graph-processing)
- [PMGraph, TACO 2024](https://doi.org/10.1145/3689337)
- [KGraph, ICDE 2025](https://gaosen.org/resources/papers/KGraph.pdf)
- [SimGQ, HiPC 2020](https://par.nsf.gov/biblio/10267911-simgq-simultaneously-evaluating-iterative-graph-queries)
- [C-Graph, ICPP 2018](https://doi.org/10.1145/3225058.3225136)
- [Q-Graph, GRADES-NDA 2018](https://doi.org/10.1145/3210259.3210265)
- [MultiQx-GPU, PVLDB 2014](https://www.vldb.org/pvldb/vol7/p1011-wang.pdf)
- [Sharing Data and Work Across Concurrent Analytical Queries, PVLDB 2013](https://www.vldb.org/pvldb/vol6/p637-psaroudakis.pdf)
