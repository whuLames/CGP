# Concurrent Graph Processing 文献索引

本目录按 papers.bib 的 BibTeX key 为每篇学术论文建立独立笔记。内容优先依据已下载原文；无法公开取得全文的条目仅依据 BibTeX 摘要和出版社、作者或机构仓储中的公开元数据概括，并在文件中明确标记。

## 汇总

- BibTeX 条目：27
- 学术论文或预印本：26
- 已下载并校验 PDF：20
- 未取得公开全文：6
- 排除的非学术网页：1（noauthor_claude_nodate）

## 按研究路线分类

### 共享图结构、数据访问与缓存局部性

- [GraphM](zhao_graphm_2019.md)：跨作业共享图结构，并通过调度复用已加载数据。
- [LCCG](zhao_lccg_2021.md)：以专用硬件规整并合并并发图访问。
- [CGraph](zhang_cgraph_nodate.md)：利用并发迭代作业间的时空访问相关性。
- [Krill](chen_krill_2021.md)：通过属性缓冲区与 kernel fusion 减少图数据访问。
- [ForkGraph](lu_cache-efficient_nodate.md)：围绕 LLC 分区执行大量源点查询。
- [Glign](yin_glign_2022.md)：对齐并发遍历，提升 CPU 上的共享和 SIMD 效率。

### GPU 并发遍历与执行方式

- [iBFS](liu_ibfs_2016.md)：GPU 上共享 frontier 的并发 BFS。
- [EGraph](zhang_egraph_2022.md)：GPU 上并发处理动态图快照作业。
- [1106 Programs](liu_choosing_2023.md)：系统比较图算法的实现与并行化方式。
- [GPU analytical queries](wang_concurrent_2014.md)：GPU 查询调度与显存换入换出。
- [GPU utilization](noauthor_measuring_nodate.md)：细化 GPU 利用率和干扰测量。

### 多实例计算、结果共享与自动生成

- [MITra](li_mitra_2023.md)：统一表达并自动合成共享计算的多实例遍历。
- [AutoMI](zhao_automating_2024.md)：把单实例顶点中心算法转换为向量化多实例算法。
- [KGraph](gao_efficient_2025.md)：通过细粒度、选择性的 memoization 复用查询计算。
- [MultiLyra](mazloumi_multilyra_2019.md)：分布式批处理中摊销通信、同步和计算。
- [BEAD](mazloumi_bead_2020.md)：面对图和查询变化增量复用历史批处理结果。

### 分布式、外存与资源调度

- [Seraph 2014](xue_seraph_2014.md) 与 [Seraph 2017](xue_processing_2017.md)：解耦图结构、运行时状态和计算逻辑。
- [GraphCP](xu_graphcp_2021.md)：面向外存 CGP 的收益感知 I/O 共享。
- [GraphScSh](tang_graphscsh_2019.md)：多存储设备上的并发 I/O 调度与图共享。
- [Quegel](yan_general-purpose_2016.md)：面向轻量查询的 query-centric 分布式框架。
- [Q-Graph](mayer_q-graph_2018.md)：保留局部查询的分区与同步局部性。
- [C-Graph](zhou_c-graph_2018.md)：分布式并发 k-hop reachability。
- [Congra](pan_congra_2017.md)：共享内存机器上的并发查询资源配置。
- [Graph query scheduling](hauck_scheduling_nodate.md)：联合控制查询内和查询间并行度。

### 综述

- [CGAQ Survey](li_survey_2024.md)：按共享机会、共享调度和系统优化梳理领域。

## 对当前论文叙事的综合启示

现有工作大致在四个层面消除并发图处理的浪费：共享图结构或 I/O、共享中间计算、调度并发作业，以及针对特定算法联合遍历。当前工作最有辨识度的落点不是笼统的“支持并发”，而是在通用 GPU 上把 Memory Access Latency 拆成两个可测量、可分别优化的因素：减少跨查询的邻接表访问次数，以及降低保留下来的每次访问成本。

因此，Related Work 中应主动承认 iBFS 的 shared-frontier、Glign 的 traversal alignment、GraphM/CGraph/Krill 的跨作业数据共享，以及 MITra/AutoMI 的多实例计算共享；随后说明当前工作同时提供 GPU 上的精确 query mask/vertex sharing、面向合并访存的 warp-level 物理执行方式，以及根据工作负载在不同执行方式间选择的机制。最终边界应由实验而不是措辞支撑：分别报告访问次数、内存事务或字节数、合并效率/带宽、stall latency，以及端到端吞吐和延迟。

## 条目说明

papers.bib 中的 noauthor_claude_nodate 指向腾讯文档中的账号教程，不是 concurrent graph processing 学术文献，因而没有下载到 papers/，也不纳入论文综述。
