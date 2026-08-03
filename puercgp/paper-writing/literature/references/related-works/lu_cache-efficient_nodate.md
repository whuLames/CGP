# Cache-Efficient Fork-Processing Patterns on Large Graphs

- 系统：ForkGraph
- 文献：Lu et al., complete version
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/lu_cache-efficient_nodate.pdf)
- 预印本：https://arxiv.org/abs/2103.14915

## 主要内容

ForkGraph 处理从许多源点启动、彼此独立但共享同一图的 fork-processing pattern。它把图划分为 LLC 可容纳的分区，把查询按分区缓冲；分区内采用顺序且无原子的执行，分区间通过 yielding 和 priority scheduling 减少无效工作。论文还给出 work-efficiency 分析，并在多核 CPU 上报告相对通用图系统最高两个数量级的提升。

## 与当前工作的关系

ForkGraph 与当前工作的工作负载高度相近，但优化目标是 CPU LLC 命中率和分区驻留。当前工作面向 GPU SIMT，重点是跨查询共同扫描邻接表、状态布局、warp 对齐和 DRAM transaction。对比时应避免只报告端到端速度，还应说明两者对查询独立性、原子操作、图分区和负载不均衡的不同处理。
