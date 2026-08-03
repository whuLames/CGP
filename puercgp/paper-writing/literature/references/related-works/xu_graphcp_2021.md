# GraphCP: An I/O-Efficient Concurrent Graph Processing Framework

- 文献：Xu et al., IWQoS 2021
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/xu_graphcp_2021.pdf)
- DOI：https://doi.org/10.1109/IWQOS52092.2021.9521293

## 主要内容

GraphCP 面向单机外存环境中的并发图作业。其 benefit-aware sharing execution model 选择性共享图分区的 I/O 与处理，并自适应决定数据装载顺序；Source-Sorted Sub-Block 布局进一步改善处理能力和局部性。论文报告相对 GridGraph、GraphZ 和 Seraph 分别最高约 10.3×、4.6× 和 2.1×。

## 与当前工作的关系

GraphCP 证明共享不是越多越好，需要按收益调度；这一点可用于论证当前工作的 online grouping/cost model。区别在于 GraphCP 优化磁盘 I/O 和外存分区，而当前工作优化 GPU DRAM 的邻接扫描和内存事务。
