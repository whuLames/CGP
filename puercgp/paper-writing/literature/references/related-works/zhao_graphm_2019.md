# GraphM: An Efficient Storage System for High Throughput of Concurrent Graph Processing

- 文献：Zhao et al., SC 2019
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/zhao_graphm_2019.pdf)
- DOI：https://doi.org/10.1145/3295500.3356143

## 主要内容

GraphM 面向多个迭代式图作业并发运行时的重复图存储和重复数据访问。它把只读图结构与作业私有状态解耦，让并发作业共享一份图结构；Share-Synchronize 运行时统一组织分区遍历，并通过调度尽量让多个作业复用已经进入缓存、内存或存储层的数据。论文把该设计集成到多类已有图系统，并报告最高约 13× 的性能提升。

## 与当前工作的关系

GraphM 是“跨作业共享图结构访问”叙事的重要前驱，但其核心是 CPU/存储层次上的静态作业和分区调度。当前工作应把差异落到 GPU 单卡上的查询级精确共享：哪些 query 在某一轮共享某个顶点的邻接扫描，以及剩余访问如何通过 warp-level 对齐转化为合并的显存事务。
