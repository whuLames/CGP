# An Efficient Memoization Engine for Concurrent Graph Query Processing

- 系统：KGraph
- 文献：Gao et al., ICDE 2025
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/gao_efficient_2025.pdf)
- DOI：https://doi.org/10.1109/ICDE65448.2025.00081

## 主要内容

KGraph 观察到并发图查询之间存在重复计算，因而引入图上的 memoization。为控制保存和查找结果的开销，它只在与查询相关的图分区内进行细粒度缓存，并优先选择具有较高共享潜力的 pivotal queries，而不对所有查询无差别缓存。五类应用上的实验显示，其相对当时系统平均加速约 4.2×。

## 与当前工作的关系

KGraph 复用的是可跨阶段或跨查询保存的计算结果；当前工作的 vertex sharing 更接近同一步中的邻接扫描和状态传播共享。两者可用“复用对象、生命周期、元数据开销、适用算法”四个维度区分。KGraph 也提醒当前系统必须证明 sharing 收益超过 query mask、grouping 和调度开销。
