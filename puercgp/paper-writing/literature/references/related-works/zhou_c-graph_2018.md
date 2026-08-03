# C-Graph: A Highly Efficient Concurrent Graph Reachability Query Framework

- 文献：Zhou et al., ICPP 2018
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/zhou_c-graph_2018.pdf)
- DOI：https://doi.org/10.1145/3225058.3225136

## 主要内容

C-Graph 是面向并发 k-hop reachability 的分布式 edge-set 遍历框架。它维护全局顶点状态，支持同步和异步通信，把查询分解为局部遍历，并通过物理 edge-set 组织和共享子图减少重复工作。实验表明其优于多个基线方案。

## 与当前工作的关系

C-Graph 针对单一查询类型和分布式通信，当前工作希望覆盖 BFS/SSSP 等多种源点图算法并优化单 GPU 访存。应将其列为 query-specific shared traversal，而把当前贡献定位为 GPU 上更通用、类型化的多查询执行机制。
