# Glign: Taming Misaligned Graph Traversals in Concurrent Graph Processing

- 文献：Yin et al., ASPLOS 2023
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/yin_glign_2022.pdf)
- DOI：https://doi.org/10.1145/3567955.3567963

## 主要内容

Glign 针对并发图遍历在迭代进度和活跃顶点上的错位。它在 Ligra 风格 CPU 执行中合并同轮 frontier，并通过延迟启动等方法让不同查询的迭代重新对齐；同时使用 affinity-aware batching 和 query-contiguous 的 V×Q 状态布局，提高共享、缓存局部性和 SIMD 利用率。

## 与当前工作的关系

Glign 是最接近的通用并发遍历基线之一。当前工作不能只把贡献表述为“对齐查询”或“共享 frontier”，而应强调 GPU 上按顶点聚合 query mask 后只扫描一次邻接表，以及让 warp 对边/状态访问形成合并事务的物理执行方式。实验需在相同算法、图和查询批次下拆分比较 grouping、sharing 与访存算子。
