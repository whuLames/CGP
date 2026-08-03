# Krill: A Compiler and Runtime System for Concurrent Graph Processing

- 文献：Chen et al., SC 2021
- 阅读状态：未取得公开全文；以下仅依据 BibTeX 与公开元数据
- DOI：https://doi.org/10.1145/3458817.3476159

## 主要内容

Krill 结合编译器与运行时处理并发图作业。公开摘要显示，它以 property buffer 统一承载多作业属性，并融合可共同执行的图 kernel，从而减少重复图数据访问和 kernel 开销；论文报告内存访问次数下降超过 6×、性能最高提升 7.67×，并缩短相对 GraphM 的响应时间。

## 与当前工作的关系

Krill 是“减少 Memory Access Count”方向上必须重点讨论的前驱。区别不应停留在 CPU/GPU：还要说明 Krill 的 compiler-driven job fusion 与当前运行时的 query mask/vertex sharing 在适用算法、动态查询到达、分支差异和状态隔离上的不同，并用访问次数消融做直接证据。
