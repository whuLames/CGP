# A Survey on Concurrent Processing of Graph Analytical Queries: Systems and Algorithms

- 文献：Li et al., IEEE TKDE 2024
- 阅读状态：未取得公开全文；以下仅依据 papers.bib 摘要与公开元数据
- DOI：https://doi.org/10.1109/TKDE.2024.3393936

## 主要内容

该综述从系统和算法两个层面梳理 Concurrent Graph Analytical Queries，并提出三个组织问题：系统利用了什么共享机会，使用什么调度方法最大化共享，以及还采用了哪些优化。它同时总结现有缺口和未来方向，是建立相关工作分类与术语边界的合适入口。

## 与当前工作的关系

当前论文可以沿用其“sharing opportunity—scheduling—optimization”框架，但应进一步突出 GPU memory-access 视角：vertex sharing 定义共享机会，query grouping/online alignment 决定共享能否发生，warp-level 物理算子则降低无法消除的访问成本。该综述本身不是需要对比性能的系统基线。
