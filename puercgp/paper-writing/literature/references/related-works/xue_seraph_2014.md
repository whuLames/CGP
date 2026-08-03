# Seraph: An Efficient, Low-Cost System for Concurrent Graph Processing

- 文献：Xue et al., HPDC 2014
- 阅读状态：未取得公开全文；以下仅依据 BibTeX 与公开元数据
- DOI：https://doi.org/10.1145/2600212.2600222

## 主要内容

Seraph 针对分布式图系统中并发作业各自保存图结构、导致内存占用和容错开销过高的问题。其解耦模型允许作业共享只读图结构，以 copy-on-write 隔离作业修改，并以 lazy snapshot 为不同时刻提交的作业提供一致视图；同时利用增量式 checkpoint 或重建机制降低容错成本。

## 与当前工作的关系

Seraph 的共享粒度是分布式作业级图结构和快照，而不是 GPU kernel 内的邻接扫描或状态事务。它适合作为“共享一份图结构并不等于共享实际访问”的对照：当前工作进一步减少每轮真实发生的内存访问，并优化访问映射。
