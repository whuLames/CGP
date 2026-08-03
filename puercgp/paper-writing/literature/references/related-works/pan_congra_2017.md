# Congra: Towards Efficient Processing of Concurrent Graph Queries on Shared-Memory Machines

- 文献：Pan and Li, ICCD 2017
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/pan_congra_2017.pdf)
- DOI：https://doi.org/10.1109/ICCD.2017.40

## 主要内容

Congra 面向共享内存多核机器上的多用户图查询。它离线刻画查询的执行时间、内存带宽、原子操作及线程扩展性，再据此决定每个查询的核心数和共置组合，以缓解不合适的 intra-query 并行度与查询间资源竞争。论文报告吞吐最高提升约 60%。

## 与当前工作的关系

Congra 通过资源配置减少竞争，不共享查询的实际图计算。当前工作则主动合并相同顶点上的查询工作，并选择 GPU 物理执行方式。Congra 可归入 resource scheduling 类，而不是 vertex/computation sharing 类。
