# Scheduling of Graph Queries: Controlling Intra- and Inter-query Parallelism for a High System Throughput

- 文献：Hauck, Oukid, and Fröning
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/hauck_scheduling_nodate.pdf)
- 预印本：https://arxiv.org/abs/2110.10797

## 主要内容

该文联合控制图查询内部并行度和查询之间的并发度。系统采样图特征，从算法与硬件约束生成 work packages，并结合主动准备与运行时反馈，在顺序/并行执行以及资源分配之间切换；在最高 16 个会话和多种 BFS/PageRank 设置下，性能接近人工调优且调度开销较低。

## 与当前工作的关系

该工作选择“给每个查询多少并行资源”，当前工作选择“哪些查询共享哪些顶点工作、用哪种 GPU memory-access path”。二者可共同放入 adaptive scheduling，但当前论文需展示 cost model 特征直接对应 memory access count/cost，而不是只预测总执行时间。
