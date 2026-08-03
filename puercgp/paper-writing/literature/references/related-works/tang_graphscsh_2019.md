# GraphScSh: Efficient I/O Scheduling and Graph Sharing for Concurrent Graph Processing

- 文献：Liu et al., NPC 2019
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/tang_graphscsh_2019.pdf)
- DOI：https://doi.org/10.1007/978-3-030-30709-7_1

## 主要内容

GraphScSh 面向多存储设备上的单机外存并发图处理。它采用 CGP-balanced 图划分，并通过 I/O 调度协调多个作业对不同存储设备和图分区的请求，既减少带宽冲突，也让作业共享已经读取的图数据。论文报告相对既有外存方案最高约 82% 的性能改进。

## 与当前工作的关系

GraphScSh 与 GraphCP 一样优化存储 I/O，而当前工作处理 GPU DRAM 访问。可将两者共同概括为“数据移动共享”，再指出当前系统首次把共享粒度推进到同轮、同顶点、query mask 和 warp memory transaction。
