# MultiLyra: Scalable Distributed Evaluation of Batches of Iterative Graph Queries

- 文献：Mazloumi et al., IEEE BigData 2019
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/mazloumi_multilyra_2019.pdf)
- DOI：https://doi.org/10.1109/BigData47090.2019.9006359

## 主要内容

MultiLyra 在分布式集群上批量执行迭代图查询，把多查询的通信、同步和部分计算成本摊销到一个批次。其细粒度查询跟踪避免整批查询被最慢实例拖住，并通过值复用和扩展性优化降低昂贵阶段的代价。四种算法、四台 32 核机器上的最高加速范围为 7.35–11.86×，并优于 Quegel 的批处理。

## 与当前工作的关系

MultiLyra 主要减少分布式通信和同步，当前工作主要减少 GPU memory traffic。它可支持“批处理能够摊销公共成本”的总体动机，但不是 warp-level 邻接共享的直接技术前驱。
