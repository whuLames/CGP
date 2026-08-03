# Q-Graph: Preserving Query Locality in Multi-Query Graph Processing

- 文献：Mayer et al., arXiv 2018
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/mayer_q-graph_2018.pdf)
- 预印本：https://arxiv.org/abs/1805.11900

## 主要内容

Q-Graph 面向分布式环境中数量动态变化、集中于热点区域的局部查询。Q-cut 根据查询工作负载划分图以降低通信，hybrid barrier synchronization 让只跨少数分区的查询避免全局等待；两者还会随负载变化自适应调整。论文报告 Q-cut 相对静态、查询无关划分将平均查询延迟最多降低 57%。

## 与当前工作的关系

Q-Graph 利用的是跨机器的查询局部性和分区局部性，当前工作利用同一 GPU 上不同查询在顶点和邻接访问上的重叠。两者都需要 workload-aware 调度，但代价模型分别针对网络/同步与 GPU 内存事务。
