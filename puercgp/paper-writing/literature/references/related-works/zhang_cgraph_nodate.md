# CGraph: A Correlations-aware Approach for Efficient Concurrent Iterative Graph Processing

- 文献：Zhang et al.
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/zhang_cgraph_nodate.pdf)

## 主要内容

CGraph 观察到多个迭代式作业反复遍历同一图时，其数据访问存在时间和空间相关性。它采用 data-centric 的 Loading–Trigger–Processing 组织，把并发作业围绕 core subgraph 调度，使已加载的图结构在缓存和内存层次被多个作业复用，从而降低数据访问相对计算的成本；实验报告吞吐最高提升 2.31×。

## 与当前工作的关系

CGraph 与当前工作的动机都来自 memory wall，但 CGraph 以 CPU 上的子图装载与静态作业调度为中心。当前工作需具体说明 vertex sharing 如何按活跃查询集合减少邻接扫描，以及 warp-level 算子如何降低每次 GPU 访问成本。此外，为避免与 CGraph/C-Graph 混淆，不宜继续使用相近系统名。
