# Measuring GPU Utilization One Level Deeper

- 文献：arXiv:2501.16909
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/noauthor_measuring_nodate.pdf)
- 预印本：https://arxiv.org/abs/2501.16909

## 主要内容

该文指出整体 GPU utilization 指标不足以解释并发 workload 的性能与干扰，并用 microbenchmark 分解计算、显存带宽和 SM 内部资源压力。它主张用更细粒度的 kernel-level 干扰估计和资源隔离来解释并发执行，而不能把高 occupancy 或高利用率直接视为高性能。

## 与当前工作的关系

该文不是 concurrent graph processing 系统，但可指导评测指标选择。当前工作除端到端时间外，应报告 DRAM bytes/transactions、global-load efficiency、带宽、memory dependency stalls 和 SM 活跃度，用数据证明性能来自访问次数与单次访问成本的下降，而非简单提高并发度。
