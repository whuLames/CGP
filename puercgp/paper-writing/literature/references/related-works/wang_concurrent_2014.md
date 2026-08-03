# Concurrent Analytical Query Processing with GPUs

- 文献：Wang et al., PVLDB 2014
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/wang_concurrent_2014.pdf)
- DOI：https://doi.org/10.14778/2732967.2732976

## 主要内容

该文研究 GPU 数据库分析查询的并发执行。作者观察到逐个专用执行查询时主要 GPU 资源利用率不足，因而设计 GPU query scheduler 与 device-memory swapping 策略，在并发共享 GPU 时控制资源竞争。原型系统相对独占式查询处理将吞吐最高提升 55%。

## 与当前工作的关系

它不是图处理系统，但提供了 GPU 多查询资源管理背景。其并发来自 kernel/查询共置，没有利用查询访问同一图的语义重叠；当前工作是在资源共享之上进一步共享拓扑读取与图计算。
