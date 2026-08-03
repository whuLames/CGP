# A General-Purpose Query-Centric Framework for Querying Big Graphs

- 系统：Quegel
- 文献：Yan et al., PVLDB 2016
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/yan_general-purpose_2016.pdf)
- DOI：https://doi.org/10.14778/2904483.2904488

## 主要内容

Quegel 将查询作为一等公民：用户编写面向单个通用查询的 Pregel 风格逻辑，系统以 superstep-sharing 模型批量执行大量只访问少量顶点的轻量查询。它还提供图索引接口，以避免传统全图 vertex-centric 系统在局部查询上的集群资源浪费，并在多类查询上取得数量级提升。

## 与当前工作的关系

Quegel 为 query-centric 编程和批处理提供语义背景，但它主要摊销分布式同步并提升集群利用率，并未解决 GPU 上跨查询的邻接表访问共享与 coalescing。当前工作可以借鉴其查询接口，但贡献应定位在物理执行层。
