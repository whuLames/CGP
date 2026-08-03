# Processing Concurrent Graph Analytics with Decoupled Computation Model

- 系统：Seraph
- 文献：Xue et al., IEEE TC 2017
- 阅读状态：未取得公开全文；以下仅依据 papers.bib 摘要与公开元数据
- DOI：https://doi.org/10.1109/TC.2016.2618923

## 主要内容

该文是 Seraph 的期刊版本，进一步系统化解耦运行时作业数据与计算逻辑的并发图处理模型。多个作业共享图结构，copy-on-write 隔离写入，lazy snapshot 支持不同提交时间；统一同步/异步接口和轻量 checkpoint 则扩大调度空间并降低容错开销。论文与 Giraph、Spark、GraphX、PowerLyra 比较内存占用和完成时间。

## 与当前工作的关系

Related Work 可把 2014 与 2017 两篇 Seraph 合并讨论，以期刊版为主引用、会议版说明起源。其核心是分布式作业状态管理；当前工作关注共享图已驻留 GPU 后，访问是否真正合并以及每次访问的硬件代价。
