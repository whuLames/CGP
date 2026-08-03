# EGraph: Efficient Concurrent GPU-based Dynamic Graph Processing

- 文献：Zhang et al., IEEE TKDE 2022
- 阅读状态：未取得公开全文；以下仅依据 BibTeX 与公开元数据
- DOI：https://doi.org/10.1109/TKDE.2022.3171588

## 主要内容

EGraph 面向图规模超过 GPU 显存、且需要同时分析多个动态图快照的场景。公开信息表明，它利用快照之间的时间和空间相似性，通过 Loading–Processing–Switching 风格的执行组织降低 CPU–GPU 数据传输并提高 GPU 利用率，报告相对集成的既有系统约 2.3–3.5× 的提升。

## 与当前工作的关系

EGraph 的并发对象是动态图快照作业，主要瓶颈是 out-of-core 传输与快照复用；当前工作处理驻留于 GPU 的共享图上的多源查询，目标是 kernel 内邻接访问次数和单次访问事务成本。二者都属于 GPU CGP，但问题设置和 memory hierarchy 层级不同。
