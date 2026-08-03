# MITra: A Framework for Multi-Instance Graph Traversal

- 文献：Li et al., PVLDB 2023
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/li_mitra_2023.pdf)
- DOI：https://doi.org/10.14778/3603581.3603594

## 主要内容

MITra 提出统一的 multi-instance traversal 模型：用户通过数值化 vertex rank 与 edge function 描述单实例语义，框架合成同时从多个源点遍历的共享算法，并利用 SIMD。该表达可覆盖多类经典遍历，包括比传统 frontier 语义更一般的情形；实验显示其平均比基于通用框架的方案快一个数量级，并接近手工优化算法。

## 与当前工作的关系

MITra 是多实例“语义与计算共享”的关键对照。它更关注算法合成、表达能力和 CPU/SIMD，而当前工作关注 GPU runtime 中的精确 vertex/query 活跃集合、邻接访问共享以及物理访存算子。若当前系统支持不同状态类型或算法，应与 MITra 的 rank/edge-function 表达边界对照。
