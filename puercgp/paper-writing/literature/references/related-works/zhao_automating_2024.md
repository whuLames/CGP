# Automating Vectorized Distributed Graph Computation

- 系统：AutoMI
- 文献：Zhao et al., PACMMOD 2024
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/zhao_automating_2024.pdf)
- DOI：https://doi.org/10.1145/3698833

## 主要内容

AutoMI 是 source-to-source 框架，将单实例 vertex-centric 图算法自动转换为向量化多实例版本，并给出哪些算法最适合简化向量化的代数刻画。TrackFree 等优化进一步消除不必要的实例跟踪。六个真实图上的转换结果相对串行、batch 和非向量化手工多实例算法分别取得 9.6–29.5×、7.1–26.4× 和 2.6–4.6× 提升。

## 与当前工作的关系

AutoMI 解决“如何正确生成多实例算法”，当前工作解决“如何在 GPU 上高效执行多实例算法”。如果当前系统需要用户手写 query-state 更新，应承认 AutoMI 在可编程性和正确性方面更强；反之可强调无需静态转换、可依据运行时共享度选择不同 GPU 执行路径。
