# iBFS: Concurrent Breadth-First Search on GPUs

- 文献：Liu et al., SIGMOD 2016
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/liu_ibfs_2016.pdf)
- DOI：https://doi.org/10.1145/2882903.2882959

## 主要内容

iBFS 在一个 GPU kernel 中联合推进多个 BFS，以 shared frontier 复用不同实例共同访问的顶点和边；outdegree-based GroupBy 选择更适合共同执行的 BFS，bitwise 状态表示让一个线程同时检查多个实例。系统同时支持 top-down/bottom-up 与多 GPU 扩展，单 GPU 相对逐个 BFS 最高加速 30×。

## 与当前工作的关系

iBFS 是最直接的 GPU 基线，当前论文不能声称首次联合 frontier、首次按查询分组或首次用 bit mask 表示实例。可主打三个扩展：从 BFS 专用到多算法/多状态语义；从 frontier sharing 到显式最小化邻接访问次数的 vertex-sharing 模型；从线程级 bitwise 检查到面向合并访存的 warp-level gather/pull 算子与在线选择。Q=16/32 等区间的同平台结果必须单独呈现。
