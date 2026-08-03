# BEAD: Batched Evaluation of Iterative Graph Queries with Evolving Analytics Demands

- 文献：Mazloumi et al., IEEE BigData 2020
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/mazloumi_bead_2020.pdf)
- DOI：https://doi.org/10.1109/BigData50022.2020.9378211

## 主要内容

BEAD 扩展批量迭代图查询到图数据持续变化、查询集合也动态变化的场景。它增量维护并复用先前批次的结果，使已有查询能够 fast-forward，新查询也能共享已有进度；anytime execution 允许批次被变化打断后继续。实验覆盖最高 256 个查询，报告相对从头计算最高 26.16×、相对 MultiLyra 最高 5.66× 的提升。

## 与当前工作的关系

BEAD 复用跨批次持久化的历史结果，适合动态图和持续查询；当前工作主要复用一个封闭并发窗口中同轮的邻接访问。二者在复用时间尺度和一致性假设上不同，可作为 future work 中支持动态到达/图更新的参照。
