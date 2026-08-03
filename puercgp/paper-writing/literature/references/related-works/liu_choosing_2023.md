# Choosing the Best Parallelization and Implementation Styles for Graph Analytics Codes

- 文献：Liu et al., SC 2023
- 阅读状态：已阅读全文
- 原文：[本地 PDF](../papers/liu_choosing_2023.pdf)
- DOI：https://doi.org/10.1145/3581784.3607038

## 主要内容

该文为六种图算法系统生成并评估 1106 个 CUDA、OpenMP 和并行 C++ 实现，跨两类 GPU、两类 CPU 和五个图比较 push/pull、vertex/edge-centric、任务粒度、同步与更新方式等组合。结果显示错误实现风格平均可慢 10× 以上，极端组合相差六个数量级，最优方式高度依赖算法、输入和硬件。

## 与当前工作的关系

这项研究为“不能固定使用一种物理算子”提供直接依据。当前工作的 online grouping offset 本质上优化 push，而 warp-aligned 方案优化 pull/gather；论文应把两条路径写成互补候选，由可解释特征或 cost model 选择，而不是强行让单一算子在所有图和查询规模上占优。
