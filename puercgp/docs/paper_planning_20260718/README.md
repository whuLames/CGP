# PuerCGP 论文规划索引

更新时间：2026-07-18

## 文档

1. [Concurrent Graph Processing 相关工作与叙事分析](related_work_narratives.md)
   - 从 TKDE 2024 survey 出发梳理 CCF A 类代表工作。
   - 分析 iBFS、CGraph、GraphM、ForkGraph、Krill、LCCG、Glign、KGraph 等论文的叙事结构。
   - 明确 PuerCGP 与 iBFS/Glign 的 novelty 冲突和可行差异化方向。

2. [PuerCGP 论文详细叙事与章节设计](paper_outline.md)
   - 给出统一 problem formulation 和建议 contributions。
   - 逐章细化 Introduction、Background、System Overview、Method 1/2、Implementation、Experiments 和 Related Work。
   - 为每个核心论点列出所需图表、实验和 reviewer risk。

3. [PuerCGP 投稿导向待办清单](submission_todo.md)
   - 按 P0/P1/P2 和执行阶段拆分实现、正确性、external baseline、mechanism profiling、完整实验、写作和 artifact 工作。
   - 包含 claim-evidence matrix 与 Go/No-Go gates。

## 推荐的论文主线

```text
真实的 concurrent source-based graph queries
  -> 独立 GPU 执行重复发起不规则 topology/state accesses
  -> query dimension 暴露两类互补机会
       1. shared push 减少 adjacency/topology access 数量
       2. query-parallel pull 提高 state access coalescing/transaction efficiency
  -> batch、offset 和 iteration 会改变两类算子的相对成本
  -> online multi-query optimizer 构造并选择低 effective-memory-cost 的执行计划
```

建议使用 `effective memory-access cost`，而不是笼统声称降低单次 DRAM latency。最终证据需要同时覆盖 topology work、memory transactions、stall 和端到端 query throughput/latency。

## 当前最关键风险

1. iBFS（SIGMOD 2016）已经覆盖 GPU concurrent BFS、joint frontier/status、bitwise、GroupBy 和 top-down/bottom-up。
2. Glign（ASPLOS 2023）已经覆盖 query-oblivious frontier、affinity batching、high-degree distance index 和 delayed start。
3. 当前 online evaluator 只在 all-push BFS、`N=128/Q=32`、单 seed 下评估，尚未接入完整 hybrid engine。
4. heterogeneous BFS/SSSP/WCC 功能已实现，但尚无论文级 correctness/performance matrix。
5. 旧 external baseline 结果使用较早代码，且 PuerCGP 在部分 Q 上仍慢于 iBFS，必须统一重跑。

因此，下一步不应继续增加 replenish、SM partition 或更多 kernel，而应先完成：

1. iBFS/Glign direct mechanism comparison。
2. 最终代码上的同 GPU external baseline。
3. online planner 与 hybrid cost model 的正式端到端集成。

## 当前 preliminary evidence

| 方向 | 初步结果 | 证据属性 |
|---|---:|---|
| online batching+offset, all-push BFS, N=128/Q=32 | 6 图 1.152x-2.309x；3 图超过 1.4x | 有潜力，非完整系统结论 |
| offline selective batching+offset | 4 图 1.318x-2.124x | opportunity/oracle study |
| hybrid scheduler regression matrix | 12/12 case 在旧实现 3% gate 内，最高局部 2.47x | 回归证据，非 external speedup |
| V*Q vs Q*V | Q*V pull 慢 7.0x-32.8x | 支撑 layout 选择，需 counters |
| replenish/SM partition/pause | 无稳定收益或回退 | 不进入主贡献 |

这些数字只能按其当前实验边界使用，不能直接拼接为 abstract 中的系统总收益。
