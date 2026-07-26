# Phase 0.1 / 0.2 进展记录

日期：2026-07-18

## 本轮目标

本目录记录 `docs/paper_planning_20260718/submission_todo.md` 中以下两项工作的第一轮完成情况：

- Phase 0.1：从论文和开源代码两个层面对比 iBFS 与 PuerCGP。
- Phase 0.2：从论文和开源代码两个层面对比 Glign 与 PuerCGP 的 batching、offset 和 sharing-maximization 机制。

本轮只完成机制、实现与创新边界分析，不运行性能实验，不修改 PuerCGP 生产代码，也不把待验证的推断写成实验结论。

## 原文可用性检查

用户要求在找不到任一原文时立即中断。检查结果如下，两篇原文均已找到，因此继续执行。

| 系统 | 论文 | 发表信息 | 原文状态 | 本地代码状态 |
|---|---|---|---|---|
| iBFS | *iBFS: Concurrent Breadth-First Search on GPUs* | SIGMOD 2016 | 已获取完整 14 页 PDF | 已找到 |
| Glign | *Glign: Taming Misaligned Graph Traversals in Concurrent Graph Processing* | ASPLOS 2023 | 已获取完整 15 页 PDF | 已找到 |

公开原文：

- [iBFS SIGMOD 2016 PDF](https://www2.seas.gwu.edu/~howie/publications/iBFS-SIGMOD16.pdf)
- [Glign ASPLOS 2023 PDF](https://par.nsf.gov/servlets/purl/10390494)

本地原文与检索文本：

- `sources/iBFS_SIGMOD16.pdf`
- `sources/Glign_ASPLOS23.pdf`
- `notes/iBFS_SIGMOD16.txt`
- `notes/Glign_ASPLOS23.txt`

原文校验值：

| 文件 | SHA-256 |
|---|---|
| `iBFS_SIGMOD16.pdf` | `180a0bb900ccd864e08a8c22457b9e19f7ae88e95f14eba528ca7cf379627012` |
| `Glign_ASPLOS23.pdf` | `2904bb96c1d4d57d00fa22ac286259136623e1c09baf7b54a671ed13d9dfb471` |

## 代码来源

### iBFS

- 路径：`/home/zyl/Projects/ocgp/baselines/iBFS`
- 上游：`git@github.com:iHeartGraph/iBFS.git`
- 当前基准提交：`08455983024543f04e1c4323b1bcc379bbd9930c`
- 主要实现入口：`graph.cuh` -> `bfs_gpu_opt.cuh` -> `expander.cuh` / `inspector.cuh`
- 本地修改主要用于现代 CUDA 构建、`.gr` 数据读取和外部 source list 输入；本轮对比基于仍保持不变的联合遍历、GroupBy、bitwise 和方向切换逻辑。

### Glign

- 路径：`/home/zyl/Projects/ocgp/baselines/Glign`
- 上游：`git@github.com:xyin014/Glign-AE.git`
- 上游分支：`ae-checkin`
- 当前基准提交：`8e5a3457570f75569654e7a1b064f6e6ee98b6b0`
- 主要实现入口：`apps/BFS_Batch.C`、`ligra/ligra.h`
- 嵌套仓库元数据备份：`/home/zyl/Projects/ocgp/.nested-git-backup/baselines/Glign.git`
- 本地修改主要用于 `.gr` 读取、OpenMP 配置和额外迭代输出；本轮分析确认这些修改没有改变 Glign 的 source grouping、delay 和 query-oblivious frontier 语义。

### PuerCGP

- 路径：`/home/zyl/Projects/ocgp/puercgp`
- 本轮重点代码：
  - `include/puercgp/kernels/push/shared_push_kernels.hxx`
  - `include/puercgp/kernels/pull/fused_pull_kernels.hxx`
  - `include/puercgp/core/layout.hxx`
  - `include/puercgp/core/atomics.hxx`
  - `include/puercgp/engine/frontier_engine.hxx`
  - `include/puercgp/scheduling/online_offset_evaluator.hxx`
  - `examples/bench_phase_schedule_n128.cu`
  - `examples/bench_online_offset_n128.cu`

## 输出文件

| 文件 | 内容 |
|---|---|
| `phase0_0.1_ibfs_vs_puercgp.md` | iBFS 论文、iBFS 开源实现与 PuerCGP 物理执行机制逐项对比 |
| `phase0_0.2_glign_vs_puercgp.md` | Glign 与 PuerCGP 的预处理、在线 evaluator、batching、offset、frontier 和目标函数对比 |
| `phase0_novelty_assessment.md` | 两项 related-work 核查后的创新边界、可安全使用的论文表述和后续验证优先级 |

## Phase 0 TODO 对照

### 0.1 iBFS

| TODO | 本轮状态 | 说明 |
|---|---|---|
| 阅读 joint traversal、top-down、bottom-up、GroupBy、bitwise、direction switching | 已完成 | 同时核查论文和本地开源实现 |
| 对比 Puer pull 与 iBFS bottom-up | 已完成分析 | 线程映射、状态布局、提前终止和访存均已列出；未做测量 |
| 对比 Puer shared push 与 iBFS JFQ | 已完成分析 | 明确重合点和实现差异 |
| 对比 selective batching 与两条 GroupBy 规则 | 已完成分析 | 包含论文规则与 artifact 实际简化 |
| 输出相同点、差异、预期硬件效果、验证实验表 | 已完成设计 | 实验仅列计划，本轮不执行 |
| 在相同输入上运行端到端实验 | 延后 | 用户明确要求暂不运行实验 |

### 0.2 Glign

| TODO | 本轮状态 | 说明 |
|---|---|---|
| 复核 closest-HV estimator | 已完成分析 | 给出论文算法、artifact 行为和可复现实验规格 |
| 复核 batching 与 delayed start | 已完成分析 | 未在 PuerCGP 中新增 Glign baseline 代码 |
| 对比 landmark、目标函数和 planner | 已完成分析 | 包含 offline oracle 与 online evaluator |
| 相同 window 下端到端实验 | 延后 | 用户明确要求暂不运行实验 |

这里的“已完成分析”不等同于原 TODO 中要求的“已实现并测量”。实现 Glign-compatible baseline 与实际比较仍属于下一阶段。

## 当前结论摘要

1. PuerCGP 的 shared push 与 iBFS 的联合 frontier/bitwise status 共享同一核心思想。论文不能把“多个 BFS 共享一次邻接表遍历”单独作为创新点。
2. PuerCGP 的 dense pull 不是 iBFS bottom-up 的直接重写：Puer 以 query lane 对 `V*Q` 状态执行逐查询归约，iBFS 以顶点线程或 warp 对 128-bit BFS mask 做 BFS 特化传播。两者的并行维度、状态语义和代价结构不同。
3. PuerCGP 的 batching 与 fixed offset 受 Glign 已覆盖的“遍历相位对齐”问题约束。landmark preprocessing、按相位分组和 delayed start 这些大概念不能作为 Puer 独占创新。
4. Puer 当前 online evaluator 相比 Glign 使用 pairwise relative-level histogram、greedy grouping、swap refinement 和 bounded coordinate search，机制更细；但目前主要在 all-push BFS benchmark 中验证，尚未形成 GPU push/pull 统一代价模型的完整证据链。
5. 最稳妥的论文定位是：面向 GPU 多查询图处理的物理执行与代价优化，而不是“首次联合 BFS”或“首次进行 traversal alignment”。

## 本轮未执行事项

- 未编译或运行 iBFS、Glign、PuerCGP。
- 未生成新的 benchmark 数字。
- 未修改 PuerCGP kernel、engine 或 scheduler。
- 未把当前 online evaluator 集成到生产 engine。
- 未更新原始 `submission_todo.md` 的勾选状态，避免把分析完成误记为实验完成。
