# Phase 0 合并判断：PuerCGP 的创新边界

## 1. 当前判定

状态：**Yellow，存在可区分的 GPU 多查询执行机制，但尚未证明它不是 iBFS 与 Glign 机制的工程组合。**

这不是单纯的写作包装问题。论文叙事可以决定读者先看到哪个问题，但不能改变已有工作已经覆盖的机制。当前真正需要回答的是：PuerCGP 的新组合是否产生了一个无法由 iBFS shared traversal 或 Glign phase alignment 单独解释的 GPU physical plan，以及这个 plan 是否稳定带来可测收益。

已有代码给出了一个合理候选：

```text
shared push:
  用 exact query mask 合并多个 query 的重复 adjacency access

query-lane pull:
  接受更多 edge-query relaxation，换取 query dimension coalescing

online planner:
  通过 grouping/offset 改变每轮 union-edge work 和 query density

hybrid optimizer:
  逐轮选择预计 memory cost 更低的 physical operator
```

但当前证据链在 `online planner -> hybrid operator cost` 处断开：online evaluator 主要在 all-push BFS wrapper 中验证，尚未证明它改善完整 hybrid engine 的 physical plan。

## 2. 已被 prior work 覆盖的内容

### 2.1 iBFS 已覆盖

- GPU concurrent BFS；
- joint traversal；
- joint/union frontier；
- query status 在同一 vertex 下连续布局；
- bitwise multi-query state；
- top-down 与 bottom-up；
- sharing-aware GroupBy；
- 通过邻接复用减少 global memory transactions。

### 2.2 Glign 已覆盖

- concurrent graph queries 的 traversal misalignment；
- query-oblivious union frontier；
- intra-iteration alignment；
- heavy-iteration alignment；
- high-degree landmark BFS preprocessing；
- affinity-oriented batching；
- fixed delayed start；
- bounded batching window；
- vertex-major `V*Q` value layout。

## 3. 不应使用的论文 claim

以下表述风险很高，应直接排除：

| 不安全表述 | 原因 |
|---|---|
| 首个 GPU concurrent BFS/system | iBFS 直接反例 |
| 首个 shared/union frontier | iBFS、Glign 均覆盖 |
| 首次一次邻接遍历服务多个 query | iBFS joint traversal |
| 首个 query-contiguous `V*Q` layout | iBFS JSA 与 Glign values 均已有 |
| 首个 sharing-aware batching | iBFS GroupBy 与 Glign batching |
| 首个 traversal phase alignment | Glign 核心贡献 |
| 首个 delayed start / offset | Glign inter-iteration alignment |
| 首个 concurrent query push/pull | iBFS 已有 top-down/bottom-up |
| landmark evaluator 本身是新算法 | Glign 已使用 high-degree BFS distance profile |

## 4. 当前可安全陈述的差异

这些是代码事实，可以写成 design differences；在完成实验前不应直接写成 superior contributions。

1. Puer pull 用 query-lane mapping 对 typed `V*Q` value matrix 做 relaxation；iBFS bottom-up 用 vertex/neighbor mapping 对 BFS bitmask 做 OR。
2. Puer sparse push 保留 exact per-level query mask，并将 cumulative visited 独立维护；iBFS BSA 以 cumulative bitset 为核心。
3. Puer 用 algorithm traits 统一 BFS、SSSP、WCC 状态初始化、relaxation、update 与 infinity；iBFS 是 BFS-specific。
4. Puer 同质 engine 根据实际迭代 work 动态选择 push/pull；iBFS artifact 使用固定 `sw_level`。
5. Puer online planner 使用 pairwise relative-level histogram、greedy clustering、swaps 和 coordinate offset；Glign 使用 closest-HV 标量排序和闭式 delay。
6. Puer exact-mask push 中 alignment 主要减少 union adjacency work；Glign alignment 还减少 query-oblivious frontier 带来的 inactive-query work。

## 5. 最可能成立的贡献组织

### Contribution 1：GPU multi-query physical operators

把两个 memory plans 放在统一模型中：

- shared push 减少重复 adjacency accesses 的数量；
- query-lane pull 改善随机 per-query state access 的 transaction efficiency；
- exact query masks 在共享与精确执行之间取得空间/计算折中。

需要证明 query-lane pull 相对 iBFS-style bitmask bottom-up 或其他合理 mapping 的硬件差异，而不只是比单查询 baseline 快。

### Contribution 2：Workload-aware operator selection

把当前 batch、iteration、union frontier、virtual query-edge work 和 query density 映射为 push/pull cost，并逐轮选 operator。

需要从 threshold 升级为可解释 cost model，并报告相对 per-iteration oracle 的选择准确率和误选代价。

### Contribution 3：GPU-cost-aware query plan construction

batching 与 offset 不单独作为新概念，而是用来构造更适合上述 GPU physical operators 的 query plan。Puer 的 pairwise estimator 应预测：

- union vertices / union edges；
- exact mask density；
- push/pull 选择变化；
- 最终 GPU transaction/runtime，而不仅是抽象 affinity。

需要直接优于 iBFS GroupBy 与 Glign closest-HV，并把 evaluator overhead、index cost 和 waiting latency计入。

## 6. “工程组合”与“新方法”的判别标准

### 6.1 会被判为工程组合的情况

- shared push 的收益可以完全由 iBFS joint traversal 解释；
- offset 的收益可以完全由 Glign closest-HV delayed start 复现；
- hybrid 只使用经验阈值，没有预测准确性；
- 各组件只分别加速，没有相互改变计划或 cost；
- 主要结果来自 offline trace oracle；
- GPU 指标只报告 utilization，没有 transaction/stall 证据。

### 6.2 可以支持新 GPU 多查询方法的情况

- query-lane pull 在相同逻辑 work 下显著降低 memory transaction cost；
- planner 对 union-edge / query density 的预测显著优于 Glign scalar heuristic；
- grouping/offset 会系统性改变 push/pull 最优选择；
- joint optimizer 明显优于“iBFS-like execution + Glign scheduler”的直接组合；
- online end-to-end 收益在多图、多 source seed、多算法上稳定；
- planner overhead、index memory 和 query waiting 不抵消收益；
- 相对 per-step oracle 的 gap 可解释且较小。

最关键的实验不是“Puer 是否快于单查询系统”，而是：

```text
Puer joint design
vs.
iBFS-like shared execution + Glign-compatible scheduling
```

只有前者持续领先，才能证明设计不是已有组件的直接拼装。

## 7. 当前 preliminary 证据的正确解读

现有记录显示：

- offline selective batching + offset 在四图上约 `1.318x-2.124x`；
- online batching + offset 在六图上约 `1.152x-2.309x`，其中三图超过 `1.4x`；
- pause/resume、replenish、SM partition 等没有形成稳定主收益；
- online 结果目前是 all-push BFS、`N=128/Q=32`、单 source seed 为主；
- 尚无同条件 Glign/iBFS direct comparison。

这些结果足以说明 sharing/phase opportunity 存在，也说明 online estimator 有潜力；它们不足以证明 novelty 或完整系统收益。特别是“超过 FIFO”不等于“超过 Glign heuristic”。

## 8. Phase 0 后的最高优先级

### P0：实现同引擎 closest-HV baseline

严格按 `phase0_0.2_glign_vs_puercgp.md` 的规格实现 Glign preprocessing、sort batching 和 fixed delay。不要先改 Puer planner，以免 baseline 不稳定。

### P0：建立 iBFS 公平输入接口

确保 iBFS 与 Puer 使用完全相同 source files、graph direction、batch-width accounting 和 correctness checking。先验证机制一致性，再跑性能。

### P0：把 online planner 接入 production hybrid engine

移除对 all-push benchmark wrapper 的主结果依赖，使 FIFO、Glign、Puer online 和 offline oracle 都通过同一个 engine 入口。

### P0：补完整正确性

对每个 query 比较全量 distance/value；aggregate visited count 与 sum 只保留为快速 smoke check。

### P1：建立 GPU cost correlation

把 estimator score 与 union edges、DRAM sectors、L2 hit、long-scoreboard stalls、kernel time 做相关性分析。若只与抽象 overlap 相关而与 runtime 无关，当前 online objective 需要重做。

### P1：operator oracle gap

每个 iteration 分别测 push-only 与 pull-only，形成真实最优 mode；比较 threshold/model/oracle，并分析 batch/offset 如何改变 mode sequence。

### P1：online arrival 与 fairness

引入 arrival-ordered bounded window，报告 throughput、batching wait、offset wait、p50/p95/p99 latency，避免 closed-window throughput 不能对应在线场景。

## 9. Phase 0.1 / 0.2 最终状态

| 项目 | 分析 | 代码实现 | 实验验证 | 当前结论 |
|---|---|---|---|---|
| 0.1 iBFS | 已完成 | 本轮无修改 | 未执行 | shared push 高度重合；pull/traits/dynamic choice 有实质差异 |
| 0.2 Glign | 已完成 | Glign-compatible baseline 尚未实现 | 未执行 | pairwise planner 不同，但 broad alignment idea 已有先例 |

本轮已经完成用户要求的逻辑和代码对比。原 `submission_todo.md` 中涉及“实现 baseline、运行相同输入、证明硬件收益”的验收项仍然开放。
