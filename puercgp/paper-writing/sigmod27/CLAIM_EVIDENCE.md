# Claim–Evidence Map

本文中的“已实现”“已有数据”和“投稿前缺失”必须保持区分。绝对路径便于从论文直接回到证据。

| Claim | 已有证据 | 状态与缺口 |
|---|---|---|
| exact per-vertex query mask 支持最多 64 个 query | `/home/zyl/Projects/ocgp/puercgp/include/puercgp/core/types.hxx`; `/home/zyl/Projects/ocgp/puercgp/include/puercgp/state/engine_workspace.hxx` | 已实现；需补内存开销曲线 |
| V×Q value layout 使同一 vertex 的 query values 连续 | `/home/zyl/Projects/ocgp/puercgp/include/puercgp/core/layout.hxx`; `/home/zyl/Projects/ocgp/puercgp/include/puercgp/state/value_matrix.hxx` | 已实现；需补 state-organization/access-cost ablation |
| VertexShare 对 union frontier vertex 的邻接表只扫描一次 | `/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/push/`; `/home/zyl/Projects/ocgp/puercgp/include/puercgp/engine/push_executor.hxx` | 已实现；需用 DRAM bytes 和 exact edge counters 建立因果链 |
| AlignedGather 按 vertex 组织连续的 query-state 访问 | `/home/zyl/Projects/ocgp/puercgp/include/puercgp/kernels/pull/fused_pull_kernels.hxx` | 已实现；当前 direct row-query kernel 是默认，smem tiled 不是默认 |
| BFS/SSSP/SSWP 复用 typed policy | `/home/zyl/Projects/ocgp/puercgp/include/puercgp/algorithms/`; `/home/zyl/Projects/ocgp/puercgp/include/puercgp/core/algorithm_traits.hxx` | 已实现；调度收益目前仅对 BFS 有证据 |
| landmark index + pairwise histograms + batching/offset search | `/home/zyl/Projects/ocgp/puercgp/include/puercgp/scheduling/`; `/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/README.md` | 已实现；需要多 seed 和 Glign-compatible ablation |
| online planner 在 all-push BFS 上取得 1.15×–2.31× | `/home/zyl/Projects/ocgp/puercgp/experiments/20260713_online_offset/README.md`; 对应 `final-*/summary.csv` | preliminary：N=128,Q=32, 单 seed, all-push，不能作为最终 end-to-end headline |
| N=256,Q=64 runner 的 schedule 执行正确 | `/home/zyl/Projects/ocgp/puercgp/experiments/20260719_online_runner_n256_q64/README.md`; 对应 `final-*/runs.csv` | 已覆盖 60 次真实图 run 与 27 组小图回归；仍需整理 artifact test |
| 当前 hybrid BFS 不优于 iBFS | `/home/zyl/Projects/ocgp/experiments/20260719-233131_external_baseline_n256/RESULTS.md` | fixed Q=64 当前 hybrid 的几何平均为 0.49× iBFS；必须解决或收缩主张 |
| gather 存在显著 memory-stall 行为 | `/home/zyl/Projects/ocgp/puercgp/experiments/20260727_slot_group_validation/FINAL_ANALYSIS.md`; `ncu/` | 仅代表性 iteration；需 stratified/full-run profile |
| runtime selector 是 calibrated cost model | 当前代码：`/home/zyl/Projects/ocgp/puercgp/include/puercgp/engine/frontier_engine.hxx` | **尚不成立**。当前仅为 `virtual_edges >= tau * Q * E` 的 threshold，并对 BFS sticky pull |
| GraphWeft 比 iBFS + Glign-compatible combination 更好 | 无 | **关键缺失实验**：必须在同一 engine、同一输入上分解比较 |

## 不应写入提交稿的 claim

- “首次实现 GPU concurrent BFS / union frontier / V×Q layout / sharing-aware batching / delayed start”。这些均有明确先行工作。
- “降低物理 DRAM latency”。当前机制减少触发高延迟访问的次数，并降低剩余访问的有效代价；不能声称改变硬件自身延迟。
- “planner 对 SSSP/SSWP 有收益”。当前 weighted workload 只验证了安全旁路。
- “best mode 比所有 baseline 快”。逐图选择 push/hybrid/pull 是 oracle，不是可部署策略。
