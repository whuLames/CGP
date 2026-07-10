# Hybrid Batch 后续工作

> **最后更新：2026-07-06**
>
> 本文档替代早期的 "pull 接入 / warp push / bench 待实现" 计划。
> 当前代码已经完成这些功能的主要实现，后续重点转为验证矩阵、性能结论、策略优化和文档/API 清理。

## 当前状态概览

Hybrid heterogeneous batch 的功能核心已经完成：

- `run_heterogeneous` 支持同一 batch 内混合 BFS / SSSP / WCC。
- `traversal_mode = push / pull / hybrid` 已接入主循环。
- `push_strategy = shared_node_warp` 已接入 hybrid push。
- `validate_hybrid_real` 已支持真实图 correctness 验证。
- `bench_hybrid` 已支持 sequential / hybrid / ideal upper 对比。
- `run_replenish_pipeline` 已实现 slot replenishment experimental path。

当前最重要的缺口不是"功能能否跑"，而是：

1. 系统性跑完真实图、多 Q、多算法组合、多 traversal mode 的验证矩阵。
2. 产出 hybrid batch 的论文级性能数据。
3. 明确 replenishment 路线的结论和取舍。
4. 清理过期注释、结果布局和文档。

## 历史计划完成状态

| 旧计划项 | 当前状态 |
|----------|----------|
| P0 真实图验证 | 工具已实现：`validate_hybrid_real.cu`；完整矩阵仍待跑 |
| P1 fused_pull_hybrid 接入 | 已完成，simple/smem pull 均已接入 `run_heterogeneous` |
| P2 `bench_hybrid.cu` | 工具已实现；系统实验和论文表格仍待产出 |
| P3 warp hybrid push | 已完成，`shared_node_warp` 可分派到 hybrid warp kernel |
| P4 完整验证矩阵 | 仍待做 |
| P5 小清理 | 部分仍待做 |
| latency/fairness replenishment | 已实现并实验评估；当前结论为 NO-GO |

## P0：完整验证矩阵

### 现状

`validate_hybrid_real` 已经具备真实图验证能力，但还没有把所有组合系统跑完并固化成记录。
大图上当前 validator 会在 `V > 5,000,000` 时跳过 CPU reference check，因此大图 correctness 证据仍不足。

### 目标

形成一张可复现的 correctness matrix，覆盖：

| 维度 | 候选 |
|------|------|
| 算法组合 | BFS+SSSP、BFS+WCC、SSSP+WCC、BFS+SSSP+WCC |
| Q | 4、16、32、48、64 |
| traversal mode | push、pull、hybrid |
| push strategy | shared_node、shared_node_warp |
| 数据集 | cit-Patents、soc-LiveJournal1、soc-orkut、soc-twitter、soc-sinaweibo |

### 实施要点

1. 用 `validate_hybrid_real` 跑完整组合。
2. 对 `V <= 5M` 的图保留完整 CPU reference。
3. 对更大图补 sampled reference 或离线 reference：
   - 随机抽样若干 source 和 vertex 检查。
   - 对可疑 case 单独导出子图或使用 CPU/GPU baseline 交叉验证。
4. 记录每组命令、GPU 型号、CUDA 版本、是否跑 CPU reference、mismatch 计数。
5. 把结果汇总成新的 `experiments/hybrid_correctness_matrix.md`。

### 风险

- WCC label propagation 在长链或弱连通长直径图上轮次很大，需要统一 `max_iterations` 规则。
- SSSP 使用 `unified_value_t=float`，长路径/大权重图需要记录 max abs/relative error。

## P1：Hybrid 性能实验矩阵

### 现状

`bench_hybrid.cu` 已实现，但还缺系统性实验结果。当前仍不能回答：

- hybrid 是否稳定快于 sequential 同质 batch。
- 哪些算法组合有收益。
- block push 和 warp push 哪个更适合作为默认路径。
- push/pull/hybrid traversal 在不同图结构上的 break-even 在哪里。

### 目标

产出论文可用的 hybrid 性能表：

| 对比 | 指标 |
|------|------|
| sequential vs hybrid | wall_ms、gpu_ms、iterations、speedup |
| block vs warp hybrid push | wall_ms、gpu_ms、speedup |
| push vs pull vs hybrid traversal | mode breakdown、speedup、失败/退化 case |
| 不同算法比例 | BFS/SSSP/WCC 混合比例对性能的影响 |
| 不同 Q | Q=4/16/32/48/64 scaling |

### 实施要点

1. 固定 sources 生成规则，保证 sequential 和 hybrid 使用相同 query set。
2. 每组至少 warmup 1-2 次，repeat 5-7 次取 median。
3. 优先用 `shared_node_warp` 避免 edge-balanced 在大图上分配 QxE 中间结构导致 OOM。
4. 输出 CSV，另存 `experiments/hybrid_benchmark_matrix.md` 记录摘要。
5. 如果 hybrid 输给 sequential，保留 negative result 并定位原因：
   - runtime algo tag 分支开销。
   - BFS/SSSP/WCC 轮次不一致导致长尾。
   - pull 全图扫描放大了异构 slot 计算。

## P2：Pull/Hybrid Traversal 策略优化

### 现状

Hybrid 主循环已经支持 push/pull/hybrid，但 hybrid mode 的切换主要看：

```text
current_unique >= pull_frontier_ratio * V
```

`pull_edge_ratio` 已写入 profile，但 hybrid 决策还没有完整使用 frontier edge work 估计。

### 目标

减少误切 pull 的概率，让 push/pull/hybrid mode 在真实图上更稳定。

### 实施要点

1. 在 hybrid path 中补 frontier edge count 或近似 degree sum。
2. 同时考虑：
   - unique frontier vertices。
   - active query pair count。
   - frontier outgoing edge work。
   - 当前 batch 的 BFS/SSSP/WCC slot 组成。
3. 记录每轮 mode、frontier size、edge count、active slot mask。
4. 用 P1 的 benchmark 结果校准默认阈值。

## P3：Replenishment 路线处置

### 现状

Replenishment 已实现：

- `run_replenish_pipeline`
- active slot convergence detection
- slot snapshot + clear + pending injection
- `discard_results`
- `bench_replenish`

实验记录在：

- `experiments/replenish_throughput_benchmark.md`
- `experiments/replenish_latency_benchmark.md`
- `experiments/replenish_longtail_benchmark.md`
- `experiments/replenish_e2e_throughput.md`

当前实验结论：

- 无长尾 BFS/SSSP workload：throughput `0.71-0.86x`，replenishment 慢。
- latency：p25 有优势，但 median/p90/p99 更差。
- WCC 长尾构造：push `0.84x`，pull `0.41x`，仍然 NO-GO。

### 目标

明确 replenishment 不作为当前主线 contribution，避免继续投入与论文主目标不匹配的优化。

### 建议处置

1. 保留代码作为 experimental feature，默认不开启。
2. 保留实验记录，作为 negative result 或 appendix 候选。
3. 不再把 replenishment 作为 ICDE main bar 的核心支撑。
4. 若以后重启，需要先改变执行模型假设；当前 push 下空 slot mask 过滤几乎免费，pull 下复用会增加全图扫描计算。

## P4：PageRank / Dense Engine

### 现状

`dense_engine` 仍是 placeholder：只分配 values、填 result metadata，不执行 PageRank kernel 或 convergence。

### 目标

如果 `puercgp` 的目标是完整 concurrent graph processing framework，而不仅是 frontier traversal，
需要补 dense algorithm path。

### 实施要点

1. 实现 PageRank 初始化、pull-style rank update、damping factor、convergence check。
2. 增加 `validate_pagerank`，与 CPU reference 或已知实现比对。
3. 决定 PageRank 是否参与 heterogeneous batch；如果参与，需要定义与 BFS/SSSP/WCC 不同轮次语义下的调度规则。
4. 如果论文主线不需要 PageRank，应在 README/API 文档中明确 dense engine 仍为 scaffold。

## P5：API 和结果布局清理

### 现状

不同路径的 result layout 不一致：

- `run_heterogeneous` 返回 vertex-major：`values[v * Q + q]`。
- `run_replenish_pipeline` 返回 row-major：`values[q * V + v]`。

这种差异对 benchmark 和外部用户都容易出错。

### 目标

明确或统一 result layout。

### 可选方案

1. 文档化现状，并提供 helper accessor：
   - `hybrid_value(result, v, q)`
   - `replenish_value(result, q, v)`
2. 统一 layout，但会带来迁移和额外转置成本。
3. 在 `run_result_t` 中增加 layout metadata，避免调用方猜测。

## P6：代码和文档清理

### 仍待清理

- `examples/validate_hybrid.cu` 中 `INF` 未使用，编译时有 warning。
- `include/puercgp/engine/hybrid_engine.hxx` 内仍有旧注释写着 "push-only 第一版"。
- `README.md` 仍说 dense engine 是 placeholder，这点属实；但 README 也没有反映 hybrid pull、
  warp push、真实图 validator、benchmark、replenishment 的最新进展。
- `docs/` 与 `experiments/` 之间缺一个总索引，后续可以补 `docs/experiment_index.md`。

## 建议执行顺序

1. P0：先跑 correctness matrix，确保功能证据稳固。
2. P1：跑 hybrid benchmark matrix，判断 hybrid contribution 的性能强度。
3. P2：根据 benchmark 结果优化 push/pull 切换策略。
4. P6：清理 warning、过期注释和 README。
5. P5：统一或文档化 result layout。
6. P4：只有在论文/系统目标需要 dense algorithms 时再投入 PageRank。

当前主线应聚焦 **hybrid heterogeneous batch 的 correctness + performance proof**。
Replenishment 已经完成实现和评估，但不建议继续作为主贡献推进。
