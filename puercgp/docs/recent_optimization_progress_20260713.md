# PuerCGP 近期优化与完整 Baseline 进展

时间范围：2026-07-09 至 2026-07-20  
实验平台：NVIDIA Tesla V100-SXM2-32GB，CUDA 12.8

## 1. 效果概览

### 本次汇报结论

1. PuerCGP 已形成 BFS、SSSP、SSWP、PageRank 和 PPR 的统一 GPU 并发执行框架，并完成四张真实图上的外部 baseline 测试。
2. 相对通用 GPU baseline，Best-Puer 对 Gunrock BFS stream1 和 sequential baseline 分别获得 11.47x 和 9.52x 几何平均加速。
3. 相对通用 CPU concurrent baseline，Best-Puer 对 Glign 获得 68.30x 几何平均加速；ForkGraph 结果更高，但受共享主机和分区质量影响，只作为 context。
4. 专用 GPU BFS 系统 iBFS 仍是最强直接 baseline：当前 `N=256,Q=64` 下 Best-Puer/iBFS 为 0.694x，Twitter 基本持平，其余三图仍落后。
5. 算子选择具有明确规律：BFS 最优模式依赖图结构，SSSP/SSWP 均由 hybrid 获得最佳结果，PageRank/PPR 均由 pull 获得最佳结果。
6. 20 组跨系统结果签名全部通过；大并发测试同时暴露了 Gunrock、Glign 和 ForkGraph 的 OOM/扩展性问题。

| 优化方向 | 初步性能表现 | 当前结论 |
|---|---:|---|
| 完整 external baseline，N=256 | 对 Gunrock seq/stream1 为 9.52x/11.47x；对 Glign 为 68.30x | 通用系统性能优势明确 |
| 专用 iBFS 对比，N=256/Q=64 | Best-Puer/iBFS 几何平均 0.694x | 高并发 BFS 尚未全面超过专用实现 |
| 多算法扩展 | BFS/SSSP/SSWP/PageRank/PPR 四图运行完成 | 已具备通用并发图查询系统雏形 |
| 跨系统正确性 | 20 个结果签名全部一致 | 当前测试未发现算法结果错误 |
| Hybrid push/pull 调度 | 四图、Q=16/32/64 共 12/12 场景通过 3% 性能门限；最高 2.47x | 有效，已进入主实现 |
| Selective batching，N=128/Q=32 | 四图提升 1.050x-1.237x | 有效，需转化为低开销策略 |
| Selective batching + fixed offset | 四图提升 1.318x-2.124x | 当前最强的调度结果 |
| Q=32 heavy/offset alignment | Offset 提升 1.035x-2.049x | 证明 iteration alignment 可以提升 sharing |
| Replenish reset/scheduling 降本 | 相比旧 replenish 减少 15.2%-31.8% | 维护开销明显降低 |
| Cohort-32 replenish | 相对静态 batching 为 0.997x-1.025x | 基本消除旧实现回退，但不是 immediate replenish 收益 |
| Immediate replenish，roadNet-CA | N=400/800 吞吐下降 7.5%/10.8% | occupancy 提升未转化为性能收益 |
| Push SM scaling | 中大型 workload 在 8-16 SM 最快；80 SM 慢 6.0%-27.1% | Push 存在资源饱和与竞争 |
| Pull SM scaling | 2 到 80 SM 加速 13.1x-24.7x | Pull 持续受益于更多 SM |
| Q=64 query-level push/pull 并发 | 最佳 oracle 仍比 all-pull 慢 1.3%-20.3% | 暂不集成生产引擎 |
| `V*Q` 改为 `Q*V` | Pull kernel 慢 7.0x-32.8x | 保留 `V*Q` 布局 |
| 项目结构重构 | Push 12/12 通过；正确性测试全部通过 | 为后续调度和资源控制提供接口 |

## 2. 完整 External Baseline 对比

### 测试配置

- 数据集：`cit-Patents`、`soc-orkut`、`soc-twitter`、`soc-sinaweibo`。
- BFS/SSSP/SSWP：`N=256,Q=64`；PageRank/PPR：`N=256,Q=32`，固定 10 轮。
- PuerCGP：all-push + online、hybrid + online、all-pull；online evaluator 时间计入端到端时间。
- 默认采用 2 次 warmup、5 次正式测量并报告中位数。
- `Best-Puer` 表示每个 case 在三种模式中事后选择的最优结果，是执行上界，不等于当前在线策略的实际结果。

### 总体效果

加速比定义为 `baseline_time / Best-Puer_time`：

| Baseline | 几何平均加速 | Case 数 | 当前判断 |
|---|---:|---:|---|
| ForkGraph | 139.99x | 7 | CPU context；暂不作为核心 claim |
| Glign | 68.30x | 11 | 通用 CPU concurrent baseline |
| Gunrock sequential | 9.52x | 8 | 同 GPU、逐 query baseline |
| Gunrock BFS stream1 | 11.47x | 4 | 同 GPU BFS baseline |
| Gunrock BFS stream64 | 5.96x | 1 | 其余三张大图 OOM |
| iBFS | 0.694x | 4 | 专用 GPU concurrent BFS 更快 1.44x |

当前最有说服力的结果是相对同 GPU Gunrock 的 9.52x-11.47x 加速。CPU baseline 的大幅加速说明 GPU 并发执行具有潜力，但不能用来单独证明算子优越性。

### 与 iBFS 的直接对比

| 数据集 | Best-Puer 模式/时间 | iBFS 时间 | Puer 相对 iBFS |
|---|---:|---:|---:|
| cit-Patents | pull / 802.66 ms | 298.36 ms | 0.372x |
| soc-orkut | push / 1511.57 ms | 1030.02 ms | 0.681x |
| soc-twitter | push / 3459.57 ms | 3463.27 ms | 1.001x |
| soc-sinaweibo | pull / 5672.51 ms | 5180.71 ms | 0.913x |
| 几何平均 | - | - | 0.694x |

当前结果不能表述为“全面超过 iBFS”，但可以表述为：PuerCGP 作为多算法系统，在高并发 BFS 上与专用 iBFS 保持竞争力，并在 Twitter 上持平。

旧实验的结论并未被完全推翻。旧版 `N=Q` 实验中，PuerCGP/iBFS 的几何平均加速随 Q 变化如下：

| Q | 2 | 4 | 8 | 16 | 32 | 64 | 全部 Q |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Puer/iBFS | 2.011x | 1.697x | 1.436x | 1.429x | 1.200x | 0.625x | 1.318x |

因此旧结果的优势主要来自 `Q=2-32`；旧版在 `Q=64` 时同样落后。新旧实验使用不同 source、`N`、计时边界和代码版本，不能直接混合，后续需要用当前代码统一重跑 Q scalability。

### 不同算法的模式选择

| 算法 | 四图观察 | 结论 |
|---|---|---|
| BFS | cit-Patents/SinaWeibo 偏向 pull；Orkut/Twitter 偏向 push | 最优模式依赖图结构和 frontier phase |
| SSSP | 四图均为 hybrid 最优 | 动态 push/pull 切换有效 |
| SSWP | 四图均为 hybrid 最优 | 与 SSSP 的 relaxation 特征一致 |
| PageRank | 四图均为 pull 最优 | 稠密全图迭代适合 query-lane pull |
| PPR | 四图均为 pull 最优 | 当前固定轮次 dense 实现适合 pull |

### 正确性、失败项与有效性

- 20 组加权跨系统签名全部一致，correctness failure 为 0。
- Gunrock `streams64` 在 Orkut、Twitter 和 SinaWeibo 上 OOM；没有通过降低 Q 掩盖失败。
- Glign 的 SinaWeibo SSWP OOM；ForkGraph 的 SinaWeibo SSSP 在观测到至少 131 GB RSS 后被 SIGKILL。
- ForkGraph 使用 48 个绑定物理核和 LLC-sized range partition；Twitter/SinaWeibo 的部分结果为单次测量，并在共享 CPU 主机上运行，投稿前必须在独占节点复测。
- PPR 暂无语义一致的外部 baseline；ForkGraph/Gunrock 的 residual PR-Nibble 不纳入 fixed-iteration dense PPR 主表。

完整表格与原始结果：

- [最新 external baseline 结果](../../experiments/20260719-233131_external_baseline_n256/RESULTS.md)
- [旧版 iBFS 并发度对比](../../experiment/experiments/20260609_forkgraph_glign_ibfs_bfs/COMBINED_SUMMARY_WITH_PUERCGP_HYBRID_GE_SPMM.md)

## 3. Hybrid Push/Pull 调度优化

### 效果

- 四个数据集、`Q=16/32/64` 共 12 个 BFS hybrid 场景全部通过性能门限。
- `soc-twitter Q=64`：`2218.47 ms -> 898.43 ms`，提升 2.47x。
- `soc-orkut Q=64`：`277.71 ms -> 213.80 ms`，提升 1.30x。
- `soc-sinaweibo Q=64`：`1288.43 ms -> 1070.83 ms`，提升 1.20x。

### 出发点

旧策略只根据 shared frontier 节点数选择 push/pull，无法反映不同节点度数和 query mask 密度，容易执行额外 pull 轮次。

### 实现策略

使用 degree-weighted virtual edge work 估计 shared push 成本：

```text
virtual_edge_count = sum(degree(v) * active_query_count(v))
pull if virtual_edge_count >= pull_edge_ratio * E * Q
```

同时实测发现 `Q<=64` 时 `fused_pull_simple_kernel` 优于 smem 版本，因此默认统一选择 simple kernel。

## 4. Phase-Aware Sharing 调度

### 效果

`N=Q=32`：

| 数据集 | Heavy alignment | Offset alignment | Greedy pause |
|---|---:|---:|---:|
| cit-Patents | 1.279x | 1.305x | 0.773x |
| soc-orkut | 1.011x | 1.035x | 0.573x |
| soc-twitter | 2.052x | 2.049x | 1.135x |
| soc-sinaweibo | 0.976x | 1.051x | 0.720x |

`N=128,Q=32`：

| 数据集 | Selective batching | Batching + offset | Batching + pause |
|---|---:|---:|---:|
| cit-Patents | 1.070x | 1.333x | 0.807x |
| soc-orkut | 1.204x | 1.318x | 0.816x |
| soc-twitter | 1.050x | 2.124x | 1.071x |
| soc-sinaweibo | 1.237x | 1.386x | 0.890x |

### 出发点

Shared push 的物理工作量接近多个 query frontier 的 degree-weighted union，而不是各 query edge work 之和。调整 query 组合和启动时间可以减少重复 neighbor-list 访问。

### 实现策略

- `heavy_alignment`：对齐各 query 最大 edge-work iteration。
- `offset_alignment`：构造跨 query、跨 level 的 degree-weighted overlap，搜索固定启动 offset。
- `selective_batching`：根据 query affinity 将 128 个 query 分为四个容量为 32 的 batch。
- `greedy_pause`：按一步 overlap 预测暂停或恢复 query。

当前保留的策略为：

```text
sharing-aware batch formation -> fixed start offset -> continuous execution
```

Greedy pause 会增加 global steps、frontier carry 和后续 union work，暂不继续采用。

## 5. Replenish 维护路径优化

### 效果

相比旧 replenish：

| 数据集 | 旧实现到优化实现的时间减少 | 相对静态 batching |
|---|---:|---:|
| cit-Patents | 15.2% | 1.014x |
| soc-orkut | 27.6% | 0.997x |
| soc-twitter | 27.2% | 1.025x |
| soc-sinaweibo | 31.8% | 1.016x |

### 出发点

旧 replenish 在 slot 完成后频繁执行全图 snapshot、values reset、visited clear、source reinit 和 host/device 同步，抵消了填补空 slot 的理论收益。

### 实现策略

- 将 snapshot、reset 和 reinit 解耦。
- 只 reset 即将复用的 slot。
- BFS slot 避免不必要的 values 全量初始化。
- 批量处理多个完成和新注入 slot。
- 完整 cohort 使用连续 fill/memset。
- 对尾部不足 Q 的 query 做 tail compaction。
- 保存每个 slot 的 BFS 起始 level，保证补槽后的局部距离正确。

优化后的接近持平结果主要来自 `cohort=32` 的整批回收和 buffer 复用，不能视为 immediate replenish 已经有效。

## 6. roadNet-CA Immediate Replenish 验证

### 效果

| Workload | Static | Immediate replenish | 吞吐变化 |
|---|---:|---:|---:|
| N=400,Q=32 | 3191.01 ms | 3450.39 ms | -7.5% |
| N=800,Q=32 | 6178.18 ms | 6924.43 ms | -10.8% |

- Slot 利用率由 79.75%/84.79% 提升到 94.85%/97.53%。
- N=400 median latency 回退约 2.6%。
- N=800 median/p90 latency回退约 8.7%/11.7%。
- N=800 的 800 个 BFS query 与 CPU 结果比较，`mismatches=0`。

### 出发点

验证长直径、query 完成深度差异较大、`N>>Q` 时，immediate replenish 是否能够利用空 slot 提升吞吐。

### 实现策略

- 下载并转换无向 `roadNet-CA`，最终 `V=1,965,206`、`E=5,533,214`。
- 离线选择 BFS level 为 508-857 的不同源节点。
- 使用 all-push、静态 batching 对比 immediate FIFO replenish。
- 不启用 selective batching 和 offset alignment。

结论是更高 occupancy 会混合不同 BFS phase，增加 union frontier 和每步内存访问；平均 step cost 上升约 25%，超过 step 数下降带来的收益。

## 7. Green Context SM 缩放实验

### 效果

- Shared push 对中大型真实 frontier 通常在 8-16 SM 达到最优。
- 使用全部 80 SM 比最优配置慢 6.0%-27.1%。
- Pull 从 2 SM 扩展到 80 SM 获得 13.1x-24.7x 加速。
- Pull 从 32 SM 扩展到 80 SM 仍有 1.24x-1.85x 收益。

### 出发点

研究 push 和 pull 对 GPU 计算资源的需求是否不同，为后续资源隔离和并发执行提供依据。

### 实现策略

- 使用 CUDA Green Context 将可用 SM 限制为 2-80。
- Push workload 从真实 Q=32 BFS iteration 提取，不使用随机 frontier。
- Pull 从真实 BFS values/visited 状态恢复后单独计时。

Push 的非单调扩展可能来自 atomic 更新和内存系统竞争；pull 更接近规则的全图 gather，持续受益于更多 SM。

## 8. Q=64 Query-Level Push/Pull 划分

### 效果

最佳 query grouping oracle 和最佳 `8/72` SM 划分仍慢于 all-pull：

| 数据集 | Push query 数 | All-pull | 最佳并发 | 回退 |
|---|---:|---:|---:|---:|
| cit-Patents | 1 | 17.454 ms | 18.470 ms | 5.8% |
| soc-orkut | 1 | 63.844 ms | 75.587 ms | 18.4% |
| soc-twitter | 8 | 209.025 ms | 211.772 ms | 1.3% |
| soc-sinaweibo | 1 | 246.531 ms | 296.655 ms | 20.3% |

### 出发点

Push 在少量 SM 上可能已经饱和，而 pull 需要更多 SM，因此尝试在同一 Q=64 batch 内将部分 query 分配给 push、其余 query 分配给 pull并发执行。

### 实现策略

- 为 pull 增加 `active_slots`，支持只计算指定 query mask。
- 使用两个不重叠 Green Context 分区并发运行 push/pull。
- 测试 threshold grouping、按实际 query work 排序的 oracle grouping 和多组 SM 划分。
- 修复 push/pull 并发更新 `visited_mask` 时的丢位竞态。

Pull 损失 SM、额外 shared-push union work、HBM/L2 干扰以及 merge/postprocess 抵消了并发收益，当前不集成完整 hybrid engine。

## 9. Values 数据布局验证

### 效果

将 pull 的 values layout 从 `V*Q` 修改为 `Q*V` 后，四图、`M=16/32` 全部正确，但 kernel 慢 7.0x-32.8x。

### 出发点

`Q*V` 能让单 query 的 result save/reset 连续访问，可能降低 replenish 的 slot I/O 成本。

### 实现策略

在 `ge_spmm_simple` 最小测试中保持相同 thread-to-query 映射，仅修改 values 输入输出布局，并对采样结果做 CPU 正确性验证。

当前 kernel 沿 query 维展开；`V*Q` 使 warp 访问连续 query values，`Q*V` 则产生跨 V 的离散访问。最终保留 `V*Q`，优先通过批量和异步 slot I/O 降低维护成本。

## 10. 代码结构重构

### 效果

- BFS push 12/12 性能场景通过，平均运行时间为重构前的 0.943x。
- BFS/SSSP/WCC、hybrid 和 replenish 正确性测试全部通过。
- 初始 standalone pull 为 8/12 通过，部分差异来自修复有向图 pull 语义。
- 新增 SSWP、PageRank 和 PPR；PageRank/PPR 已支持 push、pull 和 hybrid，并完成四图固定轮次测试。

### 出发点

原实现将 engine、kernel、初始化、算法分派、workspace 和 replenish I/O 集中在少数大文件中，难以支持后续调度、slot 管理和资源控制。

### 实现策略

- 拆分 core helpers、workspace、push/pull executor 和 kernel 目录。
- 增加静态 `algorithm_traits`，统一 BFS/SSSP/SSWP/WCC 初始化和 candidate 计算。
- 增加 rank workspace 和 dense engine，封装 PageRank/PPR 的双缓冲区、sum reduction、收敛状态与迭代流程。
- 增加 query partition 和 `active_slots` 接口。
- 抽取 `slot_io_manager`，封装 snapshot/reset/reinit。
- 删除 frontier list、bitmap pull 和 GE-SpMM 等旧实验路径。
- 保留 shared-frontier push、fused pull 和实验性 query-parallel push。
- 修复有向图 pull 错误读取 outgoing CSR 的问题，改用 incoming CSR。

## 11. 当前总体认识与下一步

近期结果可以统一为一个 data-movement-aware 的优化方向：

1. Shared push 通过跨 query topology sharing 减少独立 neighbor-list access。
2. Query-parallel pull 通过 `V*Q` 布局提高 memory coalescing 和 transaction utilization。
3. Selective batching 和 offset alignment 减少 degree-weighted union frontier。
4. Hybrid scheduler 在 push sharing 与 pull coalescing 之间选择预计内存代价更低的执行路径。
5. Immediate replenish、任意 pause/resume 和 query-level push/pull 并发目前均未获得稳定收益。

完整 baseline 进一步明确了论文的性能边界：

- PuerCGP 相对通用 CPU/GPU baseline 已有明显端到端优势。
- iBFS 证明专用 BFS 算子仍然可以更快，因此贡献不能表述为“全面最快的 GPU BFS”。
- 当前 `Best-Puer` 与实际 hybrid/online policy 之间仍有差距，尤其是 Twitter BFS 的 hybrid 端到端结果没有逼近 all-push；系统不能依赖事后挑选结果。
- PuerCGP 更可信的定位是：以统一 physical execution framework 支持多种并发图算法，并围绕 topology sharing、query-dimensional coalescing 和动态路径选择降低数据移动成本。

### 投稿前优先事项

1. 使用当前代码、相同 source 和相同端到端计时边界，补齐 iBFS 的 `Q=2,4,8,16,32,64` scalability 对比。
2. 修正或重构在线 push/pull selector，使默认执行结果稳定逼近 Best-Puer，重点分析 Twitter BFS。
3. 在独占 CPU 节点上用统一 2 warmups + 5 repeats 复测 Glign/ForkGraph，保留严格 Q 下的 OOM 结果。
4. 补充 shared push、query-lane pull 和 online scheduling 的分项 ablation 与 memory-traffic counter，建立可验证的 data-movement 证据链。
5. 明确 PPR 的对比口径：寻找固定轮次 dense baseline，或将 PPR 限定为功能完整性而非主要性能 claim。
6. 将 selective batching、fixed offset 和 hybrid switching 整合为一个真实在线执行流程，分别报告 planner overhead、吞吐和 tail latency。
