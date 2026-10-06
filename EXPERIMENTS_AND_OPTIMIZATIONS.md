# GraphWeft 实验与系统优化完整总结

> 最后核对：2026-10-05（Asia/Taipei）  
> 实验平台以 NVIDIA Tesla V100-SXM2-32GB 为主。本文只总结仓库中已经产生结果的实验；仍在运行的五张主图 SSWP Gunrock/Groute 补测不计入完成结果。原始命令、日志、逐轮数据、fingerprint 和统计文件均保留在 [`experiments/`](experiments/) 下。

## 1. 阅读说明与统一口径

本文把实验分为四类：

1. **正式端到端实验**：有独立 warmup、多次正式重复、完整配置和 correctness fingerprint，可用于论文主表。
2. **机制/逐轮实验**：在冻结迭代状态上测 kernel，适合解释机制，不能直接当端到端收益。
3. **探索性或受控合成长尾实验**：用于验证调度假设，必须与自然 workload 结果分开报告。
4. **失败、预检或混杂实验**：用于说明设计演进，不作为最终性能结论。

默认速度提升均为 `baseline_time / optimized_time`，大于 1 才表示优化。`workload_ms` 是 GraphWeft 完整 workload 时间；逐轮 `kernel_gpu_ms` 只包含 CUDA event 所围住的 kernel。不同项目的计时边界不同，外部系统比较处会单独说明。

核心符号：

- `N`：workload 中总查询数。
- `Q` 或 `M`：同时驻留的查询槽位数；历史脚本有时使用 `M` 表示容量。
- `G`：group 宽度。当前主配置为 grouped layout、`G=32`。
- `q1/q2/.../q32`：每条边分配的 query lane 数。
- `W1/W2/W4`：一个顶点分别由一个 block 内的 1/2/4 个 warp 处理。
- `B2/B4`：一个顶点由 2/4 个 block 处理，每个 block 4 个 warp，并在 timed kernel 内合并。
- `Shared`：所有 Push 轮次使用原有 shared Push kernel。
- `Static`：每张图使用独立冻结的单一 partition candidate。
- `Adaptive`：vertex-level、按顶点特征分配 W/B 粒度的旧自适应实现。
- `Iteration`：每个 Push 迭代用聚合特征从 30 个 `qN × W/B` candidate 中选一个，全轮只 launch 该 candidate。
- `Hybrid`：每轮先依据阈值选择 Push 或 Pull；只有 Push 轮才使用 Push mapping。
- `refill`：group 完成后用等待 group 补空槽，消除 Q-sized batch barrier。
- `LSSS bridge`：干扰感知、兼容性约束的 bridge refill；仍保持每轮一个全局 Hybrid 决策和一个全局 Iteration Push kernel。

## 2. 数据集、算法与主要测试范围

| 类别 | 数据集 | 主要用途 |
|---|---|---|
| 五张论文主图 | cit-Patents、soc-LiveJournal1、indochina、soc-orkut、soc-twitter | BFS、SSSP、SSWP 正式性能与自适应 mapping |
| 早期补充图 | soc-sinaweibo、UK | 早期 SSSP、Push/Pull partition 与 layout 机制实验 |
| 道路图 | roadNet-CA、roadNet-TX | 稀疏长迭代、Iteration 外推、外部 baseline、refill 场景 |
| 单元/合成小图 | V=2--4096、E=1--49152 | CPU reference、边界、replay、sanitizer、索引 parity |
| 强长尾派生图 | LiveJournal、Orkut、Twitter 等原图加不相连路径 | 受控 straggler/refill 实验；不代表自然图查询分布 |

正式 seed-42 workload 与 Iteration 模型的 seed-45 calibration workload 分离。主实验通常为 `N=1024`，Iteration matrix 为 `Q=64,G=32`，统一在线消融为 `Q=128,G=32`。自然 workload 审计表明五张主图的 BFS/SSSP 长短查询轮数中位数比只有 **1.125--1.269**，达不到 2 倍强长尾标准，因此强长尾收益不能冒充自然 workload 收益。

## 3. 当前可以直接用于论文的总结果

| 优化/系统结论 | 正式范围 | 主要结果 | 当前判断 |
|---|---|---|---|
| Hybrid Push/Pull | 早期六图 SSSP Q32；P0 五主图三算法 | 早期 GraphWeft Hybrid 相对 All-Push 为 1.36--5.31×；P0 Iteration 下五主图几何平均约 1.426×，15 项中 14 项获益 | 核心优化，成立 |
| 64-bit frontier publication | cit-Patents BFS，N1024/M64/G8 | kernel 1.767×，workload 1.599× | 强机制收益，已保留 |
| 长度预测分组 | 五主图三 workload；P0 七图三算法 | 早期全局排序 1.178×；P0 cohort-local 自然在线场景 1.040×，21/21 获益；五主图 1.055× | 稳定但收益取决于排序自由度 |
| Iteration-level Push mapping | 七图 × BFS/SSSP/SSWP × All-Push/Hybrid | 570 次正式测量；对 Shared 1.414×，对五主图 Static 1.081×；fingerprint 全匹配 | 核心特色贡献 |
| 自然 workload 的 LSSS refill | 七图三算法，Q128 Hybrid | 相对 Iteration+length 为 0.911×；五主图 0.877×，道路图约 0.999× | 当前不能作为自然 workload 正收益 |
| 受控强长尾 LSSS refill | 5 图扩展稀疏长尾 SSSP | E2E 1.444×，平均 latency 1.280×，P99 1.443× | 证明机会存在，但必须标注 synthetic/controlled |
| 完整系统 A→D | P0 七图三算法 | 全部几何平均 1.134×；五主图 1.204×；道路图 0.976× | Mapping+length 抵消部分 refill 回退；不能称 refill 普遍有效 |
| 正确性 | P0 570+420 正式样本及各类专项测试 | 所有对应 fingerprint 匹配；sanitizer 0 error | 通过 |

## 4. 正确性、边界与实验基础设施

### 4.1 目的

在进行性能优化前建立可重复的同步图算法语义：CPU reference、Push/Pull 同轮等价、multi-batch/tail tile、checkpoint/replay、计时字段、CLI 容量保护、frontier 精度边界、CUDA 内存和同步安全。

### 4.2 数据与范围

- 合成图 V=2--7、E=1--10，以及四顶点五边有向图。
- BFS、SSSP、SSWP；vertex-major/grouped；Q=1/8/31/32/33/63/64/65/127/128/129/256。
- 延迟启动、多 batch、尾 batch、重复边、自环、孤立点、反序 frontier、phase offset。
- cit-Patents 实图 Q128/Q256，及 SSSP/SSWP Q32。
- `compute-sanitizer --tool memcheck` 与 `synccheck`。

### 4.3 结果及其反映的系统改动

- 初始 isolated-source、无向 reverse-edge、BFS 精度边界、weighted-boundary uint16 溢出、cold-cache 输入方向等问题均先被测试捕获，随后在对应的下一目录修复并通过。
- checkpoint 可在匹配 round 上复现 Shared Push 和 Dense Pull；frozen plan 导入后与导出计划完全一致，Q 不匹配会明确拒绝。
- GPU acceptance 的 N=130/Q=129、multi-batch、grouped replay、Push/Pull 通过，memcheck/synccheck 均为 0 error。
- core-distance 与旧索引 parity 通过；weighted-boundary 在扩大键宽后通过 V4096、E8192/49152 parity。
- 逐轮 timing、feature、selector、frontier trace 字段完整；warp-reserved unordered compaction 通过同步检查。

关键证据目录：[`20260921-155526_core_validation`](experiments/20260921-155526_core_validation/)、[`20260921-155657_full_validation`](experiments/20260921-155657_full_validation/)、[`20260921-155818_gpu_acceptance`](experiments/20260921-155818_gpu_acceptance/)、[`20260921-160021_kernel_replay`](experiments/20260921-160021_kernel_replay/)、[`20260921-161801_final_gate`](experiments/20260921-161801_final_gate/)。

## 5. 初始端到端基线与 Hybrid Push/Pull

### 5.1 固定 Push 对旧 puercgp

**目的。** 建立 GraphWeft 初始端到端位置，定位主要开销。  
**数据与范围。** 六张非道路图，SSSP，64 个不同 source，Q32 FIFO，固定 Push，五次 fresh-process 中位数，并逐图验证 query 0 的全顶点结果。  
**结果。** GraphWeft task time 比 puercgp 慢 1.39--2.33×：cit 2.00×、LJ 1.85×、indochina 2.33×、orkut 1.80×、sinaweibo 1.39×、twitter 2.21×。GraphWeft compute kernel 是所有图最大阶段，frontier 重建在五张图占 19--37%。这直接促成了 Push partition、Hybrid 和 frontier 优化。详见 [`20260921-173921_sssp_e2e_nonroad`](experiments/20260921-173921_sssp_e2e_nonroad/README.md)。

### 5.2 Hybrid 阈值选择

**目的。** 避免稠密 frontier 继续用 Push，以 Pull 降低 edge-query 工作。  
**范围。** 同六图 SSSP，N=Q=32，阈值 0.20，三次 fresh-process。  
**结果。** GraphWeft Hybrid 对固定 Push 的 speedup 为 cit 1.64×、LJ 1.70×、indochina 1.36×、orkut 5.31×、sinaweibo 2.29×、twitter 1.93×。收益几乎全部来自 compute kernel；frontier 阶段基本不变。对应 Push/Pull round 数与 puercgp 一致。详见 [`20260921-201450_sssp_n32_hybrid`](experiments/20260921-201450_sssp_n32_hybrid/README.md)。

### 5.3 全 V×Q compare 诊断

**目的。** 确认 frontier 阶段究竟慢在 compare 还是 compaction。  
**范围。** 六图、同一 N32/Q32 SSSP Hybrid，三次。  
**结果。** `compare_kernel` 占完整 task 的 **25.89--60.09%**，占 frontier 阶段 **99.34--99.64%**。因此瓶颈是全 V×Q 比较/掩码/计数，不是后续 compaction；这推动 fused/direct frontier 与 mask64。详见 [`20260921-211343_compare_kernel_n32_hybrid`](experiments/20260921-211343_compare_kernel_n32_hybrid/README.md)。

## 6. Push 并行划分与候选空间

### 6.1 目的和实现

测试 query lanes 与顶点并行粒度的联合选择：`q{1,2,4,8,16,32} × {W1,W2,W4,B2,B4}` 共 30 个 candidate。`W` 在单 block 内增加 warp，`B` 用多个 block 共同处理一个顶点并合并。目标不是假设一个配置全局最优，而是量化不同图、不同 round 的差异。

### 6.2 数据和范围

- 六图真实 SSSP，N=Q=32，共 211 个真实迭代状态。
- 每个 candidate 预热 1 次、正式 10 次；噪声轮次追加 20 次。
- 每个 candidate 输出逐字节对原始 Push；计时只含核心 kernel 和 candidate 内 query-list 压缩。

### 6.3 结果

| 图 | 最佳固定 candidate | 对原始 Push | 对 q1_w1 | 逐轮 oracle 对最佳固定的额外空间 |
|---|---|---:|---:|---:|
| cit-Patents | q16_w2 | 3.40× | 1.76× | 1.05× |
| soc-LiveJournal1 | q16_w2 | 2.52× | 1.29× | 1.09× |
| indochina | q2_w4 | 3.78× | 1.46× | 1.05× |
| soc-orkut | q16_w4 | 7.04× | 2.20× | 1.06× |
| soc-sinaweibo | q32_w4 | 5.52× | 2.21× | 1.08× |
| soc-twitter | q4_b2 | 4.99× | 1.51× | 1.18× |

峰值工作轮次的赢家也从 q2_w4 到 q32_b2 不等。真实负载分析显示选择主要与活跃 query/vertex、edge-query 加权活跃度、平均和尾部 degree、工作集中度有关；仅用平均 degree 和活跃 query 数不足。211 轮中有 56 轮最快 kernel 小于 20 μs，赢家标签也存在近似平局，因此后来采用相对 cost 回归，而不是单纯 winner 分类。详见 [`push_partition_rounds`](experiments/20260921-225549_push_partition_rounds/README.md) 与 [`workload_analysis`](experiments/20260921-225549_push_partition_rounds/workload_analysis/README.md)。

## 7. Pull kernel：frontier check、query grouping 与并行粒度

### 7.1 目的

比较相同 30 种划分下的 check-free Pull 与 source-query frontier check Pull。稀疏轮次可能因跳过 source value 读取获益；稠密轮次可能被 mask 读取和分支发散拖慢。

### 7.2 数据与范围

六图、与 Push 实验相同的 Q32 SSSP source，共 211 轮；61 个候选/轮，129,930 个 CUDA event 样本；另用 NCU 分析 cit-Patents 的代表性稀疏/稠密轮次。小图扩展套件覆盖 3,660 条 trajectory、36,960 轮。

### 7.3 结果与策略

- 同配置的 6,330 个 round pair 中，check/free 比的 P10/P50/P90 为 **0.323/0.699/1.255**，check 在 70.7% pair 上获胜。
- density <1% 时 check 中位数为 0.504×、99.5% pair 获胜；density ≥50% 时为 1.205×、只赢 4.9%。
- 最稳健 check mapping 是 `q4_w2`，六图相对各图最佳 check 的几何平均仅慢 2.8%。B2/B4 没有赢得任何图的固定比较，多 block merge 成本过高。
- 固定总时间上，check 在 LJ、indochina、twitter 最快；Dense Pull 在 cit、orkut、sinaweibo 最快。不存在全局单一 Pull winner。
- NCU 表明 q1 的大量 sector request 多为 L1 重复访问，DRAM bytes 与 q8 接近；query lane grouping 的效果不能只由 DRAM 流量解释。

第一次 q1 实现含过多 block barrier，pilot 未完成任何 round，已废弃。正式结果见 [`pull_partition_final`](experiments/20260922-144225_pull_partition_final/ANALYSIS.md)，失败 pilot 见 [`pull_partition_check`](experiments/20260922-143718_pull_partition_check/README.md)。

## 8. Value layout 与 G=8 Pull 特化

### 8.1 Grouped layout

**目的。** 比较 `[ceil(Q/G)][V][G]` 与 `[V][Q]` 在 Pull 下的 locality。  
**范围。** 六图 SSSP，Q32/64/96，84 个配置，每项一次 warmup + 五次正式 complete-batch。  
**结果。** 在相同 mapping 下，grouped G8 比 vertex-major 慢 **27.7--77.5%**（不同图/Q），G16 慢 **5.4--23.3%**，G32 在 Q64/96 慢 **7.8--16.1%**。Grouped layout 是 refill/分组所需的系统组织方式，但旧 Pull kernel 不能直接获得 locality 收益。详见 [`grouped_pull_layout`](experiments/20260926_grouped_pull_layout/REPORT.md)。

### 8.2 G=8 四 edge-stream Pull

**目的。** 在 grouped G8 下，用四个 warp 分担 incoming edge stream，弥补原 q8-w1 的串行 edge 遍历。  
**范围。** 同六图 Q32/64/96，一次 warmup + 五次正式。  
**结果。** 对 g8_q8_w1，特化 kernel 在 cit/LJ/orkut 提升约 **3.1--7.6%**；indochina 提升 **50.4--52.9%**，UK **30.6--34.2%**，twitter **42.1--43.2%**。generic q32-w4 已取得大部分收益，特化版再有小幅改善。这说明高入度/长 edge stream 图需要更多顶点内 edge parallelism。证据见 [`grouped_g8_edge4_warp4`](experiments/20260927_grouped_g8_edge4_warp4/artifacts/summary.csv)。

## 9. 容量、frontier 与 publication 优化

### 9.1 M=64 容量

六张图实际分配、执行、释放均在 V100 32GB 的 80% 预算内完成；最大 soc-twitter 使用 20,322,388,388 bytes，低于 27,258,047,692-byte budget。该实验是容量正确性，不是性能样本。见 [`fixed_m64_capacity`](experiments/20260927_fixed_m64_capacity/README.md)。

### 9.2 scan、fused、direct frontier

**目的。** 去除独立全 V×Q scan，把 frontier 发布融合进 relax kernel。  
**范围。** 五张非道路图 BFS，N1024/M64，fixed Push、degree mapping、一次 warmup + 五次，scan/fused/direct。  
**结果。** direct 没有整体获胜：cit 比 fused 慢 1.66%，LJ 与 scan 近似平局，orkut 虽比 fused 快 0.56% 但比 scan 慢 19.51%，indochina 比 fused 慢 3.53%，twitter 比 scan 慢 14.02%。原因是 direct 把全局 claim/append atomic 放入 hot relaxation path。尽管端到端没有胜出，direct 提供了 unordered、M>64 正确的基础，后续与 mask64 一起使用。见 [`direct_frontier`](experiments/20260928_direct_frontier/README.md)。

### 9.3 64-bit frontier publication

**目的。** 将一个 vertex 的 query improvement 聚合成一个 64-bit word，减少逐 `(vertex,slot)` publication。  
**范围。** cit-Patents BFS，N1024/M64/G8，fixed Push Shared、direct unordered frontier，一次 warmup + 五次 A/B。  
**结果。** median kernel 从 14,533.8 ms 降到 8,223.64 ms，提升 **1.767×**；workload 从 15,708.4 ms 降到 9,826.84 ms，提升 **1.599×**。这是后续正式配置默认启用 mask64 的依据。见 [`mask64_ab`](experiments/20260928_mask64_ab/README.md)。

## 10. 长度预测分组

### 10.1 目的和实现

用 `core_distance`（BFS）或 `weighted_boundary`（SSSP/SSWP）的静态预测键，将预计长度接近的 query 放在同一 admission cohort/group 中，减少短 query 被长 query 的 batch barrier 拖住。计划可冻结、导入和 replay；P0 在线口径只允许在当前 Q=128 cohort 内排序，不预看未来 arrival。

### 10.2 数据范围与结果

- 早期 G8 campaign：五主图 × BFS/SSSP/mixed-tail，N1024/M64/G8 Hybrid，共 15 对照项、每项五次。
- P0 统一消融：七图 × BFS/SSSP/SSWP，N1024/Q128/G32 Hybrid，共 21 项、每 variant 五次。

早期全局排序的 workload 几何平均收益为 **1.178×**：BFS 1.229×、SSSP 1.116×、mixed 1.191×；P50 latency 1.213×、round 数 1.103×。更严格的 P0 cohort-local 在线口径仍有 **1.040×**，21/21 项获益；五主图为 **1.055×**，道路图为 **1.003×**。因此长度分组是稳定正收益，但早期较大的数字包含更自由的全局重排。

weighted-boundary 在道路图出现超过 65535 的键，最终将已有 uint64 存储链路上的不必要 uint16 限制移除；这保持了长道路查询的排序语义。

### 10.3 1.178× 全局长度分组实验的精确设置

这组数字来自 [`20260928-234229_hybrid_mask64_campaign`](experiments/20260928-234229_hybrid_mask64_campaign/README.md) 的 **A→B 单变量对照**，不是 P0 的 cohort-local 在线排序：

| 项目 | 设置 |
|---|---|
| 硬件 | 5 张 Tesla V100-SXM2-32GB；cit/LJ/indochina/orkut/twitter 分别绑定 GPU 0/1/5/3/4 |
| 图 | cit-Patents（3.77M V/33.04M E）、LiveJournal（4.85M/137.99M）、indochina（7.41M/304.47M）、Orkut（3.00M/212.70M）、Twitter（21.30M/530.05M） |
| workload | BFS、SSSP、mixed-tail；每图每类 N=1024 |
| 并发组织 | M=Q=64，grouped layout，G=8；每批 8 个 group |
| 图执行 | Hybrid threshold 0.20、Shared Push mapping、direct unordered frontier、mask64 开启 |
| 调度限制 | `--same_algorithm_groups`；不开启 `--group_refill`，group 只在整个 Q64 batch 完成后回收 |
| A baseline | `--planner=fifo`：保留每个算法队列中的原始输入顺序 |
| B candidate | `--planner=length --predictor=import_key`：先在**整个 1024-query workload 的同算法队列内**按冻结 feature key 升序稳定排序，再连续切成 G=8 group |
| predictor | BFS 使用 graph-only core-distance key；SSSP 使用 weighted-boundary-v4 key；不读取正式执行得到的 `reference_rounds` |
| mixed-tail | 每种算法 448 个 short + 64 个 longest，共 512 BFS + 512 SSSP，seed 44 打乱；先分别全局排序两个算法队列，再以 8-query group 交替拼接 |
| 重复 | 每个 case 一次 warmup + 五次正式 fresh process；A/B 顺序按 campaign 方案交错；取五次中位数 |
| 正确性 | campaign 的 validation/fingerprint 文件覆盖各 variant；A/B 使用相同冻结 query 文件、图、kernel mapping 和 Hybrid 参数 |

这里的“全局”是指测试开始时 1024 个 query 全部已知，调度器可以跨未来 Q64 batch 重排同算法 query；它不是带真实 arrival timestamp 的在线队列。因此 1.178× 表示一个较宽松的离线/静态 backlog 场景上限，论文若讨论在线查询，应同时给出 P0 cohort-local 的 1.040×。

### 10.4 A→B 的逐项端到端结果

表中时间是五次正式运行的 `workload_ms` 中位数；P50 和 rounds speedup 分别是 A/B 的 query latency 与总轮数之比。

| 图 | workload | A FIFO ms | B length ms | E2E speedup | P50 latency | rounds |
|---|---|---:|---:|---:|---:|---:|
| cit-Patents | BFS | 7,636.48 | 6,465.85 | 1.181× | 1.186× | 1.055× |
| cit-Patents | SSSP | 25,752.4 | 24,439.5 | 1.054× | 1.106× | 1.048× |
| cit-Patents | mixed-tail | 22,447.7 | 21,323.5 | 1.053× | 1.113× | 1.122× |
| soc-LiveJournal1 | BFS | 9,433.55 | 7,742.18 | 1.218× | 1.192× | 1.071× |
| soc-LiveJournal1 | SSSP | 26,426.6 | 24,589.5 | 1.075× | 1.150× | 1.069× |
| soc-LiveJournal1 | mixed-tail | 26,059.9 | 21,915.1 | 1.189× | 1.232× | 1.159× |
| indochina | BFS | 30,752.4 | 23,454.2 | 1.311× | 1.238× | 1.178× |
| indochina | SSSP | 90,422.2 | 78,552.5 | 1.151× | 1.196× | 1.105× |
| indochina | mixed-tail | 90,443.0 | 71,629.1 | 1.263× | 1.328× | 1.202× |
| soc-orkut | BFS | 16,608.2 | 15,992.0 | 1.039× | 1.003× | 1.021× |
| soc-orkut | SSSP | 56,615.7 | 54,942.9 | 1.030× | 1.092× | 1.035× |
| soc-orkut | mixed-tail | 42,783.4 | 40,425.3 | 1.058× | 1.108× | 1.095× |
| soc-twitter | BFS | 78,036.4 | 54,435.9 | 1.434× | 1.363× | 1.115× |
| soc-twitter | SSSP | 149,715 | 116,108 | 1.289× | 1.453× | 1.115× |
| soc-twitter | mixed-tail | 146,280 | 102,121 | 1.432× | 1.554× | 1.179× |

| 聚合范围 | E2E | P50 latency | P95 latency | P99 latency | rounds | P50 waiting |
|---|---:|---:|---:|---:|---:|---:|
| BFS（5图） | 1.229× | 1.191× | 1.224× | 1.224× | 1.087× | 1.201× |
| SSSP（5图） | 1.116× | 1.193× | 1.114× | 1.114× | 1.074× | 1.211× |
| mixed-tail（5图） | 1.191× | 1.257× | 1.188× | 1.186× | 1.151× | 1.286× |
| 全部 15 项 | **1.178×** | **1.213×** | **1.174×** | **1.174×** | **1.103×** | **1.232×** |

15/15 项 E2E 都有正收益，但分布很不均匀：Orkut 只有 1.030--1.058×，Twitter 为 1.289--1.434×。收益来源是相似长度 query 同组后减少组内空等和 batch 总轮数；它不是 kernel mapping 改进，因为 A/B 都固定 Shared Push，也不是 refill 收益，因为两者都关闭 refill。原始逐项比值见 [`comparisons.csv`](experiments/20260928-234229_hybrid_mask64_campaign/comparisons.csv)，中位数和阶段数据见 [`result.csv`](experiments/20260928-234229_hybrid_mask64_campaign/result.csv)。

## 11. 早期完整 G8 Hybrid + mask64 campaign

**目的。** 在当时最完整的 G8 系统上同时验证 Push/Hybrid、长度分组和 refill，并与外部 baseline 对比。  
**范围。** 五主图 × BFS/SSSP/mixed-tail，N1024/M64/G8，Hybrid 0.20；90/90 cases、450/450 正式样本。  
**结果。** 相对更快 Gunrock 模式，GraphWeft 几何平均 **2.046×**（BFS 2.482×、SSSP 2.107×、mixed 1.639×），未达到当时约 3× 的目标。kernel GPU 时间占 workload 中位数的 88.0%/91.2%/90.7%。长度分组为正，但该版本 eager refill 使完整 A→D 只有 0.839×，说明减少逻辑轮数不等于降低墙钟时间。见 [`hybrid_mask64_campaign`](experiments/20260928-234229_hybrid_mask64_campaign/README.md)。

## 12. G=32 自然 workload refill：负结果

**目的。** 在论文目标配置 G32 下检验普通 group refill 是否提高吞吐和 query latency。  
**范围。** 五主图 × BFS/SSSP/mixed-tail，N1024/M64/G32、Hybrid/direct/mask64，一次 warmup + 三次正式，30 cases/90 measurements。  
**结果。** refill 的 throughput speedup 只有 **0.853×**，P50 latency **0.872×**；虽有 rounds ratio 0.903，但 kernel/frontier 干扰超过了少运行轮数和少等待的收益。按 workload 为 BFS 0.759×、SSSP 0.913×、mixed 0.895×。这否定了“只要补空槽就会更快”的设计。见 [`refill_g32_campaign`](experiments/20260929-111853_refill_g32_campaign/README.md)。

## 13. Vertex-Adaptive Push：为什么没有作为最终策略

### 13.1 目的和范围

按每个 frontier vertex 的工作特征选择 W1/W2/W4/B2/B4；五主图 × BFS/SSSP/SSWP，N1024/M64/G32 Hybrid，Shared/Static/Adaptive，一次 warmup + 五次，所有 mapping 做 fingerprint。

### 13.2 结果

15 项中 Adaptive 对 Static 只有 soc-orkut SSWP 获得 1.030×，其余均为 0.733--0.988×。对 Shared 在 indochina BFS、orkut 三算法和 twitter 三算法有收益，但仍经常不如冻结 Static。主要原因是 vertex bucket/preparation 是 O(frontier) 工作且改变执行组织；例如 cit 的 preparation 达 0.78--2.53 s。该结果促成了 **Iteration-level** 设计：复用已有聚合特征，每轮只选择并 launch 一个 kernel，避免 bucket、scatter 和额外同步。完整表见 [`adaptive_push_g32`](experiments/20260929-175832_adaptive_push_g32/README.md)。

## 14. Iteration-level Push mapping 的训练

### 14.1 模型目的与特征

只在本轮已被 Hybrid 判为 Push 时运行。模型用以下七个输入：

1. `log1p(frontier_vertices)`；
2. `log1p(live_queries)`；
3. `log1p(vertex_pairs/frontier_vertices)`；
4. `log1p(edge_pairs/vertex_pairs)`；
5. `log1p(edge_pairs/frontier_vertices)`；
6. `log1p(graph_edges/graph_vertices)`；
7. density。

特征裁剪到 calibration 范围、标准化并进行二阶展开（36 维）。30 个 Ridge 回归器预测相对 log-cost，选预测最小的 candidate；零 edge-pair 回退 q1_w1，同成本按 candidate 编号确定性打破平局。

### 14.2 Calibration 数据与拆分

- 五主图 × BFS/SSSP/SSWP，seed 45，N1024/Q64/G32。
- All-Push、direct unordered frontier、mask64、无 refill。
- 30 candidates，每项一次 warmup + 五次正式；以同 round 五次 kernel GPU 时间中位数作为样本。
- `batch % 4 == 3` 为 validation，其余为 training；另生成与 seed45 train/validation source 不重叠的 test 切分。冻结清单为 train 11,520 行、validation 3,840 行、test 15,360 行；训练器过滤无有效 edge work 后使用 9,422 个 round-level 样本。
- 目标是 candidate 相对同轮 30 candidate 几何平均的 log-cost；五张图/三算法等权，内部按 q1_w1 逐轮时间占比加权。
- Ridge λ 候选 `{1e-6,1e-4,1e-2,1,100}`；validation 上被选择 kernel 的总时间分别为 148534.9、**148165.5**、150063.6、154508.7、168637.5 ms，故选 `1e-4`，再用全部 calibration 数据重拟合。
- 内置模型版本：`iteration-ridge-v1-1d09ead173fb655b`；raw CSV SHA-256 为 `f51c2d4c...58aac`。

### 14.3 审计说明

目录中的 [`INCOMPLETE.md`](experiments/20260929_iteration_calibration/INCOMPLETE.md) 描述的是最早被主动中止的“一进程一 sample”采样试验，不能作为训练语料；后续 parallel resume、resample、dataset preparation 和 `model_audit.json` 才形成当前模型。正式 seed-42 结果没有用于重新拟合或调参。

## 15. Iteration mapping 端到端结果

### 15.1 预实验

- SSSP All-Push 五主图：Iteration 对 Static 每图 1.044/1.104/1.163/1.063/1.148×，几何平均 **1.103×**；对 Shared 的每图收益 1.391--5.419×；fingerprint 全匹配。由于 GPU4 有无关驻留 allocation 且 cit Shared 波动较大，这批只作为 preliminary。见 [`iteration_sssp_allpush_e2e`](experiments/20260930-223628_iteration_sssp_allpush_e2e/README.md)。
- SSSP Hybrid 五主图：Iteration 对 Static 1.040/1.093/1.057/1.048/1.059×，几何平均 **1.059×**；fingerprint 匹配。见 [`iteration_sssp_hybrid_e2e`](experiments/20261001-112124_iteration_sssp_hybrid_e2e/summary.csv)。

### 15.2 P0 正式七图矩阵

**范围。** 七图 × BFS/SSSP/SSWP × All-Push/Hybrid；Q64/G32；主图 Shared/Static/Iteration，道路图 Shared/Iteration（没有用 seed42 给道路图调 Static）；一次 warmup + 五次，共 **570** 次正式测量。  
**结果。** fingerprint 全匹配；Iteration 对 Shared 全矩阵几何平均 **1.414×**，对五主图 Static **1.081×**。

| 模式 | 五主图 vs Shared | 五主图 vs Static | 两道路图 vs Shared |
|---|---:|---:|---:|
| All-Push | 2.128× | 1.091× | 0.980× |
| Hybrid | 1.259× | 1.071× | 0.982× |

按算法、跨五主图和两种模式，Iteration 对 Static：BFS **1.077×**、SSSP **1.087×**、SSWP **1.079×**。道路图上的约 2% 回退说明 V100 五图训练模型没有自然外推到超稀疏、极长直径道路图；当前策略应保留非 V100/域外未标定警告，不能宣称普适收益。

Hybrid 相对 All-Push 在五主图的 Iteration 配置上几何平均约 **1.426×**，15 项中 14 项获益；唯一回退是 indochina BFS 0.946×。道路图上两者基本持平。正式文件见 [`P0 iteration_matrix`](experiments/20261003-174459_p0_full_matrix/iteration_matrix/summary.csv)。

## 16. 强长尾 workload 的构造与验证

### 16.1 第一版强长尾

原图 CSR 完整保留为前缀，追加 64 个互不相连的 path component；路径长度为 Pareto(α=1.5) 后裁剪到 64--256。N1024 中 960 个自然短 query + 64 个 path-head 长 query，long fraction 6.25%。LJ 的 BFS/SSSP long/short median 分别 7.615×/3.667×，Orkut 为 12.375×/4.950×；6144 个 completion 都与冻结 reference round 相同。该 workload 诚实控制 service rounds，但每轮 frontier 稀疏、edge work 很小，不能声称是原 social graph 的自然长尾。见 [`strong_tail_workloads`](experiments/20260929_strong_tail_workloads/README.md)。

### 16.2 扩展稀疏长尾

为 cit/LJ/indochina/roadNet-CA/roadNet-TX 构造 256 个长 query，保证八个 Q128 batch 中每个都有一个 32-query long group。social path 258--1024 edges（median 446），roads 2063--8192（median 约 3569）。这是为 refill 机会而设计的 controlled trace。

## 17. Refill 设计演进

### 17.1 FIFO、预测顺序和初步 kernel 优化

**目的。** 在 Orkut/Twitter 强长尾上降低 batch barrier 等待和长 query latency。  
**范围。** BFS/SSSP/mixed，N1024/M64/G32，五次正式。  
**结果。**

| 版本 | E2E refill speedup | 全 query latency | 长 query latency | 结论 |
|---|---:|---:|---:|---|
| FIFO | 0.853× | 0.900× | 0.864× | 全面回退 |
| predicted order | 0.913× | 0.929× | 0.953× | 分组缓解但仍回退 |
| predicted + optimized | 0.919× | 0.948× | 1.057× | 长 query latency 首次改善，E2E 未过门槛 |
| wave + fast reset screen | 0.991× | 0.992× | 1.037× | 接近中性，仍未验收 |

证据见 [`strong_tail_refill_comparison`](experiments/20261001-231000_strong_tail_refill_comparison/README.md) 和 [`wave_fastreset_screen`](experiments/20261002-064700_strong_tail_refill_wave_fastreset_screen/metrics.json)。

### 17.2 Q64/Q128 与 per-group Iteration

**目的。** 验证 Q 越大是否有更多 refill 机会，以及能否为同轮不同 group launch 不同 mapping。  
**范围。** Orkut/Twitter SSSP 强长尾，G32，Q64/Q128；global no-refill、global eager refill、group-aware mapping；一次 warmup + 三次。  
**结果。** Q128 的 active-slot ratio 确实从约 0.52/0.55 提升至 0.71/0.75，但 E2E 仍回退。per-group candidate disagreement 在 Q128 达 94.4%/98.4%，却因 per-group feature 成本增长约 4.9--5.2×、kernel 多 launch、模型分布不匹配而更差；D/A 只有 0.549×（Orkut）和 0.611×（Twitter）。这证明当前 aggregate Q64 Ridge 不能直接下放到 group 粒度，也支持继续保持每轮一个 global Iteration kernel。见 [`group_iteration_refill`](experiments/20261002_group_iteration_refill_sssp_q64_q128/README.md)。

### 17.3 Head-of-line eager refill

**目的。** 构造每个 batch 一个 long group，直接验证“长 group 阻塞后三个短 group/后续 batch”的直觉。  
**范围。** Orkut/Twitter，SSSP、N1024/Q128/G32，`tail128` 和每批都有 long group 的 `tail256_hol`；global Iteration + Hybrid。  
**结果。** eager refill 将 rounds 649/659 降到 417/425（tail128），1030 降到 509/519（tail256），896/1024 query 的等待轮数减少；但 active-slot ratio 上升导致 graph-kernel GPU time 增长 1.78--1.85×。tail256 E2E 仅 0.653×/0.801×，平均 latency 0.632×/0.743×。这明确表明共运行干扰是核心问题。见 [`sssp_hol_refill_online_proxy`](experiments/20261002_sssp_hol_refill_online_proxy/README.md)。

### 17.4 干扰感知 compatible bridge（LSSS）

**机制。** 仅当剩一个 resident group、上一轮为 Push、等待 group 的预测长度不短于 resident group 且 feature key 兼容时，允许下一个 cohort 的一个 group 作为 bridge；旧 straggler 完成后一次补满，避免长期 2/3-group 低宽度 pipeline。它不读取 reference rounds、tail 标签或未来运行结果。

**受控 Orkut/Twitter 结果。** tail128 的 12 次不兼容机会被拒绝，性能约 1.000×；tail256_hol 上 Orkut E2E **1.075×**、mean/P99 latency **1.040×/1.075×**，Twitter **1.176×/1.103×/1.173×**。rounds 1030→649/658，kernel GPU 时间基本不变，copy 显著下降。见 [`interference_compatible_bridge_formal`](experiments/20261002_interference_compatible_bridge_formal/README.md)。

**扩展五图稀疏长尾。** cit/LJ/indochina/CA/TX 的 E2E 分别 **1.418/1.466/1.246/1.579/1.538×**；几何平均 E2E **1.444×**、全 query mean latency **1.280×**、P99 **1.443×**、短/长 query mean **1.258/1.338×**；fingerprint 全匹配。见 [`extended_sparse_tail_lsss_results`](experiments/20261002_extended_sparse_tail_lsss_results/README.md)。

### 17.5 自然 seed-42 在线场景：最终负结果

P0 统一消融保持 arrival order，仅在每个 Q128 cohort 内做 length ordering：

- A：Shared + FIFO + no refill；
- B：Iteration + FIFO；
- C：Iteration + cohort-local length；
- D：C + compatible LSSS bridge。

七图 × 三算法 × 四 variants × 五次，共 **420** 正式测量和 **430,080** 条 query latency，fingerprint 全匹配。

| 范围 | Mapping A/B | Length B/C | Refill C/D | 完整 A/D |
|---|---:|---:|---:|---:|
| 五主图（15项） | 1.301×，15/15 | 1.055×，15/15 | **0.877×，0/15** | 1.204×，12/15 |
| 两道路图（6项） | 0.974×，0/6 | 1.003×，6/6 | 0.999×，1/6 | 0.976×，0/6 |
| 全七图（21项） | 1.198× | 1.040× | **0.911×** | 1.134× |

自然 workload 的 refill latency 几何平均也只有 0.922×，仅 5/21 项获益。结论是：LSSS 的 insight 和受控长尾收益成立，但当前 admission guard 对普通 seed-42 arrivals 不够严格，不能把 synthetic 1.444× 当作一般 workload 收益。正式证据见 [`P0 unified_ablation`](experiments/20261003-174459_p0_full_matrix/unified_ablation/summary.csv)。

## 18. Refill 探索中被否定或混杂的版本

| 实验族 | 目的/范围 | 结果 | 使用方式 |
|---|---|---|---|
| `extremes_matched[_lead1/_lead2]` screens | Orkut/Twitter，BFS/SSSP/mixed，极值顺序和 matched refill | 结果高度不稳定；如 Orkut BFS E2E 约 1.10×但 long latency 约 0.41×，Twitter 多数回退 | 仅设计探索，不入论文主结果 |
| `iteration_increment` screens | A、C 都固定 Iteration，隔离 refill 本身 | 六项 E2E 0.996--1.001×，基本中性 | 证明此前大收益不是 refill |
| `safe_iteration_screen/formal` | 试图用 Iteration 保护 refill | 报告 E2E 1.148--1.488× | **混杂**：A 未设 `--push_mapping=iteration`，C 设置了；数字主要含 Shared→Iteration，不能归因 refill |
| continuous interference pipeline v1 | 2/3 group 持续流水 | 干扰和低宽度导致回退 | 被 bridge 取代 |
| bridge v2 | 无 compatibility guard 的一次 bridge | 仍会接纳有害组合 | 被 compatible bridge v3/formal 取代 |

这些目录仍保留完整 config/metrics，以便审计设计路径；最终 refill 结论只采用相同 mapping、相同 workload 顺序的对照。

## 19. 外部系统 Gunrock/Groute

### 19.1 冻结 workload 的早期结果

cit-Patents、N1024，Gunrock/Groute 分别在 M=1 sequential 和最大可接受 power-of-two concurrent（该图为 M256）运行，一次 warmup + 五次。Gunrock concurrent 对 sequential：BFS 1.731×、SSSP 0.972×、mixed 1.119×；Groute：1.062×、1.038×、1.039×。Mixed 被拆成 BFS/SSSP 两次 native invocation 后求和，不代表异构 query 同时共驻。见 [`external_baselines`](experiments/20260928_external_baselines/README.md)。

早期 G8 Hybrid campaign 复用可用 baseline，GraphWeft 对更快 Gunrock 模式的五图三 workload 几何平均为 2.046×；Groute 只有 8/15 项可用，不能用缺失项或 capacity probe 代替正式时间。

### 19.2 道路图正式 baseline

道路图 × BFS/SSSP/SSWP × sequential/concurrent × Gunrock/Groute，每项五次，共 24 configurations。Groute concurrent 为 CA 7.41/10.82/10.28 s、TX 8.52/8.44/8.76 s；sequential 约 21.9--25.5 s。Gunrock concurrent 为 CA 117--166 s、TX 141--207 s；BFS sequential 678/772 s，而 SSSP/SSWP sequential 约 68--75 s。结果见 [`external_baselines_roads`](experiments/20261003-174459_p0_full_matrix/external_baselines_roads/summary.csv)。

五主图 SSWP 外部 baseline 补测在本文最后核对时仍运行，因此不列为已完成结果，也不写入最终 cross-system 几何平均。

## 20. P0 正式矩阵的最终系统解释

当前最稳妥的论文叙事不是“每个优化都普遍提升”，而是：

1. **Hybrid** 处理 Push/Pull 算法阶段差异；五主图上通常显著获益。
2. **Iteration mapping** 处理 Push 轮次内部随 frontier/workload 变化的 query×vertex parallelism；相对冻结 Static 在正式矩阵上约 8.1% 几何平均收益，且无 vertex bucketing。
3. **Length grouping** 处理查询服务长度差异；严格在线 cohort-local 口径仍有约 4.0% 总体收益。
4. **Refill/LSSS** 处理在线到达或强 head-of-line blocking；它只在存在足够强、稀疏且兼容的长尾时获益。当前自然 seed-42 workload 上没有普遍收益，应该作为条件性机制或 future work，而不是无条件默认策略。
5. 道路图暴露出域外泛化问题：五图训练的 Iteration 略回退，普通自然 workload refill 也近中性；但人为构造的超长稀疏 query 又给 refill 大量机会。这三者必须分开陈述。

## 21. 完整实验目录索引

以下索引覆盖当前 `experiments/` 下所有顶层实验目录。带“失败/中止/预检”的目录不进入正式数字，但属于已经做过的实验。

### 21.1 2026-09-21：正确性与初始机制

- `155526_core_validation`（首次失败）→ `155556_core_validation`（修复通过）。
- `155644_full_validation`（无向 reverse-edge 失败）→ `155657_full_validation`（通过）。
- `155818_gpu_acceptance`：真实 GPU、多 batch、replay、sanitizer。
- `160021_kernel_replay`：checkpoint 上 Push/Pull replay。
- `160136_frontier_validation`：frontier 单测通过。
- `160159_frontier_precision`（BFS precision guard 失败）→ `160219_frontier_precision`（通过）。
- `160310_post_grid_validation`：grid 修改后回归。
- `160320_cit_patents_q128`、`160333_cit_patents_q256`：cit 实图大 Q。
- `160445_cli_guards`：显式 memory budget 拒绝与合法 Q 成功。
- `160737_phase_offsets`：SSSP phase offsets/landmarks。
- `160809_final_validation`、`161801_final_gate`：阶段性完整 gate。
- `160902_kernel_resources`、`161943_kernel_resources`：寄存器/spill 审计。
- `161030_round_trace`、`162132_round_timing`、`162157_timing_smoke`：trace/timing 字段。
- `161236_auto_predictors`、`161302_predictor_validation`：core-distance/weighted-boundary 自动预测。
- `161422_frozen_plan`、`161514_plan_capacity`：计划 replay 和 Q guard。
- `161543_core_index_parity`：core 索引 parity。
- `161631_weighted_index_parity`（uint16 overflow 失败）→ `161657_weighted_index_parity`（修复通过）。
- `161834_cit_patents_weighted`：cit 的 SSSP/SSWP 实图回归。
- `161934_warp_compaction`：warp unordered compaction + synccheck。
- `162033_cold_cache`（输入方向失败）→ `162048_cold_cache`（通过）。
- `173921_sssp_e2e_nonroad`：六图初始固定 Push 端到端。
- `201450_sssp_n32_hybrid`：六图 Hybrid。
- `211343_compare_kernel_n32_hybrid`：全 V×Q compare 诊断。
- `225549_push_partition_rounds`：30 种 Push candidate 逐轮实验与 workload analysis。

### 21.2 2026-09-22 至 09-28：Pull、layout、frontier 和第一轮系统 campaign

- `20260922-143718_pull_partition_check`：失败/中止 pilot。
- `20260922-144225_pull_partition_final`：正式 Pull check/check-free + NCU。
- `20260926_grouped_pull_layout`：84-config layout 实验。
- `20260927_fixed_m64_capacity`：M64 容量。
- `20260927_fixed_n1024_m64`：五图自然 tail audit、ABCD、G sensitivity、kernel mapping；大部分完成，twitter 的部分 sensitivity/kernel 子矩阵未完成，报告中有明确 measured/expected。
- `20260927_grouped_g8_edge4_warp4`：G8 Pull 特化。
- `20260928_direct_frontier`、`20260928_mask64_ab`：frontier 设计和 64-bit publication。
- `20260928_mask64`、`mask64_v2`、`mask64_sssp`：mask64 开发/补测目录；正式引用 `mask64_ab`。
- `20260928-232422/232554/233229/234051_hybrid_mask64_campaign`、`232500_smoke`、`234200_gpu_fingerprint_smoke`：失败、重启或 smoke；正式 campaign 为 `234229`。
- `20260928-234229_hybrid_mask64_campaign`：90 cases/450 samples 正式 G8 campaign。
- `20260928_external_baselines`：早期 native baseline；`external_baselines_matched`、`matched_full`、`remaining` 为后续匹配/补充目录。

### 21.3 2026-09-29 至 09-30：G32、Adaptive 与 Iteration

- `20260929-111853_refill_g32_campaign`：自然 workload eager refill 负结果。
- `20260929-175832_adaptive_push_g32`：Vertex-Adaptive 正式 campaign。
- `20260929_iteration_calibration`：初始中止采样、parallel resume、dataset split、Ridge 训练与 model audit 共存；当前模型由后续完整数据生成。
- `20260929_strong_tail_workloads`：LJ/Orkut 派生图和六类 workload 验证。
- `20260930-223628_iteration_sssp_allpush_e2e`：Iteration SSSP All-Push preliminary。

### 21.4 2026-10-01 至 10-02：refill 优化与干扰感知

- `20261001-112124_iteration_sssp_hybrid_e2e`：Iteration SSSP Hybrid。
- `191748_strong_tail_refill_orkut_twitter`：FIFO strong-tail refill。
- `225500_strong_tail_refill_predicted_orkut_twitter`：预测顺序。
- `230100_strong_tail_refill_predicted_optimized_orkut_twitter`：预测顺序 + kernel 优化。
- `231000_strong_tail_refill_comparison`：上述三者汇总。
- `20261002-064700_strong_tail_refill_wave_fastreset_screen`：wave/fast reset screen。
- `extended_sparse_tail_lsss`：扩展 workload 准备；`extended_sparse_tail_lsss_results`：五图正式结果。
- `group_iteration_refill_q_screen`：Q/per-group mapping 预检；`group_iteration_refill_sssp_q64_q128`：正式消融。
- `sssp_hol_refill_preflight`：HoL 预检；`sssp_hol_refill_online_proxy`：eager refill 正式负结果。
- `interference_refill_screen_v1`：continuous pipeline；`interference_bridge_screen_v2`：无 guard bridge；`interference_compatible_bridge_screen_v3`：compatibility screen；`interference_compatible_bridge_formal`：正式结果。
- `v100_refill_extremes_matched*`：extremes/matched/lead1/lead2 探索，含 Orkut/Twitter BFS/SSSP/mixed；未形成稳健统一收益。
- `v100_refill_iteration_increment*`：相同 Iteration mapping 下隔离 refill，结果约中性。
- `v100_refill_safe_iteration_screen*`、`formal*`：A/C mapping 不一致的混杂实验，仅作审计，不作 refill 证据。

### 21.5 2026-10-03 至今：P0 paper matrix

- [`20261003-174459_p0_full_matrix`](experiments/20261003-174459_p0_full_matrix/README.md)：七图 Iteration matrix、七图统一消融、两道路图 Gunrock/Groute 已完成；五主图 SSWP 外部 baseline 仍在补测。

### 21.6 顶层实验目录逐项清单

为避免分组写法漏掉 smoke、失败重跑或参数 screen，下面是本文核对时
`experiments/` 下全部 **112** 个顶层目录/链接。每个名称都已归入上面的正式、
机制、探索、预检、失败或混杂类别；目录内原始文件仍是最终证据源。

```text
20260921-155526_core_validation
20260921-155556_core_validation
20260921-155644_full_validation
20260921-155657_full_validation
20260921-155818_gpu_acceptance
20260921-160021_kernel_replay
20260921-160136_frontier_validation
20260921-160159_frontier_precision
20260921-160219_frontier_precision
20260921-160310_post_grid_validation
20260921-160320_cit_patents_q128
20260921-160333_cit_patents_q256
20260921-160445_cli_guards
20260921-160737_phase_offsets
20260921-160809_final_validation
20260921-160902_kernel_resources
20260921-161030_round_trace
20260921-161236_auto_predictors
20260921-161302_predictor_validation
20260921-161422_frozen_plan
20260921-161514_plan_capacity
20260921-161543_core_index_parity
20260921-161631_weighted_index_parity
20260921-161657_weighted_index_parity
20260921-161801_final_gate
20260921-161834_cit_patents_weighted
20260921-161934_warp_compaction
20260921-161943_kernel_resources
20260921-162033_cold_cache
20260921-162048_cold_cache
20260921-162132_round_timing
20260921-162157_timing_smoke
20260921-173921_sssp_e2e_nonroad
20260921-201450_sssp_n32_hybrid
20260921-211343_compare_kernel_n32_hybrid
20260921-225549_push_partition_rounds
20260922-143718_pull_partition_check
20260922-144225_pull_partition_final
20260926_grouped_pull_layout
20260927_fixed_m64_capacity
20260927_fixed_n1024_m64
20260927_grouped_g8_edge4_warp4
20260928-232422_hybrid_mask64_campaign
20260928-232500_hybrid_mask64_smoke
20260928-232554_hybrid_mask64_campaign
20260928-233229_hybrid_mask64_campaign
20260928-234051_hybrid_mask64_campaign
20260928-234200_gpu_fingerprint_smoke
20260928-234229_hybrid_mask64_campaign
20260928_direct_frontier
20260928_external_baselines
20260928_external_baselines_matched
20260928_external_baselines_matched_full
20260928_external_baselines_remaining
20260928_mask64
20260928_mask64_ab
20260928_mask64_sssp
20260928_mask64_v2
20260929-111853_refill_g32_campaign
20260929-175832_adaptive_push_g32
20260929_iteration_calibration
20260929_strong_tail_workloads
20260930-223628_iteration_sssp_allpush_e2e
20261001-112124_iteration_sssp_hybrid_e2e
20261001-191748_strong_tail_refill_orkut_twitter
20261001-225500_strong_tail_refill_predicted_orkut_twitter
20261001-230100_strong_tail_refill_predicted_optimized_orkut_twitter
20261001-231000_strong_tail_refill_comparison
20261002-064700_strong_tail_refill_wave_fastreset_screen
20261002_extended_sparse_tail_lsss
20261002_extended_sparse_tail_lsss_results
20261002_group_iteration_refill_q_screen
20261002_group_iteration_refill_sssp_q64_q128
20261002_interference_bridge_screen_v2
20261002_interference_compatible_bridge_formal
20261002_interference_compatible_bridge_screen_v3
20261002_interference_refill_screen_v1
20261002_sssp_hol_refill_online_proxy
20261002_sssp_hol_refill_preflight
20261002_v100_refill_extremes_matched_lead1_screen_soc-orkut_bfs
20261002_v100_refill_extremes_matched_lead1_screen_soc-orkut_mixed
20261002_v100_refill_extremes_matched_lead1_screen_soc-twitter_bfs
20261002_v100_refill_extremes_matched_lead1_screen_soc-twitter_mixed
20261002_v100_refill_extremes_matched_lead2_screen_soc-orkut_bfs
20261002_v100_refill_extremes_matched_lead2_screen_soc-orkut_mixed
20261002_v100_refill_extremes_matched_lead2_screen_soc-twitter_bfs
20261002_v100_refill_extremes_matched_lead2_screen_soc-twitter_mixed
20261002_v100_refill_extremes_matched_screen_soc-orkut_bfs
20261002_v100_refill_extremes_matched_screen_soc-orkut_mixed
20261002_v100_refill_extremes_matched_screen_soc-orkut_sssp
20261002_v100_refill_extremes_matched_screen_soc-twitter_bfs
20261002_v100_refill_extremes_matched_screen_soc-twitter_mixed
20261002_v100_refill_extremes_matched_screen_soc-twitter_sssp
20261002_v100_refill_iteration_increment_screen_soc-orkut_bfs
20261002_v100_refill_iteration_increment_screen_soc-orkut_mixed
20261002_v100_refill_iteration_increment_screen_soc-orkut_sssp
20261002_v100_refill_iteration_increment_screen_soc-twitter_bfs
20261002_v100_refill_iteration_increment_screen_soc-twitter_mixed
20261002_v100_refill_iteration_increment_screen_soc-twitter_sssp
20261002_v100_refill_safe_iteration_formal_soc-orkut_bfs
20261002_v100_refill_safe_iteration_formal_soc-orkut_mixed
20261002_v100_refill_safe_iteration_formal_soc-orkut_sssp
20261002_v100_refill_safe_iteration_formal_soc-twitter_bfs
20261002_v100_refill_safe_iteration_formal_soc-twitter_mixed
20261002_v100_refill_safe_iteration_formal_soc-twitter_sssp
20261002_v100_refill_safe_iteration_screen_soc-orkut_bfs
20261002_v100_refill_safe_iteration_screen_soc-orkut_mixed
20261002_v100_refill_safe_iteration_screen_soc-orkut_sssp
20261002_v100_refill_safe_iteration_screen_soc-twitter_bfs
20261002_v100_refill_safe_iteration_screen_soc-twitter_mixed
20261002_v100_refill_safe_iteration_screen_soc-twitter_sssp
20261003-174459_p0_full_matrix
```

## 22. 尚未完成或不能声称的内容

- 不能声称自然 workload 上 refill 普遍改善 latency/throughput；当前正式结果相反。
- 不能将 `safe_iteration_formal` 的 1.15--1.49× 写成 refill 收益，因为 baseline mapping 不一致。
- 不能将受控 path-tail 的 1.444× 外推为原图自然 query distribution。
- 不能声称 Iteration 对道路图有收益；当前约回退 2%。需要独立 road calibration 或 domain-aware guard，但不得用 seed42 正式结果反向调参。
- 五主图完整 Gunrock/Groute SSWP baseline 尚未结束，最终 cross-system 表需等该任务完成。
- Static 只对五主图有独立冻结配置；道路图不应从正式 seed42 数据挑 Static。

## 23. 推荐的论文实验引用层级

1. **主表**：P0 `iteration_matrix`，报告 Shared/Static/Iteration、All-Push/Hybrid、五主图三算法；道路图作为外推/敏感性。
2. **组件消融**：P0 `unified_ablation`，诚实报告 Mapping 和 length 正收益、natural refill 负收益。
3. **机制图**：Push/Pull partition 的逐轮 candidate heatmap、density/check tradeoff、NCU evidence。
4. **条件性调度实验**：compatible bridge 的 tail256_hol 和 extended sparse-tail，标题和正文均明确 controlled synthetic long-tail。
5. **正确性**：P0 fingerprint、Q/mask-word/tail tile、replay、memcheck/synccheck。
6. **外部 baseline**：只合并计时边界明确且已经完成的行；缺失项继续标缺失，不能以 capacity probe 代替。
