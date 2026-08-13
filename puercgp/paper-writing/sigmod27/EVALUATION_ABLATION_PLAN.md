# 6.3–6.5 节消融实验规划

> 状态：等待 review 的初稿。本文件只规划实验，不启动实验，也不修改论文正文。
> 范围：6.3 Sharing-aware Scheduler、6.4 Hybrid Computation Strategy、6.5 Multi-hop Propagation。

## 1. 总体实验逻辑

这三个章节应当依次回答三个不同的问题：

1. **6.3：调度器能否创造更多可共享的计算？** 研究哪些查询被放在一起，以及它们何时开始执行。
2. **6.4：运行时能否高效利用已经暴露出的共享机会？** 研究每轮应采用哪一种计算引擎。
3. **6.5：传播本身能否用更少的轮次和工作量完成？** 研究 in-place 更新与多跳预处理图。

三节应形成如下因果链，而不是展示三组互不相关的 speedup：

```text
查询组织方式 → 每轮计算引擎 → 传播轮次与单轮代价
```

每个实验都同时提供两类证据：

- **端到端证据**：runtime、throughput 或总体加速比，证明机制具有实际价值；
- **机制证据**：真实共享率、边访问量、活跃查询对、迭代数、硬件指标等，解释性能变化的原因。

例如，调度器加速必须同时展示真实 frontier/邻接共享量；in-place 减少轮次必须同时展示 runtime 或边检查量。不能仅凭迭代数下结论，因为一次 in-place 轮次可能包含比同步轮次更多的依赖传播。

整体论证方式可以参考 MITra、Glign 和 TGraph 中“总体效果—机制拆解—敏感性”的系统论文逻辑，但图形、指标和对照组必须围绕我们的三个组件设计，不能直接照搬其图表。

## 2. 统一实验协议与数据口径

### 2.1 工作负载与控制变量

- 数据集：**CP**（cit-Patents）、**LJ**（soc-LiveJournal1）、**IN**（indochina-2004）、**UK**（uk-2002）、**TW**（soc-Twitter）、**OK**（soc-Orkut）。
- 算法：BFS、SSSP、SSWP。
- 查询：继续使用现有统一的 256-query 清单；较小 N 必须是同一清单的严格前缀。
- 消融主场景：`N=256`，因为它具有最大的跨查询共享空间。
- 规模趋势：机制预期随并发查询规模变化时，补充 `N ∈ {32,64,128,256}`。
- 默认驻留批大小：`Q=32`。6.3 不改变 Q；仅 6.4.2 的 pull-kernel 微基准可选补充少量 Q 点，用于分析访存合并，而不作为调度敏感性实验。
- 默认调度参数：32 个 landmarks、`max_offset=16`、`batch_swaps=16`。
- 重复协议：复用现有 PuerCGP C0–C9 的协议，即 1 次预热、2 次正式运行，正式结果报告两次中位数。新增消融采用相同协议，保证横向可比。
- 主时间：排除图加载、静态 H2D 和离线预处理；包含 query 初始化/reset、在线 planner/evaluator、kernel、批次交接与同步。
- 正确性：继续使用完整向量和 fingerprint；出现 OOM、超时或不一致时保留原 query，不更换查询规避失败。
- 聚合方式：加速比跨图聚合使用几何平均，同时展示每张图的数据点或 win/total，不能只报平均数。

### 2.2 主文可视化的相对比值口径

原始 CSV、日志与 artifact 继续保存绝对 runtime、边数、内存和迭代数，以保证结果可复现；但消融实验的**主文表格和图片优先展示加入组件前后的相对比值**，让读者直接判断组件是否有效。

统一方向如下：

| 含义 | 计算方式 | 图中判定 |
|---|---|---|
| Runtime speedup | `T_before/T_after` | `>1` 表示组件带来加速 |
| Throughput gain | `Throughput_after/Throughput_before` | `>1` 表示吞吐提高 |
| Sharing gain | `ASF_after/ASF_before` | `>1` 表示逻辑 sharing 增强 |
| Iteration/work reduction factor | `Work_before/Work_after` | `>1` 表示工作量减少 |
| Memory/planner overhead | `Cost_after/Cost_before` 或占 end-to-end 的比例 | `>1` 表示额外成本 |

所有 before/after 必须来自相同 dataset、algorithm、N、Q 和 query 清单，仅改变被研究的组件。主图统一画 `y=1` 参考线。绝对值只在实验设置、复现文件或确有必要的附录中出现。

### 2.3 C0–C9 的统一定义

后续所有匹配对照均以此表为准：

| 配置 | 图 | Pull 更新 | Query batching | Start offset |
|---|---|---|---|---|
| C0 | 原图 | 同步 | FIFO/输入顺序 | 0 |
| C1 | 原图 | 同步 | Sharing-aware | 0 |
| C2 | 原图 | 同步 | FIFO/输入顺序 | 开启 |
| C3 | 原图 | 同步 | Sharing-aware | 开启 |
| C4 | 原图 | In-place | FIFO/输入顺序 | 0 |
| C5 | 预处理图 | 同步 | FIFO/输入顺序 | 0 |
| C6 | 预处理图 | In-place | FIFO/输入顺序 | 0 |
| C7 | 预处理图 | 同步 | Sharing-aware | 开启 |
| C8 | 原图 | In-place | Sharing-aware | 开启 |
| C9 | 预处理图 | In-place | Sharing-aware | 开启 |

### 2.4 可直接复用的已有数据

已有实验目录：

`/home/zyl/Projects/ocgp/experiments/20260808_six_system_n_scaling/`

可用数据包括：

- `puercgp_ablation.csv`：720 个 PASS 中位数，覆盖 6 图 × 3 算法 × 4 个 N × C0–C9；
- `raw_results.csv`：1,440 条 PuerCGP 正式运行原始记录；
- `artifacts/runs/puercgp-<dataset>-<algorithm>-n<N>/<case>/runs.csv`：每次运行的 runtime、evaluator/grouping/offset 时间、alignment score、总迭代数以及 push/pull 迭代数；
- `artifacts/preprocessing/<dataset>/<algorithm>.json`：18 份预处理 manifest，包含预处理时间、产物大小、checksum、mapping 和 recipe；
- `artifacts/calibration/PuerCGP/<dataset>/<algorithm>/selected.json`：现有 threshold sweep 与独立校准源点上的选值；
- `correctness.csv` 及完整向量/fingerprint 产物。

这些数据足以完成匹配配置的 runtime/iteration 初步消融，但还不能支持全部机制性结论。当前缺少：

- 每轮精确的 `E_union`、`E_pair` 和 active-pair 数；
- 调度后的真实 frontier/邻接共享率；
- 固定 Push-only/Pull-only 的运行结果；
- 重排和 shortcut 两个预处理阶段各自的在线收益；
- 部分硬件计数器与预处理阶段 breakdown。

### 2.5 新实验的产物目录

新增实验不覆盖历史 CSV，使用独立目录：

```text
/home/zyl/Projects/ocgp/experiments/<timestamp>_paper_ablation_6_3_6_5/
├── README.md
├── config.json
├── metrics.csv
├── stdout/
├── stderr/
├── artifacts/
│   ├── traces/
│   ├── profiles/
│   ├── preprocessing/
│   └── figures/
└── result.csv
```

每个结果必须保留 dataset、algorithm、N、Q、配置、repeat、command ID、图和 query checksum、正确性状态及 external-overlap 信息。

## 3. Section 6.3 — Sharing-aware Scheduler

建议节标题改为 **Sharing-aware Scheduling**。本节需要证明：在固定 `N=256,Q=32` 下，batching 确实提高真实共享，bounded offset 能进一步对齐传播阶段，并且两者的在线开销没有抵消收益。

### 3.0 统一的 Sharing 指标

本节采用“逻辑指标 + 实际指标”的两层评价口径：

| 层次 | 主指标 | 回答的问题 | 结论边界 |
|---|---|---|---|
| 逻辑 sharing | ARR/ASF；主图使用 `ASF_after/ASF_before` | 调度后形成了多少可合并的重复邻接工作？ | 只解释共享机会，不能直接等价为加速或物理流量下降 |
| 实际性能 | 包含在线 planner 的 end-to-end speedup `T_before/T_after` | 这些共享机会最终是否转化为实际性能收益？ | 是判断配置优劣与论文性能结论的最终依据 |

其中，**Adjacency Reuse Ratio（ARR，邻接复用率）**定义如下。对第 `t` 轮和同一 resident batch：

```text
E_pair(t)  = Σ_q Σ_{v∈F_q(t)} degree(v)
E_union(t) = Σ_{v∈∪_q F_q(t)} degree(v)

ARR = 1 - Σ_t E_union(t) / Σ_t E_pair(t)
```

其中，`E_pair` 表示所有 query 分别执行时需要读取的邻接项数量；`E_union` 表示合并相同 active vertex、每个邻接表只读取一次时需要读取的邻接项数量。对于以入边为中心的 kernel，`degree(v)` 对应实际访问的入度；对于以出边为中心的 kernel，则使用出度。

实现时可直接复用每个 active vertex 的 query mask。若 `m(v)=popcount(mask(v))`，则该顶点对 `E_pair` 的贡献为 `degree(v)·m(v)`，对 `E_union` 的贡献为 `degree(v)`，无需真的执行一遍独立查询 baseline。

- `ARR=0`：没有可复用的邻接访问；
- `ARR=0.5`：相对于 query 独立执行，理论上避免了 50% 的重复邻接读取；
- ARR 越大，执行期实际暴露出的共享机会越多。

同时可以报告等价的 **Adjacency Sharing Factor**：

```text
ASF = Σ_t E_pair(t) / Σ_t E_union(t) = 1 / (1-ARR)
```

`ASF=2` 表示一次邻接遍历平均服务于两个 query frontier。单个配置仍记录 ARR/ASF 原始逻辑指标；组件消融的主图使用 `ASF_after/ASF_before` 作为 sharing gain。ARR 接近 0 时直接计算 `ARR_after/ARR_before` 不稳定，因此不使用 ARR 比值。

ARR 衡量的是由执行期真实 active frontiers 推导出的**逻辑邻接复用量**，不是 GPU cache 命中、物理 DRAM transaction，也不是性能收益本身。最终是否有效以包含在线调度开销的 end-to-end runtime 为准。硬件流量则在 topology-kernel profiling 中用实测 DRAM bytes/L2 hit rate 单独验证，避免混淆三个层次。

计算时必须先跨轮次累加 `E_pair/E_union` 再求比值，不能简单平均每轮 ARR，否则工作量很小的轮次会获得过高权重。跨数据集汇总时展示每张图的独立点，并对 ASF 使用几何平均，避免大图按边数主导整个结论。

现有 `alignment_score` 仍然保留，但它只表示 scheduler 的**预测分数**；ARR 是执行 trace 得到的**真实共享量**。本节的完整证据链为：

```text
alignment score（预测）→ ARR/ASF（逻辑共享）→ end-to-end runtime（实际收益）
```

### 3.1 Query Batching

#### 实验目的

验证基于 landmark/relative-level 的成对 affinity 是否能把具有真实 frontier 和邻接重叠的查询放入同一批次，并确认节省的遍历工作量能够覆盖在线 grouping 开销。

目标因果链为：

```text
更高的预测 affinity
→ 更高的真实 frontier/邻接共享率
→ 更少的重复邻接访问
→ 更低的端到端时间
```

#### 对照组

可立即复用的核心匹配对照：

- **C0 vs. C1**：原图、同步更新、零 offset，仅改变 query batching。

以下两个控制组降为**最低优先级的可选实验**，不阻塞 6.3 主结论和论文初稿：

1. **Random batching**：固定 3 个随机种子，用于排除“任意打乱输入顺序就可能得到相同收益”的解释；
2. **Scalar landmark order（暂缓）**：原设想是只按单一 landmark/level 标量排序，用来判断 pairwise affinity 是否优于更简单的一维排序启发式。该 baseline 的算法定义和研究价值尚未达成共识，当前不实现、不运行，待后续讨论。

核心正式比较只有 **FIFO vs. Pairwise sharing-aware（C0 vs. C1）**。Random 仅在所有主要消融完成后执行；若其结果提供了清晰且不可由 C0/C1 替代的机制信息，再决定放入正文或附录。无论结果方向如何，原始结果均保留，不能只保留有利随机种子。Scalar 不进入当前实验矩阵。

#### 数据来源

- **可直接复用**：C0/C1 runtime、iterations、`grouping_ms`、现有 `alignment_score`；
- **最低优先级可选新增运行**：Random batching；
- **暂不运行**：Scalar landmark order；
- **需要新增计数器**：
  - 每批/每轮 active queries 和 active `(vertex,query)` pairs；
  - `E_pair`：按 query 独立展开时应检查的边数；
  - `E_union`：合并 active vertices 后实际检查的边数；
  - 按 3.0 节定义计算的 ARR/ASF；
  - 预测 affinity、pair swaps 数量及 scheduler 时间。

#### 数据展示格式

**Figure 9(a)，双面板：**

- 左图：`N=256` 下加入 sharing-aware batching 的 end-to-end speedup，即 `T(C0)/T(C1)`。x 轴为六张图，每张图展示 BFS/SSSP/SSWP；1.0 水平线明确标出负收益，planner 时间包含在 C1 内。
- 右图：相同配置的 sharing gain，即 `ASF(C1)/ASF(C0)`；与左图使用完全相同的横坐标，便于判断逻辑 sharing 增长是否转化为实际加速。

预测 alignment score 与真实 ARR/ASF 的相关性散点图移入附录，作为 predictor validity 证据，不占用主图的 before/after 比值叙事。Random 若最终执行且能提供额外解释，可作为附录控制组或主图中的浅色参考点；Scalar 当前不展示。

**配套小表：** 每个算法一行，展示 24 个 `(dataset,N)` 中的 win 数、几何平均 runtime speedup、几何平均 sharing gain 和 planner/end-to-end 比例。

#### 论文叙事与边界

- 小 N 下如果 grouping 开销大于收益，必须保留并解释；
- 如果 affinity 与共享率相关但与 runtime 相关性弱，应检查负载不均衡或 divergence，而不是直接宣称模型错误；
- SSSP/SSWP 当前使用无权 landmark 结构，应明确标注为 **algorithm-independent scheduling proxy**，不能描述成 weighted-distance oracle。

### 3.2 Query Delay

建议将子标题改为 **Start-offset Alignment**。目前的 “Query Delay” 容易被理解为在线服务中的 query arrival latency，而我们的机制是主动改变 resident query 的逻辑起始 phase。

#### 实验目的

验证 bounded start offsets 能否在 batching 之外进一步对齐传播阶段，并同时量化这种对齐给单查询完成时间带来的逻辑延迟。

#### 对照组与参数

现有匹配对照：

- **C0 vs. C2**：FIFO 下的 offset-only 效果；
- **C1 vs. C3**：sharing-aware batching 后 offset 的边际效果；

以上两组对照已经足以回答本小节的问题。**C2 vs. C3 不放入 Query Delay 小节**：两者都启用了相同的 offset，唯一差异是 FIFO 与 sharing-aware batching，因此它隔离的是 batching，而不是 delay。若后续需要证明 batching 在开启 offset 后仍然有效，可将 C2/C3 作为 6.3.1 的附录稳健性结果；主文不必重复展示。

本小节的工作负载固定为：

- 总查询数 `N=256`；
- 固定 batch size `Q=32`，因此完整 workload 由 8 个 batch 组成；
- query 清单和前后顺序保持不变；
- offset 关闭时使用 C0/C1，offset 开启时使用当前生产设置 `max_offset=16` 的 C2/C3；
- 覆盖六图和三算法。

本小节不进行 batch-size sweep，也不进行 `max_offset` sweep。此前使用的 `D` 表示最大 offset，不是 batch size；为避免符号混淆，本规划删除该符号和对应敏感性实验。

#### 数据来源

- **可直接复用**：C0–C3 runtime、iterations、`grouping_ms`、`offset_ms` 和 alignment score；
- **不需要新增参数 sweep**：现有 C0–C3 已覆盖固定 Q 下 offset 关闭/开启的核心对照；
- **需要新增计数器或 trace**：每个 query 的实际 start offset、logical completion step、相对 offset-disabled 配置的 p50/p95 completion stretch，以及 offset 前后的 ARR。

#### 数据展示格式

**Figure 9(b)，三个纵向对齐的紧凑子面板：**

1. Offset speedup：`T(C0)/T(C2)` 与 `T(C1)/T(C3)`；
2. Sharing gain：`ASF(C2)/ASF(C0)` 与 `ASF(C3)/ASF(C1)`；
3. Completion stretch：offset-enabled 相对 matched offset-disabled 配置的 p50/p95 completion-time ratio。

三个面板使用相同的 dataset/algorithm 横坐标，不展示 C2/C3，也不展示 Q 或 `max_offset` 的参数曲线。

正文或小表补充：上述两组 speedup/sharing gain 的几何平均、planner/end-to-end 比例，以及获得非零 offset 的 query 比例。

#### 判定原则

- 全局轮次减少不能掩盖过大的单 query completion delay；两者必须同时报告；
- offset 只改变 resident query 的开始时间，不改变算法语义。

### 3.3 Impact of Query Batch Size（暂缓）

本轮消融**不执行 batch-size 实验，也不将其放入 6.3 主文**。6.3 的所有结果固定总查询数 `N=256`、驻留 batch size `Q=32`，只研究 batching policy 与 start offset。

若后续篇幅和实验资源允许，可将 Q sensitivity 独立放入 sensitivity experiment；届时再固定 `N=256`，改变 Q，并明确它不属于 Query Delay 的因果对照。当前不为该实验预留 Figure 9 子图，也不提前声称 Q=32 最优。

## 4. Section 6.4 — Hybrid Computation Strategy

先修正原文的 `Hybird` 拼写。本节统一在完整 **C9** 环境下研究 push/pull 及 pull kernel，不再额外构造原图、同步、FIFO 的隔离环境。默认总查询数 `N=256`、驻留 batch size `Q=32`，使用同一份 256-query 清单，覆盖六图和三算法。

### 4.0 术语边界与本文实现映射

根据 [Liu et al., *Choosing the Best Parallelization and Implementation Styles for Graph Analytics Codes*, §2.2 和 §2.4](https://userweb.cs.txstate.edu/~burtscher/papers/sc23a.pdf)，以下两组概念在一般图计算文献中是不同维度：

- **Topology-driven vs. data-driven** 描述每轮处理范围：前者扫描全部图元素，后者只处理 worklist/frontier 中可能发生变化的元素；
- **Push vs. pull** 描述数据流方向：push 从活跃源点向外传播，pull 由目的点读取邻居状态。

GraphWeft 的当前实现把这两个维度具体绑定为：

- **Pull-only**：full/topology scan + pull/gather；
- **Push-only**：active frontier/worklist + push/scatter；
- **Hybrid**：运行时在上述两个具体执行模式之间选择。

因此正文和图例优先使用 **Push-only、Pull-only、Hybrid**，避免把一般意义上的 topology/data 与 push/pull 写成同义词；文字中再说明 GraphWeft 当前实现的具体绑定。这样既符合本文实现，也不会把一种实现选择误写成通用定义。

### 4.1 Push vs. Pull Execution

#### 实验目的

本小节回答两个问题：

1. 在完整 C9 设置下，强制 push、强制 pull 和生产 Hybrid 的端到端性能分别如何？
2. 哪些**运行时状态**使 push 或 pull 更合适，Hybrid 的选择是否与真实 crossover 一致？该问题暂缓，不进入本轮实验。

Hybrid 从机制上可以退化为某个固定模式，但“理论上可以调到不差于二者”不等于当前生产 selector 在未知正式 workload 上必然胜出。若使用正式测试结果反向调参，会引入 test-set leakage。因此本节报告当前 C9 selector 的观察结果；参数/oracle 调优留待 4.3，不预先承诺 Hybrid 在每个点都严格更快。

#### 端到端对照

保持 C9 的预处理图、in-place pull、sharing-aware batching、start offset、query 清单及其他参数不变，只强制执行模式：

1. **Push-only**：每轮只执行 frontier/worklist push；
2. **Pull-only**：每轮只执行 full-scan pull；
3. **Hybrid**：使用当前生产 selector；threshold 使用独立 calibration sources 已冻结的 C9 参数，不能根据这组正式结果重新选择。

主场景固定 `N=256,Q=32`，覆盖六图 × 三算法。push 模式本身不存在“in-place pull”，这里的 C9 表示除被强制的计算模式外，图、调度、offset、状态布局等控制变量均取 C9 设置；不能把 pull 特有语义硬套到 push 上。

正式结果报告：

- Hybrid 相对 Push-only 的 speedup：`T_push/T_hybrid`；
- Hybrid 相对 Pull-only 的 speedup：`T_pull/T_hybrid`；
- Push/Pull 直接比值：`T_push/T_pull`，大于 1 表示 pull 更快。

#### “何时更好”的诊断实验（暂缓）

仅比较三个完整运行可以回答总体性能，但不足以严谨解释“什么状态下应选择 push 或 pull”，因为两个 fixed-mode 运行会沿不同中间状态与迭代轨迹前进。此前提出的 matched-state shadow replay 暂不实现、不运行，也不在本轮结果中声称已找到 crossover。

论文 6.4 将使用红色 TODO 明确提醒：后续需要补充一种能够公平衡量单步推进量与执行代价的诊断方法，再回答 push/pull 的适用条件。当前正文只报告 Push-only、Pull-only 和 Hybrid 的端到端对照，以及 Hybrid 中两种模式的实际迭代/时间占比；不能依据端到端结果反推逐轮决策规律。

#### 数据展示格式

**Figure 10(a)：端到端 Hybrid 收益。** 展示 `T_push/T_hybrid` 和 `T_pull/T_hybrid`，按算法分组并叠加六图点，统一以 1 为参考线。

正文小表报告 Hybrid 中 push/pull 的迭代占比、时间占比和 evaluator/end-to-end 比例。“何时更好”的 crossover 图暂不生成，并在正文保留红色 TODO。

### 4.2 Comparison of Different Pull Kernels

#### 实验目的

在完全相同的 pull 状态、query masks 和 C9 数据布局下，比较不同 pull 实现，验证 GE-SpMM-like GraphWeft pull 是否确实提高全局内存访问合并度，并将底层访存改善转化为 kernel 与端到端收益。

#### 候选 kernel 与代码来源门禁

当前确认的生产实现位于：

- `include/puercgp/kernels/pull/fused_pull_kernels.hxx`：GraphWeft 的 degree-aware、query-lane cooperative、GE-SpMM-like gather；
- `src/kernels/ge_spmm_bench.cu` 与 `src/kernels/kernel_compare.cu`：已有独立 GE-SpMM/SegSumVec 对照基础；
- `src/kernels/segment_sum_bench_vec.cu`：用户确认其为此前抽取的 Gunrock-derived segment-reduce pull 实现。

当前 `segment_sum_bench_vec.cu` 的文件头和现有 git 提交仍没有记录原始 Gunrock 文件、commit 或移植说明；官方 Gunrock V2.2 的 `neighborreduce` pull 路径在本地源码中因 ModernGPU 移除而被标记为不再支持。正式实验前应为该文件补充 provenance metadata，至少记录“Gunrock-derived”、对应上游版本/文件（若能恢复）以及本地语义适配。论文图例暂写 **Gunrock-derived Segment-reduce pull**，避免暗示它是未经修改的官方现成二进制。

正式比较只保留两个语义等价并通过相同 correctness gate 的实现：

1. **Segment-reduce pull**；
2. **GE-SpMM-like pull（GraphWeft）**。

若候选 kernel 当前只支持 float sum/SpMM，必须先改成与 BFS/SSSP/SSWP 相同的 32-bit value layout、query masks 和 semiring，再进入正式实验；不能用不同计算语义的 standalone SpMM 时间替代图遍历 pull。

#### 实验矩阵

- **正式集成实验**：完整 C9 环境、`N=256,Q=32`、六图 × 三算法，强制 Pull-only，仅替换 pull backend；报告端到端和累计 pull-kernel 时间。
- **受控 kernel 微基准**：给两个 pull kernel 输入完全相同的 C9-format values、CSR 与 query columns，比较纯执行时间与硬件计数器。该实验只隔离 pull 实现，不用于推断 Push/Pull crossover。
- **Q 微基准**：仅在 kernel replay 中测试 `Q∈{8,16,32,64}`，用于观察 query-column 宽度对合并访问的影响；正式端到端仍只用 Q=32。Q=64 必须先通过显存容量、kernel launch 和正确性预检；若发生 OOM 或内部资源上限失败，则保留失败记录并将正式可用集合记为 `{8,16,32}`，不通过拆批或更换 query 规避。该实验不属于 6.3 的 batch-size sensitivity，也不用于声称调度层最佳 Q。

#### NCU 内存访问效率指标

主指标选择 V100/Nsight Compute 可直接采集的：

```text
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
```

它表示 global-load 每个 32-byte sector 中平均被实际请求的数据字节比例，越接近 100% 表示 sector 利用率越高、访问越合并。它最直接对应“GE-SpMM-like query-lane layout 改善 coalescing”的设计主张。

同时采集以下辅助指标，避免用单一 counter 过度解释：

- `dram__bytes_read.sum / useful neighbor-query operations`：每个有效邻居-query 工作消耗的 DRAM 字节；
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：是否接近带宽上限；
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`：长延迟内存依赖造成的 stall；
- `lts__t_sector_hit_rate.pct`：L2 sector hit rate；
- `smsp__warps_active.avg.pct_of_peak_sustained_active`：活跃 warp/occupancy 上下文；
- 若 segment-reduce 需要 atomic partial results，再采集 `l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum` 与 `smsp__sass_inst_executed_op_atom.sum`。

`dram throughput` 高并不自动表示访问高效，也可能只是产生了更多流量。因此论文中的核心证据采用 **global-load sector utilization gain** 和 **DRAM bytes per useful operation reduction**；throughput、L2 hit 和 scoreboard stall 只作解释性指标。`ge_spmm_bench.cu` 现有的 full/load-only/graph-only 模式可用于区分 value-column load 与 CSR graph-index load。

Profiler 至少覆盖 CP、IN、TW 三张预先固定的代表图及三算法的 Q=32 快照；若开销可接受再扩展六图。所有 NCU run 只作机制分析，不进入正式 timing。

#### 数据展示格式

**Figure 10(b)，双面板：**

- 左：GraphWeft pull 相对 segment-reduce pull 的端到端 speedup `T_segment/T_ge-spmm`，按算法展示六图点与几何平均；
- 右：相同 matched-state workload 的 global-load sector-utilization gain `Eff_ge-spmm/Eff_segment`，并叠加 DRAM-byte reduction factor `Bytes_segment/Bytes_ge-spmm`。

**Table 8：** 对 CP/IN/TW 分算法报告 kernel speedup、sector-utilization gain、DRAM-byte reduction、long-scoreboard-stall reduction 和 L2-hit ratio。主文全部使用 matched ratio，原始 counter 进入 artifact/附录。

### 4.3 Evaluation of the Decision Parameter（暂缓）

本轮不执行 threshold/decision-parameter sweep，不将 oracle、regret 或参数敏感性写入 6.4 主文。现有 calibration 结果继续作为运行配置的一部分保留，但本节只评价当前 C9 Hybrid 的实际结果，不把它描述成理论最优 selector。

若后续恢复该实验，必须使用与正式 query 不重叠的 calibration sources 冻结参数，再在 held-out workload 上比较；不能为了保证 Hybrid 胜过两个 fixed modes 而在正式结果上反向选参。

## 5. Section 6.5 — Multi-hop Propagation

本节需要拆开两个机制：

1. **In-place pull**：按处理顺序允许更新在同一轮内继续传播；
2. **预处理图**：重排顶点并加入选定的 multi-hop/forest shortcuts，在线执行前缩短依赖路径。

因此核心实验应采用 factorial design，而不是只画 C9/C0。C9/C0 同时改变图、更新方式和 scheduler，无法给出组件级归因。

### 5.1 Ablation Study on Topology-driven Implementation

建议子标题改为 **Effect of In-place Pull**。

#### 实验目的

验证 in-place pull 在原图和预处理图上是否减少收敛轮次及执行时间，并判断 preprocessing 与 in-place 是否存在协同作用。

#### 对照组

现有结果已经包含两组 2×2 factorial：

**关闭 scheduler：**

| | 同步 | In-place |
|---|---:|---:|
| 原图 | C0 | C4 |
| 预处理图 | C5 | C6 |

**开启完整 scheduler：**

| | 同步 | In-place |
|---|---:|---:|
| 原图 | C3 | C8 |
| 预处理图 | C7 | C9 |

覆盖六图 × 三算法 × 四个 N；主图展示 `N=256`，完整 N scaling 放附录。

#### 数据来源与指标

- **完全可复用**：C0/C4/C5/C6、C3/C8/C7/C9 的 runtime 与 iterations；
- **可选新增 trace**：same-round cascaded updates 数量、单轮推进的依赖距离；
- 指标：end-to-end runtime、throughput、global rounds、runtime per round、可选 edge inspections、正确性 fingerprint。

计算以下匹配效果：

- 原图 in-place：`T(C0)/T(C4)`；
- 预处理图 in-place：`T(C5)/T(C6)`；
- 同步模式下 preprocessing：`T(C0)/T(C5)`；
- in-place 模式下 preprocessing：`T(C4)/T(C6)`；
- 乘法交互项：`T(C4)·T(C5)/(T(C0)·T(C6))`，大于 1 表示组合收益超过两个独立收益的乘积。

再对 C3/C8/C7/C9 重复相同分析，确认结论在完整调度环境中仍成立。

#### 数据展示格式

**Figure 11(a)：2×2 propagation ablation。**

- 第一行：加入 in-place 前后的 runtime speedup，例如 `T(C0)/T(C4)` 和 `T(C5)/T(C6)`；
- 第二行：相同 matched pairs 的 iteration reduction factor，即 `I_before/I_after`；
- 列为原图与预处理图；
- 每算法展示六图几何平均，并叠加六个单图点；
- 旁边增加小面板展示 interaction ratio。

这种布局可以同时展示 in-place、preprocessing 及二者协同，而不是用 C9/C0 混合归因。

#### 解释边界

不能把一次 in-place round 等同于同步 superstep。若轮次下降但每轮工作量上升，结论应以 runtime 与 edge inspections 为主。正文说明处理顺序影响收敛速度，但 monotone fixed point 不变，并由 correctness 验证。

### 5.2 Ablation Study on Preprocessing Method

#### 实验目的

拆分 vertex reordering 与 multi-hop/forest shortcut 的贡献，同时报告离线时间、存储膨胀和收益摊销点。

#### 图变体

现有 raw/preprocessed 端点把多个阶段捆绑在一起。建议增加：

1. **Original**：无重排、无 shortcut；
2. **Reorder-only**：使用生产 mapping 重排，但不增加 shortcut；
3. **Shortcut-only**：在 builder 能保证语义正确时，在原始 ID 顺序上加入相同 shortcut；
4. **Reorder + shortcut**：完整生产预处理图。

如果 shortcut-only 无法在不改变算法的前提下构造，则采用三阶段序列：

```text
Original → Reorder-only → Reorder + shortcut
```

不能把不存在的 shortcut-only 图包装成独立因子。

隔离实验先统一使用同步/FIFO；再只对 Original 和完整预处理图使用 in-place，量化协同。主场景为六图 × 三算法 × `N=256`。

可选敏感性实验：将保留的 shortcut budget 设置为生产配置的约 `{0,25%,50%,100%}`。如果内部 landmark level 不能直接解释为存储成本，则优先用 shortcut ratio，而不是强行展示 L16/L64/L256。

#### 数据来源

- **可直接复用**：C0/C5、C4/C6、C3/C7、C8/C9 的 bundled endpoint；
- **可直接复用**：18 份 manifest 中的总 preprocessing time 与 final graph bytes；
- **必须新增**：reorder-only，以及语义可行时的 shortcut-only 图和运行；
- **新增预处理 breakdown**：landmark/index、reordering、candidate generation、pruning/deduplication、GR conversion；同时记录原始边数、shortcut 数和最终 edge expansion ratio。

#### 数据展示格式

**Figure 11(b)：预处理收益来源。** 所有变体都与 Original 做 matched comparison：展示 `T_original/T_variant` 和 `I_original/I_variant`。每算法报告六图几何平均并叠加单图点；x 轴标签旁标注相对 Original 的 graph-size expansion ratio。

**Table 9：预处理成本与摊销。**

| Dataset | Alg. | Reorder-only speedup | Full-preprocess speedup | Iteration reduction | Edge expansion ratio | Storage expansion ratio | Prep. cost (equivalent raw windows) | Break-even windows |
|---|---|---:|---:|---:|---:|---:|---:|---:|

摊销窗口数：

```text
ceil(preprocessing_time / (raw_window_runtime - preprocessed_window_runtime))
```

其中 `Prep. cost (equivalent raw windows)=preprocessing_time/raw_window_runtime`，以相对 workload 成本代替分钟数。若 break-even 分母不为正，报告 `∞` 或 “not amortized”，不能删除负结果。phase-index 与图转换的绝对时间仍保存在 artifact/附录复现表中，主文只展示相对成本。

#### 与设计示例衔接的真实图 trace

在附录为 CP、IN、TW 各选一个预先固定的 query，展示：

- x 轴：propagation round；
- y 轴：剩余 unsettled vertices 或 active `(vertex,query)` pairs；
- 四条曲线：Original+同步、Original+in-place、完整预处理+同步、完整预处理+in-place。

这组 trace 用来把设计章节的小图示例连接到真实数据；主图仍承担定量结论。

## 6. 主文图表组织建议

| 章节 | 主文产物 | 核心结论 |
|---|---|---|
| 6.3 | Figure 9(a–b) + 一个小汇总表 | batching 与 bounded offset 提高真实共享并转化为端到端收益 |
| 6.4 | Figure 10(a–b) + Table 8 | Hybrid 的端到端表现，以及 GE-SpMM-like pull 的访存合并收益；Push/Pull 适用区间暂缓 |
| 6.5 | Figure 11(a–b) + Table 9 | in-place 与 multi-hop preprocessing 独立减少传播工作，并可能相互增强 |

以下内容放入附录：

- 六图 × 全 N 的 scheduler 明细；
- pull-kernel 的 `Q={8,16,32}` 微基准，以及容量允许时的 Q=64 结果；
- 完整 pull-kernel NCU counter；
- 完整 GPU profiler counter；
- 每张图的 preprocessing phase breakdown；
- representative query 的 convergence traces。

这样可以避免消融章节堆满柱状图，却没有清晰研究问题。

## 7. 建议执行顺序

### Phase A：先复用现有数据

1. 用 C0/C1/C2/C3 生成 batching/offset 的 preliminary matched-pair 图；
2. 用 C0/C4/C5/C6 和 C3/C8/C7/C9 生成 2×2 in-place/preprocessing 图；
3. 汇总已有 preprocessing cost/size 与 calibration 数据。

此阶段不需要重跑 GPU，可以先让我们 review 图形布局与论证方向。

### Phase B：补机制计数器

1. 增加每轮 `E_union`、`E_pair`、active-pair、selected-mode 与时间计数；
2. 增加 query start/completion offset 统计；
3. 增加预处理 phase 与 shortcut 数量统计；
4. 比较 instrumented/non-instrumented runtime；若计数器开销明显，trace run 不参与正式 timing。

### Phase C：执行主要组件隔离实验

1. 固定 Q=32 的 C0/C1 batching 与 C0/C2、C1/C3 offset 对照；
2. C9 下 Push-only/Pull-only/Hybrid 端到端运行；
3. Segment-reduce/GE-SpMM-like pull 对照与定向 NCU profiler；
4. reorder-only/shortcut-only 图变体。

### Phase D：最低优先级可选控制组

1. 在所有主要消融完成后执行 3 个固定种子的 Random batching；
2. 根据 Random 是否提供新的机制信息，决定进入正文、附录或仅保留 artifact；
3. Scalar landmark order 暂不执行，待其定义和研究问题讨论清楚后再决定是否加入。

### Phase E：最终一致性审计

1. 每个图点追溯到 raw command ID 和 correctness 状态；
2. 确认 calibration sources 与 formal queries 不重叠；
3. 统一 Section 4 与真实 scheduler/selector 实现；
4. 保留 regressions 和 counterexamples；
5. 生成 PDF/SVG、输入 CSV、caption 与可复现脚本。

## 8. 等待 review 的关键决策

1. 是否把 “Query Delay” 改成 “Start-offset Alignment”；
2. 6.4.1 的标题是否直接改成 “Push vs. Pull Execution”；正文仍准确说明本实现中 Pull=full/topology scan、Push=frontier/data-driven 的绑定；
3. 是否确认 SSSP/SSWP 继续明确使用 algorithm-independent unweighted-landmark proxy；
4. builder 是否能够构造语义有效的 shortcut-only 图，否则采用三阶段分解；
5. 主文是否能容纳 3 张组合图与 2 个小表；若篇幅不足，Table 8 可移入附录，正文只保留核心数值。
6. Scalar baseline 暂不需要现在决策；后续若恢复，先明确它代表的简化 scheduler 和希望排除的替代解释，再设计实验。
