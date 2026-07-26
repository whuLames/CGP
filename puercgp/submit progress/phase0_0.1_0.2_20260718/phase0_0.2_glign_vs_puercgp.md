# Phase 0.2：Glign 与 PuerCGP batching / offset 对比

## 1. 结论先行

Glign 已经系统提出了以下三层 traversal alignment：

1. **Intra-iteration alignment**：用 query-oblivious frontier 让一个全局迭代只遍历一次 union frontier。
2. **Inter-iteration alignment**：估计各 query heavy iteration 的到达时间，并延迟部分 query 的启动。
3. **Affinity-oriented batching**：在有限 batching window 中，把 heavy phase 到达时间相近的 query 分入同一 batch。

因此，PuerCGP 不能声称首次提出：

- traversal phase alignment；
- sharing-aware batching；
- 通过 BFS distance index 估计 query phase；
- 用 delayed start / fixed offset 对齐 query；
- 在 `V*Q` vertex-major value layout 中联合执行 query。

PuerCGP 当前 online evaluator 的实现细节确实不同于 Glign：它用多个 degree-biased landmarks 构造 source-pair 的 relative-level histogram，再用 greedy grouping、cross-batch swaps 和 multi-start coordinate ascent 共同规划 batch 与 offset。Glign 将每个 source 压缩为一个 `closestHV` 标量，只做排序分组和直接的标量差延迟。

但这一区别目前还不足以独立支撑强 novelty：Puer 的 planner 主要在 all-push BFS benchmark 中验证，在线 score 只是 degree-biased overlap proxy，尚未直接建模完整 GPU push/pull 代价，也未与 Glign closest-HV baseline 在同一执行引擎、同一 source window 下比较。

## 2. 对比范围

### 2.1 Glign 证据

- 论文：*Glign: Taming Misaligned Graph Traversals in Concurrent Graph Processing*, ASPLOS 2023。
- 本地 PDF：`sources/Glign_ASPLOS23.pdf`
- 本地文本：`notes/Glign_ASPLOS23.txt`
- 开源代码：`/home/zyl/Projects/ocgp/baselines/Glign`
- 主要实现：`apps/BFS_Batch.C`、`ligra/ligra.h`

### 2.2 PuerCGP 证据

- online index / planner：`include/puercgp/scheduling/online_offset_evaluator.hxx`
- online benchmark：`examples/bench_online_offset_n128.cu`
- offline trace/oracle benchmark：`examples/bench_phase_schedule.cu`、`examples/bench_phase_schedule_n128.cu`
- 既有结果：`experiments/20260711-152345_phase_schedule_q32/` 及对应 N=128 实验目录和 paper-planning 汇总。

本轮不运行任何实验。下面把“已有代码行为”“已有 preliminary 结果”和“尚待验证的推断”明确分开。

## 3. Glign 的完整设计

### 3.1 它解决的核心问题

Glign 关注 CPU concurrent graph processing 中的 traversal misalignment。多个 source query 即便访问相似图区域，也可能在不同 global iterations 到达这些区域，导致：

- union frontier 变大；
- 同一邻接表不能在同一轮复用；
- cache locality 下降；
- query-oblivious execution 对未真正 active 的 query 做额外计算。

Glign 将每个 query 的局部迭代映射到一个共享 global timeline，并试图让 heavy iterations 在这个时间线上重合。

### 3.2 Intra-iteration：query-oblivious frontier

Glign 不保留每个 `(vertex, query)` 的精确 active flag，而只保留一个长度为 `V` 的 union frontier：

```text
global_frontier(t) = union_i private_frontier_i(t)
```

当顶点 `v` 在 union frontier 中时，Glign 对 batch 内所有 query slots 执行 vertex/edge function，即使 `v` 对某些 query 本轮并不 active。它用额外 value checks 和 monotonic update 保证最终结果正确。

artifact 中的对应实现是：

- `BFSLV_F`：Ligra-C two-level frontier，检查 `CurrActiveArray[v, q]`；
- `BFSLV_SKIP_F`：Glign query-oblivious path，不再检查 per-query active array，对 union frontier 的每条边遍历所有 query slots；
- `Compute_Base_Skipping`：以一个 `frontier[V]` 和 `BFSLV_SKIP_F` 运行。

代码位置：`apps/BFS_Batch.C:27-105`、`apps/BFS_Batch.C:238-292`。

它与 Puer shared frontier 的关键区别是：

- Glign 只知道“这个 vertex 对 batch 中至少一个 query active”；
- Puer `frontier_mask[v]` 精确知道“这个 vertex 对哪些 query active”；
- Glign 主动放弃 query-level frontier precision 以删除 `V*Q` frontier arrays；
- Puer 保留 `V` 个 64-bit mask，以低得多的空间保留 query precision。

### 3.3 Heavy iteration

Glign 把 frontier 较大、访问顶点/边较多的迭代称为 heavy iterations。论文基于 power-law graph 的观察：

- traversal 早期 frontier 快速增长；
- 激活 high-degree vertex 往往标志 heavy phase 开始；
- heavy iterations 占总图访问的大部分；
- heavy frontier 更大，也更可能出现跨 query overlap。

因此，Glign 不尝试预测完整 frontier 序列，而只近似预测“query 何时到达 high-degree region”。

### 3.4 图预处理

Glign 论文的预处理流程为：

1. 按 outdegree 选择 top-K high-degree vertices，默认 `K=4`；
2. 对 directed graph 构造 edge-reversed graph；
3. 从每个 high-degree vertex 在 reversed graph 上运行 BFS；
4. 得到任意 source `s` 到每个 high-degree vertex `h` 的 hop distance；
5. 将它们压缩为：

```text
closestHV[s] = min_h distance(s, h)
```

在 artifact 的 `streamingPreprocessingReturnHops` 中：

- 扫描全图并按 outdegree 排序；
- 选择前 `n_high_deg` 个顶点；
- 调用 `G.transpose()`；
- 批量运行 `Compute_Eval`；
- 再 transpose 回原图；
- 对每个 vertex 只保留所有 hub distance 的最小值；
- 按该标量对当前 query window 排序。

代码位置：`ligra/ligra.h:768-827`。

论文从概念上产生 `K*V` 距离，artifact 在 reduce 后只持续保留 `closestHV[V]`，因此 runtime index 为 `O(V)`。预处理时间在 artifact 中单独输出为 `Profiling cost`。

### 3.5 Inter-iteration：delayed start

对一个已确定的 batch，Glign 计算：

```text
d_i       = closestHV[source_i]
latest    = max_i d_i
delay_i   = latest - d_i
```

距离 high-degree region 更近的 query 延迟更多；距离最远的 query 立即启动。这样所有 query 的估计 heavy phase 在 global timeline 上同时到达。

artifact `bufferStreamingSkipping` 直接执行上述计算，然后调用 `Compute_Delay_Skipping`：

- delay 为 0 的 source 初始进入 union frontier；
- 其余 source 的 value 已初始化，但到指定 global iteration 才加入 frontier；
- query 启动后每轮连续推进；
- 没有中途 pause/resume；
- unreachable source 被映射到 batch 内最大距离，因此 delay 为 0。

代码位置：`ligra/ligra.h:895-950`、`apps/BFS_Batch.C:548-610`。

论文讨论过动态 pause/resume，但明确指出它需要保留 per-query frontier context，会破坏 query-oblivious frontier 的主要优势，因此没有采用。

### 3.6 Affinity-oriented batching

Glign 用 affinity 衡量 concurrent execution 减少了多少独立 frontier work。其 vertex-based 形式可概括为：

```text
Affinity = 1 - sum_t |union_i F_i(t)|
                 / sum_t sum_i |F_i(t)|
```

论文也给出 active-edge based 定义，并称 CPU 实验中二者趋势接近。Glign 的 planner 没有在运行前求解这个真实目标，因为真实 frontier 在执行前未知；它用 `closestHV` 作为 proxy。

batching 算法为：

1. 从 query buffer 中取最早到达的 `Bw` 个 query，形成有限 batching window；
2. 按 `closestHV[source]` 排序；
3. 每连续 `B` 个 query 形成一个 batch；
4. batch 内再用 `latest - closestHV` 生成 fixed delay。

`Bw` 用于避免 affinity 较差的 query 被无限推迟。论文默认 batch size `B=64`，实验总 query 数常为 512。artifact 的 `combination_max` 对应当前处理 window，`bSize` 对应 batch size；预处理先排序整个 window，随后 `bufferStreamingSkipping` 逐 batch 同步执行。

Glign 没有：

- 构造 source-pair affinity matrix；
- 求解 graph partitioning / clustering；
- 在 batch 之间交换 query 做局部搜索；
- 枚举多个 offset candidate；
- 根据已完成 batch 的 runtime feedback 更新后续 batch；
- 联合决定 push/pull operator。

## 4. PuerCGP 当前 sharing-aware 策略

Puer 当前有两套相关机制，必须分开：

1. **offline trace/oracle study**：先运行 query 采集真实 traversal samples，再离线构造 grouping/offset，用于证明 opportunity，不是在线系统。
2. **landmark online evaluator**：图加载前建立可复用 index，运行前只读取 source descriptors 规划 batch/offset；这是当前拟进入系统的机制。

### 4.1 Offline trace / oracle

`bench_phase_schedule*.cu` 的基本流程是：

1. 对固定 source set 跑 baseline traversal；
2. 记录 `(query_id, sampled_vertex, bfs_level, degree)`；
3. 对同一 sampled vertex 在不同 query 中出现的 level difference 建 histogram；
4. 用 degree-weighted overlap 构造 pairwise affinity；
5. greedy 组 batch；
6. 对选定 batch 再根据 trace 做 multi-start coordinate offset search；
7. schedule construction 不计入 GPU execution time。

该方案能够接近“如果已知道真实 traversal phase，可以减少多少 union-edge work”，但不能作为 online system 收益。它与 Glign 的主要区别是利用了实际 traversal trace，而 Glign 只用 source-to-hub distance proxy。

当前保留的主方向是 selective batching + fixed offset。pause/resume 虽然也做过实验，但总体不稳定或回退，不属于当前 online planner 的主要机制。

### 4.2 Online landmark index

`landmark_phase_index::build` 执行：

1. 第一个 phase landmark 选全图最大 degree vertex；
2. 最多选四个 phase landmarks；后续 phase landmark 用 farthest-point 规则，即选择到已选 landmark 集合最近距离最大的 vertex；
3. 对每个 phase landmark 做 BFS，保存到全部 vertex 的 `uint16_t` distance；
4. 剩余 landmarks 通过随机抽取 edge id，再取该 edge 的 source vertex，形成近似 degree-proportional sampling；
5. 对每个 sampled landmark 做 BFS并持久化完整距离数组；
6. phase landmark 的 score weight 为 0，sample landmarks 的 weight 为 1。

对应代码：`online_offset_evaluator.hxx:18-119`。

这形成 `L*V` 的 persistent index，空间约为：

```text
2 * L * V bytes + landmark metadata
```

与 Glign 相比，Puer 保留每个 landmark 的独立距离，而不是 reduce 为每 vertex 一个标量，因此表达能力更强，但 index build time 和空间明显更大。

**Directed graph 风险：** Glign 明确 transpose graph，以获得 `source -> hub` 距离。Puer 当前 builder 直接沿传入 CSR 从 landmark 做 BFS；在当前无向/双向数据上距离对称，因此没有问题，但对一般 directed graph，除非调用者显式传入 reversed CSR，否则该距离不是所需的 `source -> landmark`。这需要在进入论文实验前修正并加正确性测试。

### 4.3 Online phase-length estimate

Puer 对 source 的 scalar phase-length estimate 是：

```text
estimated_length(source)
    = 1 + max distance(source, phase_landmark_j)
```

在当前无向图上成立；代码实际读取 `distance(landmark_j, source)`。该值描述 source 相对几个远处分散 anchor 的跨度，更接近 traversal extent/completion proxy，而不是 Glign 的“到最近 high-degree vertex 的 heavy-arrival time”。

初始 offset 为：

```text
completion_offset_i = max_j estimated_length_j - estimated_length_i
```

即先尝试对齐估计完成时间。代码位置：`online_offset_evaluator.hxx:185-192`、`online_offset_evaluator.hxx:402-421`。

### 4.4 Online pairwise affinity

对 source pair `(a,b)`，Puer 对 degree-biased sampled landmarks 计算：

```text
delta_l = distance(l, b) - distance(l, a)
histogram[delta_l] += weight_l
affinity(a,b) = max_delta histogram[delta]
```

只考虑 `delta in [-max_offset, max_offset]`。直观含义是：如果很多 sampled vertices 对两个 source 的 BFS level difference 相同，那么通过相对延迟这两个 query，它们可能在同一个 global iteration 到达这些 vertices。

代码位置：`online_offset_evaluator.hxx:194-212`。

当前 sampled landmarks 是通过 edge-source sampling 进行 degree-biased 选择，但每个样本的显式 `weight_l=1`。因此准确表述应为“degree-biased approximation of overlap”，而不是“精确 degree-weighted union-edge objective”。phase landmarks 的 weight 为 0，只用于 completion estimate，不参与 pairwise overlap score。

### 4.5 Online selective batching

`plan_batches(sources, batch_size)` 对整个传入 source window 一次性规划：

1. 计算所有 source pairs 的 affinity matrix；
2. 在未分配 query 中选 total affinity 最大者作为 seed；
3. 反复加入对当前 batch affinity sum 最大的 query；
4. 形成所有 batches；
5. 最多做 16 次正收益 cross-batch swap；
6. 返回所有 batch assignments 和 evaluator time。

代码位置：`online_offset_evaluator.hxx:282-399`。

当前 N=128 benchmark 会先为整个 128-query closed window 生成四个 Q=32 batches，然后依次执行这些 batches。它不是“第一个 batch 完成后才构造第二个 batch”，也没有基于第一个 batch 的 runtime trace 调整后续 batch。

当前实现尚未包含：

- 明确的 query arrival timestamp；
- Glign 式 `Bw` 等待上限；
- tail-latency / fairness penalty；
- 动态滑动 window；
- batch 执行反馈。

### 4.6 Online offset search

对已经选好的一个 batch，`evaluate`：

1. 生成 zero-offset start；
2. 生成 completion-aligned start；
3. 用固定 seed 42 生成六组 random starts；
4. 对每组 start 做最多 8 轮 coordinate ascent；
5. 对每个 query 枚举 `[0, max_offset]` 中所有 offset；
6. 最大化其与其他 query 的 pairwise histogram score；
7. score 相同时偏向 completion-aligned offset；
8. 最后归一化，使最小 offset 为 0。

默认 `max_offset=16`。代码位置：`online_offset_evaluator.hxx:402-516`。

与 Glign 一样，它最终产生的是 fixed delayed start：query 一旦启动就连续执行，不在中途暂停。不同之处在于 Glign 的 offset 是一个闭式标量差，Puer 用 pairwise objective 和局部搜索选择 offset vector。

### 4.7 当前 online benchmark 边界

`bench_online_offset_n128.cu` 当前是独立 benchmark wrapper：

- `N=128`、`Q=32`；
- BFS；
- all-push；
- source 预先全部可用；
- batching 与 offset 在 GPU 计时前生成，但 evaluator time 单独加入 end-to-end；
- index build 不计入 query evaluation，只单独报告；
- correctness 主要比较 visited count 与 distance sum checksum；
- 尚未接入 production `frontier_engine` 的 hybrid push/pull path。

因此，当前 planner 是“可执行的 online prototype”，不是已经完成的 production online scheduler。

## 5. 逐项详细对比

### 5.1 预处理与索引

| 维度 | Glign | Puer online evaluator | 影响 |
|---|---|---|---|
| landmark 选择 | top-K outdegree，默认 K=4 | 1 个 max-degree + 最多 3 个 farthest anchors + 多个 degree-biased samples | Puer 覆盖图的多个区域和 pairwise phase，Glign聚焦 heavy-entry |
| BFS 方向 | directed graph 上显式 reversed graph | 直接使用传入 CSR；无向图正确，directed 需调用者反转 | Puer 当前有 directed correctness/semantics 缺口 |
| 每顶点 descriptor | `min_h dist(source,h)` 一个标量 | 对所有 L 个 landmarks 的 distance vector | Puer 表达力高，内存更大 |
| index 空间 | reduce 后 `O(V)` | `O(LV)`，distance 为 `uint16_t` | 大图上 Puer index 可达较大容量 |
| build work | K 次 BFS + degree sort | L 次 BFS + landmark selection | Puer build 明显更重 |
| 持久化 | artifact 当前进程内数组 | 支持 binary save/load | Puer 工程化更完整 |
| directed semantics | 明确 source-to-hub | 当前仅对对称图自然成立 | 必须修复后再声称通用 |

### 5.2 Runtime evaluator

| 维度 | Glign | Puer |
|---|---|---|
| query descriptor lookup | `closestHV[source]` | `L` 个 landmark distances/source |
| pairwise model | 无 | `N*N` affinity matrix |
| batching | scalar sort + consecutive chunks | greedy affinity clustering + swaps |
| offset | `max(closestHV)-closestHV` | multi-start coordinate ascent |
| planning complexity | 主要为 `O(N log N)` | 至少包含 `O(N^2 L)` score 构建和局部搜索 |
| runtime feedback | 无 | 当前也无 |
| planner overhead | 很小，主要是 table lookup/sort | 毫秒级 prototype overhead，必须计入 end-to-end |
| fairness | 有 bounded batching window 概念 | 当前 closed input window，无等待/SLA约束 |

### 5.3 Batching 目标

| 项目 | Glign | Puer offline oracle | Puer online estimator |
|---|---|---|---|
| 理想目标 | 最大化 frontier/edge affinity | 最大化真实 trace 中 degree-weighted overlap | 最大化 sampled landmark 的 pairwise level alignment |
| 可观察数据 | source 的 closest-HV | 已运行 traversal trace | 预建 landmark distance index |
| 是否真正 online | 是，图 profile 后 query lookup/sort | 否 | 是，针对已到达 closed window |
| 是否直接估计 union edges | heuristic，不直接 | 较接近 | 不直接，是 degree-biased proxy |
| 是否考虑 offset | batching 用 scalar proximity；batch 内 direct delay | 是 | pairwise affinity 对 offset bucket 取最大 |
| GPU specificity | 无，CPU locality/LLC | objective 可按 edge work 构造 | 采样偏向高度点，但尚未包含实际 GPU operator cost |

这里最重要的边界是：Puer online objective 目前不是精确的 GPU memory-transaction model。它预测 query phase overlap，期望 overlap 转化为 union-edge reduction。要声称“GPU-specific objective 优于 Glign heuristic”，必须再证明该 score 对 GPU runtime / memory transactions 的预测更准确。

### 5.4 Offset 语义

| 项目 | Glign delayed start | Puer online offset |
|---|---|---|
| 对齐目标 | 最近 high-degree vertex 到达时间 | sampled vertices 的 pairwise relative levels；completion estimate 作为初值/平局项 |
| 解法 | 闭式 `latest-d_i` | bounded combinatorial local search |
| offset 上界 | 由 batch distance range产生 | 默认显式上界 16 |
| 中途暂停 | 否 | 否 |
| query 启动后 | 每轮推进 | 每轮推进 |
| latency 影响 | 等待应计入 query latency | 同样必须计入，当前主实验仍需完整报告 p50/p95/p99 |

### 5.5 Frontier 与共享语义

| 维度 | Glign | Puer push |
|---|---|---|
| union vertex list | 是 | 是 |
| per-vertex query active set | 删除 | 64-bit exact mask |
| 对 union edge 执行哪些 query | 全部 batch slots | mask 中 active slots |
| misalignment 代价 | union edges + 对 inactive query 的额外 update checks | 主要是 union edge 重复读取和较低 mask density |
| pause/resume context | 与 query-oblivious frontier 冲突 | mask 能表达，但实测 carry/extra steps 代价高 |
| state layout | `Levels[v*B+q]` | `values[v*Q+q]` | 两者均为 `V*Q` |

因此，即使二者都最大化 sharing，收益来源并不完全相同。Glign alignment 还在减少 query-oblivious execution 引入的冗余 query work；Puer exact mask 已消除这部分 inactive-query push work，alignment 更直接地服务于 adjacency/union-edge sharing。

### 5.6 算法与硬件范围

| 维度 | Glign | Puer 当前状态 |
|---|---|---|
| 平台 | multicore CPU / Ligra | NVIDIA GPU / CUDA |
| operator | push-style `edgeMap(no_dense)` | shared push + query-lane pull + hybrid |
| 算法 | BFS、SSSP、SSWP、SSNP、Viterbi 等 monotonic functions | production traits 有 BFS、SSSP、WCC；online scheduler目前主要验证 BFS |
| heterogeneous query type | 论文有异构 buffer 实验 | engine 支持 algorithm slots，scheduler 尚未验证异构 batching |
| cache/memory目标 | 减少 LLC miss 和 union accesses | 目标是减少 union edges / random memory transactions；证据尚不完整 |
| push/pull 联合规划 | 无 | 系统有 hybrid，但 online planner 尚未接入 |

## 6. Glign 与 Puer 的共同点

以下重合必须在 related work 中明确承认：

1. 都把 graph traversal 的 iteration phase 当作可调度对象。
2. 都认为 heavy/dense phase 对总访问成本贡献最大，也是 sharing 的主要来源。
3. 都用图预处理得到 source 的 BFS-distance descriptor。
4. 都在一个 query window 内重新分组，而不是完全 FIFO。
5. 都用 fixed delayed start 对齐 query，query 启动后不再 pause。
6. 都使用 vertex-major `V*Q` values 以改善同一 vertex 的 query locality。
7. 都以 union frontier / union graph access 的减少作为性能来源。

## 7. Puer 可以主张的具体差异

以下可以作为“机制差异”，但是否能升级为“论文贡献”取决于实验：

### 7.1 从 scalar phase 到 pairwise phase relation

Glign 每个 source 只有一个 scalar `closestHV`。Puer 保留 source 对多个 sampled landmarks 的 distance vector，并显式估计 source pair 在不同 relative offset 下的 phase relation。它能表达：两个 source 到最近 high-degree vertex 的距离不相近，但在某个 offset 下仍会经过大量相同或相似高度区域。

### 7.2 从 sort 到 affinity optimization

Glign 通过标量排序后连续切 batch。Puer 构造 pairwise affinity matrix，执行 greedy cluster 与 cross-batch swap。它可以优化非传递关系，例如 A 与 B 高 affinity、B 与 C 高 affinity，但 A 与 C 较低的情况；标量排序只能表达一维邻近关系。

### 7.3 从 direct difference 到 bounded offset-vector search

Glign 的 offset 只对齐一个 predicted event。Puer 的 coordinate ascent 同时考虑 batch 中所有 source pairs 和多个 sampled regions，优化整个 offset vector，而非每个 query 独立对齐到一个 hub。

### 7.4 与 GPU exact-mask execution 的潜在结合

Puer planner 的最终目标可以是减少 shared push 的 union outgoing edges，或改变每轮 push/pull 的相对代价；Glign 只优化 CPU query-oblivious push。这是最有潜力的系统差异，但当前代码尚未完成 planner 与 hybrid engine 的联合 cost model，现阶段只能写成设计方向，不能写成已证实贡献。

## 8. Puer 当前弱于或不完整于 Glign 的部分

1. **缺少 bounded online arrival window。** Glign 明确用 `Bw` 限制等待；Puer 当前对预先给定 N-query closed window 全局规划。
2. **缺少 directed preprocessing contract。** Glign 显式 reverse graph；Puer builder 当前未处理。
3. **索引成本更高。** Glign runtime descriptor 为 `O(V)`；Puer 为 `O(LV)`，需要证明额外信息值得。
4. **尚未直接复现 Glign baseline。** 当前没有同一 Puer execution path 下的 closest-HV sort + delayed start 实现与测量。
5. **online evaluator 尚未接入 production hybrid engine。** 当前主要依赖 all-push BFS benchmark wrapper。
6. **算法覆盖不足。** scheduler 收益主要来自 BFS；不能用 engine 支持 SSSP/WCC 代替 scheduler 的多算法验证。
7. **正确性检查偏弱。** aggregate checksum 应升级为完整 per-query value comparison或强 hash。
8. **缺少 latency/SLA 证据。** offset 等待和 batching window 等待都应进入 per-query latency。
9. **score 不是显式 GPU cost。** degree-biased samples 不等价于 memory transactions、cache line requests 或 hybrid operator latency。

## 9. Glign-compatible baseline 的精确定义（后续实现）

原 TODO 要求实现 Glign baseline。本轮按用户要求不写代码、不跑实验，但为下一阶段固定如下规格，避免实现偏差。

### 9.1 Preprocessing

1. 对当前 graph execution direction 计算 outdegree；
2. 选 top-4 outdegree vertices，tie 按 vertex id；
3. 对无向图直接 BFS；对 directed 图在 reversed CSR 上从 hub BFS；
4. 对每个 vertex 保存 `closest_hv = min_h distance_to_h`；
5. 对 unreachable 使用单独 sentinel；
6. index build time、peak memory、persistent bytes 单独记录。

### 9.2 Batching

1. 输入与 Puer planner 完全相同的 arrival-ordered source window；
2. 固定 `Bw=N` 做 closed-window comparison；后续再测试 bounded `Bw`；
3. 按 `closest_hv` 排序，tie 保持 arrival order；
4. 每连续 Q 个 query 组成 batch；
5. 不加入 pairwise swaps，确保是 Glign heuristic 而非混合版本。

### 9.3 Delayed start

1. batch 内 `latest=max(reachable closest_hv)`；
2. reachable query 的 offset 为 `latest-d_i`；
3. unreachable query 的 offset 按 artifact 语义设为 0；
4. query 启动后每轮连续执行；
5. 不加入 pause/resume；
6. offset waiting 计入 per-query latency。

### 9.4 公平计时

至少分别报告：

- index build time 与 bytes；
- runtime planner time；
- GPU execution time；
- planner + GPU end-to-end；
- 若模拟在线 arrival，额外报告 batching wait；
- throughput 与 p50/p95/p99 latency；
- full per-query correctness。

## 10. 后续对比矩阵（本轮不执行）

所有方法使用相同 graph、source window、arrival order、Q、engine 和 correctness checker：

| 方法 | Grouping | Offset | 用途 |
|---|---|---|---|
| FIFO | arrival order | 0 | 基础 baseline |
| Random | fixed seed random | 0 | 排除偶然 source order |
| iBFS GroupBy | degree/common-hub rule | 0 | 早期 GPU sharing heuristic |
| Glign-Batch | closest-HV sort | 0 | 隔离 batching |
| Glign | closest-HV sort | closest-HV difference | 最接近的 alignment baseline |
| Puer-Batch | pairwise greedy + swaps | 0 | 隔离 Puer batching |
| Puer-Online | pairwise greedy + swaps | coordinate offset | 当前 online design |
| Puer-Offline | trace-based grouping | trace-based offset | opportunity upper-bound proxy |
| Per-window oracle | exact/sampled search | exact/sampled search | 小规模上界，仅用于 gap |

必须同时输出：

- 每轮 union frontier vertices；
- 每轮 union outgoing edges；
- private/virtual edge pairs；
- actual sharing ratio；
- push/pull mode；
- kernel time；
- DRAM/L2 transactions；
- planner prediction score；
- query completion step 和 latency。

## 11. Phase 0.2 验收判断

### 11.1 当前 evaluator 是否与 Glign 完全相同？

不是。Glign 是 top-K hub distance 的一维排序和闭式 delay；Puer 是多 landmark pairwise relative-level estimator、greedy clustering、swap refinement 与 bounded coordinate search。算法结构和规划表达能力有实质差异。

### 11.2 当前 evaluator 是否已经证明了 GPU-specific novelty？

没有。它目前预测 sampled phase overlap，并只在 all-push BFS prototype 中给出 preliminary 结果。尚缺 Glign direct baseline、hybrid integration、GPU counter correlation、online arrival/fairness 和多算法验证。

### 11.3 论文中应如何定位？

在现有证据下，batching/offset 应定位为 GPU multi-query plan construction 的一个组件：借鉴 Glign 已提出的 phase alignment 问题，但用 pairwise estimator 适配 exact-mask shared execution，并最终服务于 GPU union-edge / physical-operator cost。只有当同输入实验表明它比 Glign heuristic 更准确、净收益更高且 planner/index 成本可控时，才能把 pairwise online evaluator 提升为独立贡献。

### 11.4 本轮完成到什么程度？

已完成论文机制、artifact 行为、Puer offline/online 实现和创新边界的逐项对比，并固定了后续 Glign-compatible baseline 的实现规格。本轮没有实现 baseline，也没有运行端到端实验，因此原 TODO 的实验验收标准仍未完成。
