# Phase 0.1：iBFS 与 PuerCGP 对比

## 1. 结论先行

PuerCGP 不能把以下内容作为相对 iBFS 的独立创新：

- 把多个 BFS query 合并成一次 GPU traversal；
- 用一个 union frontier 消除重复的顶点和邻接表访问；
- 用每个顶点的 query bitmask 表示多个 BFS 的状态；
- 通过让同一顶点的 query 状态连续存放来改善合并访存；
- 对并发 BFS 的 source 进行 sharing-aware grouping；
- 在 top-down 与 bottom-up 之间切换。

iBFS 已经明确提出并实现了这些思想。PuerCGP 与 iBFS 的主要物理执行差异是：

1. PuerCGP 的 sparse push 使用“精确的本轮 frontier mask + 独立 cumulative visited mask”，而 iBFS 的 BSA 主要依赖累计 visited bitset 及前后状态差生成 frontier。
2. PuerCGP 的 dense pull 把 query 映射到 warp lane，对 `V*Q` value matrix 做逐查询归约；iBFS bottom-up 把线程或 warp 映射到顶点，对该顶点邻居的 128-bit BFS mask 做 OR 归约。两者的并行维度、状态含义和内存流量不同。
3. PuerCGP 通过 `algorithm_traits` 支持 BFS、SSSP、WCC 等 relaxation；iBFS 的 bitwise propagation 是 BFS 特化的可达性传播。
4. PuerCGP 的同质 engine 每轮测量 union-edge、virtual-edge 等 workload，并以 virtual-edge threshold 动态决定 push 或 pull；本地 iBFS artifact 使用命令行给定的固定 `sw_level`，不是在线代价决策。
5. PuerCGP 当前 online batching 使用 pairwise phase-overlap estimator；iBFS GroupBy 是 source 一跳局部规则。

因此，PuerCGP 不是简单的“现代 CUDA 版 iBFS”，但其 shared push 的核心复用思想与 iBFS 高度重合。论文的新颖性必须建立在 GPU 多算法物理算子、query-lane pull、GPU 代价目标和统一动态选择上，而不能建立在 shared frontier 本身。

## 2. 对比范围与证据

### 2.1 论文

- iBFS：*iBFS: Concurrent Breadth-First Search on GPUs*, SIGMOD 2016。
- 本地 PDF：`sources/iBFS_SIGMOD16.pdf`
- 本地可检索文本：`notes/iBFS_SIGMOD16.txt`
- 重点章节：Section 4 Joint Traversal、Section 5 GroupBy、Section 6 Bitwise Operations、Section 8 Evaluation。

### 2.2 iBFS 开源实现

代码根目录：`/home/zyl/Projects/ocgp/baselines/iBFS`

| 文件 | 作用 |
|---|---|
| `graph.cuh` | 图对象、source 重排、GPU BFS 总入口 |
| `bfs_gpu_opt.cuh` | 每轮 expand / inspect / frontier 切换 orchestration |
| `expander.cuh` | top-down 与 bottom-up kernel 及固定 level 切换 |
| `inspector.cuh` | 状态差检测、frontier 分类、scan/gather、distance 输出 |
| `GroupBy.cuh` | source degree / common hub 规则的 artifact 实现 |
| `comm.h` | `uint4` bitmask、固定 block/grid 参数和数据结构 |

### 2.3 PuerCGP 实现

代码根目录：`/home/zyl/Projects/ocgp/puercgp`

| 文件 | 作用 |
|---|---|
| `include/puercgp/kernels/push/shared_push_kernels.hxx` | shared-node push 的 thread / warp / query-parallel 版本 |
| `include/puercgp/kernels/pull/fused_pull_kernels.hxx` | query-lane simple pull 与保留的 smem pull |
| `include/puercgp/core/atomics.hxx` | 原子 mask 更新和 next shared frontier 生成 |
| `include/puercgp/core/layout.hxx` | `V*Q` value layout |
| `include/puercgp/engine/frontier_engine.hxx` | 同质 query 的迭代和动态 push/pull 选择 |
| `include/puercgp/engine/hybrid_engine.hxx` | 异构 algorithm slot 执行 |
| `include/puercgp/scheduling/online_offset_evaluator.hxx` | 当前 online batching / offset evaluator |

## 3. iBFS 的计算模型

### 3.1 Joint Status Array

iBFS 首先把多个独立 BFS 的 per-vertex status 合并为 Joint Status Array（JSA）：

- 逻辑坐标是 `(vertex, bfs_id)`；
- 同一顶点的多个 BFS status 连续存放；
- 对同一顶点并行处理多个 query 时，可以把原先散乱的 status load/store 合并为较少的 memory transactions。

这已经覆盖了“vertex-major / query-contiguous 状态布局改善跨 query 合并访存”的一般思想。PuerCGP 的 `values[vertex * Q + query]` 不能单独作为相对 iBFS 的新点。

### 3.2 Joint Frontier Queue

Joint Frontier Queue（JFQ）是所有并发 BFS 当前 frontier 的顶点并集：

```text
JFQ(k) = union_i F_i(k)
```

每个顶点在 JFQ 中最多出现一次。处理该顶点时，再读取它对应的多查询状态来识别哪些 BFS 需要传播。这样可使多个 query 对同一个 frontier vertex 的邻接表读取合并为一次。

iBFS 论文强调 JFQ 的最大长度是 `|V|`，而不是所有 private frontier 长度之和。JFQ 是 PuerCGP `shared_frontier_vertices + frontier_mask` 最直接的相关工作。

### 3.3 Top-down joint traversal

iBFS top-down 的逻辑是：

1. 从 JFQ 取出共享 frontier vertex `f`；
2. 读取 `f` 的多 BFS status；
3. 只加载一次 `f` 的 adjacency list；
4. 把 `f` 对应的 BFS bits OR 到每个邻居的 status；
5. 通过本轮前后状态差识别下一层新 frontier。

bitwise 版本以 `uint4` 表示 128 个 BFS。对每条边需要对四个 32-bit word 执行条件检查和 `atomicOr`。开源 kernel 按 source degree 分工：

- `td_expand_thd`：低度 frontier，由一个 thread 处理；
- `td_expand_warp`：中度 frontier，由一个 warp 处理；
- `td_expand_cta`：高度 frontier，由一个 CTA 处理；
- 三类 queue 通过三个 CUDA stream 发射。

对应代码：`expander.cuh:140`、`expander.cuh:208`、`expander.cuh:276`、`expander.cuh:505-550`。

### 3.4 Bottom-up joint traversal

iBFS bottom-up 不从“当前已访问 frontier”向外扩展，而是扫描仍可能未完成的顶点：

1. candidate JFQ 包含至少对一个 BFS 尚未访问的顶点；
2. thread 或 warp 负责一个 candidate vertex；
3. 扫描其邻居，读取邻居的 cumulative visited bitmask；
4. 对邻居 bitmask 做 OR，得到当前顶点在本轮可被哪些 BFS 访问；
5. 写回该顶点的 bitmask。

论文中的 early termination 条件是当前 vertex 的所有 BFS bits 均已置 1，此后继续检查邻居不会再增加状态，因此可以停止。这里的 early termination 不是“某个 query 找到第一个 parent 就停止”，而是 bitmask 已覆盖 batch 中全部 query 后停止。

必须注意论文和 artifact 的差异：本地 `expander.cuh:341-486` 中 `sw_expand_warp` 与 `sw_expand_thd` 的 all-ones break 均被注释。因此：

- 论文算法支持 all-query early termination；
- 当前本地开源代码没有实际启用这个 break；
- 后续实验需要分别报告“原 artifact”和“恢复论文 early termination”的结果，不能默认二者等价。

### 3.5 Bitwise Status Array

iBFS 的 Bitwise Status Array（BSA）用一 bit 表示一个 `(vertex, BFS)` 的 visited 状态：

- 本地代码 `comp_t = uint4`，每顶点 128 bits；
- BSA 记录累计 visited，而不是只记录本层 frontier；
- top-down 通过 `atomicOr(BSA_next[neighbor], BSA_current[source])` 传播；
- top-down 的新 frontier 可由前后 BSA 差异识别；
- bottom-up 通过 OR 邻居 BSA 形成当前顶点的新状态；
- bottom-up candidate 可由 `~BSA[v] != 0` 判断。

BSA 的收益是把 128 个布尔状态压缩到 16 bytes，并把跨 query 的布尔传播变成少量整数 OR。其限制是状态转移必须能表示为 bitwise reachability，天然适合 BFS，但不能直接表达 weighted min-relaxation。

### 3.6 GroupBy

iBFS 用 Sharing Degree 描述联合 frontier 的复用程度，本质形式为：

```text
sharing degree = sum_i |F_i| / |union_i F_i|
```

论文观察到前若干层的 sharing 可预测整个 traversal 的 sharing 趋势，因此提出两条 source outdegree 规则：

1. 两个 source 的 outdegree 都小于阈值 `p`；
2. 两个 source 可到达同一个 outdegree 大于阈值 `q` 的顶点。

论文默认 `q=128`，并从 `p in {4, 16, 64, 128}` 中选择。意图是把会较早汇入同一个 hub、从而产生相似 frontier 的 BFS 放进同一组。

开源 `GroupBy.cuh` 是上述规则的简化实现：

- 固定要求 source degree `<= 128`，没有运行时搜索多个 `p`；
- 只检查 source 的一跳邻居；
- 找到第一个 degree `> 128` 的邻居后立即停止；
- 以该第一个 hub 为 key 对 source 排列；
- 没有构造任意 source pair 的 affinity matrix；
- 没有全局组合优化、batch swap 或 offset 优化。

对应代码：`GroupBy.cuh:19-36` 和 `GroupBy.cuh:48-83`。

### 3.7 Direction switching

iBFS 论文采用 top-down / bottom-up 组合，但本地 artifact 的切换不是在线 workload predictor：

- `sw_level` 从命令行读入；
- `level <= sw_level` 使用 top-down；
- 随后若干 level 用 transition path 建立 bottom-up candidate queues；
- 后续持续使用 bottom-up；
- 没有根据当轮 frontier edges 动态切换，也没有 pull-to-push 切回。

对应代码：`ibfs.cu:67-82`、`expander.cuh:691-711`、`bfs_gpu_opt.cuh:11-35`。

## 4. PuerCGP 的对应机制

### 4.1 状态布局

`include/puercgp/core/layout.hxx:10` 定义：

```text
value_index(vertex, query, Q) = vertex * Q + query
```

即 value matrix 为 `V*Q`，同一顶点的多个 query value 连续。另有：

- `frontier_mask[V]`：每个顶点本轮真正 active 的 query bits；
- `visited_mask[V]`：BFS 等策略的累计访问状态；
- `frontier_vertices[V]`：只存 union frontier 中的唯一顶点；
- `active_slots`：本次 launch 允许执行的 query slots。

### 4.2 Shared push

PuerCGP 的默认 push 路径是 `expand_shared_node_warp_kernel`：

1. 一个 warp 负责一个 `frontier_vertices[i]`；
2. 读取 `frontier_mask[source] & active_slots` 得到精确 active query set；
3. warp lanes 对 source 的 outgoing edges 分工；
4. BFS path 对邻居的 `visited_mask` 执行 64-bit atomic OR；
5. `improved = active_mask & ~old_visited` 得到首次访问该邻居的 query bits；
6. 写入这些 query 的 value；
7. `mark_next_shared_frontier` 在 kernel 内原子合并 `next_frontier_mask`，并只让第一个写入者把 neighbor append 到 `next_frontier_vertices`。

对应代码：

- thread version：`shared_push_kernels.hxx:18-101`
- query-parallel experimental version：`shared_push_kernels.hxx:103-192`
- default warp version：`shared_push_kernels.hxx:194-282`
- next frontier primitive：`core/atomics.hxx:22-44`

### 4.3 Query-lane pull

`fused_pull_simple_kernel` 的 block 是：

```text
dim3(query_count, tile_row), tile_row = 128 / query_count
```

在 `Q=32` 时：

- block 为 `(32, 4)`，共 128 threads；
- 每个 warp 对应一个 vertex；
- `threadIdx.x` 是 query id；
- warp 中每个 lane 负责同一个 vertex 的一个 query；
- 每个 lane 扫描该 vertex 的全部 incoming neighbors；
- 对固定 neighbor，32 lanes 访问 `values[neighbor, 0..31]`，形成 query 维连续访问。

kernel 随后把每个 query 的 update bit 写入 shared memory，由 `threadIdx.x == 0` 的 lane 合并为 `improved_mask`，输出 `next_frontier_mask`、`unique_flags` 和 `pair_counts`。代码见 `fused_pull_kernels.hxx:18-81`。

当前 launcher 对所有 `Q <= 64` 默认选择 simple kernel；smem tiled kernel 保留用于实验但未进入默认路径。代码见 `fused_pull_kernels.hxx:193-216`。

## 5. Shared push 与 iBFS top-down/JFQ 对比

| 维度 | iBFS | PuerCGP | 判断 |
|---|---|---|---|
| union frontier | JFQ，每个顶点一次 | `frontier_vertices`，每个顶点一次 | 核心思想相同 |
| query membership | 顶点的 BSA/JSA 状态 | 精确 `frontier_mask[vertex]` | Puer 状态语义更明确 |
| visited 状态 | cumulative BSA | 独立 `visited_mask` | 逻辑不同 |
| bit width | `uint4`, 128 BFS | `uint64_t`, 最多 64 slots | 工程取舍不同，不构成算法创新 |
| 邻接复用 | 一个 JFQ vertex 的邻接表加载一次 | 一个 shared frontier vertex 的邻接表由一个 warp/CTA协作 | 相同目标 |
| BFS 更新 | 四次 32-bit `atomicOr` | 一次 64-bit mask atomic OR | Puer 原子指令数可能更低，但 batch 宽度也减半 |
| next frontier | inspect + count/scan/gather；按 degree 分三类 queue | push kernel 内 atomic append 到一个 queue | 生成路径明显不同 |
| degree 调度 | thread / warp / CTA 三类，三 stream | 当前默认单一 warp mapping | iBFS 对 degree skew 的处理更完整 |
| query work | mask-wide BFS propagation | BFS mask-wide；其他算法逐 bit relaxation | Puer 更通用，但非 BFS 代价更高 |
| source grouping | 一跳 degree/hub GroupBy | pairwise phase estimator + greedy grouping | Puer scheduler 更细，目标仍需实验证明 |

### 5.1 相同点

两者最重要的共同点是：union frontier vertex 只读取一次邻接表，再把这次 edge traversal 的结果应用到多个 query。该思想正是 iBFS joint traversal 的中心，因此 Puer shared push 不能被描述为全新的多查询执行模式。

### 5.2 实质差异

**精确 frontier 与累计状态分离。** Puer 的 `frontier_mask` 只包含本轮需要扩展的 query；`visited_mask` 只负责首次访问判定。iBFS BSA 主要保存累计状态，并通过前后 BSA 差异恢复新增状态。这会影响重复传播、frontier 生成和每轮扫描成本。

**frontier 生成位置。** Puer push 在发现 neighbor 首次变 active 时直接通过 atomic append 生成下一轮 union frontier。iBFS 通过 inspector 在顶点状态数组上识别变化，然后 count/scan/gather，并进一步按 degree 分类。这是“原子竞争与全图/候选扫描”之间的物理设计取舍。

**degree imbalance。** iBFS 根据 frontier degree 使用 thread、warp、CTA 三种粒度。Puer 当前 production push 默认 warp-per-frontier，在低度顶点多时可能浪费 lanes，在超高度顶点上也缺少 CTA 级集中协作。这一项目前 iBFS 反而更完整。

**算法范围。** Puer 对 BFS 做了 mask-wide 特判，对 SSSP/WCC 通过 policy relaxation 逐 active bit 更新。iBFS 的 BSA OR 无法直接表达带权或任意 value 类型，因此是 BFS-specific。

## 6. Puer pull 与 iBFS bottom-up 对比

### 6.1 线程映射

| 项目 | iBFS bottom-up | Puer `fused_pull_simple_kernel` |
|---|---|---|
| 主要并行单位 | 一个 thread 或 warp负责一个 candidate vertex | query 映射到 `threadIdx.x`；Q=32 时一 warp/vertex，Q=16 时一 warp含两个半 warp vertex rows，Q=64 时两 warps/vertex |
| 邻居并行 | thread 串行扫描，或 warp lanes 分摊邻居 | 每个 query lane 各自扫描全部邻居 |
| query 并行 | query 压缩在 `uint4` 的 bitwise OR 中 | query 显式映射到 lanes |
| 归约方向 | 对邻居的 128-bit visited mask 做 OR | 每个 query 对邻居 value 做 `relax + min` |
| 工作共享 | adjacency load 在 warp-neighbor mapping 中共享 | simple 版本会由 query lanes 重复发出 adjacency load，主要依赖 cache/broadcast；value load 在 query 维连续 |

iBFS 让“邻居”占主要并行维度，query 被压缩到寄存器 bitmask；Puer 让“query”占 `x` 维，邻居循环留在线程内部。在本文当前重点 `Q=32` 下，这恰好形成一 warp/vertex、每 lane/query；`Q=16/64` 的 warp-to-vertex 比例分别变成 `1:2` 和 `2:1`。它们不是同一个 kernel 换了数据类型。

### 6.2 状态布局与语义

| 项目 | iBFS | Puer |
|---|---|---|
| 传播状态 | 1 bit/query，表示累计 visited | `value_type/query`，表示 BFS level、distance、component 等 |
| 主状态大小 | 16 B/vertex 对应 128 BFS | `sizeof(value_type) * Q`/vertex，另加 masks |
| query 连续性 | BSA bit-packed；JSA 中 query state 连续 | value matrix 明确 `V*Q` 连续 |
| frontier 语义 | bottom-up candidate 是“至少一个 query 未访问” | 默认 simple pull 扫描全部 V，再由 `should_update` 过滤 |
| 输出 | 更新 cumulative BSA，再由 inspector 形成后续集合 | 直接写 value、next mask、flag、pair count，再 scan/compact |

Puer 的通用 value matrix 明显更大，但允许统一支持 min-relaxation。iBFS 的 bitset 极紧凑，但只携带 reachability，不携带每个 query 的完整数值状态；distance 另写到 `depth_merge[V*Q]`。

### 6.3 提前终止

iBFS 论文的 early termination 是：当前顶点的 128 bits 全为 1 后停止扫描邻居。本地 artifact 中该 break 被注释。

Puer simple pull 当前没有对应的 BFS parent early termination：

- 每个 `(vertex, query)` lane 扫描全部 incoming neighbors；
- 计算所有可达邻居候选的最小值；
- 即使 BFS 已找到一个能够产生 `level+1` 的 parent，也不会提前 break；
- 对 SSSP/WCC，通常也不能在未证明下界的情况下提前停止。

因此，Puer 的 pull 通用性更高，但 BFS 特化优化尚不完整。后续若实现 BFS-only early exit，应单独评估 warp divergence、邻居顺序和 active-slot 密度对收益的影响。

### 6.4 内存 transaction 预期

**iBFS bottom-up：**

- adjacency：warp 版本由 lanes 连续读取邻居，容易形成合并访问；
- state：不同 lane 访问不同 neighbor 的 `uint4`，地址取决于图邻接，属于随机 load，但每次获得 128 query 状态；
- write：每 candidate vertex 写一个 `uint4`；
- 优点是 bit-packed 状态流量很小；缺点是只能做 BFS bitwise propagation。

**Puer pull：**

- adjacency：同一个 warp 的 32 query lanes 在同一循环位置读取同一个 neighbor，硬件可能 broadcast/cache 命中，但指令仍由各 lane 执行；
- value：固定 neighbor 时访问 `values[neighbor, q]`，是连续的 32 个 value，能够合并；
- write：更新时同一 vertex 的 query values 连续写入；
- 代价是为每个 edge-query pair 执行 value load 和 relaxation，work 通常远大于 iBFS 的 bitwise OR。

因此，Puer pull 的卖点不能写成“比 iBFS 少做工作”，而应写成“接受更多逻辑工作，用 query-lane mapping 把随机 per-query value access 转化为可合并 transaction，并为非 bitwise 算法提供统一 dense operator”。这一硬件效果仍需要 transaction、sector、stall 和 instruction count 实测支持。

## 7. Selective batching 与 iBFS GroupBy 对比

### 7.1 当前 Puer online batching

`online_offset_evaluator.hxx` 当前执行：

1. 建立 landmark distance index；
2. 对每对 source 构造 sampled landmark 上的 relative-level histogram；
3. 取最大 offset bucket 作为 pairwise alignment affinity；
4. 从未分配 query 中选择 total affinity 最大者作为 batch seed；
5. 逐个加入对当前 batch affinity sum 最大的 query；
6. 执行最多 16 次正收益 cross-batch swap refinement。

这与 iBFS GroupBy 的差异不是“是否考虑 sharing”，而是 sharing estimator 的表达能力：

| 维度 | iBFS GroupBy | Puer online batching |
|---|---|---|
| 观察范围 | source 及一跳邻居 | source 到多个 landmark/sample 的 BFS distance |
| 描述符 | source degree、首个高 degree neighbor | pairwise relative-level histogram |
| affinity | 同 hub 的离散规则 | 每个 source pair 的连续 score |
| offset awareness | 无 | pairwise score 对多个相对 offset 取最大值 |
| 组 batch | 以 hub key 排列 | greedy construction + swaps |
| 复杂度 | 接近 source 邻接检查 | index build `O(L(V+E))`；planning 约 `O(N^2 L)` |
| 存储 | `O(V)` 临时 hub map | 持久化 `O(LV)` distance index |
| 公平性/window | 输入 source list 上重排 | 当前 benchmark 对给定 N-query closed window 全局规划；尚无在线等待上限模型 |

### 7.2 预期差异

Puer estimator 能识别“source 不共享一跳 hub，但若干迭代后在某个 offset 下经过相似区域”的 query pair；iBFS GroupBy 识别不了这类关系。代价是 Puer 需要较大的离线索引和二次 pairwise planning。

iBFS GroupBy 在 source 邻近 hub 的图上可能以极低开销得到足够好的 batch。Puer 必须通过同窗口、同 sources 的直接比较证明更复杂 estimator 的净收益，否则容易被评价为以高预处理代价替换一个简单有效的 rule。

## 8. Direction switching 对比

| 项目 | iBFS artifact | Puer 同质 `frontier_engine` | Puer 异构 `hybrid_engine` |
|---|---|---|---|
| 决策粒度 | 运行前给定 level | 每个 iteration | 每个 iteration |
| 指标 | `sw_level` | 同时测量 actual/virtual edges，选择条件使用 virtual query-edge work | 当前主要使用 union frontier ratio |
| 默认阈值 | 用户输入 | `virtual_edges >= 0.20 * Q * |E|` | `unique_frontier >= 0.15 * |V|` |
| 是否可切回 | artifact 中无 | 可逐轮重新判断 | 可逐轮重新判断 |
| query 粒度 | 整个 group | 当前整个 batch 同一方向 | 当前整个 active batch 同一方向 |

同质 Puer engine 的判断见 `frontier_engine.hxx:273-328`；阈值定义见 `core/types.hxx:28-29`。异构 engine 当前规则更粗，不能把同质 engine 的 virtual-edge cost model 无条件推广到异构模式。

## 9. 汇总表：相同点、差异、硬件影响与验证

| 机制 | 相同点 | 关键差异 | 预期硬件影响 | 后续必须验证 |
|---|---|---|---|---|
| union frontier | 都消除重复 frontier vertex | Puer exact per-level mask；iBFS cumulative BSA + inspect | 都减少 adjacency loads；Puer inline atomic append 可能省 scan，但有竞争 | union vertices/edges、atomic contention、scan time |
| bitmask | 都用 bit 操作聚合 query | iBFS 128-bit BFS-only；Puer 64-bit mask + full values | iBFS 状态流量更低；Puer 单次 atomic word 更少 | bytes/edge、atomic transactions、instruction count |
| push mapping | 都按共享 vertex 扩展 | iBFS degree triage；Puer 默认 warp-only | iBFS 对 degree skew 可能更稳；Puer launch 路径更简单 | 按 degree bucket 的 kernel time 与 lane efficiency |
| bottom-up/pull | 都检查 incoming neighbors | iBFS query bit-packed；Puer query-lane value reduction | iBFS 少 bytes/ops；Puer value loads 在 query 维合并 | DRAM sectors、L2 hit、long scoreboard、edges-query/s |
| early termination | 都存在理论可能 | iBFS 论文支持但 artifact 注释；Puer 未实现 | BFS 可能减少邻居扫描，也可能导致 divergence | 原 artifact、恢复 iBFS break、Puer BFS break 三方对照 |
| state layout | 同顶点多 query 相邻 | iBFS BSA/JSA；Puer typed `V*Q` | 都利于 query dimension coalescing；容量不同 | transaction/request、working-set、L2 miss |
| direction selection | 都支持 push/top-down 与 pull/bottom-up | iBFS fixed level；Puer dynamic workload threshold | Puer 可适应 source/graph phase，增加统计和切换开销 | per-step oracle、fixed-level、Puer policy 对照 |
| grouping | 都试图提高 sharing | iBFS 一跳 hub rule；Puer pairwise phase estimator | Puer 可能降低更多 union edges，但 CPU/index 成本高 | 同 source window 的 iBFS rule、Puer、offline oracle |
| algorithm support | 联合执行同构 queries | iBFS BFS-only；Puer traits 支持多算法 | Puer 更通用，但非 BFS 无法 bitwise 压缩 | BFS、SSSP、WCC 分别测，不用 BFS 结果替代 |

## 10. 后续实验规格（本轮不执行）

### 10.1 公平的端到端对比

固定以下条件：

- 同一张图、同一 directed/undirected 语义；
- 完全相同的 source list 和 source order；
- 相同 batch size，至少覆盖 `Q=32/64/128` 中双方可运行的交集；
- 相同 warm-up、重复次数和计时边界；
- 单 GPU；
- 除 throughput 外报告完整结果一致性，而不是只比较 aggregate checksum。

基线至少包括：

1. iBFS 原 artifact；
2. iBFS 恢复论文 all-ones early termination；
3. Puer all-push；
4. Puer all-pull；
5. Puer hybrid；
6. Puer shared push + iBFS GroupBy；
7. Puer shared push + current selective batching。

由于 iBFS 固定 128-bit 而 Puer 最大 64 slots，不能直接把 `Q=128` 的单次 iBFS 与两次 `Q=64` Puer 只比较 kernel latency。应同时报告 total query throughput、每 query latency 和实际处理轮数，并明确 batch-width 差异。

### 10.2 Kernel-level profile

建议采集：

- `dram__sectors_read.sum` / `dram__sectors_write.sum`；
- L1/L2 hit rate；
- global load/store transactions per request；
- `smsp__warp_issue_stalled_long_scoreboard`；
- active warps、eligible warps、achieved occupancy；
- integer/atomic instruction count；
- branch divergence；
- adjacency bytes、value/status bytes、next-frontier bytes；
- union vertices、union edges、virtual query-edge pairs。

### 10.3 机制消融

- Puer exact `frontier_mask` 对比从 cumulative visited 差生成 frontier；
- inline atomic frontier append 对比 count/scan/gather；
- warp-only push 对比 iBFS degree triage；
- Puer pull query-lane mapping 对比 bitmask-neighbor mapping；
- Puer BFS pull full scan 对比 parent/all-mask early exit；
- fixed `sw_level`、Puer threshold、离线 per-step oracle；
- random batching、iBFS GroupBy、Puer online batching、offline trace oracle。

## 11. 对 Phase 0.1 验收问题的回答

### 为什么 PuerCGP 的物理执行不是“对 iBFS 的现代化重写”？

PuerCGP 的 dense operator 不是 iBFS bitwise bottom-up：它把 query 映射到 warp lane，在 `V*Q` typed value matrix 上执行算法定义的 relaxation，以增加 edge-query 计算量换取 query 维合并访存；iBFS 则把 128 个 BFS 压缩进 `uint4`，以顶点/邻居为并行维度执行 BFS-specific OR propagation。Puer 还把 sparse push、dense pull、algorithm traits 和逐轮 workload choice 放在统一 engine 中，而 iBFS artifact 是 BFS-specific joint traversal 加固定 level 切换。

### 需要主动承认什么？

Puer shared push 的基本结构，即 union frontier、一次 adjacency traversal 服务多个 BFS、bitmask 表示 query membership，与 iBFS 的 JFQ/BSA 属于同一思想族。论文应把 iBFS 作为最接近的 GPU baseline，并将贡献限定为不同的 physical operator、通用状态语义、GPU cost model 与在线规划，而不是重新命名 shared frontier。

### 当前证据是否足以证明差异带来收益？

不足。本轮完成的是逻辑与代码边界核查。query-lane pull 是否减少实际 memory transaction、dynamic choice 是否优于 fixed switch、pairwise batching 是否值得其索引/planner 成本，都需要上述同输入实验和硬件计数器验证。
