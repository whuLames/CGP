# Hybrid Batch 已完成工作

> **最后更新：2026-07-06**
>
> 本文档汇总 `puercgp` 当前 hybrid heterogeneous batch 的已完成实现。
> 早期 Step 1-7 对应 `/home/zyl/.claude/plans/sunny-dancing-fog.md`；
> 后续 pull、warp push、真实图验证、benchmark 和 replenishment 是在该基础上的增量。

## 总体目标

为 `puercgp` 实现 **heterogeneous query batch processing**：一个 batch 内同时执行
BFS、SSSP、WCC 等不同算法类型的 query，而不是把不同算法拆成多个同质 batch 串行运行。

核心障碍是旧的 `frontier_engine.hxx` 以 `template <typename Policy>` 单态贯穿，
`Policy::value_type` 在 kernel 签名中固化，导致 BFS、SSSP、WCC 无法自然混入同一个
batch。本实现把同质路径保留为回归保护，同时新建运行期 per-query algo tag 的异构路径。

当前 hybrid batch 已支持：

- 同一 `hybrid_query_batch` 内混合 BFS / SSSP / WCC。
- `run_heterogeneous(graph, batch, context, options)` 一次执行异构 batch。
- `traversal_mode = push / pull / hybrid`。
- `push_strategy = shared_node / shared_node_warp`。
- 最多 64 个 query slot（`query_mask_t = uint64_t`）。

## 已完成交付

### Step 1 - 基础类型

文件：`include/puercgp/algorithms/hybrid.hxx`

- `using unified_value_t = float`：统一表示 BFS level、SSSP distance、WCC label。
- `enum class algo_kind_t : uint8_t { bfs, sssp, wcc }`。
- `unified_infinity()` / `unified_source_value()` device helper。
- `static_assert(sizeof(unified_value_t)==4)`，支撑 float atomic CAS 路径。

### Step 2 - Device Reduce 原语

文件：`include/puercgp/core/reduce_ops.hxx`

| 函数 | 语义 |
|------|------|
| `apply_first_write(slot, cand)` | BFS first-write CAS |
| `apply_min_reduce(slot, cand)` | SSSP/WCC min-reduce CAS |
| `compute_candidate(kind, src, w, level)` | push 模式 candidate |
| `compute_candidate_pull(kind, nb, w)` | pull 模式 candidate |

设计要点：BFS 在 push 下继续保留 visited-mask 批量 first-write；SSSP/WCC 统一走
min-reduce。WCC 与 SSSP 的差异主要在 init：WCC 是 per-vertex label 初始化。

### Step 3 - Query 描述符

文件：`include/puercgp/core/query_descriptor.hxx`

- `query_descriptor_t { source, kind, source_value }`。
- `hybrid_query_batch` host 容器和 device upload 缓存。
- `bfs_slot_mask()` / `nonbfs_slot_mask()` 在 host 端预算 Q-bit mask，避免 kernel inner loop 查表。
- `has_bfs()` / `has_wcc()` 支撑 init 分支。
- `update_slot()` 支撑 replenishment slot 复用时替换 descriptor。
- `validate()` 检查非空、`max_queries`、`Q<=64`、source 非负。

### Step 4 - 异构 Init Kernels

文件：`include/puercgp/engine/hybrid_engine.hxx`

- `fill_unified_kernel`：填充 `unified_infinity()`。
- `init_hybrid_sources_kernel`：初始化 BFS/SSSP source；BFS 同时设置 visited bit。
- `init_wcc_all_vertices_kernel`：WCC label 初始化为 vertex id，并让全顶点入 frontier。

### Step 5 - Heterogeneous Shared-Node Push

文件：`include/puercgp/engine/hybrid_engine.hxx`

`expand_shared_node_hybrid_kernel` 已实现双路 update：

- BFS slot：`active_mask & bfs_slot_mask` 通过 `atomic_or_query_mask(visited+nbr, ...)`
  批量判重，保留 BFS batch 的 bit-mask 优化。
- SSSP/WCC slot：遍历 `active_mask & nonbfs_slot_mask` 的 set bits，逐 slot 调
  `apply_min_reduce()`。
- 所有改进合并为 `improved` mask，并写入下一轮 shared frontier。

### Step 6 - `run_heterogeneous` 主循环

文件：`include/puercgp/engine/hybrid_engine.hxx`

`hybrid_frontier_engine::run()` 已接入完整异构主循环：

- init values / masks / frontier。
- 每轮根据 `run_options.traversal_mode` 选择 push、pull 或 hybrid 切换。
- 维护 `active_union`，记录每轮产生 frontier 写入的 slot mask。
- 收集 `run_result_t<int, unified_value_t>`。

顶层入口：

```cpp
auto result = puercgp::run_heterogeneous(graph, batch, context, options);
```

### Step 7 - WCC 同质基线

文件：`include/puercgp/algorithms/wcc.hxx`、`include/puercgp/engine/frontier_engine.hxx`

- `wcc_policy` 使用 `float` label，relax 语义为 identity，update 语义为 min label。
- 同质 `frontier_engine` 增加 WCC init 分支，保留 BFS/SSSP 回归路径。
- `examples/validate_wcc.cu` 覆盖 WCC 同质 correctness。

### Step 8 - Hybrid Pull 接入

文件：`include/puercgp/engine/hybrid_engine.hxx`

早期文档中 pull 只实现 kernel、未接入主循环；当前状态已完成接入。

- `fused_pull_hybrid_simple_kernel`：处理 `Q<=32`。
- `fused_pull_hybrid_smem_kernel`：处理 `33<=Q<=64`。
- `launch_fused_pull_hybrid` 统一分派 simple/smem。
- 主循环 pull 路径执行：
  - fused pull kernel 扫全图，产生 `next_frontier_mask`、`unique_flags`、`pair_counts`。
  - `thrust::inclusive_scan` 聚合 `unique_flags`。
  - 复用 `detail::compact_shared_pull_frontier_kernel` 重建 shared frontier list。

当前 hybrid 切 pull 的主要判据是 `current_unique >= pull_frontier_ratio * V`；
`pull_edge_ratio` 已记录在 profile 中，但 hybrid 分派还没有完整使用 edge work 估计。

### Step 9 - Warp Hybrid Push

文件：`include/puercgp/engine/hybrid_engine.hxx`

已实现 `expand_shared_node_warp_hybrid_kernel`，并在 `hybrid_frontier_engine::run()` 中通过
`push_strategy_t::shared_node_warp` 分派。

该路径每个 warp 处理一个 frontier vertex，update 段沿用 BFS 批量 first-write +
SSSP/WCC min-reduce 的双路设计。

### Step 10 - 真实图验证与 Hybrid Benchmark 工具

文件：

- `examples/validate_hybrid_real.cu`
- `examples/bench_hybrid.cu`
- `examples/mtx_loader.hxx`

`validate_hybrid_real` 支持：

- CSR bin 目录或 Matrix Market `.mtx`。
- CLI 指定 BFS sources、SSSP sources、WCC count。
- `mode=push|pull|hybrid` 和 `push_strategy=shared_node|shared_node_warp`。
- CPU reference：BFS、SSSP、WCC label propagation。
- 结果比对：BFS 取整、SSSP `1e-3` epsilon、WCC exact label。

`bench_hybrid` 支持三组对比：

- sequential：同质 `run<bfs_policy>` / `run<sssp_policy>` / `run<wcc_policy>` 串行。
- hybrid：`run_heterogeneous` 一次混合 batch。
- ideal upper：同质 BFS batch 作为共享遍历上界参考。

### Step 11 - Replenishment 实验路径

文件：

- `include/puercgp/engine/replenish_engine.hxx`
- `examples/smoke_replenish.cu`
- `examples/validate_replenish.cu`
- `examples/bench_replenish.cu`
- `experiments/replenish_*.md`

已实现 `run_replenish_pipeline(graph, all_queries, context, options)`：

- 一次提交 `N` 个 query，内部最多使用 64 个 active slot。
- 收敛 slot 通过 `active_union` 检测。
- 回收 slot 前 snapshot values 到最终 row-major buffer。
- 从 pending queue 注入新 BFS/SSSP query。
- WCC 必须进入首批 slot，不能进入 pending，避免 label 污染。
- 支持 `discard_results`，用于 latency/throughput benchmark 降低显存压力。

实验结论记录在 `experiments/`：

- 标准 BFS/SSSP workload 下，replenishment throughput 全面慢于 sequential 分批。
- 长尾 WCC 构造下，replenishment 仍未翻盘。
- 当前判断：replenishment 是已实现的 experimental path，但不适合作为主 contribution。

## 当前文件清单

### 核心 headers

| 路径 | 内容 |
|------|------|
| `include/puercgp/algorithms/hybrid.hxx` | unified value 和算法 tag |
| `include/puercgp/algorithms/wcc.hxx` | WCC policy |
| `include/puercgp/core/reduce_ops.hxx` | BFS/SSSP/WCC device reduce 原语 |
| `include/puercgp/core/query_descriptor.hxx` | hybrid query descriptor 和 batch |
| `include/puercgp/engine/hybrid_engine.hxx` | hybrid kernels、push/pull 主循环 |
| `include/puercgp/engine/replenish_engine.hxx` | slot replenishment experimental engine |

### 验证与 benchmark

| 路径 | 内容 |
|------|------|
| `examples/smoke_reduce_ops.cu` | device 原语 smoke |
| `examples/smoke_hybrid_init.cu` | hybrid init smoke |
| `examples/smoke_hybrid_push.cu` | 单轮 hybrid push smoke |
| `examples/validate_hybrid.cu` | toy graph BFS+SSSP+WCC 端到端 |
| `examples/validate_hybrid_real.cu` | 真实图 hybrid correctness |
| `examples/validate_wcc.cu` | WCC 同质 correctness |
| `examples/bench_hybrid.cu` | hybrid vs sequential benchmark |
| `examples/smoke_replenish.cu` | active_union / reinit / snapshot smoke |
| `examples/validate_replenish.cu` | replenishment toy graph 端到端 |
| `examples/bench_replenish.cu` | replenishment throughput/latency benchmark |

## 验证状态

已知验证工具：

```bash
cd /home/zyl/Projects/ocgp/puercgp
cmake -S . -B build && cmake --build build
./build/smoke_reduce_ops
./build/smoke_hybrid_init
./build/smoke_hybrid_push
./build/validate_hybrid
./build/validate_wcc
./build/validate_hybrid_real <graph> <bfs_srcs> <sssp_srcs> <wcc_count>
./build/bench_hybrid <graph> <bfs_srcs> <sssp_srcs> <wcc_count>
./build/smoke_replenish
./build/validate_replenish
./build/bench_replenish <graph>
```

截至 2026-07-06，本机检查结果：

- `cmake --build build -j 8` 编译通过。
- 当前会话环境没有 CUDA device，运行 smoke/validate 会报
  `cudaErrorNoDevice: no CUDA-capable device is detected`，因此本机只确认了编译状态。
- 历史 GPU 运行记录显示 toy hybrid smoke/validate 已通过；真实图和 replenishment 的实验记录见
  `experiments/`。

## 关键设计决策

| 决策 | 当前选择 | 说明 |
|------|----------|------|
| value 统一 | `float` | BFS level、SSSP distance、WCC label 共用 4-byte value |
| 算法区分 | runtime `algo_kind_t` | 支持单 batch 多算法 |
| 同质路径 | 保留 `frontier_engine<Policy>` | 回归保护，避免影响 BFS/SSSP 原路径 |
| BFS update | visited-mask first-write | 保留 batch bit-mask 优化 |
| SSSP/WCC update | min-reduce | WCC 与 SSSP update 同构 |
| frontier 表示 | shared frontier list + per-vertex mask | push/pull 后都回到 shared frontier |
| max query | 64 | `query_mask_t = uint64_t` 硬限制 |
| replenishment | experimental | 功能实现完成，但当前实验结论为 NO-GO |
