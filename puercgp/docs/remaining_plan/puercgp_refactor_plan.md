# puercgp refactor plan

## 目标

本次重构的目标是把当前堆积在 `frontier_engine.hxx` 和
`hybrid_engine.hxx` 中的实现拆成清晰层次，同时删除早期实验阶段留下的
legacy kernel。重构后应只保留同构和异构场景下的核心 push / pull 路径，
并为后续 replenish 的异步 save/reset 与 kernel 并行预留接口。

约束：

- 保持现有 correctness 不变。
- 核心性能不低于当前 baseline。
- 优先保持 public API 稳定，避免一次性破坏 examples 和 benchmark。
- 先做结构拆分，再删除 legacy 路径，最后做 Algorithm / replenish 抽象。

当前 baseline commit：

```text
9dddd10 chore: checkpoint puercgp before refactor
```

## 当前主要问题

1. `frontier_engine.hxx` 文件过大，混合了基础工具函数、初始化 kernel、
   frontier 表示转换、push kernel、pull kernel、host 调度逻辑和 profile。
2. 同构 pull 路径中同时存在 bitmap pull、legacy shared pull、dense GE-SpMM
   pull 和 fused pull，多数已经不是目标实现。
3. push 和 pull 生成 next frontier 的路径不一致：
   - push kernel 内直接生成 `next_frontier_mask` 和 `next_frontier_vertices`。
   - pull kernel 先生成 `unique_flags`，再通过 scan + compact 生成
     `next_frontier_vertices`。
4. hybrid kernel 中算法逻辑通过 `algo_kind_t` + `switch` 绑定在
   `compute_candidate` / `compute_candidate_pull` 中，不利于后续用户自定义算法。
5. replenish 的 slot save/reset 因当前 `V * Q` layout 对单 slot 访问不连续，
   后续需要异步 stream 和 event 接口来掩盖开销。

## 建议文件结构

```text
include/puercgp/
  core/
    types.hxx
    cuda_utils.hxx
    mask.hxx
    atomics.hxx
    layout.hxx
    algorithm_id.hxx

  algorithms/
    algorithm_traits.hxx
    init_traits.hxx
    bfs.hxx
    sssp.hxx
    wcc.hxx
    registry.hxx
    dispatcher.hxx

  state/
    value_matrix.hxx
    frontier_storage.hxx
    engine_workspace.hxx
    pull_workspace.hxx
    replenish_workspace.hxx

  kernels/
    common/init_kernels.hxx
    common/frontier_kernels.hxx
    common/pull_postprocess.hxx
    push/shared_push_kernels.hxx
    pull/fused_pull_kernels.hxx
    hybrid/hybrid_push_kernels.hxx
    hybrid/hybrid_pull_kernels.hxx
    replenish/slot_io_kernels.hxx

  engine/
    frontier_engine.hxx
    hybrid_engine.hxx
    replenish_engine.hxx
    traversal_scheduler.hxx
    query_partition.hxx
    execution_lane.hxx
    push_executor.hxx
    pull_executor.hxx
```

## 基础工具拆分

从 `frontier_engine.hxx` 中抽出以下基础函数：

- `core/cuda_utils.hxx`
  - `throw_if_cuda_error`
  - `grid_for`
  - `timed_gpu`

- `core/mask.hxx`
  - `query_bit`
  - `mask_popcount`
  - `mask_ffs`

- `core/atomics.hxx`
  - `atomic_or_query_mask`
  - `atomic_min_value`
  - hybrid 使用的 first-write / min-reduce 原语

- `core/layout.hxx`
  - `value_index(v, q, Q)`
  - 当前默认仍为 `V * Q`
  - 后续如果测试 `Q * V` 或 slot-staging，只从这里扩展

- `core/algorithm_id.hxx`
  - `algo_kind_t`
  - `unified_value_t`
  - 与异构 slot 描述相关的轻量类型

说明：

原计划里把 `hybrid.hxx` / `dispatcher.hxx` 都放在 `algorithms/` 下不够准确。
更合理的划分是：

- 算法本身的语义放在 `algorithms/`。
- 异构 slot 的 ID / value 类型放在 `core/algorithm_id.hxx`。
- dispatcher 如果只负责算法语义分派，放在 `algorithms/dispatcher.hxx`。
- dispatcher 如果和 hybrid kernel 强绑定，后续应移动到
  `kernels/hybrid/algorithm_dispatch.hxx`。

## Workspace 抽象

### `engine_workspace`

职责：

- 持有同构/异构主计算状态。
- 管理 values、visited mask、frontier A/B、next frontier、计数器。
- 封装初始化、清空 next frontier、swap frontier。

建议成员：

```cpp
template <typename vertex_t, typename value_t>
class engine_workspace {
 public:
  void resize(std::size_t vertex_count, std::size_t query_count);
  void clear_next(cudaStream_t stream);
  void swap_frontiers();

  value_t* values();
  query_mask_t* visited_mask();

  vertex_t* current_vertices();
  query_mask_t* current_mask();
  vertex_t* next_vertices();
  query_mask_t* next_mask();

  unsigned long long* next_unique_count();
  unsigned long long* next_pair_count();
};
```

### `pull_workspace`

职责：

- 持有 pull 专用 `unique_flags`、`pair_counts` 和 scan buffer。
- 提供 pull postprocessing 的 host wrapper。

注意：

pull postprocessing 不建议强行合并成单个 CUDA kernel，因为 compact 需要全局
prefix sum。合理做法是封装为一个 host 算子：

```cpp
launch_pull_postprocess(...)
```

内部包含：

1. inclusive scan `unique_flags`
2. inclusive scan `pair_counts`
3. compact kernel 生成 `next_frontier_vertices`
4. copy/count 更新 `next_unique_count` 和 `next_pair_count`

这样可以隐藏实现细节，同时保留当前避免大量 atomic append 的优势。

### `workspace` 和 `engine` 的边界

`workspace` 是状态和显存资源容器，不决定算法流程。它应该回答：

- values 存在哪里？
- visited/frontier mask 存在哪里？
- 当前 frontier 和 next frontier 是哪一组 buffer？
- 如何清空 next frontier？
- 如何 swap current/next？

`engine` 是控制器，不直接管理大量裸 buffer。它应该回答：

- 初始化用哪个算法逻辑？
- 本轮走 push 还是 pull？
- 调哪个 executor？
- 何时结束？
- profile 如何记录？

因此，`workspace` 不应该知道“现在是 BFS 还是 SSSP”，也不应该决定
push/pull；`engine` 不应该到处散落 `thrust::device_vector`、`cudaMemsetAsync`
和 counter reset。

## Engine / Executor 抽象

### `frontier_engine`

只保留同构主流程：

1. 初始化 workspace。
2. 初始化 source / WCC label。
3. 每轮根据 scheduler 选择 push 或 pull。
4. 调用 `push_executor` 或 `pull_executor`。
5. swap frontier。
6. 收集 profile。

### `hybrid_engine`

只保留异构主流程：

1. 上传 `hybrid_query_batch`。
2. 初始化 slot kind、source value、WCC label。
3. 每轮选择 hybrid push 或 hybrid pull。
4. 调用 hybrid push/pull executor。
5. 维护 active slot 与收敛状态。

### `push_executor`

职责：

- 选择并 launch 当前保留的 push kernel。
- engine 不直接感知 block/grid 和具体 kernel 名。

同构接口示例：

```cpp
template <typename Algorithm, typename graph_t, typename vertex_t>
void launch_push_shared(...);
```

异构接口示例：

```cpp
template <typename graph_t>
void launch_hybrid_push_shared(...);
```

### `pull_executor`

职责：

- launch fused pull kernel。
- 调用 pull postprocessing。
- 返回下一轮 frontier 统计。

同构流程：

```text
fused_pull_simple/smem
  -> unique_flags / pair_counts / next_frontier_mask
  -> pull_postprocess
  -> next_frontier_vertices / next counts
```

异构流程相同，只是 kernel 换成 hybrid fused pull。

## Algorithm 抽象

需要补充一个明确的 Algorithm static traits 接口定义，但不使用 C++ 虚类。
原因是 device kernel 中 runtime virtual dispatch 不适合性能路径，也会影响内联。

当前采用 static traits。Algorithm 类型直接提供固定名字的 `static` 成员，
engine 和 kernel 通过模板参数调用这些成员。

```cpp
enum class init_mode_t {
  single_source,
  all_vertices_label
};

struct bfs_algorithm {
  using value_type = int;

  static constexpr init_mode_t init_mode = init_mode_t::single_source;
  static constexpr bool mark_source_visited = true;

  __host__ __device__ static value_type infinity();
  __host__ __device__ static value_type source_value();

  __host__ __device__ static value_type initial_value(int vertex,
                                                      int source,
                                                      int query_id);

  template <typename weight_t>
  __host__ __device__ static value_type candidate_push(value_type src,
                                                       weight_t weight,
                                                       int level);

  template <typename weight_t>
  __host__ __device__ static value_type candidate_pull(value_type nbr,
                                                       weight_t weight);

  __host__ __device__ static bool should_update(value_type candidate,
                                                value_type current);
};
```

同构路径通过模板参数 `Algorithm` 静态分派。

CRTP 暂不作为第一版方案。只有当 BFS/SSSP/WCC 之间出现大量重复默认逻辑，
并且 static traits 已经无法清晰表达时，再考虑引入 CRTP base。

### 初始化抽象

算法初始化也必须归入 Algorithm 语义，而不是散落在 engine 中。

当前代码的初始化逻辑是：

- 同构 BFS/SSSP：
  - `fill_values_kernel` 将整个 `values[V * Q]` 填成 infinity。
  - `init_shared_sources_kernel` 对每个 query 的 source 写入 source value。
  - BFS 额外设置 `visited_mask[source]`。
  - source 写入 `frontier_mask`，并通过 atomic append 生成 unique frontier。

- 同构 WCC：
  - `init_wcc_labels_kernel` 对每个 `(vertex, query)` 写 `label = vertex_id`。
  - 所有 vertex 都进入初始 frontier。

- 异构 BFS/SSSP/WCC：
  - `fill_unified_kernel` 将整个 `values[V * Q]` 填成 unified infinity。
  - `init_hybrid_sources_kernel` 处理 BFS/SSSP slot。
  - `init_wcc_all_vertices_kernel` 只处理 WCC slot。

重构后应保留这个性能形态：不要为了统一接口，把所有算法都退化成一个
`V * Q` 大 kernel 分支。建议做法：

1. Algorithm 暴露 `init_mode`、`source_value`、`initial_value`、
   `mark_source_visited`。
2. host 侧根据 slot 的 `init_mode` 分组。
3. 初始化 executor 选择对应优化 kernel：
   - `single_source` 走 source init kernel。
   - `all_vertices_label` 走 WCC 这类 all-vertices init kernel。
4. 异构场景仍可以用 slot metadata 驱动统一调度，但 kernel 内只处理对应分组。

### 异构 dispatcher

异构路径保留 `algo_kind_t`，但把 `switch` 从裸函数移动到
`AlgorithmDispatcher`。dispatcher 只负责算法语义，不负责 kernel launch：

```cpp
template <typename AlgorithmSet>
struct AlgorithmDispatcher {
  __device__ static unified_value_t candidate_push(algo_kind_t kind, ...);
  __device__ static unified_value_t candidate_pull(algo_kind_t kind, ...);
  __device__ static unified_value_t infinity(algo_kind_t kind);
  __device__ static init_mode_t init_mode(algo_kind_t kind);
};
```

这样后续用户自定义算法时，不直接修改 kernel 主体，而是扩展 algorithm set。

## 保留的 kernel

### 同构 push

保留：

- `expand_shared_node_warp_kernel`
- `expand_shared_node_kernel`
- `expand_shared_node_query_parallel_kernel`

说明：

`expand_shared_node_warp_kernel` 作为主路径；`expand_shared_node_kernel` 暂时作为
simple/debug fallback。

`expand_shared_node_query_parallel_kernel` 暂时不删除。它代表“一个 warp 内按
query 维度并行”的 push 实现，需要单独 benchmark，比较它与“一个 thread 串行
处理 active queries”的性能差异后再决定是否作为正式路径保留。

### 同构 pull

保留：

- `fused_pull_simple_kernel`
- `fused_pull_smem_kernel`
- `compact_shared_pull_frontier_kernel`

说明：

`compact_shared_pull_frontier_kernel` 后续移动到
`kernels/common/pull_postprocess.hxx`，并通过 `pull_executor` 封装。

### 异构 push

保留：

- `expand_shared_node_warp_hybrid_kernel`
- `expand_shared_node_hybrid_kernel`

说明：

warp 版作为主路径；simple 版暂时作为 fallback。

### 异构 pull

保留：

- `fused_pull_hybrid_simple_kernel`
- `fused_pull_hybrid_smem_kernel`

### Replenish

保留：

- `snapshot_slot_values_kernel`
- `snapshot_and_clear_multi_slot_kernel`
- `set_sources_multi_kernel`
- `clear_single_slot_kernel`
- `set_slot_source_kernel`

后续统一移动到 `kernels/replenish/slot_io_kernels.hxx`。

## 建议删除的 kernel / 路径

### bitmap pull 相关

删除：

- `list_to_bitmap_kernel`
- `shared_to_bitmap_kernel`
- `bitmap_to_list_kernel`
- `bitmap_to_shared_kernel`
- `compute_bitmap_degrees_kernel`
- `pull_expand_kernel`

原因：

bitmap pull 已不是目标实现，且会导致 engine 中保留额外 frontier 表示转换。

### legacy shared pull 相关

删除：

- `pull_values_simple_kernel`
- `pull_bfs_source_mask_kernel`
- `pull_values_warp_coarsened_kernel`
- `launch_shared_pull_values`

原因：

当前目标 pull 路径是 fused pull simple/smem。

### dense / GE-SpMM legacy pull 相关

删除：

- `list_to_dense_kernel`
- `shared_to_dense_kernel`
- `dense_to_list_kernel`
- `dense_to_shared_kernel`
- `ge_spmm_bfs_simple_kernel`
- `ge_spmm_sssp_simple_kernel`
- `ge_spmm_postprocess_kernel`
- `launch_ge_spmm_bfs`
- `launch_ge_spmm_sssp`

原因：

dense GE-SpMM 是早期实验路径，不是当前目标主线。

### 实验 push 相关

删除：

- `expand_edge_balanced_kernel`
- `expand_shared_node_degree_low_bfs_kernel`
- `expand_shared_node_degree_medium_bfs_kernel`
- `expand_shared_node_degree_high_bfs_kernel`
- `launch_expand_shared_node_degree_bfs`

原因：

这些路径目前不是核心实现，且会显著增加调度和维护复杂度。

暂不删除：

- `expand_shared_node_query_parallel_kernel`

原因：

该 kernel 用于验证 query-level parallelism 是否能提升 push 性能，应先加入
benchmark，再决定是否删除或正式保留。

## enum 和 option 清理

建议最终状态：

```cpp
enum class traversal_mode_t { push, pull, hybrid };

enum class push_strategy_t {
  shared_node,
  shared_node_warp
};

enum class pull_strategy_t {
  fused
};

enum class frontier_repr_t {
  shared
};
```

第一轮直接删除旧 `pull_strategy_t` 和 `frontier_repr_t` 枚举值，不做
deprecation 兼容层。相关 examples/bench 同步迁移到 fused pull 和 shared
frontier。

## Query 粒度调度与并发扩展

后续如果希望同一个 batch 内部分 query 走 push，部分 query 走 pull，需要显式
引入 query partition，而不是只用一个全局 traversal mode。

建议抽象：

```cpp
struct query_partition_t {
  query_mask_t active_slots;
  traversal_mode_t mode;
};

struct execution_lane_t {
  cudaStream_t stream;
  int priority = 0;
  int sm_quota = 0;  // 0 表示不限制；后续可接 green context/libsmctrl
};
```

kernel launch 侧统一接收：

```cpp
query_mask_t active_slots
```

### 当前 kernel 对 query partition 的支持情况

push 路径相对容易支持。当前 shared push 本质上读取：

```cpp
active_mask = frontier_mask[source];
```

如果改为：

```cpp
active_mask = frontier_mask[source] & active_slots;
```

就可以让同一个 frontier 中只有部分 query slot 参与本次 push。

pull 路径当前还不完整支持 query partition。原因是 fused pull kernel 当前按
`query_id < query_count` 扫所有 query，没有传入 `active_slots`。后续需要改为：

```cpp
if ((active_slots & query_bit(query_id)) == 0) return;
```

或者在 block/thread 映射上只映射 active slot。

### 并发 push/pull 的 next frontier 问题

如果 push partition 和 pull partition 在不同 stream 上并发执行，不能简单共享
同一个 `next_frontier_mask` 并让 pull 做普通赋值：

```cpp
next_frontier_mask[vertex] = improved_mask;
```

否则会覆盖另一个 partition 写入的 bit。

更稳妥的设计：

1. 每个 partition 使用独立的 `next_frontier_mask` / `unique_flags`。
2. 所有 partition 完成后做一次 merge + compact。
3. 或者所有写入统一使用 atomic OR，但仍需要解决 unique flag / compact 的一致性。

建议第一版采用独立输出 buffer，减少并发写同一 mask 的竞态复杂度。

### stream / green context / libsmctrl 关系

CUDA 中同一个 stream 内 kernel 按顺序执行；要让两个独立 kernel 有机会并发，
通常需要把它们提交到不同 stream。

Green Context 的作用不是替代 stream，而是选择一部分 GPU 资源并通过 CUDA stream
operations / kernel launches 去使用这些资源。因此后续资源控制接口仍应以
`execution_lane_t` 为核心：每个 lane 持有 stream，并可选绑定资源限制。

规划接口时不要让 engine 直接持有单一 `cudaStream_t`，而是让 executor 接收
`execution_lane_t`。这样后续可以扩展为：

```text
partition A: slots mask A, push, stream 0, SM quota 40%
partition B: slots mask B, pull, stream 1, SM quota 60%
```

第一轮重构只预留接口，不实现 green context/libsmctrl 绑定。

## Replenish slot save/reset 设计

第一轮将当前已有的 slot save/reset 实现纳入重构后的结构中，但不默认开启
dynamic schedule。目标是先把 slot I/O 路径整理清楚，并完成 correctness 验证。

新增 `slot_io_manager`，用于封装 snapshot/reset/reinit。第一版可以使用同步语义
或当前已有 kernel 实现；异步 stream/event 作为接口预留，不作为第一轮性能目标。

职责：

- 复用现有 `snapshot_and_clear_multi_slot_kernel` 和 source reinit kernel。
- 统一封装 slot snapshot、reset、reinit 的 host 调用。
- dynamic schedule 默认关闭。
- 第一轮只验证 slot save/reset correctness。
- 预留独立 `cudaStream_t` 和 `cudaEvent_t` 字段，后续再启用异步执行。

接口草案：

```cpp
class slot_io_manager {
 public:
  void snapshot_and_reset(..., cudaStream_t stream);
  void reinit_slots(..., cudaStream_t stream);

  // 后续异步扩展接口，第一轮不默认启用。
  void snapshot_and_reset_async(..., cudaStream_t compute_stream);
  void wait_before_reuse(cudaStream_t compute_stream);
};
```

第一轮流程：

```text
detect converged slots
  -> snapshot_and_clear_multi_slot_kernel
  -> set_sources_multi_kernel / set_slot_source_kernel
  -> correctness validation
```

后续异步流程预留：

```text
compute stream detects converged slots
  -> record compute_done event
  -> io stream waits compute_done
  -> snapshot_and_clear_multi_slot_kernel
  -> set_sources_multi_kernel
  -> record slot_ready event
  -> compute stream waits slot_ready only before reusing those slots
```

这样第一轮可以利用当前已有 slot reset 实现，同时不把 dynamic schedule 和异步
资源控制混入本次核心重构的性能验收。

## 详细执行计划

执行原则：

- 每个阶段只解决一个层次的问题，避免结构拆分、kernel 删除和算法抽象混在同一
  次改动里。
- 每个阶段结束都要能编译，通过对应 smoke/correctness 验证。
- 性能验证分层执行：小规模性能门禁用于发现明显回退，最终阶段执行完整性能对比。
- 如果某个阶段出现 correctness 回归，先修正该阶段，不继续推进后续阶段。

### 阶段 0：冻结 baseline 和验证脚本

目标：

- 固定当前可回退基线。
- 明确后续所有性能对比的命令、数据集、输出位置。
- 避免重构过程中才发现没有可复现 baseline。

执行项：

1. 确认 baseline commit：
   - `9dddd10 chore: checkpoint puercgp before refactor`
2. 新建重构分支：
   - 建议分支名：`refactor/puercgp-structure`
3. 记录当前编译命令：
   - CMake configure/build 命令
   - nvcc / CUDA / GPU 信息
   - 是否开启 profiling option
4. 记录当前 correctness 命令：
   - `smoke_hybrid_push`
   - `validate_bfs`
   - `validate_replenish`
   - 现有其他 smoke/validate examples
5. 记录当前 performance 命令：
   - fused pull benchmark
   - hybrid benchmark
   - replenish benchmark
   - query_parallel 对照 benchmark 如果已有则记录，没有则阶段 4 后补
6. 创建结果目录：
   - `puercgp/experiments/refactor_baseline/`
   - `puercgp/experiments/refactor_reports/`

不做：

- 不改源码。
- 不删除 kernel。
- 不修改 CMake 目标。

阶段验收：

- baseline correctness 全部通过，或明确记录当前已知失败项。
- baseline performance 至少保存一份可对比结果。
- 输出 baseline report，后续阶段统一引用。

### 阶段 1：拆分基础工具层

目标：

- 只把基础工具函数从 `frontier_engine.hxx` / `hybrid_engine.hxx` 中移出。
- 不改变任何 kernel 逻辑和 host 调度逻辑。

新增文件：

```text
include/puercgp/core/cuda_utils.hxx
include/puercgp/core/mask.hxx
include/puercgp/core/atomics.hxx
include/puercgp/core/layout.hxx
include/puercgp/core/algorithm_id.hxx
```

迁移内容：

- `cuda_utils.hxx`
  - `throw_if_cuda_error`
  - `grid_for`
  - `cuda_event_timer`
  - `timed_gpu`

- `mask.hxx`
  - `query_bit`
  - `mask_popcount`
  - `mask_ffs`

- `atomics.hxx`
  - `atomic_or_query_mask`
  - `atomic_min_value`
  - hybrid first-write / min-reduce 基础原语

- `layout.hxx`
  - `value_index`
  - 保持当前 `V * Q` layout，不改变访问方式

- `algorithm_id.hxx`
  - `algo_kind_t`
  - `unified_value_t`
  - `unified_infinity`
  - `unified_source_value`

需要修改：

- `frontier_engine.hxx` 改为 include 新工具头。
- `hybrid_engine.hxx` 改为 include 新工具头。
- `core/reduce_ops.hxx` 依赖新的 `algorithm_id.hxx` / `atomics.hxx`。

不做：

- 不移动 push/pull kernel。
- 不删除 legacy kernel。
- 不修改 enum。
- 不修改 examples。

阶段验收：

- 全项目编译通过。
- `git diff --check` 通过。
- smoke correctness：
  - BFS smoke/validate
  - hybrid smoke
  - reduce ops smoke
- 小规模性能 sanity：
  - 选一个小图或已有默认 benchmark，确认 runtime 没有明显异常。

通过标准：

- correctness 与阶段 0 一致。
- 编译产物无新增 warning/error。
- 小规模 runtime 无明显级别回退；该阶段理论上不应有 kernel 性能变化。

### 阶段 2：拆分 workspace 状态管理

目标：

- 把 engine 中裸 `thrust::device_vector` 状态集中到 workspace。
- 让 engine 主流程减少显存资源管理细节。
- 不改变 kernel launch 参数和执行顺序。

新增文件：

```text
include/puercgp/state/engine_workspace.hxx
include/puercgp/state/pull_workspace.hxx
include/puercgp/state/replenish_workspace.hxx
```

`engine_workspace` 职责：

- values
- visited mask
- current/next frontier mask
- current/next frontier vertices
- unique/pair counters
- degree/profile 临时 buffer
- clear next frontier
- swap current/next frontier

`pull_workspace` 职责：

- `unique_flags`
- `pair_counts`
- scan offsets
- compact 输出相关 buffer

`replenish_workspace` 职责：

- final buffer
- slot 状态
- pending/active slot metadata
- dynamic schedule 默认关闭时只作为结构容器

需要修改：

- `frontier_engine.hxx` 中的 device_vector 创建迁移到 `engine_workspace`。
- `hybrid_engine.hxx` 中的 values/frontier/counter buffer 迁移到 workspace。
- `replenish_engine.hxx` 中 slot 相关 buffer 逐步使用 `replenish_workspace`。

不做：

- 不移动 kernel 定义。
- 不删除任何 kernel。
- 不改变 push/pull 决策。
- 不改变 pull postprocess 实现。

阶段验收：

- 全项目编译通过。
- correctness：
  - 同构 BFS push/pull/hybrid
  - 异构 hybrid smoke
  - WCC smoke 或 validate
- 检查 engine 主流程：
  - 不应再散落大量 device_vector 初始化。
  - kernel launch 参数值应与阶段 1 等价。

小性能门禁：

- 选 `cit-Patents` 或一个中等数据集。
- 跑 Q=16 的 push/pull/hybrid。
- 允许小幅 host overhead 波动，但不应出现 `>3%` median runtime 回退。

### 阶段 3：拆分 executor 和 pull postprocess

目标：

- 将 push/pull 的 host launch 逻辑从 engine 中分离。
- 将 pull kernel + scan + compact 包装为明确的 host 算子。
- 不改变 fused pull kernel 本体逻辑。

新增文件：

```text
include/puercgp/engine/push_executor.hxx
include/puercgp/engine/pull_executor.hxx
include/puercgp/kernels/common/pull_postprocess.hxx
include/puercgp/kernels/push/shared_push_kernels.hxx
include/puercgp/kernels/pull/fused_pull_kernels.hxx
```

迁移内容：

- `shared_push_kernels.hxx`
  - `expand_shared_node_kernel`
  - `expand_shared_node_warp_kernel`
  - `expand_shared_node_query_parallel_kernel`

- `fused_pull_kernels.hxx`
  - `fused_pull_simple_kernel`
  - `fused_pull_smem_kernel`
  - `launch_fused_pull`

- `pull_postprocess.hxx`
  - `compact_shared_pull_frontier_kernel`
  - inclusive scan + compact + count copy 的 host wrapper

`push_executor` 职责：

- 根据 push strategy 选择 simple / warp / query_parallel。
- 默认策略仍为 warp。
- query_parallel 仅 experimental，不作为默认。

`pull_executor` 职责：

- 调 fused pull simple/smem。
- 调 pull postprocess。
- 返回 next unique count / pair count。

不做：

- 不删除 bitmap/dense/legacy pull。
- 不改 `pull_strategy_t`。
- 不改 `frontier_repr_t`。
- 不改 Algorithm 抽象。

阶段验收：

- 全项目编译通过。
- correctness：
  - 同构 BFS pull
  - 同构 SSSP pull
  - 异构 hybrid pull
  - push/hybrid smoke 确认 executor 分离无副作用
- pull postprocess 单独检查：
  - `next_frontier_mask`
  - `next_frontier_vertices`
  - `next_unique_count`
  - `next_pair_count`

分组性能门禁 A：阶段 1-3 后执行。

- 目标：验证“纯结构拆分 + executor 封装”没有引入性能回退。
- 数据集：至少 `cit-Patents`、`soc-orkut`。
- Q：16、32。
- 模式：push、pull、hybrid。
- 判定：
  - median total runtime 回退 `<=3%`。
  - fused pull kernel time 基本一致。
  - pull postprocess time 基本一致。
  - 如果回退明显，优先检查是否新增 stream synchronize 或多余 device-host copy。

### 阶段 4：删除 legacy kernel 和旧执行路径

目标：

- 删除非目标路径。
- 主流程收敛为 shared frontier + fused pull。
- `pull_strategy_t` 第一轮直接变为单一 `fused`。
- `frontier_repr_t` 第一轮只保留 `shared`。

删除内容：

- bitmap pull：
  - `list_to_bitmap_kernel`
  - `shared_to_bitmap_kernel`
  - `bitmap_to_list_kernel`
  - `bitmap_to_shared_kernel`
  - `compute_bitmap_degrees_kernel`
  - `pull_expand_kernel`

- legacy shared pull：
  - `pull_values_simple_kernel`
  - `pull_bfs_source_mask_kernel`
  - `pull_values_warp_coarsened_kernel`
  - `launch_shared_pull_values`

- dense / GE-SpMM legacy pull：
  - `list_to_dense_kernel`
  - `shared_to_dense_kernel`
  - `dense_to_list_kernel`
  - `dense_to_shared_kernel`
  - `ge_spmm_bfs_simple_kernel`
  - `ge_spmm_sssp_simple_kernel`
  - `ge_spmm_postprocess_kernel`
  - `launch_ge_spmm_bfs`
  - `launch_ge_spmm_sssp`

- 实验 push 中删除：
  - `expand_edge_balanced_kernel`
  - `expand_shared_node_degree_low_bfs_kernel`
  - `expand_shared_node_degree_medium_bfs_kernel`
  - `expand_shared_node_degree_high_bfs_kernel`
  - `launch_expand_shared_node_degree_bfs`

保留：

- `expand_shared_node_query_parallel_kernel`
  - benchmark-only / experimental
  - 默认不启用

需要修改：

- `core/types.hxx`
  - `pull_strategy_t` 只保留 `fused`
  - `frontier_repr_t` 只保留 `shared`
  - `push_strategy_t` 保留 `shared_node`、`shared_node_warp`、
    `shared_node_query_parallel`
- examples/bench 中旧 option 同步迁移。
- docs 中旧路径说明同步标记为 removed。

阶段验收：

- 编译通过。
- `rg` 检查旧 symbol 不再被引用。
- examples/bench 不再传 `bitmap` / `ge_spmm`。
- correctness：
  - 同构 BFS/SSSP/WCC
  - 异构 BFS+SSSP+WCC
  - push/pull/hybrid

分组性能门禁 B：阶段 4 后执行。

- 目标：确认删除 legacy 分支没有改变核心 fused 路径性能。
- 数据集：`cit-Patents`、`soc-orkut`。
- Q：16、32。
- 场景：
  - 同构 BFS pull/hybrid
  - 异构 hybrid pull/hybrid
- 判定：
  - fused pull kernel time 与阶段 3 基本一致。
  - total runtime 不低于阶段 3。
  - 如果变慢，重点检查 enum 清理是否改变默认策略或阈值。

### 阶段 5：Algorithm static traits 抽象

目标：

- 将 BFS/SSSP/WCC 的计算语义、初始化语义集中到 Algorithm traits。
- 同构路径继续编译期静态分派。
- 异构路径通过 `AlgorithmDispatcher` 做 runtime tag 到 traits 的语义分派。

新增文件：

```text
include/puercgp/algorithms/algorithm_traits.hxx
include/puercgp/algorithms/init_traits.hxx
include/puercgp/algorithms/dispatcher.hxx
```

Algorithm traits 内容：

- `value_type`
- `init_mode`
- `mark_source_visited`
- `infinity()`
- `source_value()`
- `initial_value(vertex, source, query_id)`
- `candidate_push(...)`
- `candidate_pull(...)`
- `should_update(candidate, current)`

初始化模式：

```cpp
enum class init_mode_t {
  single_source,
  all_vertices_label
};
```

需要修改：

- `bfs.hxx`
- `sssp.hxx`
- `wcc.hxx`
- `hybrid.hxx`
- `core/reduce_ops.hxx`
- hybrid candidate 逻辑迁移到 `algorithms/dispatcher.hxx`
- init executor 根据 `init_mode` 分组调用优化 kernel

不做：

- 不做 host-side algorithm registry。
- 不做用户自定义算法公开 API。
- 不改变 `algo_kind_t` 的现有 BFS/SSSP/WCC 表示。

阶段验收：

- 编译通过。
- correctness 必须覆盖：
  - BFS source init
  - SSSP source init
  - WCC all-vertices label init
  - hybrid batch 中 BFS/SSSP/WCC 混合 init
  - push candidate
  - pull candidate
- `smoke_reduce_ops` 或等价测试必须更新并通过。

分组性能门禁 C：阶段 5 后执行。

- 目标：确认 traits/dispatcher 没有破坏内联和 kernel 性能。
- 数据集：至少 `cit-Patents`、`soc-orkut`。
- Q：16、32。
- 场景：
  - 同构 BFS/SSSP/WCC
  - 异构 BFS+SSSP+WCC
- 判定：
  - 同构 kernel 不应出现明显回退。
  - 异构 kernel 如出现回退，检查 dispatcher switch 是否进入 inner loop，必要时把
    slot-invariant 信息提前读入寄存器。

### 阶段 6：query partition 接口预留

目标：

- 为后续部分 slot push、部分 slot pull、green context/libsmctrl 资源控制预留接口。
- 第一轮不实现部分 slot push / 部分 slot pull 并发调度。

新增文件：

```text
include/puercgp/engine/query_partition.hxx
include/puercgp/engine/execution_lane.hxx
```

新增类型：

```cpp
struct query_partition_t {
  query_mask_t active_slots;
  traversal_mode_t mode;
};

struct execution_lane_t {
  cudaStream_t stream;
  int priority;
  int sm_quota;
};
```

需要修改：

- push executor 接口增加 `active_slots`，默认等于当前 batch 的全部有效 bit。
- push kernel 内使用：
  - `active_mask = frontier_mask[source] & active_slots`
- pull executor 接口预留 `active_slots`。
- pull kernel 可先保留默认 all-slots 行为；如果加判断，必须确认无性能回退。

不做：

- 不实现多个 stream 并发 push/pull。
- 不实现 green context。
- 不接 libsmctrl。
- 不实现多个 partition 的 next frontier merge。

阶段验收：

- 默认单 partition correctness 与阶段 5 一致。
- `active_slots = all_slots` 时结果完全一致。
- 可选单元测试：
  - 构造一个 mask，只让部分 slot 参与 push，验证未参与 slot 不被更新。
  - 该测试不进入默认执行路径。

性能检查：

- 只做小门禁。
- 目标：确认 `& active_slots` 没有造成主路径明显回退。
- 判定：push kernel time 回退 `<=3%`；如超过，考虑让 all-slots 默认路径使用
  compile-time fast path。

### 阶段 7：replenish slot I/O 结构化封装

目标：

- 复用当前已有 slot reset/save 实现。
- 将 slot snapshot/reset/reinit kernel 移动到独立文件。
- 引入 `slot_io_manager`。
- dynamic schedule 默认关闭。
- 第一轮只做 correctness 验证，不做 dynamic schedule 性能验收。

新增文件：

```text
include/puercgp/kernels/replenish/slot_io_kernels.hxx
include/puercgp/engine/slot_io_manager.hxx
```

迁移内容：

- `snapshot_slot_values_kernel`
- `snapshot_and_clear_multi_slot_kernel`
- `set_sources_multi_kernel`
- `clear_single_slot_kernel`
- `set_slot_source_kernel`
- 对应 host launch wrapper

`slot_io_manager` 第一版接口：

```cpp
void snapshot_and_reset(..., cudaStream_t stream);
void reinit_slots(..., cudaStream_t stream);
```

预留但默认不启用：

```cpp
void snapshot_and_reset_async(..., cudaStream_t compute_stream);
void wait_before_reuse(cudaStream_t compute_stream);
```

需要修改：

- `replenish_engine.hxx` 调用 `slot_io_manager`。
- `hybrid_engine.hxx` 中可复用的 slot I/O helper 移到新文件。
- options 中 dynamic schedule 默认关闭。

不做：

- 不开启异步 stream/event save/reset。
- 不做部分 slot push/pull 调度。
- 不把 dynamic schedule 性能纳入第一轮验收。

阶段验收：

- 编译通过。
- correctness：
  - `smoke_replenish`
  - `validate_replenish`
  - hybrid batch correctness
  - slot 被 reset 后重新注入 query，旧 slot 数据不能污染新 query
- 检查默认路径：
  - dynamic schedule 默认关闭。
  - 未开启 replenish 时，普通 hybrid run 不应调用 slot I/O。

性能检查：

- 默认非 replenish 路径做一次 sanity benchmark，确认没有明显回退。
- slot save/reset 本阶段只要求 correctness；性能优化留到后续异步版本。

### 阶段 8：完整 correctness / performance 验证

目标：

- 对重构结果做最终验收。
- 与 baseline commit `9dddd10` 做同环境对比。
- 输出可归档的 performance report。

验证步骤：

1. checkout baseline commit `9dddd10`。
2. 使用固定编译参数重新 build。
3. 跑 baseline correctness。
4. 跑 baseline performance，保存到：
   - `puercgp/experiments/refactor_baseline/`
5. checkout refactor branch。
6. 使用同样编译参数重新 build。
7. 跑 refactor correctness。
8. 跑 refactor performance，保存到：
   - `puercgp/experiments/refactor_reports/`
9. 生成对比报告：
   - correctness summary
   - runtime ratio
   - kernel time ratio
   - pull postprocess ratio
   - memory usage
   - 已删除 kernel / 已保留 experimental kernel 列表

完整 correctness 场景：

- 同构 BFS
- 同构 SSSP
- 同构 WCC
- 异构 BFS + SSSP + WCC
- push
- pull
- hybrid
- replenish slot save/reset correctness

完整 performance 场景：

- 数据集：
  - `cit-Patents`
  - `soc-sinaweibo`
  - `soc-twitter`
  - `soc-orkut`
- Q：
  - 16
  - 32
  - 64
- 模式：
  - push
  - pull
  - hybrid
- 算法：
  - BFS
  - SSSP
  - WCC
  - hybrid BFS+SSSP+WCC

最终验收标准：

- correctness 必须全部通过。
- median total runtime 不低于 baseline，允许 `<=3%` 波动。
- 单个核心 kernel 若变慢超过 `5%`，必须定位原因。
- pull postprocess time 应与 baseline scan + compact 等价。
- 非 replenish 默认路径不能因为 slot I/O 封装变慢。
- replenish dynamic schedule 默认关闭；slot save/reset 第一轮只做 correctness 验证。

回退规则：

- correctness 回归：必须修复，不接受带回归进入下一阶段。
- 结构拆分导致性能回退：优先检查额外同步、额外 device-host copy、多余 memset。
- kernel 性能回退：检查 launch 配置、寄存器、shared memory、内联、分支位置。
- 无法解释的核心路径回退：回退对应阶段改动，不继续叠加后续重构。

## 性能验证指标

数据集：

- `cit-Patents`
- `soc-sinaweibo`
- `soc-twitter`
- `soc-orkut`

query 数：

- `Q = 16`
- `Q = 32`
- `Q = 64`

场景：

- 同构 BFS
- 同构 SSSP
- 同构 WCC
- 异构 BFS + SSSP + WCC
- replenish enabled / disabled

模式：

- push
- pull
- hybrid

指标：

- total runtime
- push kernel ms
- pull kernel ms
- pull postprocess ms
- iteration count
- frontier unique count
- frontier pair count
- GPU memory usage
- replenish snapshot/reset time
- slot idle time

判定标准：

- correctness 必须一致。
- median total runtime 不低于 baseline，允许 `<= 3%` 波动。
- 单个核心 kernel 若变慢超过 `5%`，需要定位原因或回退。
- pull postprocessing 封装后总耗时应与原 scan + compact 路径一致。
- replenish dynamic schedule 第一轮默认关闭。
- slot save/reset 第一轮只做 correctness 验证；性能优化留到后续异步版本。

## Review 决策点

需要在正式动手前确认：

1. 已确认：同构 push 保留 `expand_shared_node_query_parallel_kernel` 作为
   benchmark-only / experimental 路径，默认主路径仍使用
   `expand_shared_node_warp_kernel`。
2. 已确认：异构 push 第一轮同时保留 `expand_shared_node_hybrid_kernel`
   和 `expand_shared_node_warp_hybrid_kernel`。默认使用 warp hybrid，simple
   hybrid 作为 correctness/debug fallback；重构和性能验证完成后再决定是否删除。
3. 已确认：`pull_strategy_t` 第一轮直接改为单一 `fused`，不保留
   `bitmap` / `ge_spmm` 旧枚举值；相关 examples/bench 同步迁移到 fused pull。
4. 已确认：`frontier_repr_t` 第一轮只保留 `shared`，删除
   `list` / `bitmap` / `dense` 旧表示及其主流程依赖。
5. 已确认：Algorithm 抽象第一版只采用 static traits，不做 host-side
   algorithm registry；用户自定义算法 API 留到后续设计。
6. 已确认：异构 `AlgorithmDispatcher` 放在 `algorithms/dispatcher.hxx`。
   它只负责 `algo_kind_t -> Algorithm traits` 的语义分派，kernel 层只调用
   dispatcher，不持有算法分派逻辑。
7. 已确认：query partition 第一轮只预留接口，不实现部分 slot push、
   部分 slot pull。
8. 已确认：第一轮实现 slot save/reset 的结构化封装并复用当前已有实现；
   dynamic schedule 默认关闭；slot save/reset 第一轮只做 correctness 验证，
   异步 stream/event 执行留到后续。
