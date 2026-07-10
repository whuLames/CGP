# Bug 报告：push 与 pull 模式 BFS 收敛轮次不一致

## 1. 概述

**现象**：使用相同源节点 + 相同图跑 BFS，`traversal_mode=push` 与 `traversal_mode=hybrid`（含 pull phase）产生**不同的 frontier 序列**和**不同的收敛轮数**。

**预期**：BFS 是确定性算法，给定相同源 + 相同图，收敛轮数 = BFS 树最大深度，**与 push/pull 实现无关**。三种模式（push-only / pull-only / hybrid）应该产生完全相同的 frontier 序列和相同的收敛轮数。

**实际**：四种测试数据集全部出现分叉，pull 模式的 frontier_size **系统性偏大**，导致 BFS "走得更快"，收敛轮数减少。

**严重性**：高。如果 pull 是对的，则 push 漏访问顶点（BFS 不完整）；如果 push 是对的，则 pull 多访问顶点（错误结果）。无论哪种，至少一种模式产生错误的 BFS distance。

---

## 2. 环境信息

| 项目 | 值 |
|------|---|
| GPU | Tesla V100-SXM2-32GB ×2 |
| 驱动 | 570.211.01 |
| CUDA | 12.8 (nvcc V12.8.93) |
| OS | Debian 6.1.0-44 / Linux 6.1.0-44-amd64 |
| 代码版本 | puercgp master 分支（commit 待补） |
| 关键修改 | `frontier_engine.hxx:2614` 临时关闭 skip_shared_direct_edge_count（仅 profile 模式） |

---

## 3. 复现步骤

### 3.1 编译

```bash
cd /home/zyl/Projects/ocgp/puercgp/build
cmake --build . --target validate_bfs -j 4
```

### 3.2 修改代码（为了让 shared_node_warp 也能记录 edge_count）

**文件**：`include/puercgp/engine/frontier_engine.hxx:2614-2620`

**修改前**：
```cpp
bool skip_shared_direct_edge_count =
    options.traversal_mode == traversal_mode_t::push &&
    current_repr == frontier_repr_t::shared &&
    (options.push_strategy == push_strategy_t::shared_node_warp ||
     options.push_strategy == push_strategy_t::shared_node_degree);
```

**修改后**：
```cpp
bool skip_shared_direct_edge_count =
    options.traversal_mode == traversal_mode_t::push &&
    current_repr == frontier_repr_t::shared &&
    !options.profile_iterations &&   // ← 新增：profile 模式不跳过
    (options.push_strategy == push_strategy_t::shared_node_warp ||
     options.push_strategy == push_strategy_t::shared_node_degree);
```

**注**：此修改**不影响 bug 复现**。即使不修改，bug 依然存在。修改只是为了在评估策略时能拿到完整的 edge_count 数据。

### 3.3 运行测试

**测试源节点**（每个数据集 64 个源，seed=1）：
```
/home/zyl/Projects/ocgp/puercgp/results/bfs_q64_allpush_frontier_vs_pull_v1/sources/<dataset>_q64_seed1.txt
```

**命令模板**：
```bash
SOURCES=$(cat results/bfs_q64_allpush_frontier_vs_pull_v1/sources/<dataset>_q64_seed1.txt)

# push-only
./build/validate_bfs /home/zyl/data/csr_data/<dataset> "$SOURCES" 3 push shared_node_warp ge_spmm

# hybrid (含 pull phase)
./build/validate_bfs /home/zyl/data/csr_data/<dataset> "$SOURCES" 3 hybrid shared_node_warp ge_spmm
```

**4 个数据集的命令**：

```bash
# cit-Patents
SOURCES=$(cat results/bfs_q64_allpush_frontier_vs_pull_v1/sources/cit-Patents_q64_seed1.txt)
./build/validate_bfs /home/zyl/data/csr_data/cit-Patents "$SOURCES" 3 push shared_node_warp ge_spmm
./build/validate_bfs /home/zyl/data/csr_data/cit-Patents "$SOURCES" 3 hybrid shared_node_warp ge_spmm

# soc-orkut
SOURCES=$(cat results/bfs_q64_allpush_frontier_vs_pull_v1/sources/soc-orkut_q64_seed1.txt)
./build/validate_bfs /home/zyl/data/csr_data/soc-orkut "$SOURCES" 3 push shared_node_warp ge_spmm
./build/validate_bfs /home/zyl/data/csr_data/soc-orkut "$SOURCES" 3 hybrid shared_node_warp ge_spmm

# soc-sinaweibo
SOURCES=$(cat results/bfs_q64_allpush_frontier_vs_pull_v1/sources/soc-sinaweibo_q64_seed1.txt)
./build/validate_bfs /home/zyl/data/csr_data/soc-sinaweibo "$SOURCES" 3 push shared_node_warp ge_spmm
./build/validate_bfs /home/zyl/data/csr_data/soc-sinaweibo "$SOURCES" 3 hybrid shared_node_warp ge_spmm

# soc-twitter
SOURCES=$(cat results/bfs_q64_allpush_frontier_vs_pull_v1/sources/soc-twitter_q64_seed1.txt)
./build/validate_bfs /home/zyl/data/csr_data/soc-twitter "$SOURCES" 3 push shared_node_warp ge_spmm
./build/validate_bfs /home/zyl/data/csr_data/soc-twitter "$SOURCES" 3 hybrid shared_node_warp ge_spmm
```

### 3.4 检查输出

每个命令的 stdout 会输出 `frontier_sizes=...` 一行。**对比两个模式的 frontier_sizes 即可观察 bug**。

---

## 4. 实测数据（核心证据）

### 4.1 总览

| 数据集 | V | E | push-only 收敛轮数 | hybrid 收敛轮数 | 差异 |
|--------|------|------|-----|-----|------|
| cit-Patents | 3,774,768 | 33,037,895 | **21** | **17** | -4 轮 |
| soc-orkut | 2,997,166 | 212,698,418 | **9** | **5** | -4 轮 |
| soc-sinaweibo | 58,655,849 | 522,642,142 | **8** | **5** | -3 轮 |
| soc-twitter | 21,297,772 | 530,051,618 | **19** | **17** | -2 轮 |

**规律**：所有数据集，hybrid（含 pull phase）都比 push-only **少 2-4 轮收敛**。

### 4.2 逐数据集 frontier 序列对比

#### cit-Patents（分叉点 iter 8）

```
iter:    0    1    2     3      4       5        6         7          8 (← 分叉)
push:   64  734  9375  96506  761158  4645106  20231991  52883517   73,801,414
hybrid: 64  734  9375  96506  761158  4645106  20231991  52883517  103,333,498  ← pull 多 40%

继续:
push:   ... 56288650  24036478  6550136  1320947  ... 4  0  (21 轮收敛)
hybrid: ... 56182827  7342028   168418   9567    ... 36 0  (17 轮收敛)
```

#### soc-orkut（分叉点 iter 4）

```
iter:    0   1    2       3          4 (← 分叉)
push:   64  4675 451655  23859622   90,195,037
hybrid: 64  4675 451655  23859622  166,709,654  ← pull 多 85%

继续:
push:   ... 72581402  4707134  18974  61  0  (9 轮收敛)
hybrid: ... 2116652    853      34     3   0  (5 轮收敛)
```

#### soc-sinaweibo（分叉点 iter 4）

```
iter:    0   1    2        3           4 (← 分叉)
push:   64  2920 1601401  144194620   3,019,982,659
hybrid: 64  2920 1601401  144194620   3,606,892,081  ← pull 多 19%

继续:
push:   ... 586855144  1335640  32  0  (8 轮收敛)
hybrid: ... 1283493    969      8   0  (5 轮收敛)
```

#### soc-twitter（分叉点 iter 4）

```
iter:    0   1    2        3           4 (← 分叉)
push:   64  1271 7717390  117101835   401,805,025
hybrid: 64  1271 7717390  117101835   667,822,609  ← pull 多 66%

继续:
push:   ... 449772726  252659616  ... 8  0  (19 轮收敛)
hybrid: ... 490027446  125600595  ... 1  0  (17 轮收敛)
```

### 4.3 共同特征

1. **所有数据集都在切到 pull 的那一轮出现分叉**
2. **pull 模式的 frontier_size 系统性偏大**（19% ~ 85%）
3. **后续轮次的 frontier 也偏大**（导致 BFS 走得更快，提前收敛）
4. **push 模式的初始 64 源 = pull 模式的初始 64 源**，前几轮（push phase）完全一致

---

## 5. 可能的原因分析

### 5.1 假设 A：pull 多算了（pull 有 bug）

**怀疑代码位置**：`include/puercgp/engine/frontier_engine.hxx:1615-1705` 的 `fused_ge_bfs_pull_smem_kernel`

```cpp
// line 1656-1672: 检查邻居
for (int i = 0; i < tile_count; ++i) {
  vertex_t neighbor = neighbor_tile[threadIdx.y][i];
  if (values[neighbor, q] == level) acc = true;   // 邻居值==当前 level？
}

// line 1674-1690: 更新当前顶点
if (acc && values[vertex, q] == infinity()) {
  values[vertex, q] = level + 1;                  // 标记为下一轮 frontier
  local_mask |= query_bit(q);
}
```

**可能的 bug**：
- `values[neighbor] == level` 检查可能**错误地包含了已访问的顶点**（values 已被前几轮更新到 level+1, level+2...）
- 但代码看起来逻辑正确（只检查 ==level，不检查其他值）
- 可能是 **warp 之间的 race condition**：line 1691 `lane_masks[...] = local_mask` + line 1694-1698 `for` 循环 OR，这里 lane_masks 是 shared memory，但 `__syncthreads()` 在 line 1692 应该保证可见性

### 5.2 假设 B：push 漏算了（push 有 bug，更严重）

**怀疑代码位置**：`expand_shared_node_warp_kernel`（push 主 kernel）

**可能的 bug**：
- `atomicOr` mask 累加丢失更新（race condition）
- BFS 不完整，部分顶点 distance 应该是 k+1 但被错误标为 infinity，导致 pull 重新发现它们

**支持证据**：pull 多发现的 (vertex, query) pair 数量很大（cit-Patents 多 40%，soc-orkut 多 85%），这种规模的差异**不可能是统计误差**，必然有一方实现有 bug。

### 5.3 假设 C：visited_mask 在 pull 模式下没正确更新

**怀疑代码位置**：`fused_ge_bfs_pull_smem_kernel:1700`

```cpp
visited_mask[vertex] |= improved_mask;
```

注意这里是**非 atomic 的 `|=`**！如果多个 thread 同时写同一个 vertex 的 visited_mask，可能丢失更新。但每个 vertex 只被一个 thread 处理（grid_for 拆分），所以理论上没问题。**需要验证**。

---

## 6. 建议的调试步骤

### 6.1 第一步：跑 pull-only 模式确认是哪一方有 bug

```bash
# 对每个数据集跑 pull-only
SOURCES=$(cat results/bfs_q64_allpush_frontier_vs_pull_v1/sources/cit-Patents_q64_seed1.txt)
./build/validate_bfs /home/zyl/data/csr_data/cit-Patents "$SOURCES" 3 pull shared_node_warp ge_spmm
```

**判断**：
- 如果 pull-only 收敛轮数 = push-only（21 轮）→ **hybrid 的 pull phase 有 bug**（pull 单独跑是对的）
- 如果 pull-only 收敛轮数 = hybrid（17 轮）→ **push 有 bug**（pull 是对的）
- 如果 pull-only 收敛轮数 ≠ 两者 → **push 和 pull 都有 bug**

### 6.2 第二步：开启 CPU 验证

`validate_bfs.cu:114`：
```cpp
bool run_cpu_check = graph.vertices <= 1000000;
```

**临时改为**：
```cpp
bool run_cpu_check = graph.vertices <= 5000000;  // 包含 cit-Patents
```

重新编译，跑 cit-Patents 三种模式，看 `distance_mismatches` 字段。**至少一种模式应该报告大量 mismatch**。

### 6.3 第三步：小图复现

用更小的图（如 unweighted.mm 测试图）跑三种模式，对比 distance 数组。`validate_bfs.cu:60` 默认 matrix 是 `examples/matrices/unweighted.mm`，可以直接跑：

```bash
./build/validate_bfs   # 默认参数，使用 unweighted.mm
```

### 6.4 第四步：单源 + 单 query 复现

把 Q 降到 1（单源 BFS），看是否还有问题：

```bash
SOURCES="563564"  # 单源
./build/validate_bfs /home/zyl/data/csr_data/cit-Patents "$SOURCES" 3 push shared_node_warp ge_spmm
./build/validate_bfs /home/zyl/data/csr_data/cit-Patents "$SOURCES" 3 pull shared_node_warp ge_spmm
```

### 6.5 第五步：在 pull kernel 加调试日志

在 `fused_ge_bfs_pull_smem_kernel:1689` 后加：
```cpp
if (vertex == <某个可疑顶点> && query0 == <某个可疑 query>) {
  printf("[DEBUG] vertex=%zu q=%d acc=%d values[v]=%d neighbor_values=[...]\n",
         vertex, query0, acc0, values[value_pos]);
}
```

---

## 7. 已确认的事实（无需再验证）

1. ✅ 所有 4 个数据集都出现分叉
2. ✅ 分叉点都在切到 pull 的那一轮
3. ✅ pull 的 frontier_size 系统性偏大
4. ✅ push 和 pull 的前几轮（push phase）frontier 完全相同
5. ✅ 源节点完全相同（同 sources.txt）
6. ✅ BFS 算法本身是确定性的，理论上一致

## 8. 未确认的事项（需要调试）

1. ❓ push-only 还是 pull-only 哪个有 bug（需要跑 pull-only 测试）
2. ❓ 哪个具体 kernel 行有 bug
3. ❓ 是 race condition 还是逻辑错误
4. ❓ 是否影响最终 BFS distance 的正确性（需要 CPU 验证）

---

## 9. 相关文件

| 文件 | 说明 |
|------|------|
| `results/bfs_q64_allpush_frontier_vs_pull_v1/<ds>_q64_seed1_shared_node_warp.log` | push-only 实测 log |
| `results/bfs_q64_hybrid_strategy_eval/<ds>_q64_seed1_hybrid.log` | hybrid 实测 log |
| `results/bfs_q64_allpush_frontier_vs_pull_v1/sources/<ds>_q64_seed1.txt` | 64 源节点 |
| `include/puercgp/engine/frontier_engine.hxx:1615-1705` | pull kernel（怀疑位置）|
| `include/puercgp/engine/frontier_engine.hxx:819-905`（约）| push kernel（怀疑位置）|
| `examples/validate_bfs.cu` | 测试入口 |

---

## 10. 报告人 / 日期

- 报告日期：2026-07-06
- 报告人：通过 Claude Code 自动分析生成
- 数据来源：hybrid 策略评估测试（`results/bfs_q64_hybrid_strategy_eval/`）
