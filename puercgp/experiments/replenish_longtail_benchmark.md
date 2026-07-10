# Replenishment 长尾场景测试（WCC 构造长尾）

**日期**：2026-06-30
**目的**：验证 replenishment 在长尾场景（WCC O(L) 轮收敛）能否翻盘（前面无长尾场景全面 NO-GO）。

## 长尾构造

- 主图：cit-Patents（V=3.77M, E=33M）
- 附加独立长链分量：双向 path，长度 L（不连主图）
- BFS/SSSP source 限制主图 [0, V_主)，只遍历主图（长链不连通），收敛快（O(直径)）
- WCC query 覆盖全图，label 沿长链逐跳传播，**O(L) 轮收敛（长尾）**
- 合并图：V = V_主 + L，通过 `attach_chain()` 在 host 端构造

## 测试配置

| 参数 | 值 |
|---|---|
| N (total) | 200（100 BFS + 96 SSSP + 4 WCC）|
| Q (batch_size) | 32 |
| chain_length L | 5000 |
| WCC 数 | 4 |
| source | 固定 seed=42 随机，BFS/SSSP ∈ [0, V_主) |
| discard_results | on |
| GPU | V100-32GB |

## 结果

| 场景 | mode | sequential_ms | replenish_ms | speedup | WCC completion_level |
|---|---|---|---|---|---|
| A0 对照（chain=0, wcc=0）| push | 3044 | 3527 | 0.86x | — |
| **A2 长尾（L=5000, wcc=4）** | **push** | 3379 | 4020 | **0.84x** | **4999** ✓ 长尾生效 |
| **A2 长尾（L=5000, wcc=4）** | **pull** | ~42000 | ~103000 | **~0.41x** | 5000 ✓ |

**WCC 长尾确认**：4 个 WCC query completion_level=4999-5000（≈ chain_length），O(L) 轮长尾构造成功。

**但 replenish 仍 NO-GO**：push 0.84x（和 A0 对照几乎一样），pull 更差（0.41x）。

## 根因分析（推翻"长尾翻盘"假设）

replenishment 理论优势："快 query 收敛后腾 slot，避免 slot 空等"。实验证明该优势在当前引擎下不成立：

### push 模式：空等免费
BFS/SSSP 收敛后 mask bit=0，push kernel `active_mask & bfs/nonbfs_mask` 过滤，空 slot 直接跳过（`active_mask=0`）。**空等几乎零开销**。
- sequential 空等不亏
- replenish 复用省不下 push 工作
- 反增每轮调度开销（active_union/dispatch × 4999 轮累积）
- → 0.84x（调度开销净亏）

### pull 模式：复用反增工作
pull 每轮扫全图，开销 ∝ query_count（32 slot）。
- sequential：BFS/SSSP 收敛后 slot 值稳定，pull 仍扫但 `should_update=false`（轻）
- replenish：slot 复用持续注入新 query（新 source values=INF 需重算），pull 持续做满 32 slot compute（重）
- → 0.41x（复用让 pull 更重，极亏）

## 最终结论

**replenishment 在 puercgp 当前架构下无性能价值**，穷尽验证：

| 维度 | 结论 |
|---|---|
| throughput（4 图，无长尾）| 0.71-0.86x NO-GO |
| latency（4 图，无长尾）| median/p99 全更差，仅 p25 亮点 |
| **长尾 push** | 0.84x NO-GO（空等免费）|
| **长尾 pull** | 0.41x 极 NO-GO（复用增工作）|

**本质**：replenishment 的"避免空等"假设前提是"空等占资源"。puercgp 的 push（mask 过滤，空等免费）/ pull（全图扫描，空等轻、复用重）都不满足。**batch 引擎的空等处理已高效**，replenishment 反引入调度开销。

## 复现命令

```bash
cd /home/zyl/Projects/ocgp/puercgp

# A0 对照（无长尾）
./build/bench_replenish /home/zyl/data/csr_data/cit-Patents --bfs=100 --sssp=100 --wcc=0 --chain-length=0 --repeats=3

# A2 push 长尾
./build/bench_replenish /home/zyl/data/csr_data/cit-Patents --bfs=100 --sssp=96 --wcc=4 --chain-length=5000 --repeats=2 --mode=push

# A2 pull 长尾
./build/bench_replenish /home/zyl/data/csr_data/cit-Patents --bfs=100 --sssp=96 --wcc=4 --chain-length=5000 --repeats=1 --mode=pull
```
