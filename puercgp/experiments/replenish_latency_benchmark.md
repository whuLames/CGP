# Replenishment Latency Benchmark

**日期**：2026-06-30
**GPU**：Tesla V100-SXM2-32GB
**目的**：从 latency（每个 query 完成时间）视角对比 sequential 分批 vs replenish slot 复用，验证 replenishment 在在线场景的 latency 优势（补充 throughput 视角）。

## 测试配置

| 参数 | 值 |
|---|---|
| Q (batch_size) | 32 |
| N (total query) | 200（100 BFS + 100 SSSP）|
| source 生成 | 固定 seed=42 随机生成 ∈ [0, V)，两组共用同一组 |
| discard_results | on（replenish 不分配 final_buffer N×V，省显存；snapshot/clear 调度开销保留）|
| mode / push | push + shared_node_warp |
| repeats / warmup | cit-Patents/orkut/twitter=3, sinaweibo=2 / 1 |
| 数据集 | cit-Patents, soc-orkut, soc-twitter, soc-sinaweibo |

## Latency 定义

- **sequential**：批次累积 wall（第 k 批 query latency = Σ前k批 wall_time_ms）。**含每批 init**（每批 `run_heterogeneous` 重新分配 buffer 的悲观估计——真实在线会复用 buffer，所以这是 sequential 的上界）。
- **replenish**：`completion_wall`（slot 收敛时刻，从第一轮 push 起算）。**不含 init**。
- **流式返回假设**：query 一算完立即可返回给用户（在线服务模型）。

> 注：sequential 的 latency 含 init 是悲观估计。即使如此，replenish 仍输（见下）→ replenish 确实差，非测量偏差。

## Throughput 结果（replenish / sequential，<1 = replenish 慢）

| 图 | V | E | sequential_ms | replenish_ms | speedup |
|---|---|---|---|---|---|
| cit-Patents | 3.77M | 33M | 3038.6 | 3524.5 | **0.86x** |
| soc-orkut | 3.0M | 212M | 6632.8 | 9309.4 | **0.71x** |
| soc-twitter | 21M | 530M | 14074.7 | 19183.0 | **0.73x** |
| soc-sinaweibo | 58M | 522M | 18035.7 | 25387.7 | **0.71x** |

**replenish 全面慢 14-41%**。密图（orkut/sinaweibo）最差（0.71x）。

## Latency 分布（ms，低为优）

| 图 | 方案 | min | p25 | median | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| cit-Patents | seq | 277.7 | 546.0 | **1476.2** | 2093.2 | 2704.7 | 3038.6 | 3038.6 | 1386.9 |
| | repl | 360.1 | **415.7** ✓ | 1966.7 | 3028.4 | 3513.9 | 3543.2 | 3546.4 | 1836.4 |
| soc-orkut | seq | 370.5 | 750.0 | **2891.5** | 4505.0 | 6172.0 | 6631.6 | 6631.6 | 2792.5 |
| | repl | 477.4 | **520.6** ✓ | 5608.9 | 7852.1 | 9277.1 | 9302.1 | 9304.3 | 4505.5 |
| soc-twitter | seq | 1529.1 | 2613.8 | **7145.4** | 9694.4 | 12959.5 | 13934.4 | 13934.4 | 6601.1 |
| | repl | 1737.5 | **2098.4** ✓ | 8561.0 | 17228.4 | 19139.8 | 19498.7 | 19521.0 | 10072.0 |
| soc-sinaweibo | seq | 1790.7 | 3378.7 | **8640.2** | 12456.7 | 16316.4 | 18035.7 | 18035.7 | 8289.1 |
| | repl | 1827.3 | **1853.3** ✓ | 13127.2 | 21349.2 | 24496.6 | 25040.8 | 25086.2 | 12345.3 |

✓ = replenish 该分位优于 sequential。

## 关键发现

### 1. throughput 全面 NO-GO（0.71-0.86x）
BFS/SSSP 无长尾，slot 复用省的 push 不够抵消调度开销（snapshot/clear/sync）。密图最差。

### 2. median / p75 / p90 / p99 latency 全面更差
因为 throughput 差，query 的**绝对完成时刻普遍推后**。replenish 的 slot 复用让 query 陆续完成，但总时间长，多数 query 完成更晚。

### 3. 唯一亮点：p25（前 25% query）replenish 更快
| 图 | seq p25 | repl p25 | replenish 优势 |
|---|---|---|---|
| cit-Patents | 546.0 | 415.7 | -24% |
| soc-orkut | 750.0 | 520.6 | -31% |
| soc-twitter | 2613.8 | 2098.4 | -20% |
| soc-sinaweibo | 3378.7 | 1853.3 | -45% |

早收敛的快 query（短路径 BFS）通过 slot 复用**一收敛就流式返回**，不用等整批；sequential 的前几批也要等批内最慢 query 一起返回。**但代价是 p75-p99 全面恶化**（tail latency 牺牲）。

### 4. 公平性说明
sequential 的 latency 含每批 init（悲观估计）。真实在线 sequential 会复用 buffer（init 摊薄），latency 更低。**即使给 sequential 悲观估计，replenish 仍输** → 结论稳健。

## 结论

replenishment 在标准 BFS/SSSP workload（无长尾）**全面 NO-GO**：
- throughput 差 14-41%
- median/p90/p99 latency 差 20-94%
- 仅 p25（早 query 流式返回）有优势，但不足以证明整体优越

**根因**：replenishment 的 slot 复用优势依赖**长尾差异**（快 query 收敛后腾 slot 给慢 query）。BFS/SSSP 在标准图上收敛都快（diameter 约束），无长尾 → slot 复用省的 push 不够抵消调度开销。

## 后续长尾验证（已完成，见 replenish_longtail_benchmark.md）

构造了 WCC 长尾（`attach_chain` 加独立长链 L=5000，WCC O(L) 轮收敛，completion_level=4999 确认），**长尾下仍 NO-GO**：
- push: 0.84x（**空等免费**：push mask 过滤，空 slot `active_mask=0` 跳过，空等零开销；replenish 复用省不下 push，反增调度）
- pull: 0.41x（**复用反增工作**：pull 扫全图开销∝query_count，slot 复用持续注入新 query 让 pull 持续满载，比 sequential 收敛后轻 pull 更重）

**根因**：replenishment 的"避免空等"假设前提是"空等占资源"。puercgp 的 push（mask 过滤，空等免费）/ pull（全图扫描，空等轻、复用重）都不满足 → batch 引擎空等处理已高效。详见 `replenish_longtail_benchmark.md`。

---

## 复现命令

```bash
cd /home/zyl/Projects/ocgp/puercgp
cmake --build build --target bench_replenish -j 8

# 单图（默认 total=200, bfs=100, sssp=100, batch=32, seed=42, discard=on）
./build/bench_replenish /home/zyl/data/csr_data/cit-Patents --repeats=3 --warmup=1

# 4 图循环
for g in cit-Patents soc-orkut soc-twitter soc-sinaweibo; do
  echo "===== $g ====="
  ./build/bench_replenish /home/zyl/data/csr_data/$g --repeats=3 --warmup=1
done
```
