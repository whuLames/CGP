# Replenishment Throughput Benchmark（总运行时间对比）

**日期**：2026-06-30
**GPU**：Tesla V100-SXM2-32GB
**指标**：总运行时间（wall time，含 init），sequential 分批 vs replenish slot 复用

## 测试配置

| 参数 | 值 |
|---|---|
| N (total query) | 200（100 BFS + 100 SSSP）|
| Q (batch_size) | 32 |
| source | 固定 seed=42 随机生成 ∈ [0, V)，两组共用 |
| mode / push | push + shared_node_warp |
| discard_results | on |
| sequential 方案 | 按 Q=32 分批（32×6 + 8 = 7 批），每批 run_heterogeneous |
| replenish 方案 | run_replenish_pipeline 一次提交，32 slot 复用 |
| repeats | cit-Patents/orkut/twitter=3, sinaweibo=2 |

## 无长尾场景：4 图总运行时间

| 图 | V | E | sequential_ms | replenish_ms | speedup (repl/seq) |
|---|---|---|---|---|---|
| cit-Patents | 3.77M | 33M | 3038.6 | 3524.5 | **0.86x** |
| soc-orkut | 3.0M | 212M | 6632.8 | 9309.4 | **0.71x** |
| soc-twitter | 21M | 530M | 14074.7 | 19183.0 | **0.73x** |
| soc-sinaweibo | 58M | 522M | 18035.7 | 25387.7 | **0.71x** |

**结论**：replenish 总时间**全面更长**（慢 14-41%）。密图（orkut/sinaweibo）最差（0.71x）。

## 长尾场景（WCC 构造，仅 cit-Patents）

| 场景 | mode | sequential_ms | replenish_ms | speedup |
|---|---|---|---|---|
| A0 对照（chain=0, wcc=0）| push | 3044 | 3527 | 0.86x |
| A2 长尾（L=5000, wcc=4）| push | 3379 | 4020 | **0.84x** |
| A2 长尾（L=5000, wcc=4）| pull | ~42000 | ~103000 | **~0.41x** |

WCC completion_level=4999-5000 确认长尾 O(L) 生效，但 replenish 仍更慢。

## 总结论

**replenishment 总运行时间在所有场景都更长**（NO-GO）：
- 无长尾 4 图：0.71-0.86x（慢 14-41%）
- 长尾 push：0.84x（push mask 过滤，空 slot 空等免费，复用不省 push）
- 长尾 pull：0.41x（pull 全图扫描，slot 复用让 pull 持续满载，比 sequential 收敛后轻 pull 更重）

**根因**：replenishment 的"避免空等"假设前提是"空等占资源"，但 puercgp 的 push（mask 过滤）/ pull（全图扫描）都不满足 → batch 引擎空等处理已高效，replenishment 反引入调度开销（snapshot/clear/sync + active_union 每轮处理）。

## 复现命令

```bash
cd /home/zyl/Projects/ocgp/puercgp
cmake --build build --target bench_replenish -j 8

# 4 图无长尾
for g in cit-Patents soc-orkut soc-twitter soc-sinaweibo; do
  echo "===== $g ====="
  ./build/bench_replenish /home/zyl/data/csr_data/$g --repeats=3 --warmup=1
done

# 长尾（cit-Patents）
./build/bench_replenish /home/zyl/data/csr_data/cit-Patents --bfs=100 --sssp=96 --wcc=4 --chain-length=5000 --mode=push
./build/bench_replenish /home/zyl/data/csr_data/cit-Patents --bfs=100 --sssp=96 --wcc=4 --chain-length=5000 --mode=pull
```
