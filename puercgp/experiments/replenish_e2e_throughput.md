# Replenishment 端到端 Throughput（4 真实图，纯 BFS+SSSP，无长尾）

**日期**：2026-06-30
**GPU**：Tesla V100-SXM2-32GB
**workload**：100 BFS + 100 SSSP（无 WCC，无长尾构造），真实图端到端

## 配置

| 参数 | 值 |
|---|---|
| N (total query) | 200 |
| Q (batch_size) | 32 |
| query 组成 | 100 BFS + 100 SSSP |
| source | 固定 seed=42 随机 ∈ [0, V)，两组共用同一组 |
| mode / push | push + shared_node_warp |
| discard_results | on |
| sequential | 按 Q=32 分批（32×6 + 8 = 7 批），每批 run_heterogeneous |
| replenish | run_replenish_pipeline 一次提交 200 query，32 slot 复用 |
| repeats | cit-Patents/orkut/twitter=3, sinaweibo=2 / warmup=1 |

## 总运行时间结果

| 图 | V | E | sequential_ms | replenish_ms | speedup (repl/seq) |
|---|---|---|---|---|---|
| cit-Patents | 3.77M | 33M | 3038.6 | 3524.5 | **0.86x** |
| soc-orkut | 3.0M | 212M | 6632.8 | 9309.4 | **0.71x** |
| soc-twitter | 21M | 530M | 14074.7 | 19183.0 | **0.73x** |
| soc-sinaweibo | 58M | 522M | 18035.7 | 25387.7 | **0.71x** |

**结论**：端到端场景 replenish 总运行时间**全面更长**（慢 14-41%），密图最差。

**根因**：BFS/SSSP 收敛都快（diameter 约束，无长尾差异），slot 复用省的 push 工作不够抵消调度开销（snapshot/clear/sync + active_union 每轮处理）。push 模式下空 slot 的 mask bit=0 被 `active_mask` 过滤跳过，空等几乎免费 → sequential 的"空等"不亏，replenish 的"复用"不赚。

## 复现命令

```bash
cd /home/zyl/Projects/ocgp/puercgp
cmake --build build --target bench_replenish -j 8

for g in cit-Patents soc-orkut soc-twitter soc-sinaweibo; do
  echo "===== $g ====="
  ./build/bench_replenish /home/zyl/data/csr_data/$g --bfs=100 --sssp=100 --wcc=0 --chain-length=0 --repeats=3 --warmup=1
done
```
