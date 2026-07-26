# Online Offset Evaluator

日期：2026-07-13  
平台：NVIDIA Tesla V100-SXM2-32GB，CUDA 12.8  
负载：all-push BFS，`N=128`，`Q=32`，随机种子 42，源节点互不重复

## 1. 结果概览

端到端时间包含在线 batching 和 offset evaluator，不包含可离线摊销的 graph index 构建。每个 query 均比较 visited vertex count 和 BFS distance sum，所有实验正确性通过。

| 数据集 | Baseline (ms) | Online (ms) | Speedup | Evaluator (ms) | Union edges 变化 |
|---|---:|---:|---:|---:|---:|
| indochina | 1116.43 | 483.42 | **2.309x** | 10.69 | 9.278B -> 3.302B |
| soc-twitter | 5160.31 | 2602.48 | **1.983x** | 11.87 | 9.462B -> 4.054B |
| soc-LiveJournal1 | 1032.91 | 693.50 | **1.489x** | 13.14 | 2.367B -> 1.422B |
| soc-orkut | 1459.51 | 1087.66 | 1.342x | 14.49 | 2.887B -> 2.016B |
| cit-Patents | 1039.62 | 806.09 | 1.290x | 14.20 | 0.745B -> 0.527B |
| soc-sinaweibo | 5571.70 | 4834.70 | 1.152x | 11.24 | 4.973B -> 4.202B |

目标在三张非道路图上达到：indochina、soc-twitter 和 soc-LiveJournal1 均超过 1.4x。该收益不是跨图保证；cit-Patents、soc-orkut 和当前仅使用 16 landmarks 的 soc-sinaweibo 未达到 1.4x。

## 2. 实现

核心实现位于：

- `include/puercgp/scheduling/online_offset_evaluator.hxx`
- `examples/bench_online_offset_n128.cu`

### Graph preprocessing

`landmark_phase_index` 对 CSR 图构建可持久化索引：

1. 选择最多 4 个 farthest-point landmarks，用于估计 BFS phase length。
2. 其余 landmarks 通过随机 edge source 采样，等价于按 degree 分布采样 vertex。
3. 对每个 landmark 执行 BFS，以 landmark-major `uint16_t` 保存到所有 vertex 的距离。
4. 索引可通过 `save()` 和 `load()` 跨运行复用。

索引构建复杂度为 `O(L * (V + E))`，空间为约 `2 * L * V` bytes。当前单线程 CPU 预处理实测约为：indochina 30.3 s / 0.88 GiB，LiveJournal 45.8 s / 0.58 GiB，twitter 116.7 s / 1.27 GiB。该成本不进入每批 query 的在线延迟。

### Online batching

`online_offset_evaluator::plan_batches()` 只读取 query source 对应的 landmark distance：

1. 对每对 query 构建相对层差直方图。
2. 取直方图最大桶作为可通过 offset 对齐的 sharing affinity。
3. 贪心构造 `Q=32` batches，最大化 batch 内 affinity。
4. 最多执行 16 次跨 batch swap，修正贪心局部次优。

这里使用 offset-aware affinity，而不是 source 之间的最短路距离；后者与“同一 global step 访问相同 vertex”的目标不一致。

### Online offset

`online_offset_evaluator::evaluate()` 对每个 batch 生成 offset：

1. 以 phase length 预测产生 completion-aligned 初始解。
2. 使用 landmark 相对层差直方图计算 pairwise alignment score。
3. 从 zero-offset、completion-aligned 和 6 个确定性随机初值出发，执行有界 coordinate ascent。
4. offset 范围限制为 `[0, 16]`；query 启动后连续运行，不执行 pause/resume。

在线 evaluator 实测为 10.7-14.5 ms，占达标数据集 baseline 的 0.2%-1.3%。

## 3. 调用方式

库接口：

```cpp
auto index = puercgp::scheduling::landmark_phase_index::load(index_path);
puercgp::scheduling::online_offset_evaluator evaluator(index, 16);

auto batches = evaluator.plan_batches(sources, 32, 16);
for (const auto& query_ids : batches.query_ids) {
  std::vector<int> batch_sources;
  for (int query_id : query_ids) batch_sources.push_back(sources[query_id]);
  auto offsets = evaluator.evaluate(batch_sources);
  // Launch each query when global_step >= offsets.offsets[slot].
}
```

构建 benchmark：

```bash
cmake --build build --target bench_online_offset_n128 -j 8
```

运行或首次构建索引：

```bash
CUDA_VISIBLE_DEVICES=2 ./build/bench_online_offset_n128 \
  /home/zyl/data/csr_data/indochina \
  /home/zyl/data/csr_data/indochina/phase_index_sample_l64.bin \
  experiments/20260713_online_offset/final-indochina \
  --seed=42 --repeats=5 --landmarks=64 --batch-swaps=16 --rebuild-index
```

去掉 `--rebuild-index` 即复用索引。输出包括：

- `summary.csv`：GPU、evaluator、端到端时间和 sharing 指标。
- `metadata.csv`：图规模、索引规模和预处理时间。
- `landmark_batches.csv`：在线 batch assignment。
- `online_plans.csv`：每个 query 的 phase prediction 和 offset。

## 4. 结论与限制

- 在线 landmark evaluator 能逼近此前依赖完整 query trace 的 offset oracle，同时把决策成本控制在十余毫秒。
- 收益主要来自 union edge accesses 的减少，符合 phase alignment 提升 push sharing 的预期。
- 图上的可对齐结构决定收益上限，因此不能保证每张图达到 1.4x。
- 当前索引仍为 `O(VL)`；soc-sinaweibo 受索引体积限制只测试了 16 landmarks，affinity 分辨率不足。
- 当前 end-to-end benchmark 固定为 BFS、all-push、`N=128/Q=32`；接入 hybrid engine 和其他 query 数量需要单独验证。
