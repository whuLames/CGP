# Online Runner: N=256, Q=64

日期：2026-07-19  
平台：NVIDIA Tesla V100-SXM2-32GB  
模式：homogeneous BFS/SSSP/SSWP，hybrid push/pull

## 1. 实验设置

- 数据集：`cit-Patents`、`soc-orkut`、`soc-twitter`、`soc-sinaweibo`。
- 每个算法执行 `N=256` 个 query，每批 `Q=64`，共 4 个 batch。
- 同一数据集的三个算法复用完全相同的 256 个 source。
- 2 次 warmup、5 次正式重复，表中报告中位数。
- baseline 与 online 在奇偶 repeat 中交换执行顺序。
- `GPU+Eval` 为所有 batch 的 CUDA event 时间之和，加在线 evaluator 时间；不包含可离线复用的 graph index 构建。
- `Wall+Eval` 为 runner 实测 host wall time，加在线 evaluator 时间。
- online 使用 `auto` 策略，`max_offset=16`，`batch_swaps=0`。

`auto` 当前采用保守门控：BFS 的平均预测 phase length 不小于 16 时启用 online batching + offset；否则保留 sequential batching 和 zero offset。SSSP/SSWP 暂时旁路，因为现有 landmark index 预测的是无权 BFS phase，强制用于带权算法在 pilot 中没有稳定收益。

## 2. 性能结果

| 数据集 | 算法 | Online 策略 | Baseline GPU+Eval (ms) | Online GPU+Eval (ms) | 加速比 | Wall 加速比 | Evaluator (ms) |
|---|---|---|---:|---:|---:|---:|---:|
| cit-Patents | BFS | full | 1102.670 | 960.697 | **1.1478x** | **1.1297x** | 26.623 |
| cit-Patents | SSSP | disabled | 1508.920 | 1507.740 | 1.0008x | 0.9928x | 0.017 |
| cit-Patents | SSWP | disabled | 1505.940 | 1512.190 | 0.9959x | 0.9925x | 0.018 |
| soc-orkut | BFS | disabled | 1502.570 | 1503.330 | 0.9995x | 0.9965x | 0.016 |
| soc-orkut | SSSP | disabled | 3013.990 | 3023.420 | 0.9969x | 0.9960x | 0.017 |
| soc-orkut | SSWP | disabled | 2891.090 | 2889.370 | 1.0006x | 0.9994x | 0.018 |
| soc-sinaweibo | BFS | disabled | 5265.450 | 5265.590 | 1.0000x | 0.9997x | 0.111 |
| soc-sinaweibo | SSSP | disabled | 7102.380 | 7095.460 | 1.0010x | 0.9989x | 0.017 |
| soc-sinaweibo | SSWP | disabled | 6507.330 | 6501.320 | 1.0009x | 0.9997x | 0.017 |
| soc-twitter | BFS | full | 11486.600 | 10264.700 | **1.1190x** | **1.1129x** | 15.103 |
| soc-twitter | SSSP | disabled | 8283.450 | 8204.570 | 1.0096x | 1.0052x | 0.016 |
| soc-twitter | SSWP | disabled | 6316.690 | 6237.320 | 1.0127x | 1.0080x | 0.017 |

几何平均：

| 算法 | GPU+Eval 加速比 | Wall+Eval 加速比 |
|---|---:|---:|
| BFS | **1.0644x** | **1.0579x** |
| SSSP | 1.0021x | 0.9982x |
| SSWP | 1.0025x | 0.9999x |
| 全部 12 组 | **1.0226x** | **1.0183x** |

结论：当前在线模型对 BFS 有选择性收益，在 `cit-Patents` 和 `soc-twitter` 上实际启用后分别取得 14.78% 和 11.90% 的 `GPU+Eval` 提升；低 phase-length 图由门控避免负收益。SSSP/SSWP 已支持在线 schedule 的正确执行，但还没有算法相关的 phase estimator，因此本轮只验证安全旁路，不应把约 1% 的波动解释为调度收益。

## 3. 实现内容

- `slot_start_schedule` 将 query 启动 offset 从固定编译期配置改为运行时 `Q=1..64`。
- `online_runner` 封装 sequential、batch-only、offset-only 和 full execution plan。
- frontier engine 在 offset 到达时延迟初始化 source，并只调度已启动且未完成的 slot。
- push/pull kernel 接收 runtime active-slot mask；pull 路径同步维护 active union。
- `run_scheduled()` 提供 host 侧统一入口，benchmark 支持 BFS、SSSP、SSWP 和 push/pull/hybrid。
- scheduled push 不再维护仅 pull 需要的 active union，zero-offset 路径不引入额外原子开销。
- 修复 BFS hybrid 从 pull 切回 push 时可能遗漏后续距离修正的问题；BFS 一旦进入 pull，后续保持 pull 到收敛。正式 baseline 与 online 均使用修复后的相同策略。

## 4. 正确性与限制

- 四图 12 组实验的 60 次 online run 均与 baseline 的 256 个 query fingerprint 一致。
- 独立小图回归覆盖 BFS/SSSP/SSWP、push/pull/hybrid、`Q=1/32/64` 共 27 组，全部逐值匹配。
- 现有 estimator 的 landmark distance 与 affinity 都基于无权 BFS。要让 SSSP/SSWP 真正获得 online scheduling 收益，需要分别定义 weighted shortest-path phase 和 widest-path phase，而不是复用 BFS offset。
- 原始结果位于各 `final-<dataset>-<algorithm>/runs.csv`；汇总数据见 `summary.csv`。
