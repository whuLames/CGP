# PuerCGP 投稿导向待办清单

更新时间：2026-07-18

## 1. 优先级定义

- **P0 / submission blocker**：不完成就不能形成可信的 ICDE/VLDB/SIGMOD 投稿。
- **P1 / strong-paper requirement**：可以先写 draft，但缺失会显著降低说服力。
- **P2 / optional**：只在主线结果稳定后投入。

总体原则：先解决 novelty 和 external baseline，再补功能。当前最危险的情况不是少一个 kernel，而是投入大量实验后才发现核心 claim 已被 iBFS/Glign 覆盖。

## 2. Phase 0：Novelty 与 scope 决策

### P0-0.1 完成 iBFS 机制级对照

- [ ] 阅读 iBFS joint traversal、top-down、bottom-up、GroupBy、bitwise 和 direction switching 的完整实现/论文。
- [ ] 对照 `fused_pull_simple_kernel` 与 iBFS bottom-up 的 thread mapping、status layout、early termination 和 memory transaction。
- [ ] 对照 shared push 与 iBFS joint frontier/JFQ。
- [ ] 对照当前 selective batching 与 iBFS 两条 outdegree GroupBy rules。
- [ ] 形成一张“相同点、不同点、预期硬件影响、验证实验”的表。
- [ ] 在相同 V100、相同 CSR orientation、相同 source lists 下跑 iBFS correctness 和性能。

验收标准：能用两三句话明确回答“PuerCGP 的 physical execution 为什么不是 iBFS 的现代 CUDA 重写”。如果答案只剩工程重构，应立即缩小或改变贡献。

### P0-0.2 完成 Glign 调度级对照

- [ ] 实现 Glign closest-high-degree arrival estimator。
- [ ] 实现 Glign affinity-oriented batching 和 delayed start baseline。
- [ ] 对照 index：top-K high-degree distances vs 当前 farthest/degree-sampled landmarks。
- [ ] 对照 objective：heavy iteration arrival vs relative-level histogram vs degree-weighted union-edge reduction。
- [ ] 在同一 query window 上报告 schedule、union edges、planner time 和 end-to-end time。

验收标准：证明当前 online evaluator 在至少一个 GPU-specific objective 上优于 Glign heuristic；否则把 offset 降级为已有思想在 GPU engine 中的适配，不作为主要 novelty。

### P0-0.3 冻结论文 scope

- [ ] 决定主文是 `concurrent BFS` 还是 `source-based traversal queries (BFS+SSSP)`。
- [ ] 推荐选择 BFS+SSSP；只有 SSSP 性能无法站住时才缩成 BFS。
- [ ] 决定 heterogeneous BFS+SSSP 是否进入主贡献。
- [ ] WCC 固定为 appendix/功能展示。
- [ ] 明确 static、single-GPU、in-memory、closed-window 是 scope。
- [ ] 将 replenish、pause/resume、SM partition、Q*V 从主设计中移除。

产出：`claim_scope.md`，包含 title、one-sentence thesis、3 个 contributions、non-goals 和禁止使用的 first claims。

## 3. Phase 1：冻结可复现实验基线

### P0-1.1 建立 paper evaluation 分支和版本

- [ ] 先处理当前工作树中的已有修改，避免把未完成在线 evaluator 与论文基线混在一起。
- [ ] 从确定 commit 创建 paper-eval 分支并记录 commit hash。
- [ ] 所有 paper runner 输出 git hash、GPU、driver、CUDA、编译 flags 和 options。
- [ ] 清理/更新过期 README，特别是 bitmap pull、GE-SpMM、dense engine 和已删除路径的描述。
- [ ] 固化 CMake preset 或单一 build script。

### P0-1.2 统一计时边界

- [ ] 定义 `kernel-only`、`GPU-engine`、`online end-to-end` 三种计时。
- [ ] 主结论使用 online end-to-end：planner、init、iteration、postprocess 全部包含。
- [ ] graph load 和 reusable index build 单独报告，不进入每 batch latency。
- [ ] result D2H copy 是否计入必须统一；建议主图给不含完整结果 materialization 的 engine time，并另报含 copy 的 service time。
- [ ] 所有 baseline 使用相同边界。

### P0-1.3 统一 query sets

- [ ] 生成 versioned query manifest：dataset、seed、source id、algorithm、weight config。
- [ ] 每个 dataset/config 至少 5 个 source seeds。
- [ ] source 必须互不相同；对 disconnected graph 记录可达顶点比例。
- [ ] 为 uniform、degree-stratified、hotspot/localized 和 application-derived workload 分别生成 manifest。
- [ ] 每个系统从同一 manifest 读取，禁止各 baseline 内部重新随机采样。

### P0-1.4 正确性基线

- [ ] BFS：逐 query 比较 visited count、distance checksum，并对中小图做 full distance compare。
- [ ] SSSP：逐 query full/reference compare，定义 absolute/relative epsilon 和 infinity。
- [ ] 大图生成一次可复用 CPU reference 文件，避免 validator 跳过 `V>5M`。
- [ ] push、pull、hybrid、online schedule 使用相同 reference。
- [ ] Q=1/4/8/16/31/32/33/48/63/64 覆盖 mask 和 warp 边界。
- [ ] directed graph 同时验证 outgoing CSR push 与 incoming CSR pull。

验收标准：最终 paper configurations 全部 `mismatches=0`；SSSP 误差在预先定义阈值内，而不是遇到结果后调整。

## 4. Phase 2：完成论文版执行引擎

### P0-2.1 将 online planner 接入正式 engine

- [ ] 提供正式 API：query window -> online batch plans -> engine execution -> aggregate result。
- [ ] 不再依赖 `bench_online_offset_n128.cu` 中的 all-push 专用 wrapper 作为主结果。
- [ ] 支持 `N>Q` 的多个 batch，固定 source/query id 映射。
- [ ] planner time 计入 online end-to-end。
- [ ] offset waiting 计入 per-query latency。
- [ ] 输出 batch assignment、offset、每轮 mode、sharing metrics 和 completion time。

验收标准：同一入口可运行 `FIFO hybrid`、`online hybrid` 和 `offline oracle hybrid`，无需 benchmark 内复制 engine 逻辑。

### P0-2.2 建立统一 push/pull cost model

- [ ] 为每轮采集 `unique_vertices`、`E_union`、`E_pair`、active pairs 和 mask density。
- [ ] pull 采集/估计 unvisited pairs、neighbor probes、active Q 和 postprocess work。
- [ ] 建立 `C_push`/`C_pull` 的最小线性模型或 lookup model。
- [ ] 模型系数使用独立 calibration set；不能在测试 query 上选择最优阈值。
- [ ] 采用 leave-one-graph-out 或 train/test graph split 验证泛化。
- [ ] 记录 predicted mode、oracle mode、误选代价和 decision overhead。
- [ ] 将该模型接入 homogeneous engine。
- [ ] 如果 heterogeneous 是主 scope，也接入 heterogeneous engine；当前后者仍主要使用 `current_unique` threshold。

验收标准：相对 virtual-edge threshold 明显缩小 per-iteration oracle gap，并且自身决策开销不抵消收益。

### P0-2.3 完成 BFS/SSSP operator matrix

- [ ] shared push 支持并验证 BFS 和 SSSP。
- [ ] coalesced pull 支持并验证 BFS 和 SSSP。
- [ ] hybrid selector 对两种算法分别校准，不假设相同 cost coefficient。
- [ ] mixed BFS/SSSP 若进入主文，验证 runtime tag 分支和统一 `float` value 的代价。
- [ ] 明确 weighted graph 的 incoming edge weight layout。

### P1-2.4 减少 host-side iteration overhead

- [ ] profile 每轮 host/device count copy 和 stream synchronization。
- [ ] 判断 frontier metrics 是否导致过多 host sync。
- [ ] 尝试 device-side decision 或异步双缓冲 metrics，仅在 measured breakdown 显著时实现。
- [ ] 保持优化前后 mode decisions 可追踪。

### P1-2.5 优化/约束 online index

- [ ] 测试 `L=8/16/32/64/128` 的精度、空间和构建时间。
- [ ] 增加 landmark distance 压缩方案评估，例如按图直径选择 uint8/uint16 或 unreachable bitmap。
- [ ] 测试只存 source candidate 的稀疏/分层摘要是否足够。
- [ ] 修复或明确距离超过 `uint16_t` 的行为。
- [ ] 明确 directed graph 是否从 reverse graph 构建。
- [ ] 索引文件加入 graph fingerprint/version，避免加载错误图的 index。

## 5. Phase 3：External baseline

### P0-3.1 GPU baselines

- [ ] Gunrock single-query sequential batch。
- [ ] Gunrock/自有 kernel naive multi-stream concurrency。
- [ ] iBFS joint traversal + 原始 GroupBy。
- [ ] GE-SpMM/TCRGraph pull operator。
- [ ] PuerCGP all-push、all-pull、hybrid、hybrid+online planner。
- [ ] 对每个 baseline 记录 unsupported/OOM，而不是静默跳过。

iBFS 特别检查：

- [ ] 适配现代 CUDA 时不改变核心算法。
- [ ] 记录它要求的 batch/group size 和当前调整。
- [ ] 输入图方向、去重、自环和边数与 PuerCGP 一致。
- [ ] 确认 timing 是否包含 status initialization/frontier generation。
- [ ] 对结果做 independent correctness check。

### P0-3.2 Scheduler baselines

- [ ] FIFO + zero offset。
- [ ] random grouping + zero offset。
- [ ] iBFS GroupBy。
- [ ] Glign closest-HV batching + delayed start。
- [ ] current landmark histogram planner。
- [ ] exact/full-trace or sampled offline oracle，明确它只是上界。

### P1-3.3 CPU systems

- [ ] ForkGraph。
- [ ] Glign 原系统。
- [ ] Krill，如果 artifact 可构建。
- [ ] KGraph，如果选用 SSSP/memoization 对比且 workload 语义一致。
- [ ] 报告 CPU 线程、NUMA、内存和编译配置。

CPU 比较只用于展示系统 landscape。论文的主要 speedup 必须由同 GPU direct baselines 支撑。

## 6. Phase 4：Motivation 和 mechanism experiments

### P0-4.1 证明 workload 的 sharing opportunity

- [ ] 每轮记录 per-query frontier、union frontier 和 degree-weighted work。
- [ ] 报告 `E_ind/E_union` 分布，而不是只报 vertex overlap。
- [ ] 分图、Q、source distribution、algorithm 展示。
- [ ] 至少 5 个 source seeds，报告 median/boxplot。
- [ ] 找到 high-sharing、medium-sharing、no-sharing 代表 case。

### P0-4.2 证明 memory bottleneck

- [ ] 使用 Nsight Compute 收集 memory workload analysis。
- [ ] 验证当前 V100/Nsight 版本实际可用的 metric 名称。
- [ ] 至少收集 DRAM read/write bytes、L2 sectors/hit、global sectors/request、long-scoreboard stall、achieved bandwidth。
- [ ] 对 sequential、naive concurrent、shared push、coalesced pull 使用完全相同 query。
- [ ] 区分 bandwidth saturation 和 latency/stall domination。

### P0-4.3 证明 shared push mechanism

- [ ] `E_union` reduction vs runtime speedup scatter/correlation。
- [ ] shared frontier vs independent frontier 的 adjacency bytes/transactions。
- [ ] atomic attempt/success 和 mask density。
- [ ] degree skew sensitivity。
- [ ] 与 iBFS joint top-down 的 counters 和 runtime 对比。

### P0-4.4 证明 query-parallel pull mechanism

- [ ] `V*Q` vs `Q*V` 最终代码重跑。
- [ ] query-parallel vs query-serial/vertex-parallel mapping。
- [ ] Q=8/16/32/48/64 的 coalescing efficiency。
- [ ] simple vs smem pull。
- [ ] 与 iBFS bottom-up 的 neighbor probes、transactions 和 kernel time 对比。
- [ ] 单独报告 scan/compact postprocess。

### P0-4.5 证明 hybrid cost selection

- [ ] 逐 iteration 同时测量 push-only 和 pull-only oracle cost，构建 ground truth。
- [ ] 比较 frontier-size、virtual-edge、proposed model。
- [ ] 报告 mode accuracy、weighted regret、额外 pull rounds 和 end-to-end time。
- [ ] 展示至少两个传统 threshold 误判而 cost model 正确的 case。

### P0-4.6 证明 online planner

- [ ] planner score 与 actual `E_union` saving 的相关性。
- [ ] planner score 与 actual runtime saving 的相关性。
- [ ] online vs Glign vs iBFS GroupBy vs oracle。
- [ ] evaluator latency 包含在最终端到端时间。
- [ ] planner 在 no-sharing workload 上不会显著回退。
- [ ] offset 对 p50/p95/p99 latency 的影响。

## 7. Phase 5：完整性能矩阵

### P0-5.1 Dataset matrix

- [ ] cit-Patents。
- [ ] indochina。
- [ ] soc-LiveJournal1。
- [ ] soc-orkut。
- [ ] soc-twitter。
- [ ] soc-sinaweibo。
- [ ] roadNet-CA 作为 long-diameter robustness case。
- [ ] 至少一组 RMAT/Kronecker controlled graph。
- [ ] 记录 graph statistics 和转换脚本。

### P0-5.2 Query scale matrix

- [ ] `Q=8,16,32,64`。
- [ ] `N=Q,4Q,16Q`。
- [ ] 每项至少 5 个 source seeds。
- [ ] uniform、degree-stratified、localized/hotspot。
- [ ] BFS 和 SSSP。

完整笛卡尔积过大时，先定义 core matrix：

```text
6 main graphs x 2 algorithms x Q={16,32,64} x N={Q,4Q} x 5 seeds
```

其余 sensitivity 选择代表图，但选择规则必须在看结果前确定。

### P0-5.3 End-to-end metrics

- [ ] GPU engine time。
- [ ] online planner + GPU end-to-end time。
- [ ] query/s。
- [ ] per-query p50/p95/p99 latency。
- [ ] peak GPU memory。
- [ ] correctness status。
- [ ] geometric mean、min、max 和置信区间。

### P1-5.4 第二 GPU 架构

- [ ] 在至少一张 Ampere/Ada/Hopper GPU 重跑 core matrix 的代表子集。
- [ ] 重新校准或直接迁移 cost model，测试 portability。
- [ ] 比较 memory system 差异是否改变 push/pull crossover。

### P1-5.5 真正 online arrival workload

只有论文要使用“online service”叙事时执行：

- [ ] Poisson/bursty arrivals。
- [ ] bounded batching window。
- [ ] queue wait + execution latency。
- [ ] fairness/starvation bound。
- [ ] throughput-latency trade-off curve。

否则主文明确是 online planning of a query window，不扩张 claim。

## 8. Phase 6：消融与边界

### P0-6.1 必需消融

- [ ] independent -> shared push。
- [ ] shared push -> +coalesced pull。
- [ ] +hybrid selector。
- [ ] +selective batching。
- [ ] +fixed offset。
- [ ] planner overhead included/excluded breakdown。

### P0-6.2 必需 sensitivity

- [ ] landmark count。
- [ ] max offset。
- [ ] batch swap count。
- [ ] pull cost coefficients/threshold。
- [ ] Q occupancy 和 active slot continuity。
- [ ] graph degree skew/diameter。

### P1-6.3 Negative/limit cases

- [ ] roadNet-CA 的 low sharing/long diameter。
- [ ] adversarial source sets。
- [ ] Q 很小导致 pull coalescing 不充分。
- [ ] index memory 受限的大图。
- [ ] planner misprediction case study。

不要隐藏负结果。主文用一小节说明何时 PuerCGP 退化到 all-push/all-pull/FIFO，以及 optimizer 如何限制回退。

## 9. Phase 7：写作与图表

### P0-7.1 Introduction 证据

- [ ] 一张真实/application workload 图。
- [ ] 一张 naive concurrency memory profiling 图。
- [ ] 一张 topology sharing + query coalescing opportunity 图。
- [ ] contributions 不使用被 iBFS/Glign 覆盖的 first claim。

### P0-7.2 Method figures

- [ ] architecture。
- [ ] shared frontier/push dataflow。
- [ ] pull warp-to-`V*Q` mapping。
- [ ] batch/offset/operator optimizer flow。
- [ ] 所有符号在第一次使用时定义。

### P0-7.3 Main evaluation figures

- [ ] overall throughput。
- [ ] latency/scaling。
- [ ] memory counters。
- [ ] ablation。
- [ ] online-vs-oracle gap。
- [ ] preprocessing/planner overhead table。

### P1-7.4 Related work delta table

列：

```text
system | hardware | algorithms | topology sharing | query-state coalescing
       | push/pull | batch planning | temporal alignment | online cost model
```

至少包括 iBFS、CGraph、GraphM、ForkGraph、Krill、Glign、LCCG、KGraph、PuerCGP。

## 10. Phase 8：Artifact

### P0-8.1 Reproducibility

- [ ] 一条命令下载/转换每张公开图。
- [ ] 固化 graph checksums。
- [ ] 提交所有 query manifests。
- [ ] 一条命令运行 correctness core matrix。
- [ ] 一条命令运行 paper core performance matrix。
- [ ] 每个 figure/table 有生成脚本。
- [ ] raw CSV 不只存在 `.gitignore` 文件中。
- [ ] 记录失败/OOM/timeout。

### P0-8.2 Baseline packaging

- [ ] baseline license 和原始 commit。
- [ ] patch 单独保存，不能静默修改算法。
- [ ] build/run commands。
- [ ] timing parser 和 correctness adapter。
- [ ] iBFS/Glign/ForkGraph source sets 对齐说明。

### P1-8.3 Documentation

- [ ] README 更新为真实支持范围。
- [ ] system API example。
- [ ] index build/load 文档。
- [ ] memory requirements calculator。
- [ ] known limitations。

## 11. Claim-Evidence 矩阵

| 计划 claim | 必须具备的证据 | 当前状态 |
|---|---|---|
| Concurrent query dimension exposes topology sharing | 多图、多 seed 的 `E_ind/E_union` | 有单 seed trace，未完成 |
| Shared push reduces topology access | adjacency traffic/counter + iBFS direct comparison | 有 runtime，counter 不足 |
| Pull improves query-state access efficiency | mapping ablation + sectors/request + iBFS bottom-up | layout microbench 有，direct/counter 不足 |
| Hybrid selects lower memory-cost plan | threshold/model/oracle gap | virtual-edge heuristic 有，论文模型不足 |
| Online planning increases sharing cheaply | online-vs-Glign/iBFS/oracle + included overhead | 6 图 BFS preliminary，有明显缺口 |
| System is general beyond BFS | SSSP correctness/performance matrix | 功能存在，论文证据不足 |
| End-to-end beats state of the art | final-code external baseline core matrix | 旧结果混合，必须重跑 |
| Preprocessing is practical | build time/space/sensitivity | 有 3 图初步数字，缺 sensitivity |

## 12. 阶段性 Go/No-Go 门

### Gate A：Novelty gate

通过条件：

- 能明确区分 PuerCGP 与 iBFS shared top-down/bottom-up。
- online planner 相对 Glign 不是仅换 landmark selection。
- 至少一个核心 operator/optimizer 有独立、稳定的机制收益。

未通过：缩小为一个强 BFS operator paper，或重新寻找核心创新；不要继续堆系统功能。

### Gate B：Generality gate

通过条件：BFS 和 SSSP 均正确，且 SSSP 不是大范围回退。

未通过：标题和 claim 限定为 concurrent BFS，不再使用 general CGAQ engine。

### Gate C：Online gate

通过条件：planner overhead included 后，相对 FIFO/Glign/iBFS scheduler 在多图、多 seed 上有统计显著收益，并控制 tail latency。

未通过：将 online planner 作为 optional optimization；论文主线保留 physical execution 和 hybrid selection。

### Gate D：Submission gate

最低条件：

- final-code external baselines 完整。
- 至少 6 图、2 算法、多个 Q/N/seeds。
- strongest GPU baseline 上有统计显著的 geometric-mean 收益。
- memory counters 能解释收益。
- 没有未说明的大范围回退。
- artifact 可复现。

## 13. 推荐执行顺序

1. **先做 Phase 0**：iBFS/Glign novelty delta 和 scope。
2. **再做 Phase 1/3**：冻结 query、timing、correctness 和 external baselines。
3. **随后做 Phase 2**：online planner 正式接入和 cost model。
4. **做 Phase 4**：先取得 mechanism evidence，再跑完整大矩阵。
5. **做 Phase 5/6**：完整性能、消融和边界。
6. **并行推进 Phase 7/8**：每个实验完成后立即生成 paper figure 和 artifact script。

不推荐先做第二 GPU、open-loop arrivals、replenish 或更多算法。它们不会解决当前最关键的 novelty 与 direct-baseline 风险。
