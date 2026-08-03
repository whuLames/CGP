# 投稿前实验清单

按阻塞论文主张的优先级排列。

1. **固定策略端到端主表**：N=256,Q=64；BFS/SSSP/SSWP；所有核心图；GraphWeft planner 与 selector 时间计入；同一 commit、2 warmups、至少 7 repeats、median 与置信区间。
2. **iBFS + Glign 区分实验**：同一 GraphWeft engine 内实现 iBFS-like exact shared BFS、Glign-compatible scalar-landmark schedule、二者组合、GraphWeft 完整设计，避免跨语言/跨硬件混杂。
3. **selector 实现与验证**：always VertexShare、always AlignedGather、现有 threshold、calibrated model、per-iteration oracle；按 graph 划分 train/test；报告 regret、模型开销和切换次数。
4. **claim–mechanism 证据链**：记录 E_ind、E_union、E_pair、active pairs；采集 DRAM bytes、L2 sectors、transactions/useful value、long scoreboard；验证 edge-count 降低是否转化为 bytes 与 time 降低。
5. **访问策略与状态组织 ablation**：V×Q vs Q×V、VertexShare vs AlignedGather、direct vs shared-memory tiled aggregation，以及实现层映射；Q=1/2/4/8/16/32/64，多 value width。
6. **planner ablation**：input/random order、degree grouping、Glign-compatible scalar landmark、pairwise histograms、+swaps、zero/completion/coordinate offsets；至少 10 个 source seeds，报告 predicted 与 realized overlap。
7. **算法边界**：为 SSSP/SSWP 设计 phase summary 并验证；若不能稳定获益，明确把 scheduling claim 缩到 BFS，把 multi-algorithm claim 限定为共享计算模型与执行策略。
8. **scalability**：N=64–1024、Q=1–64；加入 roadNet-CA 与可控 synthetic graphs；至少 V100 和一张较新 GPU；报告 peak GPU memory 和 OOM 边界。
9. **preprocessing/overhead**：index build/load/size、planner 分阶段时间、scan/compact/counter 时间、break-even query windows。
10. **正确性与 artifact**：随机小图、disconnected、duplicate sources、unreachable、weighted ties、offset 边界、N%Q≠0；为每个 paper number 建立原始文件 manifest。
