# OCGP: Out-of-GPU-Memory Concurrent Graph Processing

## Target Venue

VLDB (primary), SIGMOD (secondary), ICDE (fallback)

## Setting

单机消费级 GPU（如 RTX 4090, 24GB VRAM）上的批量离线图分析。

- 图规模超出 GPU 显存，数据存储在 Host Memory，按需传输到 GPU
- 用户同时提交多个图算法（BFS, SSSP, PR, CC, GNN SAG 等）
- 目标：**最大化批量查询的总吞吐**

### Setting 合理性

- 无需额外论证：消费级 GPU + 大图 = 大多数研究者和工程师的现实环境
- 类比 Subway (EuroSys'20)：其 motivation 仅为 "graphs grow, GPU memory is limited"
- 类比 Grapin (VLDB'25)：字节跳动场景 "thousands of graph analysis tasks daily"
- **TODO:** 补充具体 workload 场景描述（如社交网络分析批量查询、推荐系统混合负载），避免 reviewer 质疑 "谁会在单卡上批量提交多个图查询"

## Core Insight

现有系统将每个 query 视为独立任务，逐一处理（sequential per-query），忽略了 **inter-query data reuse** 机会。

我们识别出三层正交的 inter-query reuse：

### 1. I/O Reuse（传输层）

- **瓶颈**：Host-GPU PCIe 带宽
- **Insight**：多 query 的活跃子图存在大量重叠，加载一次子图可同时服务多个 query
- **对应技术**：Shared subgraph scheduling strategy
- **类比**：DB shared scan

### 2. Cache Reuse（缓存层）

- **瓶颈**：GPU 缓存（L1/L2/smem）容量有限
- **Insight**：多 query 的热点顶点数据高度重叠，可合并缓存
- **对应技术**：Cross-query neighbor deduplication + smem 联合缓存
- **类比**：DB buffer pool sharing

### 3. Graph Access Reuse（访问层）← 核心贡献

- **瓶颈**：图遍历的不规则访存（irregular memory access, cache miss 率高）
- **Insight**：**Decouple query state from graph structure**
  - 将 vertex state 从 scalar 扩展到 vector：`vertex.state = [s_BFS, s_SSSP, s_PR, s_CC]`
  - 一次图遍历（edge traversal）同时服务所有 query
  - 一次昂贵的 irregular graph access 的代价被 N 个 query 分摊
- **对应技术**：Vectorized query execution
- **类比**：DB vectorized execution / batch processing

### Graph Access Reuse 与 GNN Aggregation 的等价性

```
// GNN: vertex 的 M 维 embedding
vertex.embedding = [e_0, e_1, ..., e_{M-1}]   // M = embedding dimension

// OCGP: vertex 的 N 个 query 状态
vertex.state    = [s_q1, s_q2, ..., s_qN]     // N = query count
```

结构完全等价，因此已有的 GNN kernel 优化技术天然适用：
- 邻居去重 + smem 缓存
- Coalesced memory access
- Warp-level data sharing

**与 GNN 框架的关键区别（Reviewer 必问）：**

| 维度 | GNN Framework (DGL/PyG) | OCGP |
|------|------------------------|------|
| 负载类型 | GNN inference/training | 经典图算法 (BFS/SSSP/PR/CC) |
| 执行模型 | 固定迭代轮次 | 不规则收敛（不同算法收敛轮次差异大） |
| 内存管理 | In-GPU assumption | Out-of-GPU，显存受限 |
| Query 语义 | 单一 GNN task | 多 query 并发，动态增减 |

定位：**借鉴 GNN kernel 优化手段，但执行模型完全不同。**

### 显存开销分析

Vectorized state 的额外显存开销在实际场景下可控：
- 典型大规模图（如 70 亿边）：顶点数通常在 1 亿以内
- 1 亿顶点 × 1 query × 4B ≈ 0.4GB
- 1 亿顶点 × 24 query × 4B ≈ 9.6GB（在 24GB 卡上可接受）
- 结论：显存开销不是致命瓶颈，graph structure 数据（CSR）才是主要占用

## Heterogeneous Query Coexistence（异构查询共存策略）

异构 query 的收敛速度差异是核心技术挑战：

```
Iteration:  1  5  10  15  20  25  50  100
BFS:        ✓  ✓  ✓   ✓   ✓   ✗  -   -
SSSP:       ✓  ✓  ✓   ✓   ✓   ✓  ✗   -
PR:         ✓  ✓  ✓   ✓   ✓   ✓  ✓   ✓
```

### 策略 1：容忍退化（Baseline）

- 完全忽略 query 之间迭代的长尾分布
- 前期多 query 共享遍历获得收益，后期自然退化为 single-query processing
- 最差情况：后续迭代轮次退化为 single-query 的 processing
- **核心理念：** 前期 N 个 query 共享遍历的累积收益足以覆盖后期退化的开销

### 策略 2：动态补充（优化策略）

- 随着 query batch size 提升，吞吐受限于硬件资源会收敛，定义收敛时的 batch size 为 M
- 将测试场景扩展到 query batch size > M 的场景
- 一旦某个 query 计算完成，立即向 running batch 中补充新 query
- 预分配 M 个 slot，空 slot 填 dummy query（nop），完成后用新 query 覆盖
- 避免 SoA layout 下的运行时内存重分配

### 组合使用

两个策略不冲突：
1. 策略 1 保证基本盘（最差情况无额外开销）
2. 策略 2 作为进阶优化，维持 GPU 利用率不下降
3. Evaluation 中展示 "无优化 vs 策略 1 vs 策略 1+2" 的渐进收益

## Preliminary Validation（初步验证结果）

- **环境：** All-pull 执行模型，kernel-level 测试
- **结果：** Query batch size 在 32 以内时，整体耗时几乎相同
- **含义：** 图遍历的 irregular memory access 是绝对瓶颈，附加 query state 的 compute 开销几乎被完全掩盖
- **推论：** Graph Access Reuse 的收益接近线性扩展，Throughput ≈ N × throughput_single (N ≤ 32)

### 待补充的验证

- [ ] Push 模型验证（frontier merge 是否同样高效）
- [ ] 端到端验证（含 I/O 传输，确认 I/O 是否成为新瓶颈）
- [ ] 异质 query 组合验证（BFS+PR 混合，目前大概率是同质 query 测试）
- [ ] Batch size > 32 后的拐点分析（瓶颈从 memory bandwidth 转移到 compute bandwidth 的 transition）

## Paper Structure

### Title

**OCGP: Efficient Batch Graph Processing on Consumer GPUs via Inter-Query Data Reuse**

### 1. Introduction

- Setting: 消费级 GPU 批量离线图分析
- Goal: 最大化吞吐
- Insight: 三层 inter-query reuse
- 核心: decouple query state from graph structure
- Contributions（分层呈现）:

| 层次 | Contribution | 性质 |
|------|-------------|------|
| **核心** | Graph Access Reuse：vectorized query execution，将 query state 从 graph structure 中解耦 | 技术 insight |
| **系统** | I/O Reuse：多 query 联合子图调度 + 显存约束下的 cost-driven 传输 | 系统设计 |
| **优化** | 动态 query 补充机制：解决异构 query 收敛差异导致的资源闲置 | 调度策略 |
| **工程** | Kernel 级 cache reuse（邻居去重 + smem 联合缓存） | 实现优化 |

### 2. Motivation Study

- 2.1 Profile 单 query 系统跑 batch 的低效性
  - 对比：串行执行 N 个 query vs 理想吞吐
  - 展示 GPU 利用率低、带宽浪费
- 2.2 量化三层 reuse 的潜在收益
  - I/O: 多 query 活跃子图重叠率
  - Cache: 多 query 热点顶点重叠率
  - Graph Access: graph traversal 占总时间的比例
- 2.3 关键数据：graph access 开销 vs 计算 vs I/O 的占比
- **TODO:** 补充真实/synthetic workload 的重叠率分析，作为 motivation 的定量支撑

### 3. System Overview

- 架构图
- 三层 reuse 的执行流程
- Query lifecycle：提交 → 分组 → 向量化执行 → 结果拆分
- **动态补充机制**需在此处体现（query pool → 调度 → vectorized execution → 结果拆分 → 新 query 补充）

### 4. Graph Access Reuse: Vectorized Query Execution ← 核心贡献

- Query state vectorization (scalar → vector)
- 与 GNN aggregation 的等价性分析（强调区别而非等同）
- 统一的 push/pull 执行模型
  - Pull: `output[v] = AGG_{u ∈ N(v)} (input[u])`，多 query 共享 neighbor traversal
  - Push: frontier 合并，多 query 共享 scatter path
  - **TODO:** 讨论 Push/Pull 混合模式：BFS(push) 和 PR(pull) 在同一轮迭代中如何协调？强制统一为 pull 会丢失 BFS 的 frontier sparsity 优势
- 异构 query 共存策略
  - 基线策略（容忍退化）+ 动态补充策略
  - 运行时序图：展示两种策略下 GPU 利用率随迭代的变化
- Query state layout: AoS vs SoA 的 trade-off

### 5. I/O Reuse: Shared Subgraph Scheduling ← 系统贡献

- 多 query 活跃子图并集的计算
- Cost-driven 的子图加载调度
  - 优化目标：最小化总传输量
  - 约束：GPU 显存预算
- 传输与计算的流水线（prefetch + compute overlap）
- 与 Subway (SubCSR) 的区别：
  - Subway: 单 query 活跃边
  - OCGP: 多 query 联合活跃边 + 跨 query 缓存复用

### 6. Cache Reuse: Cross-Query Data Caching ← Kernel 贡献

- 邻居去重 + smem 缓存（已有工作基础）
  - Graph reorder 后同 block 内多 query 的邻居集重叠
  - 去重后的唯一邻居 embedding 缓存到 smem
  - 索引映射：每个 warp 通过预计算索引从 smem 读取
- 跨 query 热点顶点联合缓存
  - L1/L2 级别的 cache-aware 数据布局
  - Bank conflict 感知的 smem 布局
- **注意：** Kernel 优化在 GNN 文献中已有大量讨论，篇幅应控制，重点放在 **跨 query 去重带来的额外收益**

### 7. Experimental Evaluation

- 7.1 Setup
  - 硬件: RTX 4090 (24GB), 可选 A100 (80GB) 对比
  - 数据集: SNAP, SuiteSparse, 合成图 (6-8 个)
  - 查询负载: 不同组合 (BFS+SSSP, PR+CC, 全混合, GNN SAG)
- 7.2 Overall Performance
  - Baselines:
    - Subway 串行执行（out-of-GPU + single query 的代表）
    - EGraph 并发执行（concurrent + in-GPU 的代表）
    - CPU 多查询并发 (Galois/Ligare)
    - ~~Grapin~~：本质是动态图工作，与 OCGP 不在同一 scope，移至 Related Work
  - 指标: 总吞吐量、加速比、GPU 显存占用
- 7.3 Ablation Study
  - 逐层启用：None → Graph Access Reuse → +I/O Reuse → +Cache Reuse → +Dynamic Replenishment
  - 展示每层的独立贡献百分比
  - 不同 query 组合下的收益分析
- 7.4 Heterogeneous Query Analysis（回应 Reviewer 最大 concern）
  - 同质组合：N×BFS（upper bound）
  - 异质组合：BFS+SSSP+PR+CC（realistic）
  - 极端异质：BFS+PR（收敛差异最大的 pair）
  - 动态补充策略在各场景下的收益
- 7.5 Break-even Analysis
  - 在什么条件下 OCGP 反而比串行慢？（query 数量太少、图太小 fit in GPU）
  - 给出 break-even point，展示方法的适用边界
- 7.6 Scalability
  - Query 数量扩展 (2→32+)
  - Batch size 拐点分析（>32 后瓶颈从 memory bandwidth 转移到 compute bandwidth）
  - 图规模扩展
  - GPU 显存限制变化

### 8. Related Work

- Out-of-GPU graph processing: Subway, EMOGI, Grapin (dynamic graph, 不同 scope), Liberator, CoreGraph
- Concurrent/multi-query graph processing: TuGraph, EGraph
- GPU graph kernel optimization: GE-SpMM, GNNAdvisor, GunRock
- Vectorized execution in DB

### 9. Conclusion

## Competitive Landscape

| Work | Venue | Single/Concurrent | In/Out-of-GPU | Reuse Type | 与 OCGP 关系 |
|------|-------|-------------------|---------------|------------|-------------|
| Subway | EuroSys'20 | Single | Out-of-GPU | Active edge filtering | I/O 层的 single-query baseline |
| EMOGI | VLDB | Single | Out-of-GPU | Zero-copy access | 另一种 out-of-GPU 策略，不涉及并发 |
| Grapin | VLDB'25 | Single | Out-of-GPU | Hot subgraph + incremental | 动态图 scope，非直接竞品，Related Work 中提及 |
| Liberator | TPDS'23 | Single | Out-of-GPU | Data reuse framework | 同 query 内 reuse，非跨 query |
| CoreGraph | EuroSys'24 | Single | Out-of-GPU | Edge centrality caching | 同 query 跨迭代复用，非跨 query |
| EGraph | TKDE'23 | Concurrent | In-GPU | Shared subgraph transfer | **最接近竞品**：concurrent 但不处理 out-of-GPU |
| GE-SpMM | SC'20 | Single | In-GPU | Row caching + warp merging | Kernel 优化参考 |
| **OCGP (ours)** | **Target: VLDB** | **Concurrent** | **Out-of-GPU** | **Three-layer reuse** | **首次结合并发与 out-of-GPU，三层 reuse** |

## Reviewer Perspective（审稿人角度的风险与建议）

### 预期 Reviewer 质疑及应对

| # | Reviewer 可能的质疑 | 应对策略 | 优先级 |
|---|-------------------|---------|--------|
| R1 | "与 GNN 框架有什么区别？为什么不直接用 DGL/PyG？" | 执行模型差异表（固定轮次 vs 不规则收敛、in-GPU vs out-of-GPU、单 task vs 多 query 并发） | P0 |
| R2 | "异构 query 收敛差异如何处理？BFS 20 轮 vs PR 100 轮" | 双策略组合：容忍退化（基线）+ 动态补充（优化），ablation 展示渐进收益 | P0 |
| R3 | "Push/Pull 混合模式怎么处理？" | 明确设计选择 + trade-off 分析，若 push 效果差则诚实报告并限定适用场景 | P1 |
| R4 | "动态补充的 slot 管理 overhead 如何？" | 预分配 M 个 slot + dummy query 填充，避免运行时 realloc，量化 overhead | P1 |
| R5 | "显存受限场景下 vectorized state 是否加剧压力？" | 1 亿顶点 × 24 query × 4B ≈ 9.6GB，在 24GB 卡上可控 | 已解决 |
| R6 | "谁会在单卡上批量提交多个图查询？场景是否真实？" | 补充 workload 特征分析（社交网络批量分析、推荐系统混合负载） | P1 |
| R7 | "端到端性能是否被 I/O 瓶颈掩盖？" | 端到端实验 + ablation，展示三层 reuse 各自的贡献 | P0 |
| R8 | "OCGP 在什么条件下反而比串行慢？" | Break-even analysis，给出适用边界 | P2 |

### Action Items（按优先级排序）

| 优先级 | 行动 | 目的 |
|--------|------|------|
| **P0** | 实现 vectorized execution 原型，验证 BFS+PR 混合执行的加速比 | 验证核心 idea 的基本可行性 |
| **P0** | 测量异质 query 组合下 multi-query 受益迭代占比 | 量化容忍退化策略的 baseline 收益 |
| **P0** | 端到端验证（含 I/O），确认 kernel 收益不被 I/O 淹没 | 确认三层 reuse 都必要 |
| **P1** | 实现 push 模型验证 | 评估 generality |
| **P1** | 实现动态补充策略并测量收益增量 | 验证优化策略的额外贡献 |
| **P1** | 补充真实/synthetic workload 的重叠率分析 | 支撑 Motivation |
| **P2** | Push/Pull 混合策略的设计决策 | 补充技术细节 |
| **P2** | Break-even analysis | 防御 reviewer 攻击 |

## Key Messages (一句话总结)

> OCGP identifies three orthogonal layers of inter-query data reuse in batch graph processing on resource-constrained GPUs, and proposes a vectorized query execution model that decouples query state from graph structure, enabling one irregular graph traversal to serve multiple queries simultaneously.
