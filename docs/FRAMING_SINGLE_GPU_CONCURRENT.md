# OCGP: Single-GPU Concurrent Graph Processing Framing

## One-Sentence Thesis

OCGP is a single-GPU execution engine for concurrent graph analytics that treats multiple graph queries over the same resident graph as a shared physical workload, amortizing irregular graph access across queries while adapting to heterogeneous algorithms, active sets, and convergence behavior.

## Target Venue

- Primary: VLDB
- Secondary: SIGMOD
- Fallback: ICDE

This version intentionally does not rely on out-of-GPU execution as the main contribution. The graph is assumed to fit in GPU memory. The paper must therefore compete on multi-query execution, scheduling, and system-level behavior, not on PCIe transfer reduction.

## Problem Setting

The target setting is batched or concurrent graph analytics over one large graph resident in a single GPU.

- The graph structure fits in GPU memory.
- Many graph analytics queries run over the same graph.
- Queries may arrive as a batch or as a stream.
- Queries can be homogeneous or heterogeneous: BFS, SSSP, PageRank, connected components, personalized PageRank, k-hop neighborhood queries, graph sampling, and GNN-style aggregation.
- The objective is high aggregate throughput while preserving acceptable per-query latency and avoiding pathological slowdown for individual queries.

The motivating workload should be repeated analytics over a shared graph, not just "many users submit queries." Stronger examples include:

- many-source BFS or SSSP for offline feature generation;
- batched personalized PageRank for recommendation and ranking;
- repeated ego-network analytics for fraud, security, or social graph analysis;
- graph sampling and neighborhood expansion for GNN preprocessing;
- interactive or near-interactive graph exploration where many queries run over a common graph snapshot.

## Main Reviewer Risk

Without the out-of-GPU angle, the main reviewer risk becomes:

> Is this just batching several graph queries into a wider GPU kernel?

A second risk is:

> Existing GPU graph systems already support multiple kernels, streams, or batched primitives. Why is OCGP a database systems contribution?

The paper must answer by making concurrency a physical execution and scheduling problem. The core contribution should not be "store N query states per vertex." It should be a general execution model that decides which queries can share graph traversal, how to handle different active sets, when to split or regroup queries, and how to trade throughput against latency.

## Revised Core Insight

Single-query GPU graph processing repeatedly pays the cost of irregular graph access for each query. When multiple queries run over the same graph, the graph structure is common and query states are independent.

OCGP separates:

- graph structure traversal, which can be shared;
- query state transition, which remains query-specific;
- scheduling decisions, which decide when sharing is beneficial.

The core insight is:

> The expensive physical operation in many graph analytics workloads is not the arithmetic in each query, but the irregular traversal of the same graph structure. A concurrent graph engine can amortize this traversal across multiple queries when their physical access patterns are compatible.

## What Makes This a VLDB/SIGMOD Paper

To be competitive at VLDB or SIGMOD, the paper should emphasize database systems ideas:

- multi-query optimization for graph analytics;
- shared physical operators over graph traversal;
- query grouping and compatibility;
- throughput-latency tradeoffs under concurrent execution;
- adaptive scheduling for heterogeneous workloads;
- predictable performance boundaries.

The GPU backend is necessary, but it should be presented as the implementation of a multi-query execution model.

## Paper Contributions

### Contribution 1: A Multi-Query Physical Execution Model for Graph Analytics

OCGP defines a shared graph traversal abstraction that decouples graph access from per-query state updates.

The model should include:

- query state vectors attached to vertices or edges;
- per-query update functions;
- shared traversal operators over vertices, edges, or frontiers;
- active masks indicating which queries participate in a traversal;
- termination masks for completed queries;
- fallback behavior for incompatible queries.

Expected reviewer-facing claim:

> OCGP turns concurrent graph analytics into a multi-query physical execution problem, analogous to shared scans in databases but adapted to irregular graph access.

What must be shown:

- The model can express several graph algorithms without rewriting graph storage.
- Sharing is not restricted to identical copies of one query.
- The model allows partial sharing and fallback instead of all-or-nothing batching.

### Contribution 2: Compatibility-Aware Query Grouping

OCGP should include a grouping policy that decides which queries should share execution.

The policy should consider:

- algorithm type;
- push versus pull traversal mode;
- active-set density;
- overlap in active vertices or edges;
- expected remaining iterations;
- per-query state footprint;
- latency sensitivity;
- resource pressure such as registers, shared memory, and occupancy.

Expected reviewer-facing claim:

> OCGP avoids blind batching by grouping queries whose physical access patterns are likely to share useful work.

What must be shown:

- Grouping improves mixed-workload throughput over naive batching.
- Grouping prevents severe slowdowns when algorithms are incompatible.
- The grouping overhead is small relative to execution time.

### Contribution 3: Vectorized Shared Traversal Backend

OCGP implements shared graph traversal on a single GPU.

The backend should support:

- dense pull-style traversal for algorithms such as PageRank;
- sparse frontier-style traversal for BFS and SSSP;
- hybrid push-pull execution when beneficial;
- query-state layout choices such as SoA, AoS, or tiled SoA;
- active query masks to avoid work for completed or inactive queries;
- warp- or block-level sharing of neighbor loads;
- optional cache and shared-memory reuse for hot vertices or repeated neighbor access.

Expected reviewer-facing claim:

> OCGP amortizes irregular graph memory access across multiple query states while keeping the additional state-update cost low.

What must be shown:

- Kernel time grows sublinearly with query count in memory-bound regimes.
- Throughput scales until a clear bottleneck appears.
- The backend remains competitive with strong batched SpMM, GNN, or algorithm-specific batched baselines.

### Contribution 4: Adaptive Scheduling for Heterogeneous Progress

Concurrent graph queries finish at different times and may change traversal density across iterations. OCGP should adapt during execution.

The scheduler should handle:

- query completion and slot reclamation;
- dynamic replenishment from a waiting queue;
- regrouping when active-set density changes;
- fallback to single-query or smaller-group execution when sharing becomes harmful;
- optional latency-aware scheduling to avoid starving short queries.

Expected reviewer-facing claim:

> OCGP maintains high throughput for large concurrent workloads without forcing every query to follow the slowest or densest query.

What must be shown:

- Dynamic replenishment improves throughput over fixed batches.
- Regrouping or fallback improves hard heterogeneous workloads.
- Short queries are not excessively delayed by long-running queries.

### Contribution 5: Cost Model and Applicability Boundaries

A VLDB/SIGMOD paper should not only report speedups. It should explain when sharing helps.

OCGP should include a simple cost model based on:

- graph traversal cost;
- per-query state update cost;
- active-set size and overlap;
- memory bandwidth pressure;
- register and occupancy pressure;
- query group size;
- scheduling overhead.

Expected reviewer-facing claim:

> OCGP can predict when multi-query sharing is beneficial and avoid regimes where it is harmful.

What must be shown:

- The model predicts the saturation point as query count grows.
- The model predicts at least coarse-grained sharing versus no-sharing decisions.
- Negative cases are reported and explained.

## Required Experimental Claims for VLDB/SIGMOD

### Claim 1: Concurrent Graph Workloads Have Shareable Physical Access

Reviewer concern:

> Why should concurrent graph queries share enough access to justify a system?

Required evidence:

- Measure active vertex and edge overlap across queries.
- Measure repeated graph-structure access under serial execution.
- Compare overlap under different query source distributions: uniform, degree-biased, community-local, and clustered.
- Include homogeneous and heterogeneous query mixes.

Minimum convincing result:

- Show that a meaningful fraction of graph traversal work is repeated under serial execution.
- Show that repeated access is not limited to a synthetic best case.

Stronger result:

- Demonstrate high overlap for application-inspired query distributions, such as many sources from the same community or repeated personalized ranking over related users.

### Claim 2: OCGP Beats Strong Concurrent Baselines

Reviewer concern:

> Why not just launch multiple GPU kernels or streams?

Required evidence:

Baselines should include:

- serial single-query execution;
- naive concurrent CUDA streams;
- fixed-size blind batching;
- batched SpMM or GNN-style dense feature execution;
- algorithm-specific batched baselines where available, such as multi-source BFS or batched PageRank;
- a mature GPU graph framework such as Gunrock if integration is feasible.

Metrics:

- aggregate throughput;
- average latency;
- tail latency;
- slowdown relative to isolated single-query execution;
- GPU memory bandwidth;
- occupancy;
- achieved SM utilization.

Minimum convincing result:

- OCGP should outperform serial and naive concurrent streams on aggregate throughput.
- OCGP should match or beat dense batched baselines in workloads where active-set and termination differences matter.

Stronger result:

- OCGP should show better throughput-latency tradeoffs than blind batching.

### Claim 3: Benefits Hold Beyond Homogeneous Batches

Reviewer concern:

> This only works for N copies of PageRank or BFS.

Required evidence:

Evaluate:

- homogeneous BFS, SSSP, PageRank, and CC batches;
- mixed BFS + SSSP;
- mixed PageRank + CC;
- hard mixed BFS + PageRank;
- full mixed BFS + SSSP + PageRank + CC;
- streaming query arrivals with replenishment.

Metrics:

- speedup over baselines;
- shared traversal ratio;
- wasted traversal ratio;
- group changes over time;
- per-query latency distribution;
- query completion timeline.

Minimum convincing result:

- Homogeneous workloads show the upper bound of sharing.
- Mixed workloads show meaningful benefit over serial and naive concurrency.
- Hard mixed cases are either improved by grouping/fallback or clearly identified as outside the sweet spot.

Stronger result:

- Compatibility-aware grouping consistently outperforms blind batching on mixed workloads.

### Claim 4: Sharing Does Not Destroy Latency or Fairness

Reviewer concern:

> Maximizing throughput may make short queries wait behind long queries.

Required evidence:

- Compare fixed batches, dynamic replenishment, and latency-aware scheduling.
- Include query queues with mixed short and long algorithms.
- Report average, p95, and p99 latency.
- Report slowdown relative to isolated execution.

Minimum convincing result:

- OCGP improves throughput without causing unacceptable tail latency.
- Dynamic replenishment reduces idle slots without starving short queries.

Stronger result:

- Provide a tunable policy that trades throughput for latency, and show the Pareto curve.

### Claim 5: The Performance Gains Come from the Claimed Mechanisms

Reviewer concern:

> The speedup may come from implementation details, not the proposed execution model.

Required evidence:

Ablation variants:

- serial baseline;
- naive streams;
- blind vectorized batching;
- plus compatibility-aware grouping;
- plus active masks and termination handling;
- plus dynamic replenishment;
- plus cache or shared-memory reuse;
- full OCGP.

Metrics:

- throughput;
- latency;
- kernel time;
- memory bandwidth;
- scheduling overhead;
- state memory footprint.

Minimum convincing result:

- Each main mechanism contributes in a clear workload regime.
- If cache reuse is minor, present it as an optimization, not a main contribution.

### Claim 6: OCGP Has Clear Break-Even Boundaries

Reviewer concern:

> When should this system not be used?

Required evidence:

Sweep:

- query count;
- graph size;
- graph degree distribution;
- active-set overlap;
- active-set density;
- algorithm mix;
- per-query state size;
- latency target.

Minimum convincing result:

- Show regimes where OCGP is beneficial and regimes where it is not.
- Show that the scheduler avoids the worst blind-batching slowdowns.

Stronger result:

- Validate a simple cost model that predicts sharing decisions.

## Required Experimental Plan

### Experiment 1: Motivation and Sharing Opportunity

Goal:

Show that concurrent graph queries repeatedly access the same graph structure and therefore create sharing opportunity.

Setup:

- Datasets: social, web, road, citation, and synthetic power-law graphs.
- Query source distributions: uniform, degree-biased, community-local, and clustered.
- Workloads: multi-source BFS, SSSP, personalized PageRank, CC, k-hop queries, and mixed analytics.

Metrics:

- active vertex overlap;
- active edge overlap;
- frontier overlap;
- dense traversal overlap;
- graph access time fraction;
- memory bandwidth consumption under serial execution.

Expected effect:

- Establish that graph traversal and memory access dominate enough to make sharing worthwhile.

### Experiment 2: Overall Concurrent Performance

Goal:

Show that OCGP improves total throughput and maintains reasonable latency.

Baselines:

- serial execution;
- CUDA streams with independent queries;
- blind vectorized batching;
- batched SpMM or GNN-style execution;
- algorithm-specific batched implementations;
- existing GPU graph framework where feasible.

Metrics:

- queries per second;
- total batch completion time;
- average latency;
- p95 and p99 latency;
- isolated-query slowdown;
- GPU bandwidth and utilization.

Expected effect:

- OCGP should provide strong throughput improvement over serial and naive streams.
- OCGP should improve or match blind batching on homogeneous workloads and outperform it on heterogeneous workloads.

### Experiment 3: Homogeneous Scaling

Goal:

Measure the maximum benefit of shared traversal.

Workloads:

- N x BFS;
- N x SSSP;
- N x PageRank;
- N x CC;
- N x personalized PageRank.

Sweep:

- N = 1, 2, 4, 8, 16, 32, 64 if memory allows.

Metrics:

- throughput scaling;
- kernel time scaling;
- memory bandwidth;
- occupancy;
- register pressure;
- state memory footprint.

Expected effect:

- Kernel time should grow much slower than query count while traversal is memory-bound.
- The experiment should identify the saturation point and explain the bottleneck.

### Experiment 4: Heterogeneous Query Execution

Goal:

Show the system handles mixed algorithms.

Workloads:

- BFS + SSSP;
- PageRank + CC;
- BFS + PageRank;
- BFS + SSSP + PageRank + CC;
- mixed query queue with arrivals and completions.

Metrics:

- speedup;
- useful shared work;
- wasted work;
- group composition over time;
- per-query completion time;
- fallback frequency.

Expected effect:

- Mixed workloads should outperform serial and naive streams.
- Compatibility-aware grouping should beat blind batching.
- For hard cases, fallback should bound slowdown.

### Experiment 5: Push, Pull, and Hybrid Behavior

Goal:

Address the biggest algorithmic concern: sparse-frontier and dense-iteration algorithms do not naturally share the same traversal mode.

Setup:

- BFS and SSSP under push, pull, and hybrid modes.
- PageRank and CC under dense or semi-dense traversal.
- Mixed BFS + PageRank and BFS + CC workloads.

Metrics:

- active-set density;
- traversal mode decisions;
- speedup versus forced-pull and forced-push baselines;
- wasted edge visits;
- kernel time.

Expected effect:

- OCGP should not rely on forcing every algorithm into pull mode.
- Hybrid or grouping decisions should preserve sparse-frontier benefits when they matter.

### Experiment 6: Dynamic Replenishment and Latency

Goal:

Show that query completion mismatch does not waste GPU slots or damage latency.

Baselines:

- fixed batch;
- blind replenishment;
- compatibility-aware replenishment;
- optional latency-aware replenishment.

Metrics:

- throughput over time;
- slot utilization;
- average latency;
- p95 and p99 latency;
- short-query slowdown;
- replacement overhead.

Expected effect:

- Replenishment should improve throughput for large query queues.
- Latency-aware scheduling should control tail latency when needed.

### Experiment 7: Ablation Study

Goal:

Show which parts of OCGP matter.

Variants:

- serial execution;
- naive streams;
- blind vectorized execution;
- plus active masks;
- plus compatibility grouping;
- plus hybrid push-pull selection;
- plus dynamic replenishment;
- plus cache or shared-memory reuse;
- full OCGP.

Metrics:

- throughput;
- latency;
- kernel time;
- memory bandwidth;
- scheduling overhead.

Expected effect:

- The largest gains should come from shared traversal and grouping.
- Dynamic replenishment should matter most for streaming or long-tail workloads.
- Cache-level optimizations should be reported honestly as secondary if their gains are small.

### Experiment 8: Cost Model and Break-Even Analysis

Goal:

Explain when OCGP helps.

Sweep:

- graph size;
- query count;
- active-set overlap;
- active-set density;
- per-query compute cost;
- per-query state size;
- algorithm mix;
- latency constraint.

Metrics:

- predicted sharing benefit;
- measured speedup;
- scheduler decision accuracy;
- slowdown avoided by fallback.

Expected effect:

- The model should predict coarse sharing decisions.
- OCGP should avoid severe slowdowns in low-sharing regimes.

## Minimum Bar for a Credible VLDB/SIGMOD Submission

The single-GPU framing is viable only if the paper can show:

1. Strong end-to-end throughput gains over serial execution and naive CUDA streams.
2. Strong baselines beyond serial execution, especially blind batching and batched SpMM or GNN-style execution.
3. Meaningful gains on heterogeneous workloads, not only homogeneous batches.
4. A real compatibility-aware grouping or fallback policy.
5. Evidence that push/pull differences are handled rather than ignored.
6. Latency and fairness analysis for concurrent workloads.
7. A break-even analysis explaining when sharing should be disabled.

If the only strong result is that N homogeneous all-pull queries run in roughly the same kernel time as one query, the paper will likely be seen as a GPU batching optimization rather than a VLDB/SIGMOD systems contribution.

## Suggested Paper Structure

### 1. Introduction

- Many graph analytics workloads issue repeated queries over a shared graph.
- Single-query GPU graph engines repeatedly pay irregular graph access cost.
- Naive concurrency and blind batching fail under heterogeneous algorithms and convergence.
- OCGP introduces compatibility-aware shared execution for concurrent graph analytics.

### 2. Motivation

- Profile serial and naive concurrent execution.
- Quantify repeated graph access.
- Show where blind batching helps and fails.
- Introduce heterogeneous active-set and termination mismatch.

### 3. System Overview

- Query lifecycle.
- Query grouping.
- Shared traversal backend.
- Dynamic replenishment.
- Fallback and latency-aware scheduling.

### 4. Multi-Query Execution Model

- Query state representation.
- Shared traversal operators.
- Active masks and termination masks.
- Push, pull, and hybrid modes.
- Compatibility rules.

### 5. Scheduling and Grouping

- Compatibility-aware grouping.
- Dynamic replenishment.
- Regrouping and fallback.
- Cost model.
- Throughput-latency policy.

### 6. GPU Execution Backend

- State layout.
- Vectorized traversal kernels.
- Frontier handling.
- Mask handling.
- Cache and shared-memory optimizations.

### 7. Evaluation

- Sharing opportunity.
- Overall throughput and latency.
- Homogeneous scaling.
- Heterogeneous workloads.
- Push/pull/hybrid analysis.
- Dynamic replenishment.
- Ablation.
- Break-even and negative cases.

### 8. Related Work

- GPU graph processing systems.
- Concurrent graph query processing.
- Multi-query optimization and shared scans.
- GNN and SpMM systems.
- GPU scheduling and kernel fusion.

### 9. Conclusion

- OCGP makes shared graph traversal a first-class physical operator for concurrent single-GPU graph analytics.

## Positioning Against Related Work

OCGP should be positioned as follows:

- Against single-query GPU graph systems: OCGP optimizes concurrent workloads over a shared graph.
- Against naive CUDA streams: OCGP shares graph access instead of competing independently for memory bandwidth.
- Against blind batching and GNN-style dense feature execution: OCGP handles active-set mismatch, algorithm heterogeneity, termination mismatch, and dynamic replenishment.
- Against database multi-query optimization: OCGP adapts shared physical execution to irregular graph traversal on GPUs.

The strongest positioning sentence is:

> OCGP brings database-style multi-query physical optimization to single-GPU graph analytics, where the shared operator is irregular graph traversal rather than a relational scan.

## Go/No-Go Checklist

Proceed toward VLDB/SIGMOD if:

- OCGP beats naive streams and blind batching on mixed workloads.
- BFS + PageRank or another hard mixed workload is handled with either speedup or principled fallback.
- Dynamic replenishment improves streaming query queues without unacceptable tail latency.
- A cost model can explain when sharing helps.

Reframe or target a more GPU-focused venue if:

- Gains only appear for homogeneous dense pull workloads.
- Strong batched SpMM or GNN-style baselines match OCGP everywhere.
- Heterogeneous workloads require so much fallback that multi-query sharing rarely helps.
- Latency degradation is large and cannot be controlled.

