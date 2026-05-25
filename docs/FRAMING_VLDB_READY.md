# OCGP: VLDB-Oriented Framing

## One-Sentence Thesis

OCGP is an out-of-GPU-memory execution engine for batched graph analytics that treats inter-query sharing as a first-class physical execution problem: it decides when graph queries should share traversal, transfer, and cache resources under GPU memory constraints.

## Target Venue

- Primary: VLDB
- Secondary: SIGMOD
- Fallback: ICDE

This paper should be framed as a database systems paper, not as a GPU kernel optimization paper. The core value must be an execution and scheduling abstraction for batched graph analytics over graphs that do not fit in GPU memory.

## Problem Setting

The target workload is offline batched graph analytics on a single commodity GPU with limited device memory.

- The graph is larger than GPU memory and resides in host memory.
- Many graph analytics requests arrive over the same graph.
- Queries may be homogeneous or heterogeneous: BFS, SSSP, PageRank, connected components, personalized PageRank, ego-network analytics, graph sampling, and GNN preprocessing or aggregation.
- The system objective is total batch throughput under GPU memory constraints.

The paper should not claim that this setting is self-evident. VLDB reviewers will expect a concrete workload story. The motivation should focus on repeated analytics over a shared large graph, such as:

- multi-source BFS or SSSP for offline reachability and distance features;
- batched personalized PageRank or random-walk style ranking;
- repeated neighborhood analytics for recommendation or fraud detection;
- graph feature generation for downstream ML pipelines;
- GNN preprocessing or sampling over a graph that exceeds GPU memory.

## Main Reviewer Risk

The main risk is that reviewers view OCGP as a straightforward batching trick:

> Put multiple query states into a wider vector, then run a GNN-style or SpMM-style kernel.

The paper must therefore make clear that the contribution is not state vectorization alone. The contribution is a physical execution engine that handles query grouping, compatibility, out-of-GPU transfer, active-set mismatch, termination mismatch, fallback, and dynamic replenishment.

## Revised Core Insight

Existing out-of-GPU graph systems optimize one query at a time. Existing concurrent graph systems usually assume the graph fits in GPU memory or do not optimize transfer and execution jointly. OCGP targets the missing intersection:

- concurrent graph analytics;
- out-of-GPU graph storage;
- shared traversal and transfer across queries;
- heterogeneous query progress and termination.

The key insight is:

> Graph structure access is often the expensive shared physical operation, while query state is logically independent. If the system separates graph traversal from per-query state transitions, multiple queries can share graph access without requiring identical query semantics.

This should be presented as a physical execution principle, not only a kernel layout choice.

## Paper Contributions

### Contribution 1: Multi-Query Graph Execution Model

OCGP introduces a multi-query execution model that separates graph structure access from query state transitions.

The model should define:

- a per-query state vector attached to vertices or edges;
- query-local update functions;
- shared traversal operators over graph tiles, frontiers, or dense vertex ranges;
- query compatibility rules for deciding when queries can share a traversal;
- termination and fallback behavior when queries progress at different rates.

Expected reviewer-facing claim:

> OCGP exposes inter-query sharing as a physical execution decision, rather than hard-coding one batched kernel for one algorithm.

What must be shown:

- The model supports both homogeneous batches and realistic heterogeneous batches.
- The model can express BFS, SSSP, PageRank, and connected components without changing graph storage.
- The system can split or stop sharing when sharing becomes harmful.

### Contribution 2: Compatibility-Aware Query Grouping and Execution

OCGP should include a runtime policy that groups queries based on whether they can profitably share graph traversal and transfer.

The grouping policy should consider:

- traversal mode: push, pull, or hybrid;
- active-set density and overlap;
- expected remaining iterations;
- per-query state footprint;
- GPU memory budget;
- transfer reuse opportunity.

This contribution is critical because heterogeneous query coexistence is the biggest technical challenge. A weak paper says "we batch BFS and PR together." A strong paper explains when batching BFS and PR is beneficial, when it is not, and how the engine adapts.

Expected reviewer-facing claim:

> OCGP avoids blindly batching incompatible queries; it uses compatibility-aware grouping to preserve sharing benefits while bounding wasted work.

What must be shown:

- Homogeneous batches achieve near-linear throughput scaling until a hardware bottleneck is reached.
- Heterogeneous batches still provide meaningful speedup over serial and naive concurrent execution.
- Incompatible combinations are detected and either split or executed with controlled fallback.

### Contribution 3: Out-of-GPU Shared Transfer Scheduling

OCGP should treat host-to-GPU transfer as part of the multi-query physical plan.

The scheduler should decide:

- which graph tile or subgraph to move to GPU;
- which query group will consume it;
- how much per-query state can reside on GPU;
- when to prefetch and evict;
- how to overlap transfer with computation.

This contribution makes the paper relevant to VLDB. Without it, the paper risks looking like a GPU kernel paper with an out-of-GPU motivation attached.

Expected reviewer-facing claim:

> OCGP reduces redundant PCIe transfer by scheduling shared graph tiles for multiple queries, while respecting GPU memory limits.

What must be shown:

- Transfer bytes are reduced compared with serial out-of-GPU execution.
- Transfer and compute overlap improves end-to-end runtime.
- The scheduler remains beneficial under realistic memory budgets, not only when GPU memory is artificially large.

### Contribution 4: Vectorized GPU Backend for Shared Traversal

OCGP implements shared traversal using a vectorized GPU backend.

This includes:

- SoA or hybrid layout for per-query state;
- shared neighbor traversal;
- coalesced state access where possible;
- active mask or slot mask for completed and replenished queries;
- frontier handling for sparse algorithms;
- cache-conscious layout and optional shared-memory reuse.

This should be presented as the backend that realizes the execution model, not as the only core contribution.

Expected reviewer-facing claim:

> Once queries are selected for sharing, OCGP amortizes irregular graph access across query states with low additional compute overhead.

What must be shown:

- Adding query slots does not proportionally increase kernel time while graph access remains the bottleneck.
- The backend handles completed query slots and dynamic replenishment with low overhead.
- The backend is competitive with strong batched SpMM or GNN-style kernels.

### Contribution 5: Dynamic Replenishment for Long-Tail Query Completion

OCGP supports dynamic replenishment: when one query in a running group finishes, a new query can occupy its slot without reallocating the execution layout.

This addresses heterogeneous convergence and long-tail execution.

Expected reviewer-facing claim:

> OCGP maintains high GPU utilization for large query queues even when individual queries finish at different times.

What must be shown:

- Replenishment improves throughput over fixed batches.
- Slot replacement overhead is small relative to total runtime.
- Replenishment does not break correctness or memory isolation between queries.

## What Must Be Proven for VLDB

The paper needs to prove five claims. Each claim should have a dedicated experiment or analysis.

### Claim 1: The Workload Is Real and Has Sharing Opportunity

Reviewer concern:

> Who submits many graph queries to one GPU, and why should their work overlap?

Required evidence:

- Provide a realistic workload generator or trace-inspired workload.
- Report active vertex, active edge, and tile overlap across queries.
- Show overlap for both homogeneous and heterogeneous workloads.

Minimum convincing result:

- For realistic batches, a non-trivial fraction of graph tiles or active edges is reused across queries.
- The paper should show that reuse is common enough to matter, not just possible in synthetic best cases.

Stronger result:

- Workload overlap remains meaningful across several graph families: social, web, road, and synthetic power-law graphs.
- Query arrivals and sources are sampled from plausible application distributions, not only uniformly random vertices.

### Claim 2: OCGP Is Not Just Batched SpMM or GNN Aggregation

Reviewer concern:

> This is equivalent to treating query id as feature dimension.

Required evidence:

- Compare against a strong batched SpMM or GNN-style implementation where query states are stored as dense feature dimensions.
- Include algorithm-specific batched baselines where available, such as multi-source BFS or batched PageRank.
- Explain which parts of OCGP cannot be captured by dense feature batching: out-of-GPU scheduling, active-set mismatch, termination mismatch, replenishment, and compatibility-aware grouping.

Minimum convincing result:

- OCGP outperforms or matches batched SpMM/GNN-style execution on homogeneous workloads.
- OCGP clearly outperforms it on heterogeneous or out-of-GPU workloads where dense batching does redundant work or transfers too much data.

Stronger result:

- Show cases where dense batching is good, and explain why OCGP still wins through scheduling or reduced transfer.
- Show cases where OCGP chooses not to share because sharing is predicted to be harmful.

### Claim 3: Heterogeneous Queries Still Benefit

Reviewer concern:

> The speedup only holds for N copies of the same algorithm.

Required evidence:

- Evaluate homogeneous batches: BFS-only, SSSP-only, PageRank-only, CC-only.
- Evaluate mixed batches: BFS+SSSP, PageRank+CC, BFS+PageRank, BFS+SSSP+PageRank+CC.
- Include both sparse-frontier and dense-iteration algorithms.
- Measure useful shared work versus wasted work.

Minimum convincing result:

- Homogeneous workloads should show near-linear throughput scaling up to the memory or compute bottleneck.
- Mixed workloads should still show a meaningful end-to-end speedup over serial execution and naive concurrent streams.
- The paper should identify at least one hard mixed case and explain why speedup is lower.

Stronger result:

- A compatibility-aware grouping policy should outperform blind batching on mixed workloads.
- Dynamic regrouping or fallback should avoid large slowdowns in incompatible combinations.

### Claim 4: Out-of-GPU Scheduling Improves End-to-End Runtime

Reviewer concern:

> Kernel speedup may disappear once PCIe transfer and scheduling overhead are included.

Required evidence:

- End-to-end experiments where graph structure exceeds GPU memory.
- Time breakdown: host-to-GPU transfer, kernel execution, scheduling overhead, materialization, result extraction.
- Transfer byte breakdown: serial baseline versus OCGP shared transfer.
- Memory budget sensitivity: 8GB, 16GB, 24GB, and larger GPU if available.

Minimum convincing result:

- OCGP improves total runtime or throughput, not only kernel time.
- Transfer bytes and/or transfer stalls are reduced relative to serial out-of-GPU execution.
- Scheduling overhead is small enough that it does not dominate.

Stronger result:

- OCGP maintains benefit as memory budget decreases.
- Prefetch and compute overlap provide measurable additional benefit.
- The paper demonstrates a break-even point where OCGP becomes beneficial as graph size or query count grows.

### Claim 5: The System Has Clear Applicability Boundaries

Reviewer concern:

> When does this fail, and are those cases common?

Required evidence:

- Break-even analysis over graph size, query count, active-set overlap, query heterogeneity, and memory budget.
- Cases where the graph fits in GPU memory.
- Cases with low-overlap query sources.
- Cases with very small query batches.
- Cases where per-query state dominates memory.

Minimum convincing result:

- The paper honestly reports regimes where OCGP is not beneficial.
- The proposed grouping policy avoids the worst slowdowns.

Stronger result:

- Provide a simple cost model that predicts whether sharing should be enabled.
- Validate the model against measured runtime.

## Required Experimental Plan

### Experiment 1: Motivation Study

Goal:

Show that repeated graph analytics over the same graph creates measurable inter-query reuse.

Setup:

- Datasets: social graphs, web graphs, road graphs, and synthetic power-law graphs.
- Workloads: multi-source BFS, multi-source SSSP, personalized PageRank, CC variants, mixed analytics.
- Query source distributions: uniform, degree-biased, community-local, and application-inspired clustered sources.

Metrics:

- active vertex overlap;
- active edge overlap;
- graph tile overlap;
- hot vertex reuse;
- graph traversal time fraction;
- PCIe transfer fraction.

Target effect:

- Establish that graph access and transfer dominate enough to justify a system for sharing them.
- Show that overlap exists in realistic workloads, not only hand-picked examples.

### Experiment 2: Overall End-to-End Performance

Goal:

Show that OCGP improves batch throughput over strong baselines on graphs larger than GPU memory.

Baselines:

- serial out-of-GPU single-query execution;
- naive concurrent GPU streams with independent query execution;
- batched SpMM or GNN-style dense feature execution;
- algorithm-specific batched implementations where available;
- CPU multicore graph system as a secondary reference.

Metrics:

- throughput in queries per second;
- total batch completion time;
- speedup over each baseline;
- GPU memory footprint;
- PCIe bytes transferred;
- GPU utilization.

Target effect:

- Clear end-to-end speedup over serial and naive concurrent execution.
- Competitive or better performance than dense batched SpMM/GNN-style baselines.
- Benefits should hold when graph data does not fit in GPU memory.

### Experiment 3: Homogeneous Scaling

Goal:

Measure the upper bound of shared traversal.

Workloads:

- N x BFS;
- N x SSSP;
- N x PageRank;
- N x CC.

Sweep:

- query count: 1, 2, 4, 8, 16, 32, 64 if feasible;
- graph size;
- memory budget.

Metrics:

- throughput scaling;
- kernel time scaling;
- memory bandwidth;
- occupancy;
- per-query state overhead.

Target effect:

- Throughput should improve close to linearly until the bottleneck shifts.
- The paper should identify the saturation point and explain whether it is caused by memory bandwidth, compute, occupancy, or state footprint.

### Experiment 4: Heterogeneous Query Analysis

Goal:

Show that OCGP is useful beyond homogeneous batches.

Workloads:

- BFS + SSSP;
- PageRank + CC;
- BFS + PageRank;
- BFS + SSSP + PageRank + CC;
- mixed query queue with dynamic arrivals.

Metrics:

- shared traversal ratio;
- wasted traversal ratio;
- per-query completion time;
- batch throughput;
- active-set density over time;
- number of regrouping or fallback decisions.

Target effect:

- Mixed workloads should outperform serial and naive concurrent baselines.
- Compatibility-aware grouping should outperform blind batching.
- Hard cases such as BFS + PageRank should be reported honestly, with explanation and fallback behavior.

### Experiment 5: Out-of-GPU Transfer and Scheduling

Goal:

Prove that the out-of-GPU part is real and beneficial.

Variants:

- no sharing;
- shared traversal only;
- shared transfer only;
- shared traversal + shared transfer;
- shared traversal + shared transfer + prefetch;
- full OCGP with dynamic replenishment.

Metrics:

- transferred bytes;
- transfer time;
- transfer-compute overlap;
- scheduler overhead;
- tile reuse count;
- eviction count;
- peak GPU memory.

Target effect:

- Shared transfer should reduce redundant PCIe traffic.
- Prefetch should reduce exposed transfer stalls.
- Full OCGP should improve total runtime, not just reduce a subcomponent.

### Experiment 6: Dynamic Replenishment

Goal:

Show that replenishment handles long-tail completion in mixed query queues.

Baselines:

- fixed batch without replenishment;
- blind slot replacement;
- OCGP compatibility-aware replenishment.

Metrics:

- throughput over time;
- slot occupancy;
- completed queries per unit time;
- replacement overhead;
- wasted work due to dummy or inactive slots.

Target effect:

- Replenishment should improve throughput for query queues larger than the saturation batch size.
- Slot replacement overhead should be small.
- Compatibility-aware replenishment should avoid replacing a finished query with an incompatible query that harms the group.

### Experiment 7: Ablation Study

Goal:

Attribute performance gains to individual system components.

Variants:

- serial baseline;
- vectorized shared traversal only;
- plus compatibility-aware grouping;
- plus shared transfer scheduling;
- plus prefetch and overlap;
- plus cache reuse;
- plus dynamic replenishment.

Metrics:

- speedup;
- transferred bytes;
- kernel time;
- scheduling overhead;
- memory footprint.

Target effect:

- Each major component should provide measurable benefit in at least one meaningful regime.
- If cache reuse contributes little, it should be demoted to an implementation optimization rather than a main contribution.

### Experiment 8: Break-Even and Negative Cases

Goal:

Show maturity by identifying when OCGP should not be used.

Sweep:

- graph fits versus does not fit in GPU memory;
- query count from very small to large;
- high-overlap versus low-overlap query sources;
- homogeneous versus highly incompatible queries;
- small versus large per-query state;
- PCIe bandwidth sensitivity if possible.

Metrics:

- speedup or slowdown;
- predicted benefit from cost model;
- actual measured benefit.

Target effect:

- The paper should provide an applicability map.
- OCGP should avoid severe slowdowns through grouping and fallback.
- The cost model should roughly predict the break-even region.

## Minimum Bar for a Credible VLDB Submission

Before submission, the paper should have evidence for all of the following:

1. End-to-end speedup on out-of-GPU graphs, not just kernel-level speedup.
2. Strong baselines including dense batched SpMM or GNN-style execution.
3. Heterogeneous workloads with at least one hard case such as BFS + PageRank.
4. A compatibility-aware grouping or fallback mechanism.
5. Transfer byte reduction and time breakdown proving the out-of-GPU scheduler matters.
6. Memory budget analysis on commodity GPU sizes.
7. Break-even analysis and honest negative cases.
8. Clear workload motivation with measured sharing opportunity.

If the final results only show that 32 homogeneous queries can run in roughly the same kernel time as one query under an all-pull model, the paper is unlikely to be competitive at VLDB or SIGMOD. That result is valuable, but it is closer to a GPU kernel optimization result than a database systems contribution.

## Suggested Paper Structure

### 1. Introduction

- Batched graph analytics over large shared graphs is common in offline data systems.
- Commodity GPUs are attractive but device memory is limited.
- Existing out-of-GPU systems optimize single-query execution.
- Existing concurrent systems do not jointly optimize sharing and out-of-GPU transfer.
- OCGP introduces multi-query graph execution under memory constraints.

### 2. Background and Motivation

- Explain out-of-GPU graph processing.
- Profile serial batch execution.
- Quantify graph access, transfer, and compute costs.
- Quantify inter-query overlap in realistic workloads.
- Show why naive batching and naive concurrency are insufficient.

### 3. OCGP Overview

- Query lifecycle.
- Query grouping.
- Shared traversal.
- Shared transfer scheduling.
- Dynamic replenishment.
- Fallback path.

### 4. Multi-Query Execution Model

- State representation.
- Query-local update functions.
- Shared physical traversal.
- Compatibility rules.
- Termination and fallback.
- Push, pull, and hybrid execution.

### 5. Out-of-GPU Shared Scheduling

- Memory budget model.
- Tile selection and eviction.
- Query group to tile mapping.
- Prefetch and compute overlap.
- Cost model.

### 6. GPU Backend

- State layout.
- Shared traversal kernels.
- Active masks and slot management.
- Cache and shared-memory reuse.
- Dynamic replenishment implementation.

### 7. Evaluation

- Motivation and workload overlap.
- Overall performance.
- Homogeneous scaling.
- Heterogeneous query analysis.
- Out-of-GPU transfer breakdown.
- Dynamic replenishment.
- Ablation.
- Break-even and negative cases.

### 8. Related Work

- Out-of-GPU graph processing.
- Multi-query and concurrent graph processing.
- GPU graph analytics.
- GNN and SpMM systems.
- Database shared scans, vectorized execution, and multi-query optimization.

### 9. Conclusion

- Reiterate that OCGP makes inter-query sharing a physical execution problem for out-of-GPU graph analytics.

## Positioning Against Related Work

OCGP should be positioned against four groups of systems:

- Out-of-GPU graph systems: they optimize data movement but typically for one query at a time.
- Concurrent graph systems: they run multiple queries but often assume the graph fits in GPU memory or do not optimize transfer sharing.
- GNN and SpMM systems: they provide optimized batched aggregation but do not handle heterogeneous graph analytics semantics, query termination, dynamic replenishment, or out-of-GPU scheduling as first-class problems.
- Database multi-query optimization: they motivate shared scans and shared operators, but do not address irregular graph traversal under GPU memory constraints.

The strongest positioning sentence is:

> OCGP brings database-style multi-query physical optimization to out-of-GPU graph analytics, where the shared resource is irregular graph access rather than a relational scan.

## Go/No-Go Checklist

Proceed toward VLDB if:

- BFS + PageRank mixed workload has meaningful end-to-end speedup or a principled fallback.
- OCGP beats dense batched SpMM or GNN-style execution on at least the workloads where OCGP claims superiority.
- Transfer scheduling reduces exposed PCIe overhead on graphs larger than GPU memory.
- The system can explain and predict when sharing is beneficial.

Reframe or target a different venue if:

- Speedup exists only for homogeneous all-pull kernels.
- Out-of-GPU transfer dominates and cannot be reduced.
- Heterogeneous workloads require so much fallback that sharing rarely helps.
- The strongest baseline is only serial single-query execution.

