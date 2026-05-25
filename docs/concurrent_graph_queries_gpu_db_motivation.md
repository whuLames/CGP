# Concurrent Graph Query Processing on GPUs: DB-Oriented Motivation

## Core Position

For a DB audience, GPU-specific terms such as memory coalescing, latency hiding, and occupancy should be explained through concrete graph-processing patterns.

The top-level argument should be:

1. Single graph queries expose specific inefficiencies on GPUs.
2. Concurrent graph queries provide an opportunity to address them.
3. However, GPU graph concurrency differs from CPU systems such as Glign and ForkGraph.
4. The solution should focus on GPU-friendly shared traversal, not simply running independent queries concurrently.

## Single-Query Problems

### Narrow Traversal Underutilization

A single graph query often has a small or phase-varying frontier. This happens in early iterations, late iterations, local graph queries, source-specific traversals, and personalized graph analytics.

Graph-level pattern:

```text
frontier too small -> too few active vertices/edges -> insufficient traversal parallelism
```

GPU interpretation:

```text
low occupancy / low SM utilization
```

DB-facing phrasing:

> A single graph query often exposes insufficient traversal parallelism at many iterations.

### Low Reuse of Topology Traversal

When multiple graph queries execute independently, each query repeatedly pays the cost of accessing the same graph topology:

```text
locate vertex -> read neighbor list -> fetch/update query state
```

Graph-level pattern:

```text
same graph topology, repeated traversal, repeated adjacency-list access
```

DB-facing phrasing:

> Independent execution repeatedly pays the cost of traversing the same graph topology.

## Why Concurrent Graph Queries Help

The opportunity is not merely to run more kernels at the same time. The more relevant opportunity is to transform many narrow traversals into a wider batched traversal.

Concurrent graph queries can help in two ways:

- Combine small frontiers to expose more active vertices and edges.
- Share topology access across multiple queries and amortize graph traversal cost.

Recommended phrasing:

> Concurrent graph queries provide an opportunity to transform many narrow traversals into a wider batched traversal, while amortizing topology access across queries.

## GPU Challenges as Graph-Processing Patterns

CPU systems such as Glign and ForkGraph mainly optimize cache locality through traversal alignment, graph partitioning, and LLC reuse. GPU graph concurrency must also consider whether the resulting batched traversal is GPU-friendly.

### Challenge 1: Misaligned Frontiers Cause Scattered State Access

Graph-level pattern:

```text
different queries visit unrelated graph regions in the same iteration
```

This causes each query to access different vertices, neighbors, and query states. Even if queries are batched, their memory accesses remain scattered.

GPU interpretation:

```text
poor memory coalescing / inefficient bulk memory access
```

DB-facing phrasing:

> When queries expand unrelated frontiers, the batched traversal issues scattered accesses to query states and neighbor features, preventing the GPU from serving them as regular bulk memory accesses.

### Challenge 2: Skewed Frontier Sizes Cause Load Imbalance

Graph-level pattern:

```text
some queries have large active frontiers, while others are in sparse phases
```

Simply batching them may not create useful parallelism. A few heavy queries dominate the work, while other queries contribute little.

GPU interpretation:

```text
warp/block imbalance, low effective occupancy, tail effects
```

DB-facing phrasing:

> Concurrent queries may be at different traversal phases. Simply batching them can create phase imbalance rather than useful parallelism.

### Challenge 3: Duplicate Neighbor Expansion Wastes Bandwidth

Graph-level pattern:

```text
multiple queries repeatedly expand the same active vertex or high-degree hub
```

Without explicit sharing, the same adjacency lists are loaded multiple times.

GPU interpretation:

```text
redundant global memory traffic / bandwidth pressure / unstable cache reuse
```

DB-facing phrasing:

> The same hot adjacency lists can be fetched multiple times unless the system explicitly shares expansion work across queries.

### Challenge 4: Query State Layout Determines Whether Batching Helps

Graph-level pattern:

```text
topology access is shared, but per-query vertex states are physically scattered
```

Even if adjacency-list traversal is shared, the system may still issue irregular state accesses if query states are not organized according to the batched traversal order.

GPU interpretation:

```text
uncoalesced state access, poor cache behavior, wasted bandwidth
```

DB-facing phrasing:

> Sharing topology alone is insufficient; the per-query vertex states must also be organized to match the batched traversal order.

## Proposed Strategy 1: Traversal-Centric Query Batching

Goal:

```text
form batches that are wide enough for GPU execution but not too scattered
```

Instead of randomly batching queries, group them according to traversal behavior:

- frontier overlap
- frontier locality
- active edge count
- traversal phase
- source locality
- frontier degree distribution

This strategy addresses:

- narrow traversal underutilization
- phase imbalance
- frontier misalignment

Recommended phrasing:

> We introduce traversal-centric batching, which groups queries according to their current frontier locality and traversal phase. The goal is to form batches that expose enough active edges while keeping their graph accesses aligned.

## Proposed Strategy 2: Shared Expansion with Query Masks

Goal:

```text
read each useful adjacency list once and reuse it across all queries that need it
```

Core idea:

```text
union frontier -> expand adjacency once -> apply updates to multiple query states
```

Each active vertex can carry a compact query mask:

```text
active_mask[v] = which queries in the batch need vertex v
```

Execution pattern:

```text
for v in union_frontier:
    load adjacency list of v once
    for each neighbor u:
        apply updates to queries in active_mask[v]
```

This strategy addresses:

- duplicate neighbor expansion
- repeated CSR access
- repeated topology traversal cost

Recommended phrasing:

> We introduce shared expansion, which expands the union frontier of a query batch and attaches a compact query mask to each active vertex. This allows the adjacency list of a vertex to be read once and reused by all queries that need it.

## Optional Strategy: Query-State Tiling

If the design needs a third component, use query-state tiling.

Goal:

```text
make query state access follow the same order as shared topology traversal
```

Potential layout:

```text
state_tile[vertex][query_in_batch]
```

This makes the states of multiple queries for the same active vertex physically close, improving bulk state access after shared topology traversal.

Recommended phrasing:

> We tile query states by active vertices and query batches, so that shared topology accesses are followed by compact state accesses.

## Recommended Paper Narrative

1. Observation: Single graph queries underutilize GPU traversal capacity because their frontiers are often narrow or phase-varying.
2. Observation: Multiple independent graph queries repeatedly traverse the same graph topology.
3. Opportunity: Concurrent graph queries can be fused into batched traversal to expose more active work and amortize topology access.
4. Challenge: GPU-friendly batching is not the same as CPU-style cache alignment.
5. Solution: Use traversal-centric batching and shared expansion with query masks.

## Thesis Statement

Recommended English version:

> We find that the bottleneck of GPU graph query processing is not merely the lack of concurrent queries, but the lack of GPU-friendly shared traversal. Independent queries repeatedly traverse the same graph yet often expose too little traversal parallelism when executed alone. We therefore batch queries by traversal phase and frontier locality, and execute them through shared expansion that amortizes topology access while preserving regular state accesses.

Recommended Chinese version:

> 我们发现，GPU 图查询处理的瓶颈不只是缺少并发，而是缺少适合 GPU 的共享遍历。单个 query 往往 frontier 窄、并行度不足；多个 query 独立执行又会重复访问同一图拓扑。因此，我们按照遍历阶段和 frontier locality 对 query 进行 batching，并通过共享 expansion 复用拓扑访问，同时保持 query state 访问尽量规整。

## Difference From CPU Systems

CPU systems such as ForkGraph and Glign mainly focus on:

```text
cache locality
LLC reuse
partitioning
traversal alignment
```

GPU concurrent graph processing must additionally ensure:

```text
enough active edges
aligned frontier access
regular query-state access
reduced duplicate expansion
balanced traversal phases
```

Recommended phrasing:

> Unlike CPU-based systems such as ForkGraph and Glign, whose primary goal is to improve cache reuse through partitioning or traversal alignment, our design must also ensure that the resulting shared traversal is GPU-efficient: it must generate enough active edges to fill the device, organize query states for bulk memory accesses, and avoid redundant expansion of high-degree vertices.

## Translation From GPU Terms to DB/Graph Terms

| GPU term | Graph-processing pattern |
| --- | --- |
| Occupancy | Enough active vertices/edges in a traversal batch |
| Memory coalescing | Regular bulk access to query states or neighbor features |
| Latency hiding | Enough independent active edges across queries |
| Bandwidth pressure | Repeated topology expansion and redundant state traffic |
| Load imbalance | Skewed frontier sizes or high-degree vertices across queries |
| Cache reuse | Reusing adjacency lists or query states across aligned frontiers |

