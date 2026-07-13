# Q=64 query-level push/pull partitioning study

## Objective

Evaluate whether a single `N=Q=64` BFS batch should be partitioned every
iteration into a push query group and a pull query group, then execute both
groups concurrently on disjoint Green Context SM partitions.

The agreed stop condition was: do not modify the complete hybrid engine unless
fixed partition replay on real iterations is consistently faster than the
existing all-push/all-pull iteration.

## Platform

- Tesla V100-SXM2-32GB, 80 SMs
- CUDA 12.8 Green Context Driver API
- Q=64 distinct BFS sources, seed 42
- Graphs: cit-Patents, soc-orkut, soc-twitter, soc-sinaweibo
- Existing hybrid threshold: pull when virtual frontier edge work is at least
  `0.20 * Q * |E|`

## Stage 1: subset kernel support

Push already accepted `active_slots`. Homogeneous fused pull was extended with
the same mask and now skips inactive query lanes for both simple and smem paths.

Q=64 validation covers:

- contiguous 32/32 query masks;
- even/odd interleaved masks;
- values, visited masks, and merged next-frontier masks;
- production `expand_shared_node_warp_kernel` and `fused_pull_simple_kernel`.

All element-wise comparisons passed.

Concurrent validation exposed a real race: pull used a non-atomic
`visited_mask[v] |= improved_mask` while push used atomic OR on the same
64-bit per-vertex mask. Pull now also uses `atomic_or_query_mask`, preventing
lost query bits during concurrent execution.

## Stage 2: real Q=64 hybrid traces

The existing batch-wide hybrid policy enters pull for only two iterations per
graph:

| Graph | Pull iterations | First pull split from per-query threshold |
|---|---|---:|
| cit-Patents | 7, 8 | 22 push / 42 pull |
| soc-orkut | 4, 5 | 5 push / 59 pull |
| soc-twitter | 3, 4 | 29 push / 35 pull |
| soc-sinaweibo | 3, 4 | 5 push / 59 pull |

Replay uses the first pull iteration. Its input is reproduced exactly by push
iterations only and is therefore independent of the current in-place pull
propagation order.

## Stage 3: fixed threshold partition replay

The replay compares:

1. all-pull on 80 SMs;
2. split push then pull serially;
3. split push/pull concurrently on disjoint `8/72`, `16/64`, `24/56`, and
   `32/48` SM partitions;
4. mask OR plus inclusive scan and compact postprocessing.

The two Green Contexts are created from one resource split. The pull context
uses the remainder, guaranteeing disjoint partitions whose SM counts sum to 80.

| Graph | All-pull ms | Best threshold split ms | Split slowdown |
|---|---:|---:|---:|
| cit-Patents | 17.64 | 52.63 | 2.98x |
| soc-orkut | 63.80 | 95.45 | 1.50x |
| soc-twitter | 208.51 | 278.47 | 1.34x |
| soc-sinaweibo | 246.56 | 507.63 | 2.06x |

The per-query threshold is unsuitable because it ignores the union push work
of the selected group. Push and pull do overlap, but the saved pull query work
does not cover the added shared-push work and memory-system interference.

## Stage 4: grouping oracle and SM search

Queries are sorted by exact frontier edge work. The oracle assigns only the
lowest-work `k` queries to push and sweeps small `k`. The best observed choices
are cit `k=1`, orkut `k=1`, twitter `k=8`, and sinaweibo `k=1`.

For all four best groups, `8/72` is the best tested SM allocation. Giving push
more SMs slows the concurrent iteration because pull loses useful SM capacity
while small push groups gain little.

Median of three key repeats:

| Graph | Push queries | All-pull ms | Concurrent 8/72 ms | Slowdown |
|---|---:|---:|---:|---:|
| cit-Patents | 1 | 17.454 | 18.470 | 5.8% |
| soc-orkut | 1 | 63.844 | 75.587 | 18.4% |
| soc-twitter | 8 | 209.025 | 211.772 | 1.3% |
| soc-sinaweibo | 1 | 246.531 | 296.655 | 20.3% |

Even the measured grouping oracle does not beat all-pull.

## Observation on the existing hybrid threshold

In the last all-push iteration before switching, fixed partitioning sometimes
beats the current all-push decision:

| Graph | Current all-push ms | Best split ms | Split vs current mode |
|---|---:|---:|---:|
| cit-Patents | 70.13 | 66.13 | 1.06x faster |
| soc-orkut | 179.21 | 173.21 | 1.03x faster |
| soc-twitter | 198.18 | 235.90 | slower |
| soc-sinaweibo | 555.79 | 434.15 | 1.28x faster |

However, all-pull at the same state is faster than the split for cit, orkut,
and sinaweibo. Twitter already prefers all-push. This means the larger issue is
the batch-wide push/pull switching threshold, not the absence of query-level
partitioning.

## Decision

The stop condition is reached. Query-level concurrent push/pull is not
integrated into the production hybrid engine because:

- no real first-pull workload beats the best uniform mode;
- no tested grouping oracle produces a positive result;
- partition merge and scan are small, but not zero;
- pull loses throughput when SMs are reserved for a small push group;
- push and pull contend for HBM/L2 while running concurrently.

The next optimization priority should be a measured batch-wide switching cost
model using actual shared-push union work and measured Q=64 pull cost. Query
partitioning should be reconsidered only if future kernels make pull cost scale
more strongly with active query count, or if another concurrent workload can
use push's underutilized SMs without reducing pull throughput.

## Artifacts

- `traces/*/hybrid_trace.csv`: existing Q=64 hybrid decisions
- `traces/*/query_work.csv`: per-query frontier work by BFS iteration
- `replay/*.csv`: threshold-based replay and SM sweeps
- `oracle/*.csv`: sorted low-work grouping search
- `sm_sweep/*.csv`: best-group SM allocation search
- `repeats/*.csv`: three key repetitions
- `examples/validate_query_partition_kernels.cu`: subset correctness
- `examples/trace_hybrid_q64.cu`: baseline trace
- `examples/trace_query_work_q64.cu`: per-query trace
- `examples/bench_query_partition_replay_q64.cu`: serial/concurrent replay
