# Phase-aware shared-frontier scheduling review

## 1. Motivation

The push kernel scans the outgoing edges of the union frontier once and applies
the query mask carried by each frontier vertex. Its physical edge work is
therefore closer to the degree-weighted union of query frontiers than to the
sum of 32 independent BFS frontiers. Query execution order can change this
union without changing BFS semantics.

The scheduling study asked two questions:

1. For N=Q=32, can iteration alignment reduce union edge work?
2. For N=128 and Q=32, can selecting the queries in each batch create more
   sharing, and can pause/resume improve further?

All experiments use push-only BFS, distinct random sources with seed 42, and
the same source set for every compared strategy. Correctness is checked per
query using both visited-vertex count and BFS-distance checksum.

## 2. Trace and metrics

Each global step records:

- local BFS iteration of every query;
- alive and scheduled query masks;
- union frontier vertex count;
- per-query frontier vertices and edge work;
- virtual edge work: sum of edge work over queries;
- union edge work: edges scanned by the shared push representation;
- sharing factor: virtual edge work / union edge work;
- measured push-kernel latency.

For scheduling search, frontier vertices are deterministically sampled. A
sample contains vertex ID, degree, query mask, and BFS iteration. Pairwise
overlap is degree weighted because sharing a high-degree vertex removes more
physical edge scans than sharing a low-degree vertex.

## 3. N=Q=32 strategies

### Lockstep baseline

All unfinished queries advance once per global step.

### Heavy alignment

For each query, identify the iteration with maximum frontier edge work. Delay
query starts so these peak iterations occur at the same global step. Once a
query starts, it never pauses.

### Offset search

For every query pair and relative iteration offset, estimate degree-weighted
overlap from the trace. Search offsets in `[0,16]` with multiple initial states
and coordinate ascent. This is a sampled offset oracle, not an exhaustive
global optimum. Queries run continuously after their offset expires.

### Greedy pause

At each step, use one-step overlap lookahead to decide whether keeping a query
at its current frontier appears better than advancing it. At least 16 queries
must advance and no query can pause for more than three consecutive steps.

### N=32 results

| Graph | Baseline ms | Heavy speedup | Offset speedup | Pause speedup |
|---|---:|---:|---:|---:|
| cit-Patents | 272.983 | 1.279x | 1.305x | 0.773x |
| soc-orkut | 368.414 | 1.011x | 1.035x | 0.573x |
| soc-twitter | 1488.960 | 2.052x | 2.049x | 1.135x |
| soc-sinaweibo | 1342.920 | 0.976x | 1.051x | 0.720x |

The result established that phase misalignment can be large, especially on
twitter, but fixed random batches can also have little remaining headroom.
Low diameter alone is not sufficient: useful alignment requires dense
frontiers, different arrival times, and unsaturated baseline sharing.

## 4. N=128, Q=32 strategies

### Sequential batching

The 128-source sequence is split consecutively into four batches. Queries in
each batch execute in lockstep.

### Selective batching

Profile the initial four batches and construct a 128-by-128 affinity matrix:

```text
A(i,j) = max over delta in [-16,16]
         degree_weighted_overlap(F_i[level], F_j[level + delta])
```

Build four capacity-32 batches greedily. Start a batch with the unassigned
query having the largest affinity to remaining queries, then repeatedly add
the query with maximum summed affinity to current batch members.

### Selective batching plus offset

After grouping, run the N=32 offset search independently inside each batch.
This separates the decisions of which queries run together and when they
start.

### Selective batching plus pause

Run the bounded one-step pause policy inside each selected batch. Paused
frontiers are explicitly carried into the next global step and later resumed.

### N=128 results

| Graph | Selective batching | Batching + offset | Batching + pause |
|---|---:|---:|---:|
| cit-Patents | 1.070x | 1.333x | 0.807x |
| soc-orkut | 1.204x | 1.318x | 0.816x |
| soc-twitter | 1.050x | 2.124x | 1.071x |
| soc-sinaweibo | 1.237x | 1.386x | 0.890x |

Selective batching improves all four graphs by 5.0%-23.7%. Fixed offset
alignment improves all four by 31.8%-112.4% and is the strongest strategy.

Pause/resume is not retained. One-step decisions can improve immediate overlap
while increasing later union work, traversal steps, and frontier-carry cost.
It only beats the original baseline on twitter and remains far behind offset
alignment there. The retained design is therefore:

```text
sharing-aware batch formation -> per-batch fixed offset -> continuous run
```

## 5. Experimental limitations

- Profiling and schedule search are excluded from kernel timing. An online
  system must amortize or predict these costs.
- Affinity uses deterministic sampled frontiers rather than exact intersections.
- The source distribution is one fixed seed; more seeds are needed for a paper
  claim.
- Experiments are push-only. Hybrid push/pull switching must compare pull cost
  against shared push union work rather than summed per-query push work.
- Offset search is an approximate oracle and is not yet an online scheduler.

## 6. Artifacts

- N=32 report: `experiments/20260711-152345_phase_schedule_q32/README.md`
- N=128 report: `experiments/20260711_phase_schedule_n128/README.md`
- N=32 benchmark: `examples/bench_phase_schedule.cu`
- N=128 benchmark: `examples/bench_phase_schedule_n128.cu`
- Per-query sources, affinity matrices, batch assignments, schedules, and
  per-step traces are stored in the corresponding experiment directories.
