# BFS phase scheduling: N=128, Q=32

## Setup

- Workload: push-only BFS
- Total queries: N=128
- Resident slots per batch: Q=32
- Sources: 128 distinct vertices generated with seed 42
- GPU: one fixed GPU (`CUDA_VISIBLE_DEVICES=2`)
- Timing: median of 5 runs; profiling and schedule construction are excluded
- Correctness: visited vertex count and BFS distance checksum for every query
- Graphs: cit-Patents, soc-orkut, soc-twitter, soc-sinaweibo

The same 128 sources are used by every method on a graph. All methods passed
the per-query correctness comparison.

## Methods

### sequential_baseline

Queries retain random source order and are split consecutively into four
32-query batches. Every unfinished query advances at every global step.

### affinity_baseline

The four initial batches are profiled once. For each sampled frontier vertex,
the query, BFS level, and vertex degree are recorded. Pairwise affinity is the
maximum degree-weighted overlap over relative level offsets in `[-16, 16]`.

Four capacity-constrained batches are built greedily. A batch starts from the
unassigned query with the largest remaining affinity, then repeatedly adds the
query with the largest summed affinity to current batch members. Queries in a
selected batch still execute in lockstep.

### affinity_offset

Uses the affinity batches. Within each batch, each query receives one start
offset in `[0, 16]`. Multi-start coordinate ascent maximizes sampled pairwise
overlap. Once started, a query runs continuously and is never paused.

### affinity_pause

Uses the affinity batches and a one-step overlap lookahead. At each step it may
pause a query when retaining its current frontier is predicted to overlap more
with the other queries' next frontiers than advancing it. At least 16 queries
advance, and a query can be paused for at most three consecutive steps. Paused
frontiers are carried to the next global step and later resumed.

## Results

| Graph | Method | GPU ms | Speedup | Union edges | Sharing | Steps |
|---|---|---:|---:|---:|---:|---:|
| cit-Patents | sequential baseline | 1032.87 | 1.000x | 744,877,019 | 5.675x | 87 |
| cit-Patents | affinity baseline | 965.21 | 1.070x | 683,099,960 | 6.188x | 85 |
| cit-Patents | affinity offset | 774.59 | 1.333x | 517,407,280 | 8.170x | 88 |
| cit-Patents | affinity pause | 1280.47 | 0.807x | 981,304,908 | 4.308x | 129 |
| soc-orkut | sequential baseline | 1447.06 | 1.000x | 2,886,721,088 | 9.431x | 36 |
| soc-orkut | affinity baseline | 1201.89 | 1.204x | 2,352,296,374 | 11.574x | 36 |
| soc-orkut | affinity offset | 1097.79 | 1.318x | 2,066,215,553 | 13.177x | 38 |
| soc-orkut | affinity pause | 1773.53 | 0.816x | 4,099,961,167 | 6.640x | 54 |
| soc-twitter | sequential baseline | 5147.22 | 1.000x | 9,462,072,564 | 7.170x | 77 |
| soc-twitter | affinity baseline | 4903.43 | 1.050x | 9,349,466,863 | 7.257x | 77 |
| soc-twitter | affinity offset | 2423.58 | 2.124x | 3,915,890,805 | 17.326x | 79 |
| soc-twitter | affinity pause | 4808.20 | 1.071x | 9,291,661,872 | 7.302x | 88 |
| soc-sinaweibo | sequential baseline | 5484.57 | 1.000x | 4,973,153,252 | 13.452x | 32 |
| soc-sinaweibo | affinity baseline | 4435.21 | 1.237x | 4,149,940,364 | 16.120x | 29 |
| soc-sinaweibo | affinity offset | 3955.83 | 1.386x | 3,636,358,509 | 18.397x | 29 |
| soc-sinaweibo | affinity pause | 6163.54 | 0.890x | 6,092,633,768 | 10.980x | 42 |

## Findings

Selective batching is effective in this N>Q experiment. It improves all four
graphs by 5.0%-23.7%, and the speedups track reductions in union edge work.
The benefit is especially notable on orkut and sinaweibo, where N=Q previously
left little room for alignment but N=128 allows choosing compatible queries.

Batching plus a fixed start offset is the strongest tested method. It improves
all four graphs by 31.8%-112.4%. Twitter has large phase misalignment: offset
alignment reduces union edge work by 58.6% and raises sharing from 7.17x to
17.33x.

The tested pause/resume heuristic is not a generally useful policy. It helps
twitter by 7.1% relative to the original baseline, but is much slower than
offset alignment and regresses the other three graphs. Its one-step objective
can pause a locally attractive query while causing later frontier divergence,
more global steps, carry-frontier overhead, and much larger cumulative union
work. A useful dynamic policy needs a multi-step benefit estimate and should
use offset alignment as its starting schedule, only pausing when predicted
savings exceed both carry cost and added critical-path work.

## Artifacts

- `*/sources.csv`: fixed source list
- `*/affinity.csv`: pairwise sampled affinity
- `*/baseline_batches.csv`: sequential batch assignment
- `*/affinity_batches.csv`: selected batch assignment
- `*/summary.csv`: median timing and work metrics
- `*/<method>/batch_*/`: per-step schedules and traces

Benchmark implementation:
`examples/bench_phase_schedule_n128.cu`.
