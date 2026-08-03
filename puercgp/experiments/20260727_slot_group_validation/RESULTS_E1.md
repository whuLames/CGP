# E1: SM-partitioned concurrent push groups (decisive experiment)

N=64 BFS queries (seed=42 unique random sources), contiguous split into G
groups, production push engine (`run_heterogeneous`, shared_node_warp,
max_iterations=20000) per group. warmup=1, repeats=5, median makespan.
Fingerprints (per-query reached count + level sum) match A0 exactly on all
four graphs for the A3 4-way split.

Implementation note: `remaining` resources from `cuDevSmResourceSplitByCount`
cannot be split again on this driver (CUDA 12.8 / V100), so
`green_context_group` performs one N-way split call; partitions must be
uniform.

## Makespan (ms, median of 5)

| config | cit-Patents | soc-orkut | soc-twitter | roadNet-CA |
|---|---:|---:|---:|---:|
| A0: 1xQ64 full GPU        | 374.3 | 487.1  | 1790.0 | 484.6 |
| A1: 2xQ32 @ 40+40         | 459.0 | 746.1  | 2158.7 | 438.6 |
| A2: 2xQ32 @ 16+16 (48 idle)| 471.5 | 769.5  | 2210.6 | 495.7 |
| A3: 4xQ16 @ 20x4          | 660.6 | 1228.6 | 3309.2 | 390.8 |
| A4: 2xQ32 sequential full GPU | 551.9 | 754.4 | 2610.8 | 549.7 |
| probe: group0 alone @ 40SM | 274.9 | 360.2  | 1440.7 | 276.1 |

Best multi-group vs A0: 0.82x / 0.65x / 0.83x / **1.24x**.

## Memory-interference factor (A1 group0 wall / alone40 wall)

| graph | T_concurrent | T_alone | factor |
|---|---:|---:|---:|
| cit-Patents | 455.3 | 271.9 | 1.67x |
| soc-orkut   | 733.3 | 357.2 | 2.05x |
| soc-twitter | 2134.2 | 1435.6 | 1.49x |
| roadNet-CA  | 428.3 | 273.4 | 1.57x |

Green Contexts partition SMs only: running the identical Q=32 workload on the
same 40-SM partition next to a concurrent sibling inflates its wall time by
1.5-2.05x. The bottleneck the groups fight over is HBM/L2, not SMs.

## Supporting observations

1. A2 ~= A1 everywhere (within 3-6%): 16 SMs per group perform like 40 SMs
   per group under concurrency — per-group push is memory-bound long before
   it is SM-bound, consistent with the isolated SM-scaling curves.
2. A4 per-group walls (~270-380 on full GPU) ~= alone40 walls (Q32 push
   saturates at <=40 SM even in isolation).
3. soc-orkut: A4 (754) ~= A1 (746): with a 2.05x interference factor,
   concurrency buys nothing over running the two halves serially; both lose
   to A0 because splitting Q64 into 2xQ32 forfeits shared-frontier work
   (368+381 serial vs 487 fused).
4. roadNet-CA is the exception: A1 = 1.11x, A3 = 1.24x over A0. Long-diameter
   graphs run ~850 iterations of tiny frontiers; each step is
   latency/launch-bound, not bandwidth-bound, so concurrent groups overlap
   step latencies profitably. This is precisely the workload class where E0
   showed replenish granularity cannot win — a different lever works there.

## Verdict against pre-registered criteria

"Best multi-group >= 1.10x A0 on >= 2 graphs" fails (1 of 4). Spatial SM
isolation is rejected as a general mechanism; retained as a special-case
optimization for high-diameter / small-frontier workloads. E3 (asymmetric
push-partition / pull-timeslice scheduling) is not pursued on social-graph
workloads.
