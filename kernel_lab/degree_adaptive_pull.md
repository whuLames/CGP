# Degree-adaptive multi-warp Pull microbenchmark

## Question

Can inter-warp load imbalance in the Q=32 SSSP Pull kernel be reduced by
assigning more warps to higher-in-degree vertices, without changing the
GraphWeft execution engine?

## Implementation

`degree_adaptive_pull.cu` is a standalone kernel microbenchmark.  It uses
32 concurrent queries, with one lane per query.  The baseline assigns one
warp to every destination vertex.  The adaptive variant statically buckets
vertices by incoming degree and assigns 1, 2, 4, or 8 warps:

```
wanted(v) = ceil(in_degree(v) / C)
warps(v)  = round_up_to_1_2_4_8(wanted(v)), capped at 8
```

Warps assigned to the same destination traverse disjoint incoming-edge
streams and reduce their 32 per-query minima in shared memory.  Static degree
bucket construction and host/device setup are outside the timed region; all
nonempty bucket kernel launches are included.  Baseline and adaptive samples
are interleaved to control GPU clock drift.  Every reported configuration is
checked cell-by-cell against the baseline output.

The input values are deterministic and finite, so this experiment measures a
dense SSSP Pull step rather than an end-to-end iteration trajectory.

## V100 results

The table reports the best tested fixed edge capacity, which was `C=32` for
the degree-skewed graphs.  Each configuration used five paired warmups and 21
paired samples (11 for soc-twitter); values are median kernel times.

| Graph | Vertices | Edges | W1 baseline (ms) | Adaptive (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| cit-Patents | 3,774,768 | 33,037,895 | 9.599 | 9.276 | 1.035x |
| soc-LiveJournal1 | 4,847,571 | 137,987,546 | 20.221 | 17.998 | 1.124x |
| soc-orkut | 2,997,166 | 212,698,418 | 37.624 | 33.867 | 1.111x |
| soc-twitter | 21,297,772 | 530,051,618 | 156.563 | 86.535 | 1.809x |
| roadNet-CA | 1,965,206 | 5,533,214 | 1.590 | 1.589 | 1.001x |

roadNet-CA has no vertex with in-degree above 32, so both paths reduce to the
same W1 kernel.  This is a negative control: after paired timing, the apparent
difference is below 0.1%.  In contrast, soc-twitter assigns 368,145,005 of its
530,051,618 edges to the W8 bucket at `C=32`, and obtains the largest speedup.

## Interpretation and limits

The experiment supports the mechanism: degree-aware multi-warp ownership can
substantially reduce Pull-kernel load imbalance on graphs whose incoming-edge
work is concentrated in hubs.  It provides little benefit when degrees are
uniformly small.  The result does not yet establish an end-to-end GraphWeft
speedup: it excludes frontier-dependent reachability, mixed algorithms,
engine overhead, and any dynamic maintenance cost.  The next validation step
should replay captured production Pull rounds while retaining the static
degree buckets.

## Reproduction

```bash
cmake --build build --target graphweft_degree_pull -j4
CUDA_VISIBLE_DEVICES=0 ./build/graphweft_degree_pull GRAPH_DIRECTORY 0 21
```
