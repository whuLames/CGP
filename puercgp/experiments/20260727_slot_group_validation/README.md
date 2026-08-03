# Slot-group validation: replenish granularity and SM-partitioned groups

Configuration fixed before measurement (2026-07-27).

## Question

Two hypotheses behind the proposed "fixed slot groups + per-group SM
partition" design:

- **E0 (granularity)**: immediate replenish (`chunk=1`) loses to static
  batching because per-slot injection mixes BFS phases; larger replenish
  cohorts should recover the loss monotonically, or expose an optimal
  intermediate granularity.
- **E1 (spatial isolation)**: shared push saturates at 8-16 SM in isolation,
  so G concurrent push groups on disjoint Green Context SM partitions should
  beat one full-GPU batch — unless the real bottleneck is the shared memory
  system (HBM/L2), which Green Contexts do not partition.

## E0 protocol

Identical to `experiments/20260712_roadnet_replenish` for direct
comparability:

- graph: roadNet-CA CSR (`/home/zyl/data/csr_data/roadNet-CA`, V=1,965,206
  E=5,533,214);
- BFS-only, N=400/800, sources from
  `experiments/20260712_roadnet_replenish/sources/sources_n{400,800}.csv`;
- Q(batch_size)=32, mode=push, push=shared_node_warp, no-discard;
- warmup=1, repeats=3, median reported;
- sweep `replenish_chunk` in {1, 4, 8, 16, 32}; sequential static batching
  measured once per N in the chunk=1 run.

## E1 protocol

- `green_context_group`: N-way `cuDevSmResourceSplitByCount` partition
  (new, generalizes the existing pair helper);
- `bench_concurrent_push_groups`: G host threads, each with private BFS
  push state (values/masks/frontiers) over its own query subset, running the
  shared-frontier warp push loop on its group's green-context stream;
  makespan = max over groups, wall-clocked from a common barrier;
- workload: N=64 BFS queries per graph, seed=42 unique sources, identical
  across configs; graphs: soc-orkut, soc-twitter, roadNet-CA;
- configs:
  - A0: 1 group x Q=64, full GPU (no green context);
  - A1: 2 x Q=32 on 40+40 SM;
  - A2: 2 x Q=32 on 16+16 SM (48 SMs left idle on purpose);
  - A3: 4 x Q=16 on 20x4 SM;
  - A4: 2 x Q=32 sequential on full GPU (grouping loss without concurrency);
  - interference probe: group 0 of A1 run alone on its 40-SM partition
    (T_alone) vs inside A1 (T_concurrent);
- warmup=1, repeats=5, median makespan; query order fixed, contiguous split
  (queries [0,31] -> group 0, [32,63] -> group 1, ...).

## Decision criteria (fixed in advance)

- E0: monotone improvement toward chunk=32 confirms phase-mixing as the
  dominant cost; an interior optimum sets the group size for the E2 engine.
- E1: best multi-group config >= 1.10x A0 on >= 2 graphs -> spatial
  isolation stays alive (E3); within +-5% -> memory-bound confirmed, drop
  SM partitioning and keep only time-shared groups; A1 loses but A4 ~= A0
  -> loss comes from concurrency interference, not from grouping.

## Outcome (2026-07-27)

- E0: monotone curve, no interior optimum; chunk=32 still 0.98x static.
  See `RESULTS_E0.md`.
- E1: multi-group beats A0 only on roadNet-CA (A3 = 1.24x); social graphs
  lose 17-35% with 1.49-2.05x memory-interference factors. Spatial SM
  isolation rejected as a general mechanism. See `RESULTS_E1.md`.
- Deviations from protocol: green_context_group requires uniform partitions
  (driver cannot re-split a `remaining` resource, one N-way split call);
  affinity-based grouping sub-variant not run (interference dominates the
  social-graph outcome regardless of grouping quality).

The validation was extended to N=256/400/800, ordinary streams, pull and
hybrid modes, immediate-replenish controls, correctness checks, and Nsight
Compute attribution. The complete Chinese report is in
[`FINAL_ANALYSIS.md`](FINAL_ANALYSIS.md); structured tables and the summary
plot are in `results/`.
