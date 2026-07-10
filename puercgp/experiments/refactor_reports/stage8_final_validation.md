# Stage 8 final validation report

Branch:

```text
refactor/puercgp-structure
```

Compared baseline:

```text
9dddd10 chore: checkpoint puercgp before refactor
```

Baseline was built in an isolated worktree:

```text
/home/zyl/Projects/ocgp_baseline_9dddd10
```

Both builds used:

```text
CMAKE_BUILD_TYPE=Release
CMAKE_CUDA_ARCHITECTURES=52
```

## Scope

Completed final validation after Stages 1-7:

- Rebuilt baseline and refactor code.
- Ran baseline correctness gate.
- Ran refactor correctness gate.
- Ran BFS performance matrix on:
  - `cit-Patents`
  - `soc-sinaweibo`
  - `soc-twitter`
  - `soc-orkut`
- Ran `Q=16/32/64` and `push/pull/hybrid` for BFS.
- Fixed `validate_sssp.cu` to load binary CSR directories through
  `load_graph_auto()`.
- Added the same large-graph CPU-reference skip behavior to `validate_sssp.cu`
  that `validate_bfs.cu` already uses.
- Ran refactor-only SSSP performance matrix after the loader fix.

Raw logs:

```text
puercgp/experiments/refactor_baseline/stage8_baseline_correctness.log
puercgp/experiments/refactor_baseline/stage8_baseline_perf_matrix.log
puercgp/experiments/refactor_reports/stage8_refactor_correctness.log
puercgp/experiments/refactor_reports/stage8_refactor_perf_matrix.log
puercgp/experiments/refactor_reports/stage8_refactor_sssp_perf_matrix_after_loader_fix.log
```

Summary tables:

```text
puercgp/experiments/refactor_reports/stage8_bfs_perf_summary.csv
puercgp/experiments/refactor_reports/stage8_sssp_refactor_perf_summary.csv
```

## Correctness

Baseline gate:

- PASS:
  - `smoke_hybrid_init`
  - `smoke_hybrid_push`
  - `smoke_frontier`
  - `validate_bfs`
  - `validate_sssp`
  - `validate_wcc`
  - `validate_hybrid`
  - `smoke_replenish`
  - `validate_replenish`
- Known baseline issue:
  - `smoke_reduce_ops` fails `push bfs level+1=4`.

Refactor gate:

- PASS:
  - `smoke_reduce_ops`
  - `smoke_frontier`
  - `validate_bfs`
  - `validate_sssp`
  - `validate_wcc`
  - `validate_hybrid`
  - `smoke_hybrid_init`
  - `smoke_hybrid_push`
  - `smoke_replenish`
  - `validate_replenish`

Refactor correctness status: PASS.

## BFS Performance Summary

Threshold used by the plan:

```text
refactor wall_ms <= baseline wall_ms * 1.03
```

Results:

```text
total BFS cases: 36
pass:            28
regression >3%:  8
```

Mode-level summary:

```text
push:   12/12 pass, average ratio 0.943x, max ratio 0.990x
pull:    8/12 pass, average ratio 1.043x, max ratio 1.416x
hybrid:  8/12 pass, average ratio 0.995x, max ratio 1.547x
```

Regressions over 3%:

```text
dataset          Q   mode    baseline_ms  refactor_ms  ratio
cit-Patents      64  pull    279.026      316.062      1.133
soc-orkut        64  pull    655.026      716.723      1.094
soc-orkut        64  hybrid  277.706      387.657      1.396
soc-sinaweibo    16  pull    703.430      996.408      1.416
soc-sinaweibo    64  pull    2098.840     2226.360     1.061
soc-twitter      16  hybrid  625.554      750.918      1.200
soc-twitter      32  hybrid  712.414      846.271      1.188
soc-twitter      64  hybrid  2218.470     3431.300     1.547
```

Performance gate status: NOT PASS.

## Interpretation

Push path is performance-safe in this run. Every BFS push case is faster than
baseline or within the 3% tolerance.

The remaining regressions are concentrated in pull and hybrid. The most severe
hybrid regression is not caused by a single slow push kernel. In
`soc-twitter Q=64 hybrid`, baseline switches back to push after three pull
iterations, while the refactor path keeps using pull for two additional rounds.
Those two extra pull rounds account for most of the wall-time increase.

Part of this is expected from Stage 4: pull now reads incoming adjacency for
directed graphs. The baseline pull path used outgoing adjacency in the hybrid
pull kernel, which was faster in some cases but was not semantically correct for
directed pull traversal. The current result is therefore not a pure kernel-level
apples-to-apples comparison.

There is still a real scheduling issue to fix: the current hybrid decision uses
only `current_unique >= pull_frontier_ratio * V`. That is not enough to switch
back to push once the active frontier is still large by unique-vertex count but
cheap enough by edge work. Restoring an edge-count / degree-scan based decision,
or an equivalent push-cost estimator, is the next required performance task.

## SSSP Performance

Baseline SSSP large-graph performance is not comparable because baseline
`validate_sssp.cu` attempts to parse `/home/zyl/data/csr_data/<dataset>` as a
MatrixMarket file and aborts with:

```text
missing matrix dimensions: /home/zyl/data/csr_data/<dataset>
```

The refactor branch fixes this CLI path by using `load_graph_auto()` and skips
CPU reference checks for large graphs. Refactor-only SSSP results are recorded
in:

```text
puercgp/experiments/refactor_reports/stage8_sssp_refactor_perf_summary.csv
```

Large-graph SSSP correctness is not proven by those performance runs because
CPU reference is intentionally disabled for graphs with more than 1M vertices.
Toy SSSP correctness remains covered by `validate_sssp`.

## Deleted And Kept Kernels

Removed from the default implementation:

- legacy list frontier path;
- bitmap pull strategy enum/path;
- GE-SpMM pull strategy enum/path;
- dense/bitmap/list frontier representations from the public options;
- old degree-classified shared-node push strategies from public options.

Kept:

- `expand_shared_node_query_parallel_kernel` as an experimental benchmark path;
- shared-frontier push block and warp kernels;
- fused pull simple/smem kernels;
- hybrid fused push/pull kernels;
- replenish slot snapshot/reset/reinit kernels behind `slot_io_manager`.

## Follow-Up Required

Before considering the refactor fully performance-clean:

1. Add a hybrid scheduling signal beyond unique frontier count.
2. Re-run the BFS matrix after that change.
3. Decide whether directed-pull correctness should be compared only against a
   corrected baseline, not against baseline commit `9dddd10`.
4. Add a real large-graph WCC benchmark CLI if WCC standalone performance is
   part of the final paper gate.
