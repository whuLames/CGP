# Stage 9 hybrid scheduler validation

## Design

The previous shared-frontier hybrid decision used only unique frontier vertices:

```text
current_unique >= pull_frontier_ratio * V
```

This is not a good proxy for push cost. In the shared frontier representation,
push work is closer to:

```text
sum(out_degree(v) * active_query_count(v)), for v in current frontier
```

Pull work is closer to a full graph scan:

```text
E * Q
```

Stage 9 therefore adds a shared-frontier metrics kernel and makes homogeneous
`frontier_engine` choose pull only when:

```text
virtual_edge_count >= pull_edge_ratio * E * Q
```

The kernel also reports:

- `actual_edge_count`: `sum(out_degree(v))`
- `virtual_edge_count`: `sum(out_degree(v) * active_query_count(v))`
- `active_pair_count`: `sum(active_query_count(v))`

`pull_frontier_threshold` is still recorded for profiling, but it is no longer
the hybrid decision gate.

## Pull kernel selection

The original refactor used `fused_pull_smem_kernel` for `Q > 32`. On the Stage 9
BFS hybrid matrix, the simple row-query layout is consistently faster for the
tested `Q=16/32/64` cases. The default fused pull launcher now uses
`fused_pull_simple_kernel` for all `Q <= 64`; the smem kernel remains available
in the source for focused experiments.

## Tested variants

```text
variant A: pair-capacity threshold + edge-work fallback
  - fixed the extra-pull regression on soc-twitter
  - remaining regression: soc-orkut Q=64, 295.135 ms vs 277.706 ms

variant B: edge-work-only scheduler
  - improved soc-twitter further by reducing unnecessary pull rounds
  - remaining regression: soc-orkut Q=64, 295.170 ms vs 277.706 ms

variant C: edge-work-only scheduler + simple pull for Q<=64
  - all hybrid BFS cases pass the 3% baseline gate
```

## Final BFS hybrid matrix

Threshold:

```text
stage9 wall_ms <= baseline wall_ms * 1.03
```

```text
dataset        Q   baseline_ms  stage9_ms  ratio  verdict
cit-Patents    16  101.690      101.150    0.995  PASS
cit-Patents    32  143.269      143.444    1.001  PASS
cit-Patents    64  224.805      200.941    0.894  PASS
soc-sinaweibo  16  346.509      350.407    1.011  PASS
soc-sinaweibo  32  772.675      768.571    0.995  PASS
soc-sinaweibo  64  1288.430     1070.830   0.831  PASS
soc-twitter    16  625.554      534.540    0.855  PASS
soc-twitter    32  712.414      621.678    0.873  PASS
soc-twitter    64  2218.470     898.428    0.405  PASS
soc-orkut      16  93.063       92.684     0.996  PASS
soc-orkut      32  124.403      124.291    0.999  PASS
soc-orkut      64  277.706      213.804    0.770  PASS
```

Final status:

```text
12/12 BFS hybrid cases pass.
```

Raw local logs:

```text
puercgp/experiments/refactor_reports/stage9_hybrid_scheduler_perf_matrix.log
puercgp/experiments/refactor_reports/stage9_hybrid_scheduler_summary.csv
```

These raw files are ignored by `.gitignore` (`*.log`, `*.csv`) and should be
added with `git add -f` only if raw artifacts need to be committed.

## Verification

Correctness:

```text
./build/smoke_frontier
./build/validate_bfs ./examples/matrices/unweighted.mm 0,1,2,3 3 hybrid shared_node_warp fused
./build/validate_hybrid
```

Performance:

```text
4 datasets x Q=16/32/64 x BFS hybrid x repeats=3
datasets: cit-Patents, soc-sinaweibo, soc-twitter, soc-orkut
```
