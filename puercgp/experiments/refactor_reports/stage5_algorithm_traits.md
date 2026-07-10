# Stage 5 algorithm traits report

Branch:

```text
refactor/puercgp-structure
```

## Scope

Completed Stage 5:

- Added static algorithm traits:
  - `include/puercgp/algorithms/algorithm_traits.hxx`
  - `include/puercgp/algorithms/init_traits.hxx`
  - `include/puercgp/algorithms/dispatcher.hxx`
- Added static `algorithm_kind` and `init_mode` metadata to BFS, SSSP, and WCC
  policies.
- Moved hybrid candidate computation from `core/reduce_ops.hxx` into the
  algorithms dispatcher.
- Kept `core/reduce_ops.hxx` focused on atomic reduce primitives.
- Updated `smoke_reduce_ops` to test dispatcher semantics directly.

## Build

Command:

```bash
cmake --build /home/zyl/Projects/ocgp/puercgp/build -j 8
```

Result: PASS.

Known warnings match previous stages:

- nvcc deprecated offline compilation target warning.
- `validate_hybrid.cu` unused `INF` warning.

## Correctness

Gate command group:

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

Result: PASS.

Additional directed pull/hybrid checks:

- BFS pull/hybrid: `distance_mismatches=0`
- SSSP pull/hybrid: `distance_mismatches=0`, `max_abs_delta=0`

Notable change:

- `smoke_reduce_ops` now passes. The BFS push dispatcher test expects
  `source_value + 1`, matching the current replenishment-safe BFS semantics.

## Performance sanity

Command output saved in:

```text
puercgp/experiments/refactor_reports/stage5_perf_light.log
```

Small graph results:

```text
bench_hybrid:    GO, hybrid speedup 1.22262x
bench_replenish: speedup 1.46005x
```

Full Stage 5 performance gate C remains part of the final reproducible
performance pass.
