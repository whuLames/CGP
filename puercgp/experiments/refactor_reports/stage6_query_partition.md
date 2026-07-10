# Stage 6 query partition report

Branch:

```text
refactor/puercgp-structure
```

## Scope

Completed Stage 6 interface work:

- Added `engine/query_partition.hxx`.
- Added `engine/execution_lane.hxx`.
- Added `query_slots_mask()` and `query_partition_t::all_slots()`.
- Added `active_slots` to homogeneous shared push executor APIs.
- Added `active_slots` to homogeneous shared push kernels:
  - `expand_shared_node_kernel`
  - `expand_shared_node_query_parallel_kernel`
  - `expand_shared_node_warp_kernel`
- Homogeneous `frontier_engine` now creates an all-slots partition and passes it
  through push launch wrappers.

Not implemented in this stage:

- No partial slot push/pull scheduling.
- No multiple stream partition execution.
- No green context / libsmctrl integration.
- No partial-partition frontier merge.

## Build

Command:

```bash
cmake --build /home/zyl/Projects/ocgp/puercgp/build -j 8
```

Result: PASS.

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

All default runs use `active_slots = query_slots_mask(query_count)`, so the
stage validates the default all-slots path.

## Performance sanity

Command output saved in:

```text
puercgp/experiments/refactor_reports/stage6_perf_light.log
```

Small graph results:

```text
bench_hybrid:    GO, hybrid speedup 1.1877x
bench_replenish: speedup 1.44573x
```

Full partial-slot performance validation is deferred until partial scheduling is
actually implemented.
