# Stage 7 slot I/O manager report

Branch:

```text
refactor/puercgp-structure
```

## Scope

Completed Stage 7 replenish slot I/O extraction:

- Added `kernels/replenish/slot_io_kernels.hxx`.
- Added `engine/slot_io_manager.hxx`.
- Moved replenish snapshot/reset/reinit helper kernels out of
  `engine/hybrid_engine.hxx`.
- Updated `engine/replenish_engine.hxx` to call `slot_io_manager` for:
  - final snapshot of still-active slots at truncation;
  - batch snapshot and reset of converged slots;
  - batch source initialization for newly replenished slots.
- Added `slot_io_manager.hxx` to the umbrella include `puercgp.hxx`.

The migration preserves the existing single-stream behavior. The async entry
point currently forwards to the synchronous same-stream implementation, and
`wait_before_reuse()` is a no-op placeholder for later stream/event based slot
reset scheduling.

Not implemented in this stage:

- No dynamic scheduling enablement.
- No separate reset stream.
- No event dependency graph between compute and slot I/O streams.
- No green context / libsmctrl resource control integration.

## Build

Command:

```bash
cmake --build /home/zyl/Projects/ocgp/puercgp/build -j 8
```

Result: PASS.

Known warnings:

- nvcc deprecated offline compilation warning for old GPU targets.
- Existing `validate_hybrid.cu` unused `INF` warning.

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

## Performance sanity

Command output saved in:

```text
puercgp/experiments/refactor_reports/stage7_perf_light.log
```

Small graph results:

```text
bench_hybrid:    GO, hybrid speedup 1.19943x
bench_replenish: speedup 1.45375x
```

This stage is expected to be performance-neutral because it only moves helper
kernels behind a manager interface and keeps the same launch sequence.
