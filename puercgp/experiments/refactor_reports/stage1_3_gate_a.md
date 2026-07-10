# Stage 1-3 gate A report

Branch:

```text
refactor/puercgp-structure
```

Latest commit before this report:

```text
1ff9965 refactor: introduce puercgp workspaces
```

## Scope

Stages covered:

- Stage 1: split core helper headers.
- Stage 2: introduce shared engine/pull workspaces and wire hybrid engine.
- Stage 3: move shared push, fused pull, and pull postprocess kernels into
  dedicated headers; introduce lightweight push/pull executor wrappers.

## Build

Command:

```bash
cmake --build /home/zyl/Projects/ocgp/puercgp/build -j 8
```

Result: PASS.

Known warnings match baseline:

- nvcc deprecated offline compilation target warning.
- `validate_hybrid.cu` unused `INF` warning.

## Correctness

Gate command group:

- `smoke_hybrid_init`
- `smoke_hybrid_push`
- `smoke_frontier`
- `validate_bfs`
- `validate_sssp`
- `validate_wcc`
- `validate_hybrid`
- `smoke_replenish`
- `validate_replenish`

Result: PASS.

Known baseline failures remain excluded from this gate:

- `smoke_reduce_ops`
- `validate_hybrid_real`

## Performance Sanity

Light small-graph sanity:

- `bench_hybrid` on `examples/matrices/unweighted.mm`: PASS.
- `bench_replenish` on `examples/matrices/unweighted.mm`: PASS.

Real-graph gate A:

```text
cit-Patents Q=16 push:   PASS
cit-Patents Q=16 pull:   PASS
cit-Patents Q=16 hybrid: PASS
cit-Patents Q=32 push:   PASS
cit-Patents Q=32 pull:   PASS
cit-Patents Q=32 hybrid: PASS
soc-orkut   Q=16 push:   benchmark reports NO-GO vs sequential, same metric
                            category as baseline algorithm comparison; not
                            treated as refactor regression.
soc-orkut   Q=16 pull:   PASS
soc-orkut   Q=16 hybrid: PASS
```

Raw logs:

```text
puercgp/experiments/refactor_reports/stage1_correctness.log
puercgp/experiments/refactor_reports/stage1_perf_light.log
puercgp/experiments/refactor_reports/stage2_correctness.log
puercgp/experiments/refactor_reports/stage2_perf_light.log
puercgp/experiments/refactor_reports/stage3_correctness.log
puercgp/experiments/refactor_reports/stage3_perf_gate_a.log
```

The `.log` files are ignored by git.
