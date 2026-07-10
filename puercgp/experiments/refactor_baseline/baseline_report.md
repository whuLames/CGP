# puercgp refactor baseline report

Baseline commit:

```text
9dddd10 chore: checkpoint puercgp before refactor
```

Refactor branch:

```text
refactor/puercgp-structure
```

Environment:

```text
GPU: Tesla V100-SXM2-32GB
CUDA compiler: nvcc 12.8.93
Build dir: /home/zyl/Projects/ocgp/puercgp/build
```

## Build

Command:

```bash
cmake --build /home/zyl/Projects/ocgp/puercgp/build -j 8
```

Result: PASS.

Known warnings:

- nvcc warns that offline compilation for architectures prior to sm_75 will be
  removed in a future release.
- `examples/validate_hybrid.cu` has an existing unused variable warning for
  `INF`.

## Correctness

Passed baseline checks:

- `smoke_hybrid_init`
- `smoke_hybrid_push`
- `smoke_frontier`
- `validate_bfs`
- `validate_sssp`
- `validate_wcc`
- `validate_hybrid`
- `smoke_replenish`
- `validate_replenish`

Known baseline issues:

- `smoke_reduce_ops` fails one existing assertion:
  - `push bfs level+1=4`
  - This happens before refactor changes and is treated as a baseline issue.
- `validate_hybrid_real` is not used as a refactor gate in the first pass:
  - with `weighted.mm`, BFS/SSSP mismatches remain at baseline.
  - with `unweighted.mm`, WCC mismatches remain at baseline.

Raw logs are saved under:

```text
puercgp/experiments/refactor_baseline/
```

The `.log` files are intentionally ignored by git.

## Light Performance Baseline

Command:

```bash
/home/zyl/Projects/ocgp/puercgp/build/bench_hybrid \
  /home/zyl/Projects/ocgp/puercgp/examples/matrices/unweighted.mm \
  0 0 1 --repeats=3 --warmup=1 --mode=push --push=warp
```

Median result:

```text
sequential wall_ms: 0.393494
hybrid wall_ms:     0.297496
speedup_vs_seq:     1.32269x
```

Command:

```bash
/home/zyl/Projects/ocgp/puercgp/build/bench_replenish \
  /home/zyl/Projects/ocgp/puercgp/examples/matrices/unweighted.mm \
  --bfs=2 --sssp=2 --wcc=0 --batch-size=2 \
  --repeats=3 --warmup=1 --mode=push --push=warp
```

Median result:

```text
sequential_total_ms: 0.56292
replenish_total_ms:  0.418042
speedup:             1.34656x
```

## Refactor Gates

The first refactor phases should use the passing checks above as the primary
correctness gate. The known baseline failures must not be counted as refactor
regressions unless their behavior changes unexpectedly.
