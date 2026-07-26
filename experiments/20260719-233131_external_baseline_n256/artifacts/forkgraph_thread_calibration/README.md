# ForkGraph CPU Configuration Audit

## Finding

The initial ForkGraph measurements used 96 OpenMP threads without affinity and
eight range partitions on every graph. Those measurements are retained only as
preliminary diagnostics. They are not valid primary baseline numbers.

ForkGraph parallelizes across active queries while processing each query's
partition-local work sequentially. With Q=64, 96 OpenMP workers cannot all do
useful query work. SMT siblings also contend on the sequential priority-queue
work and OpenMP critical sections.

## Thread Calibration

The calibration uses the same `cit-Patents` Q=64 source batch and the original
P=8 input. Values are median internal compute time over three measurements.

| Configuration | Median (ms) | Relative to old setting |
|---|---:|---:|
| 24 physical cores, bound | 16,264 | 3.65x |
| 32 physical cores, bound | 15,787 | 3.76x |
| 40 physical cores, bound | 17,499 | 3.39x |
| 48 physical cores, bound | **15,328** | **3.87x** |
| 64 threads, bound | 22,635 | 2.62x |
| 96 logical threads, bound | 56,797 | 1.04x |
| 96 logical threads, unbound | 59,342 | 1.00x |

The corrected formal setting is:

```text
OMP_NUM_THREADS=48
OMP_DYNAMIC=FALSE
OMP_PLACES=cores
OMP_PROC_BIND=spread
```

## Partition Calibration

ForkGraph requires graph partitions to be approximately LLC-sized. This host
has two 35.8 MiB L3 caches. The corrected preprocessing creates contiguous,
edge-balanced range partitions capped at 35.75 MiB per partition:

| Dataset | Partitions |
|---|---:|
| cit-Patents | 9 |
| soc-orkut | 47 |
| soc-twitter | 123 |
| soc-sinaweibo | 137 |

The source artifact does not ship its paper partitioning pipeline. The local
preprocessor therefore preserves vertex IDs and uses contiguous edge-balanced
range partitions rather than METIS or a random partition plus renumbering.
This limitation must be disclosed with final results.

## Validity

- All four split GR files pass header, edge-count, file-size, and sampled edge
  classification checks.
- `cit-Patents` BFS and weighted SSSP signatures exactly match PuerCGP after
  repartitioning.
- The machine was shared during calibration. The 48-vs-96 gap is large and
  consistent enough for configuration selection, but publication timing must
  be repeated on an exclusive node.

Raw samples are in `runs.csv`, `summary.json`, and the sibling
`forkgraph_thread_calibration_low` directory.
