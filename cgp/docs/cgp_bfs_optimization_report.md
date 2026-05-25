# CGP BFS Optimization Report

## Current Implementation

This round adds an in-repository multi-stream baseline and updates `cgp_bfs` with
the first three optimization points from the plan:

- edge-balanced expansion over the combined frontier;
- delayed per-query completion reporting;
- block-local output-count aggregation.

The distance layout remains `distances[query_id * |V| + vertex]` for correctness
and validation compatibility.

## Multi-Stream Baseline

Added `examples/algorithms/bfs/cgp_bfs_multi_stream.cu` and registered it in the
BFS example CMake file.

The baseline accepts the same source input style as `cgp_bfs`, including
`--market`, `--query_file`, `--src`, `--json_dir`, and `--json_file`. It creates
one non-blocking stream and context per query, then calls `gunrock::bfs::run` for
each source. JSON output includes batch wall time, per-query GPU time, sources,
graph metadata, command line, and system/GPU metadata.

Important caveat: the existing Gunrock enactor and merge-path advance path
synchronize the context stream internally. This baseline is still useful as a
same-repo Gunrock BFS baseline, but it is not a pure fully-overlapped stream
measurement.

## Optimization 1: Edge-Balanced Expansion

Changed `include/gunrock/algorithms/cgp_bfs.hxx`.

The old expansion mapped one thread to one frontier vertex and then looped over
all outgoing edges from that vertex. High-degree vertices created long per-thread
serial work while neighboring threads could sit idle.

The new path computes per-frontier degrees, exclusive edge offsets, and
inclusive edge ends. The expansion kernel maps thread work to edge ordinals and
uses a binary search over edge ends to recover the `(query_id, vertex)` item.
This distributes large high-degree adjacency lists across many threads.

Expected benefit: lower duration for levels where a few frontier vertices own a
large fraction of edges. The `level_edge_counts` JSON field records per-level
edge work for correlating runtime with frontier shape.

## Optimization 2: Reduced Per-Level Host Work

The original loop reset two counters with Thrust fills, synchronized the stream,
copied a per-query next-frontier count array to host, and used that to mark each
query complete.

The new loop resets only the global next-frontier count with a one-thread kernel.
It no longer maintains or copies `per_query_next_counts`. Per-query completion
records are delayed until batch completion and use the final batch wall time.

There are still two scalar synchronization points per level:

- total edge count after degree scans, needed to size the expansion launch;
- next frontier count, needed for BFS termination.

Expected benefit: fewer host-device round trips and less Thrust temporary work.
The remaining scalar syncs are the main residual issue for future work.

## Optimization 3: Lower Atomic Output Contention

The expansion kernel now accumulates discoveries in a block-local shared counter
for each block chunk. One thread performs a global `atomicAdd` to reserve the
output span, then discovered items write to that contiguous span.

The previous per-discovered-vertex global `out_count` atomic is replaced by one
global atomic per block chunk with discoveries. The previous per-query global
atomic is removed.

Expected benefit: lower global atomic pressure on large levels with many
discoveries.

## Optimization 4: Memory And Initialization

No structural memory-layout change was made in this round. The implementation
still initializes and stores `num_queries * |V|` distances. This keeps the
correctness surface small, but it means large graph and high-query-count runs
remain memory-bound.

Recommended next step: add explicit `--batch_size` or memory-budget splitting,
then evaluate visited stamps or bitmaps.

## Benchmark Table

Build verification completed for both targets:

- `cgp_bfs`
- `cgp_bfs_multi_stream`

Smoke validation completed on `datasets/chesapeake/chesapeake.mtx` with
`--src 0,1,2 --validate`:

| driver | wall_time_ms | gpu_time_ms | validation |
| --- | ---: | ---: | --- |
| `cgp_bfs` | 0.452 | 0.435296 | 0 errors vs CPU BFS for final query |
| `cgp_bfs_multi_stream` | 1.132 | 0.874496 summed | 0 errors vs CPU BFS for final query |

The generated JSON files are under `build/cgp_bfs_optimization_results/`:

- `cgp_bfs_chesapeake_q3.json`
- `cgp_bfs_multi_stream_chesapeake_q3.json`

## GunrockV2.2 Comparison

Large-graph comparison used GunrockV2.2
`/home/zyl/Projects/ocgp/baselines/gunrockV2.2/build_cuda/bin/bfs_concurrent`
as the multi-stream baseline. Both drivers used `.gr` inputs from
`/home/zyl/data/ggr_data/singlegpu`, the same query files, and
`CUDA_VISIBLE_DEVICES=2`. Each row reports the median of 3 runs. Raw JSON/logs
are under `build/cgp_bfs_optimization_results/v22_compare/`.

| graph | q | cgp wall ms | v2.2 total wall ms | v2.2 batch wall ms | speedup vs total | speedup vs batch | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cit-Patents | 2 | 23.536 | 46.779 | 44.314 | 1.988 | 1.883 | ok |
| cit-Patents | 4 | 49.470 | 70.633 | 67.527 | 1.428 | 1.365 | ok |
| cit-Patents | 8 | 108.445 | 135.663 | 129.559 | 1.251 | 1.195 | ok |
| cit-Patents | 16 | 227.535 | 276.393 | 264.303 | 1.215 | 1.162 | ok |
| soc-orkut | 2 | 89.725 | 116.763 | 114.957 | 1.301 | 1.281 | ok |
| soc-orkut | 4 | 227.587 | 217.115 | 213.880 | 0.954 | 0.940 | ok |
| soc-orkut | 8 | 527.026 | 407.534 | 402.749 | 0.773 | 0.764 | ok |
| soc-orkut | 16 | 1142.672 | 800.693 | 789.279 | 0.701 | 0.691 | ok |
| soc-twitter | 2 | 198.913 | 343.894 | 341.945 | 1.729 | 1.719 | ok |
| soc-twitter | 4 | 426.145 | 651.855 | 644.757 | 1.530 | 1.513 | ok |
| soc-twitter | 8 | 931.188 | 1287.616 | 1278.828 | 1.383 | 1.373 | ok |
| soc-sinaweibo | 2 | 408.148 | 502.343 | 499.705 | 1.231 | 1.224 | ok |
| soc-sinaweibo | 4 | 877.400 | 957.799 | 952.467 | 1.092 | 1.086 | ok |

CGP BFS beats GunrockV2.2 on `cit-Patents`, `soc-twitter`, `soc-sinaweibo`,
and `soc-orkut q2`. It loses on `soc-orkut q4/q8/q16`; this is the clearest
remaining regression and likely reflects the cost of per-level degree scans,
edge-offset scans, and binary search on a graph/query mix where Gunrock's
per-query BFS kernels already expose enough parallelism.

## Suggested Commands

Example build:

```bash
cmake --build build --target cgp_bfs
cmake --build build --target cgp_bfs_multi_stream
```

Example correctness run:

```bash
./build/bin/cgp_bfs --market ../datasets/chesapeake/chesapeake.mtx --src 0,1,2 --validate --json_dir build/cgp_bfs_optimization_results
./build/bin/cgp_bfs_multi_stream --market ../datasets/chesapeake/chesapeake.mtx --src 0,1,2 --validate --json_dir build/cgp_bfs_optimization_results
```

Example benchmark run:

```bash
./build/bin/cgp_bfs --market /data/cit-Patents.gr --query_file queries/cit-Patents_q8.txt --json_dir build/cgp_bfs_optimization_results
./build/bin/cgp_bfs_multi_stream --market /data/cit-Patents.gr --query_file queries/cit-Patents_q8.txt --json_dir build/cgp_bfs_optimization_results
```

## Residual Issues

The edge-balanced implementation uses binary search per edge ordinal. That is
simple and robust, but a tiled merge-path implementation can reduce search
overhead on very large frontiers.

The loop still synchronizes per BFS level for two scalar values. A future
device-side persistent loop or graph-captured loop could reduce CPU control
overhead.

Memory footprint is still proportional to `num_queries * |V|`. That remains the
primary limiter for high query counts on large graphs.

## Hybrid Push/Pull Update

This round adds a heavy-frontier pull path to `cgp_bfs` while preserving the
existing edge-balanced push path.

New CLI options:

```bash
--traversal_mode=push|pull|hybrid
--pull_frontier_ratio=<float>
--pull_edge_ratio=<float>
--profile_levels
```

The default traversal mode is `hybrid`. Hybrid switches to pull when either the
active `(query, vertex)` frontier reaches `pull_frontier_ratio * queries * |V|`
or the active frontier edge count reaches `pull_edge_ratio * queries * |E|`.
Defaults are `0.15` and `0.20`.

The pull implementation uses a dense vertex-major frontier bitmap:
`frontier_bitmap[vertex * num_queries + query]`. The kernel maps one warp to one
`(vertex, query)` candidate, skips already visited distances, scans the CSR row,
and exits as soon as any lane finds a neighbor in the current frontier. It writes
the existing query-major distance layout:
`distances[query_id * |V| + vertex]`.

State conversion is explicit:

- push levels keep the combined `(query_id, vertex)` frontier list;
- switching push to pull builds the dense bitmap from the list;
- pull levels produce the next dense bitmap directly;
- switching pull back to push compacts the bitmap into the list.

JSON output now includes `traversal_mode`, threshold values, `level_modes`,
`level_wall_times_ms`, and `level_profiles`. With `--profile_levels`, CUDA event
timings are recorded for degree scans, bitmap build, push kernel, pull kernel,
bitmap compaction, and count synchronization. Consecutive pull levels avoid
list compaction solely for accounting; exact source-side edge counts for those
bitmap-only levels are therefore computed only when `--profile_levels` is set.

Smoke validation completed on `datasets/chesapeake/chesapeake.mtx` with
`--src 0,1,2 --validate`:

| traversal mode | wall_time_ms | gpu_time_ms | level modes | validation |
| --- | ---: | ---: | --- | --- |
| `push` | 0.633 | 0.609 | push,push,push,push | 0 errors |
| `pull --profile_levels` | 0.793 | 0.770 | pull,pull,pull,pull | 0 errors |
| `hybrid` | 0.480 | 0.461 | push,pull,pull,push | 0 errors |
| `hybrid --pull_frontier_ratio=0.01 --pull_edge_ratio=0.01 --profile_levels` | 0.538 | 0.519 | pull,pull,pull,pull | 0 errors |
| `hybrid --pull_frontier_ratio=0.3 --pull_edge_ratio=1.0 --profile_levels` | 1.335 | 1.317 | push,push,pull,push | 0 errors |

Raw smoke JSON files are under `build/cgp_bfs_optimization_results/`:

- `cgp_bfs_chesapeake_q3_push.json`
- `cgp_bfs_chesapeake_q3_pull.json`
- `cgp_bfs_chesapeake_q3_hybrid_default.json`
- `cgp_bfs_chesapeake_q3_hybrid.json`
- `cgp_bfs_chesapeake_q3_hybrid_transitions.json`

Large-graph medians were collected on `CUDA_VISIBLE_DEVICES=2` with the
GunrockV2.2 query files in
`/home/zyl/Projects/ocgp/baselines/gunrockV2.2/`. Each row is the median of 3
measured runs after 1 warmup. Raw JSON/logs are under
`build/cgp_bfs_optimization_results/hybrid_compare/`.

| graph | q | push ms | hybrid ms | speedup vs push | V2.2 total ms | speedup vs V2.2 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| soc-orkut | 2 | 89.436 | 8.741 | 10.232 | 116.763 | 13.358 |
| soc-orkut | 4 | 227.390 | 36.046 | 6.308 | 217.115 | 6.023 |
| soc-orkut | 8 | 526.094 | 95.199 | 5.526 | 407.534 | 4.281 |
| soc-orkut | 16 | 1143.424 | 310.275 | 3.685 | 800.693 | 2.581 |
| soc-twitter | 2 | 198.796 | 44.701 | 4.447 | 343.894 | 7.693 |
| soc-twitter | 4 | 425.688 | 136.563 | 3.117 | 651.855 | 4.773 |
| soc-twitter | 8 | 931.917 | 331.529 | 2.811 | 1287.616 | 3.884 |
| soc-sinaweibo | 2 | 408.375 | 138.807 | 2.942 | 502.343 | 3.619 |
| soc-sinaweibo | 4 | 877.559 | 254.874 | 3.443 | 957.799 | 3.758 |

The hybrid path improves every tested large-graph case. The original regression
target, `soc-orkut q8/q16`, improves from `526.094 -> 95.199 ms` and
`1143.424 -> 310.275 ms`, and now beats the existing GunrockV2.2 multi-stream
`total_wall_ms` baseline for both query counts.
