# CGP BFS Optimization Plan

## Goal

Optimize `cgp_bfs` against a same-repository multi-stream Gunrock BFS baseline.
The baseline runs one Gunrock BFS query per source on non-blocking streams where
the existing Gunrock API allows it. The optimized CGP BFS should be evaluated by
graph type, query count, and frontier shape rather than by a single aggregate
number.

## Root-Cause Analysis

The initial CGP BFS implementation combined all query frontiers but expanded
them with one thread per `(query_id, vertex)`. This underutilizes the GPU when a
frontier contains high-degree vertices and leaves many threads with little or no
work.

Each BFS level also forced host synchronization, copied per-query next-frontier
counts back to the CPU, and used Thrust fills for small counters. That made the
CPU a level-by-level control bottleneck.

The output path performed one global `atomicAdd` per discovered vertex, plus one
global per-query atomic. Large levels with many discoveries therefore hit global
atomic contention.

The distance layout was `distances[query_id * |V| + vertex]`. That is simple and
validation-friendly but creates `num_queries * |V|` initialization and memory
traffic, which limits maximum batch size on large graphs.

## Baseline

`cgp_bfs_multi_stream` is the in-repository baseline. It accepts the same
high-level input shape as `cgp_bfs`:

- `--market`
- `--query_file`
- `--src`
- `--json_dir`
- `--json_file`

It reuses `gunrock::bfs::run` for each query and creates a non-blocking stream
and context per query. Existing Gunrock BFS internals may synchronize their own
context stream, so the baseline is reproducible and comparable inside this
repository, but it is not guaranteed to be a perfectly asynchronous launch-only
measurement.

## Optimization Priorities

1. Edge-balanced expansion.
   Compute degree and prefix data for the combined frontier, then distribute
   edge work across threads. This targets high-degree imbalance and large
   frontier levels.

2. Reduce level-level host synchronization.
   Keep the performance path dependent only on batch next-frontier count.
   Per-query completion is delayed instead of copied every level. Small counter
   reset uses a kernel rather than a Thrust fill.

3. Reduce global atomic pressure.
   Aggregate discovered output count at block scope, then perform one global
   `atomicAdd` per block chunk. Disable per-query next-frontier atomics by
   default.

4. Memory and initialization.
   Keep the existing distance layout for correctness and low implementation
   risk. Add batch-size and memory-budget controls later if large-graph runs are
   limited by `num_queries * |V|` storage.

## Measurement Plan

Record these fields per stage:

- `baseline_multi_stream_wall_ms`
- `cgp_before_wall_ms`
- `cgp_after_wall_ms`
- `speedup_vs_previous`
- `speedup_vs_multi_stream`
- `gpu_time_ms`
- `frontier_sizes`
- `level_edge_counts`
- correctness validation status

Recommended benchmarks:

- `cit-Patents`: `q=2,4,8,16`
- `soc-orkut`: `q=2,4,8,16`
- `soc-twitter`: `q=2,4,8`
- `soc-sinaweibo`: `q=2,4` if memory allows

Use the same `.gr` or Matrix Market data and the same query files for both
drivers. Warm up once, then record three runs and report the median. Save JSON
or CSV output under `build/cgp_bfs_optimization_results/`.

## Correctness Plan

Validate `chesapeake` with multiple sources. For large graphs, validate at
least the final query in each query file against CPU BFS or Gunrock BFS output.
The optimized CGP BFS keeps the original distance layout, so validation can
compare one query slice directly.

## Profiling Plan

Profile at least:

- `soc-orkut q8`
- `soc-twitter q8`

Capture kernel timeline, host synchronization, global atomic behavior, memory
throughput, and occupancy with Nsight Systems or Nsight Compute.

## Assumptions

- CUDA or HIP build tools are available in the benchmark environment.
- Keeping `distances[query_id * |V| + vertex]` is acceptable for the first
  optimization round.
- If full merge-path traversal is too expensive to maintain, an edge-balanced
  binary-search mapping is acceptable as a first implementation.
- The final report must state where optimized CGP BFS beats the multi-stream
  baseline and where it does not.
