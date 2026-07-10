# Stage 10 replenish benchmark

## Scope

Re-tested replenishment after the refactor on four real graphs:

```text
cit-Patents
soc-orkut
soc-twitter
soc-sinaweibo
```

Workload:

```text
100 BFS + 100 SSSP
batch_size = 32
mode = push
push = shared_node_warp
repeats = 3
warmup = 1
discard_results = on
```

`discard_results=on` means replenish does not write the final `N * V` result
buffer. Snapshot writeback is therefore removed from this benchmark, but slot
clear/reset and scheduling overhead remain.

## Correctness

Toy correctness gate passed:

```text
./build/smoke_replenish
./build/validate_replenish
```

Coverage:

- active slot convergence mask
- slot reinitialization
- slot snapshot helper
- end-to-end slot reuse with BFS and SSSP values

Large graph runs completed successfully, but they are performance/completion
runs only; they do not include CPU reference checks for all 200 queries.

## Throughput

`speedup = sequential_total_ms / replenish_total_ms`.

```text
dataset        sequential_ms  replenish_ms  speedup  replenish_slowdown
cit-Patents    3039.19        3547.82       0.857x   +16.7%
soc-orkut      6664.17        9212.36       0.723x   +38.2%
soc-twitter    14153.90       19133.80      0.740x   +35.2%
soc-sinaweibo  18025.40       25404.20      0.710x   +40.9%
```

Result:

```text
0/4 throughput cases pass.
```

Replenish is slower on every graph in this workload.

## Latency

Ratio below is `replenish / sequential`, so `< 1` means replenish is better.

```text
dataset        p25_ratio  median_ratio  p90_ratio  p99_ratio
cit-Patents    0.757      1.358         1.296      1.168
soc-orkut      0.698      1.912         1.488      1.388
soc-twitter    0.806      1.150         1.419      1.330
soc-sinaweibo  0.546      1.516         1.511      1.391
```

Pattern:

- p25 latency improves: early queries can return sooner.
- median, p90, and p99 all regress: later queries pay the accumulated slot
  reuse overhead.
- total throughput is worse, so the p25 win does not carry the workload.

## Bottleneck

The dominant structural bottleneck is slot reset/clear, not final result
snapshot:

```text
replenish_engine.hxx:
  every iteration copies active_union back to host and synchronizes
  every converged-slot round calls dispatch_converged()

dispatch_converged():
  uploads converged slot lists
  launches snapshot_and_reset()
  launches reinit_slots()
  copies unique_count back and synchronizes

snapshot_and_clear_multi_slot_kernel:
  scans all V vertices for every converged-slot group
  clears values for each converged slot
  clears visited/frontier/next_frontier masks with atomicAnd
```

Even with `discard_results=on`, this kernel still scans all vertices and clears
slot state. On large graphs this O(V) reset cost is paid repeatedly as slots
converge and are refilled.

Push mode also weakens the theoretical benefit of replenish. Once a slot has no
active frontier bit, existing push kernels mostly skip it through mask filtering,
so "empty slot waiting" is already cheap in the sequential batch engine.
Replenish therefore adds clear/reset/sync work without removing much push work.

## Optimization Direction

Highest-priority candidates:

1. Replace full-graph slot reset with dirty-vertex reset.
   Track vertices touched by each slot and clear only those positions.

2. Replace eager clear with epoch/generation tags.
   Avoid writing `INF` over all `V` values when a slot is reused.

3. Move slot I/O to a real async stream.
   `slot_io_manager::snapshot_and_reset_async()` is currently just a same-stream
   wrapper. Event-based reset could overlap with compute when dependencies allow.

4. Reduce host synchronization in `dispatch_converged()`.
   Current flow synchronizes after every iteration and after reinit to read
   `active_union` / `unique_count`.

5. Re-evaluate replenish only for workloads with real long-tail imbalance or
   strict early-result latency goals.
   For standard BFS+SSSP throughput, current results are a clear no-go.

Raw local artifacts:

```text
puercgp/experiments/refactor_reports/stage10_replenish_benchmark.log
puercgp/experiments/refactor_reports/stage10_replenish_summary.csv
```

These files are ignored by `.gitignore` (`*.log`, `*.csv`).
