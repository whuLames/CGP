# Gunrock BFS/PageRank Iteration Profiling

This directory documents the per-iteration profiling added for Gunrock V2.2 BFS
and PageRank.

## Goal

The goal is to correlate per-iteration graph workload with GPU execution
behavior:

- workload: frontier size, graph size, and per-iteration elapsed time
- hardware behavior: achieved occupancy and IPC from Nsight Compute

This is intended for iteration-level trend analysis, such as how BFS frontier
growth/shrinkage affects occupancy/IPC, or how PageRank edge-spread behaves
across fixed iterations.

## Datasets and Sources

Graphs are loaded from:

```text
/home/zyl/data/ggr_data/singlegpu
```

Datasets:

```text
cit-Patents.gr
soc-orkut.gr
soc-sinaweibo.gr
soc-twitter.gr
```

BFS uses fixed single-source runs:

```text
cit-Patents:    0
soc-orkut:      1506298
soc-sinaweibo:  53297474
soc-twitter:    11702603
```

PageRank uses `--max_iterations 10`.

## Instrumentation

The implementation adds optional profiling via:

```text
--iter_profile <csv_path>
```

When this option is omitted, normal benchmark behavior is preserved.

Instrumentation is implemented in:

```text
include/gunrock/util/iteration_profiler.hxx
include/gunrock/algorithms/bfs.hxx
include/gunrock/algorithms/pr.hxx
examples/algorithms/bfs/bfs.cu
examples/algorithms/pr/pr.cu
```

NVTX ranges:

```text
BFS whole iteration:       gunrock:bfs:iter:<i>
PageRank whole iteration:  gunrock:pr:iter:<i>
PageRank edge-spread:      gunrock:pr:iter:<i>:edge_spread
```

The CSV schema emitted by Gunrock is:

```text
algorithm,dataset,source,run,iteration,range_name,
active_vertices,active_edges,input_frontier,output_frontier,elapsed_ms
```

For BFS:

- `active_vertices = input_frontier`
- `active_edges = -1`
- `input_frontier` and `output_frontier` are the frontier sizes around the
  advance step

For PageRank:

- `active_vertices = number_of_vertices`
- `active_edges = number_of_edges`
- both whole-iteration and `edge_spread` rows are emitted

## Collection Method

Profiling is driven by:

```text
scripts/profile_gunrock_iterations.py
```

The current collection uses:

```text
/home/zyl/.conda/envs/torch2.8/bin/ncu
```

NCU metrics:

```text
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__inst_issued.avg.per_cycle_active
```

The lock-file issue from `/tmp/nsight-compute-lock` was avoided by using a
private temp directory:

```bash
TMPDIR=/home/zyl/tmp
```

Example command:

```bash
cd /home/zyl/Projects/baselines/gunrockV2.2

CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/home/zyl/tmp \
LD_LIBRARY_PATH=/home/zyl/.conda/envs/torch2.8/lib:${LD_LIBRARY_PATH:-} \
python3 scripts/profile_gunrock_iterations.py
```

## Two-Stage Timing and Metrics

Do not use elapsed times measured inside an NCU run. Nsight Compute replay and
counter collection inflate kernel and iteration time.

The script therefore uses a two-stage flow:

1. Run Gunrock normally with `--iter_profile` to generate normal
   `*_iterations.csv` timing/workload data.
2. Run Gunrock under NCU with a separate profiling CSV path to collect NVTX
   range metrics and `*_ncu_raw.csv`.
3. Merge normal timing/workload rows with NCU hardware metrics into
   `*_merged.csv`.

This means:

```text
*_iterations.csv = normal-run workload and elapsed_ms
*_ncu_raw.csv    = raw Nsight Compute CSV output
*_ncu.ncu-rep    = Nsight Compute report
*_merged.csv     = normal timing/workload + NCU achieved_occupancy/ipc
```

Use `*_merged.csv` for analysis.

## Metric Aggregation Semantics

NCU reports metrics per kernel. A single NVTX iteration range may contain
multiple kernels.

For `*_merged.csv`, each iteration range aggregates all kernels inside that
range using GPU-duration-weighted averages:

```text
weighted_metric = sum(kernel_metric * gpu__time_duration.sum)
                  / sum(gpu__time_duration.sum)
```

This is a practical iteration-level approximation.

Important interpretation notes:

- BFS rows are not strictly "main advance kernel only"; they are per-iteration
  NVTX range aggregates.
- BFS iteration ranges can include CUB/Thrust helper kernels and the advance
  kernel, such as `block_mapped_kernel`.
- PR `gunrock:pr:iter:<i>` rows are whole-iteration aggregates.
- PR `gunrock:pr:iter:<i>:edge_spread` rows isolate the main edge-spread graph
  computation kernel and are the better PR main-kernel view.

Strict total IPC would require summing base counters, for example:

```text
sum(smsp__inst_issued.sum) / sum(smsp__cycles_active.sum)
```

The current IPC column should be treated as a GPU-time-weighted approximation
for iteration-level trend analysis.

## Generated Results

Generated files are under:

```text
iteration_profiles/
```

Expected merged files:

```text
bfs_cit-Patents_merged.csv
bfs_soc-orkut_merged.csv
bfs_soc-sinaweibo_merged.csv
bfs_soc-twitter_merged.csv
pr_cit-Patents_merged.csv
pr_soc-orkut_merged.csv
pr_soc-sinaweibo_merged.csv
pr_soc-twitter_merged.csv
```

After the two-stage correction, all merged rows have non-empty
`achieved_occupancy` and `ipc` columns.

Sanity-check summary from the completed run:

```text
bfs_cit-Patents:    rows=21, filled=21, elapsed_ms_max=5.767168
bfs_soc-orkut:      rows=9,  filled=9,  elapsed_ms_max=22.215679
bfs_soc-sinaweibo:  rows=9,  filled=9,  elapsed_ms_max=434.492401
bfs_soc-twitter:    rows=17, filled=17, elapsed_ms_max=80.667648
pr_cit-Patents:     rows=20, filled=20, elapsed_ms_max=10.368000
pr_soc-orkut:       rows=20, filled=20, elapsed_ms_max=24.344576
pr_soc-sinaweibo:   rows=20, filled=20, elapsed_ms_max=167.255035
pr_soc-twitter:     rows=20, filled=20, elapsed_ms_max=70.131714
```

## Caveats

- `TMPDIR=/home/zyl/tmp` avoids the stale `/tmp/nsight-compute-lock` permission
  problem. Only use this when no other user is running NCU concurrently on the
  same machine.
- BFS `active_edges` is currently `-1`; frontier edge-volume is not computed to
  avoid adding extra GPU work to every BFS iteration.
- NCU metrics in whole-iteration rows are range-level approximations, not exact
  algorithmic totals.
- For precise kernel-level analysis, inspect `*_ncu_raw.csv` or the
  `*_ncu.ncu-rep` report directly.
