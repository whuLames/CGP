# Push/Pull Microbench

Standalone CUDA microbenchmark for studying push scatter and GE-SpMM pull cost
models outside the `puercgp` framework.

## Scope

This benchmark is intentionally smaller than `puercgp`:

- CSR/transpose-CSR graph storage.
- Random frontier generation.
- Push scatter-sum aggregation with `atomicAdd`.
- Pull gather-sum aggregation with a GE-SpMM-style kernel.
- All-push, all-pull, mixed serial, and mixed concurrent execution.
- CSV output for sweeps.

The only supported update mode is `spmm_sum`:

```text
push: scatter aggregation with atomicAdd
pull: GE-SpMM gather aggregation
```

This is the path intended for push/pull locality and concurrency experiments.
The benchmark intentionally does not include a naive pull kernel, because the
pull side of these experiments is meant to model the GE-SpMM/GEMM-style gather
operator.

## Build

```bash
cd /home/zyl/Projects/ocgp/push_pull_microbench
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/home/zyl/.conda/envs/torch2.8/bin/nvcc
cmake --build build -j 8
```

## Basic Runs

```bash
# Correctness smoke on a built-in toy graph.
./build/push_pull_bench --self-test

# One push scatter measurement on the toy graph.
./build/push_pull_bench \
  --graph=toy \
  --mode=density \
  --exec=all_push \
  --update=spmm_sum \
  --Q=32 \
  --rho-v=0.01 \
  --rho-q=0.5 \
  --repeat=7 \
  --warmup=2

# Mixed serial split: first 16 query lanes push, next 16 query lanes pull.
./build/push_pull_bench \
  --graph=/path/to/graph.mtx \
  --mode=mixed \
  --exec=mixed_serial \
  --update=spmm_sum \
  --Q-push=16 \
  --Q-pull=16 \
  --rho-push=0.005 \
  --rho-pull=0.2 \
  --csv=results.csv
```

## Graph Input

Supported inputs:

- `toy`: built-in 4-vertex weighted graph.
- Matrix Market `.mtx`.
- Galois GR / GGR binary `.gr`.
- Existing CSR directory containing:
  - `csr_vlist.bin`
  - `csr_elist.bin`
  - optional `csr_weightlist.bin`

The loader builds both outgoing CSR and transpose CSR. Push uses outgoing CSR;
pull uses transpose CSR for GE-SpMM gather.

The `.gr` loader expects the local Galois GR layout:

```text
uint64 version
uint64 size_edge_ty
uint64 nvtxs
uint64 nedges
int64  row_start[nvtxs]
int32  edge_dst[nedges]
optional padding if weighted and nedges is odd
optional int32 adjwgt[nedges]
```

On the local datasets, any nonzero `size_edge_ty` is treated as a weighted marker;
the payload is read as `int32 adjwgt[nedges]`.

## Frontier Modes

### Density

```text
--mode=density --rho-v=<union frontier vertex ratio> --rho-q=<query occupancy>
```

`rho_v` controls how many vertices are active in the union frontier. `rho_q`
controls how many query bits are active per active vertex.

### Overlap

```text
--mode=overlap --frontier-size-per-query=<count> --overlap=<0..1>
```

If `frontier-size-per-query` is omitted, `rho-v * V` is used. Higher overlap
reduces the union frontier and increases average popcount per active vertex.

### Mixed

```text
--mode=mixed --Q-push=16 --Q-pull=16 --rho-push=0.005 --rho-pull=0.2
```

The first `Q_push` lanes form the push group. The next `Q_pull` lanes form the
pull group. This mode is meant to test query-level push/pull splitting.

## Execution Modes

- `all_push`
- `all_pull`
- `mixed_serial`
- `mixed_concurrent`

For `mixed_concurrent`, push and pull launch into separate CUDA streams and wait
on a shared start event. The reported `total_ms` is the max of the two stream
completion times.

## Scripts

```bash
scripts/run_smoke.sh
scripts/run_density_sweep.sh <graph> <out.csv>
scripts/run_overlap_sweep.sh <graph> <out.csv>
scripts/run_push_pull_characterization.sh <graph> <out_dir> [per_query_frontier_size]
scripts/run_mixed_serial_sweep.sh <graph> <out.csv>
scripts/run_concurrent_sweep.sh <graph> <out.csv>
scripts/run_ncu_cases.sh <graph>
```

## Push/Pull Characterization Experiments

Use this script for the first-pass push/pull cost characterization:

```bash
scripts/run_push_pull_characterization.sh \
  /home/zyl/data/ggr_data/singlegpu/cit-Patents.gr \
  results/cit_patents_characterization \
  3775
```

It writes three CSV files:

- `push_frontier_size.csv`: `all_push`, density mode, fixed `Q` and `rho_q`,
  sweeping `rho_v`. This measures how push changes as the union active vertex
  set grows. Use `unique_vertices`, `active_query_pairs`, `edges_scanned`, and
  `atomic_ops` from the CSV to normalize timing.
- `push_overlap.csv`: `all_push`, overlap mode, fixed per-query frontier size,
  sweeping `overlap`. This keeps per-query work approximately fixed while
  changing how many frontier vertices are shared by different queries. The CSV
  fields `unique_vertices`, `avg_popcount`, and `real_overlap` are the key
  sanity checks.
- `pull_q.csv`: `all_pull`, density mode, sweeping `Q`. Pull GE-SpMM scans
  transpose CSR for each query lane, so this experiment is intentionally much
  smaller; activation density is fixed only to build a valid input matrix.

Optional environment overrides:

```bash
Q=32 RHO_Q=0.5 REPEAT=7 WARMUP=2 SEEDS="1 2 3 4 5" \
  scripts/run_push_pull_characterization.sh <graph> <out_dir> [per_query_frontier_size]
```

## Current MVP Limits

- `libsmctrl` is not integrated yet; the first concurrency implementation uses
  ordinary CUDA streams.
- Mixed frontier generation currently uses density parameters for push/pull
  groups. `overlap_push`, `overlap_pull`, and `cross_overlap` are reserved for
  the next generator refinement.
- Kernel stats for the `spmm_sum` path are host-side estimates, so the measured
  CUDA kernels are not polluted by device-side statistic atomics.
