# ForkGraph / Glign / iBFS BFS Benchmark

Date: 2026-06-09

## Configuration

- Datasets: `cit-Patents`, `soc-sinaweibo`, `soc-twitter`, `soc-orkut`
- Input graphs: `/home/zyl/data/ggr_data/singlegpu/<dataset>.gr`
- Concurrency: `2, 4, 8, 16, 32, 64`
- Repeats: 3 per dataset/baseline/concurrency
- Query count: equal to concurrency for each run
- Query sets: fixed per dataset/concurrency and shared by all baselines
- Main metric: internal algorithm time in milliseconds
- Secondary metric: wall-clock time in seconds
- ForkGraph partitions: 8 for every dataset
- Glign mode: `-option glign -mode 3 -delay`
- iBFS switch level: `8`
- iBFS GPU id: existing build uses `USE_GPU_ID 4`

## Artifacts

- `query_sets/`: fixed source lists and JSON metadata
- `raw_logs/`: stdout/stderr and command metadata for each run
- `results.csv`: one row per run
- `summary_by_config.csv`: median/min/max by dataset/baseline/concurrency
- `SUMMARY.md`: compact median-time tables
- `scripts/generate_query_sets.py`: deterministic query generation
- `scripts/run_benchmarks.py`: runner, log parser, and summarizer

## Notes

- iBFS was patched in `/home/zyl/Projects/ocgp/baselines/iBFS/graph.cuh` to read `input.dat` when present. This is required so all baselines use the same source vertices.
- ForkGraph input files are under `/home/zyl/Projects/ocgp/baselines/ForkGraph/inputs_gr_partitioned`.
- One run timed out with the 1800s per-run cap: `soc-sinaweibo / forkgraph / q64 / repeat 1`. The other two q64 ForkGraph repeats completed, so the summary uses 2/3 successful runs for that row.

## Reproduction

Generate query sets:

```bash
python3 scripts/generate_query_sets.py --out-dir query_sets
```

Run all benchmarks with resume:

```bash
python3 scripts/run_benchmarks.py --mode full --resume --timeout-sec 7200
```

Summarize existing results:

```bash
python3 scripts/run_benchmarks.py --mode summarize
```
