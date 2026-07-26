# Experiment: Gunrock BFS sinaweibo Nsight Systems 4 streams

## Purpose

Verify whether Gunrock `bfs_concurrent` runs BFS kernels with real GPU overlap across 4 CUDA streams on `soc-sinaweibo.gr`.

## Hypothesis

With 4 BFS queries and `--num_streams 4`, Nsight Systems should show kernel execution intervals from 4 distinct CUDA streams overlapping in time.

## Dataset

- Name: soc-sinaweibo
- Source: `/home/zyl/data/ggr_data/singlegpu/soc-sinaweibo.gr`
- Size: 4.4G graph file
- Split: single graph input, 4 source queries
- Preprocessing: Gunrock `.gr` binary input used directly

## Scenario

- Task: BFS concurrent profiling
- Workload: 4 BFS source queries
- Baseline: single process Gunrock `bfs_concurrent`
- Candidate: Gunrock `bfs_concurrent --num_streams 4`
- Comparison: Nsight Systems CUDA kernel overlap by stream

## Environment

- OS: Linux V100 6.1.0-44-amd64 Debian
- Hardware: NVIDIA Tesla V100-SXM2-32GB, visible device 0
- Runtime: CUDA via NVIDIA driver 570.211.01
- Dependencies: NVIDIA Nsight Systems 2025.1.1.0
- Git commit: `1a42381e025ddcb75b5711dc7d4ae5d90d380385`

## Command

Profile:

```bash
CUDA_VISIBLE_DEVICES=0 /home/zyl/.conda/envs/torch2.8/nsight-compute-2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,osrt \
  --sample=none \
  --cpuctxsw=process-tree \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --stats=false \
  --export=sqlite \
  --output=/home/zyl/Projects/ocgp/experiment/experiments/20260604-151325_gunrock_bfs_sinaweibo_nsys4/artifacts/bfs_sinaweibo_q4_streams4 \
  /home/zyl/Projects/ocgp/baselines/gunrockV2.2/build_cuda/bin/bfs_concurrent \
  -m /home/zyl/data/ggr_data/singlegpu/soc-sinaweibo.gr \
  --query_file /home/zyl/Projects/ocgp/baselines/gunrockV2.2/bfs_concurrent_soc-sinaweibo_q4.txt \
  --num_streams 4 \
  --json_dir /home/zyl/Projects/ocgp/experiment/experiments/20260604-151325_gunrock_bfs_sinaweibo_nsys4/artifacts \
  --json_file bfs_concurrent.json
```

Nsight CSV export:

```bash
/home/zyl/.conda/envs/torch2.8/nsight-compute-2025.1.1/host/target-linux-x64/nsys stats \
  --force-export true \
  -r cuda_gpu_trace,cuda_kern_exec_trace,cuda_api_trace \
  -f csv \
  -o /home/zyl/Projects/ocgp/experiment/experiments/20260604-151325_gunrock_bfs_sinaweibo_nsys4/artifacts/nsys_stats \
  /home/zyl/Projects/ocgp/experiment/experiments/20260604-151325_gunrock_bfs_sinaweibo_nsys4/artifacts/bfs_sinaweibo_q4_streams4.nsys-rep
```

Analysis:

- Source: `artifacts/nsys_stats_cuda_gpu_trace.csv`
- Method: keep CUDA kernel rows with launch dimensions, exclude `[CUDA memcpy ...]` and `[CUDA memset ...]`, then sweep kernel intervals by stream.
- Query streams: `13`, `14`, `15`, `16`
- Additional setup/default stream observed: `7`

## Results

| Metric | Value |
|---|---:|
| Gunrock total wall time (ms) | 956.8570 |
| Batch 0 wall time (ms) | 950.2535 |
| Kernel rows analyzed | 138 |
| Query kernel rows analyzed | 93 |
| Observed kernel streams | 5 |
| Query work streams | 4 |
| Max distinct stream overlap, all kernels | 4 |
| Max distinct stream overlap, query streams only | 4 |
| 4-stream overlap, all kernels (ms) | 0.693117 |
| 4-stream overlap, query streams only (ms) | 0.588477 |
| Total kernel active window, all kernels (ms) | 919.651565 |
| Total kernel active window, query streams only (ms) | 912.356768 |

| Query ID | Source | GPU ms |
|---:|---:|---:|
| 0 | 53297474 | 935.7907 |
| 1 | 23176989 | 931.4605 |
| 2 | 23515621 | 924.8740 |
| 3 | 47808273 | 929.5107 |

| Stream | Kernel count |
|---:|---:|
| 7 | 45 |
| 13 | 24 |
| 14 | 22 |
| 15 | 22 |
| 16 | 25 |

## Observations

- Nsight Systems generated both `artifacts/bfs_sinaweibo_q4_streams4.nsys-rep` and `artifacts/bfs_sinaweibo_q4_streams4.sqlite`.
- CSV exports were generated under `artifacts/nsys_stats_*.csv`.
- The four BFS query streams `13`, `14`, `15`, and `16` overlapped simultaneously for `0.588477 ms`.
- Overall CUDA kernel activity also included stream `7`, which appears to be setup/default work. The all-kernel trace had `0.693117 ms` at 4-stream overlap.
- Most kernel-active time was single-stream execution; query-stream overlap by distinct stream count was `819.143399 ms` at 1 stream, `90.575820 ms` at 2 streams, `2.049072 ms` at 3 streams, and `0.588477 ms` at 4 streams.
- CPU context switch tracing was unavailable on this host, but CUDA tracing and GPU kernel traces were collected successfully.

## Conclusion

The experiment supports the hypothesis that 4 BFS work streams can execute kernels simultaneously. Nsight Systems observed real GPU kernel overlap across query streams `13`, `14`, `15`, and `16`, with `max_distinct_stream_overlap = 4` and `query_four_stream_overlap_ms = 0.588477`.

## Issues

- Nsight warning: `CPU context switch tracing not supported, disabling.`
- `sqlite3` CLI was not installed, so overlap analysis used the exported `cuda_gpu_trace` CSV instead of direct SQLite queries.
- Four-stream overlap exists but is brief relative to the full BFS wall time.

## Next Steps

- Inspect the Nsight timeline visually if a screenshot is required for reporting.
- Compare with `--num_streams 1`, `2`, and `3` in separate experiment directories to quantify whether stream overlap improves wall time.
- Repeat on additional graphs or query sets to check whether the short 4-stream overlap is workload-specific.
