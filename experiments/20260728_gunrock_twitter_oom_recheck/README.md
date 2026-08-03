# Gunrock soc-twitter OOM Recheck

## Corrected Results

The workload is unchanged: N=256 and source SHA-256
`8095797d5d57b5ca3ba8fd6bbd4862b1f9e6829250a073ff645ccdd8e7a74e28`.
Times exclude graph loading and report the median of five measured runs after
two warmups.

| Algorithm | Effective Q | Median (ms) | Baseline (ms) | Speedup |
|---|---:|---:|---:|---:|
| BFS | 8 | 54,459.590 | 45,178.313 | 0.830x |
| SSSP | 4 | 118,774.633 | 83,004.485 | 0.699x |

BFS Q=16 and SSSP Q=8 fail from genuine capacity pressure. BFS Q=8 and SSSP
Q=4 are therefore the largest successful powers-of-two candidates tested on a
32 GiB V100. SSSP Q=4 peaks at approximately 29,692 MiB.

One SSSP Q=4 measurement encountered an initialization-time OOM after six
successful runs at the same Q. Its replacement run completed at 118,539.516 ms
while one-second process monitoring observed only `sssp_concurrent` on GPU 7.
The result table uses five successful measured runs; the incident is treated as
shared-environment instability, not as a lower capacity limit.

## Root Cause

The old conclusion that both algorithms failed at Q=2 was incorrect:

1. The original `soc-twitter` and `soc-orkut` BFS workers were both assigned to
   GPU 1. Their file timestamps overlap from 14:36 through 14:40 on 2026-07-26.
2. Later `soc-twitter` attempts also ran with contaminated device memory. For
   example, BFS Q=8 reported only 23,362,797,568 free bytes after graph loading,
   versus 29,419,372,544 bytes on a clean GPU. BFS Q=2 and SSSP Q=2 failed
   before producing normal initialization output.
3. Every Q attempt ran in a new process, so this was not retained allocation or
   fragmentation within Gunrock's fallback loop.
4. Clean probes succeeded at BFS Q=2/Q=8 and SSSP Q=2/Q=4. Formal clean runs
   then completed with the same 256 sources.

The original harness treated any `cudaErrorMemoryAllocation` as intrinsic
capacity pressure and immediately reduced Q. It did not check whether another
process already occupied the target GPU.

## Runner Changes

`../20260725_gunrock_multistream_n256/run_case.py` now:

- checks target-GPU memory before every invocation;
- uses a per-GPU file lock so experiment workers cannot share one GPU;
- reports external occupancy as `environment_busy`, without reducing Q;
- treats an OOM after successful runs at the same Q as an unstable environment;
- accepts `--artifact-root` so corrective runs do not overwrite old evidence.

Raw results are under `bfs_artifacts/` and `sssp_artifacts/`. Clean failure
evidence for BFS Q=16 and SSSP Q=8 is under `bfs_capacity_artifacts/` and
`sssp_capacity_artifacts/`. The original failed logs remain under the 20260725
experiment for audit.
