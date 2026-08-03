# Experiment: GE-SpMM pull group-slots validation

## Purpose

Compare a single 64-query gather-min GE-SpMM kernel with two concurrent
32-query kernels assigned to disjoint Green Context SM partitions.

## Hypothesis

Splitting the independent query dimension can reduce pull-kernel makespan on
graphs whose degree distribution creates a long execution tail.

## Dataset

- Names: cit-Patents, soc-orkut, soc-twitter, soc-sinaweibo, roadNet-CA
- Source: `/home/zyl/data/csr_data`
- Representation: original CSR; int32 row offsets are promoted to int64
- Split and preprocessing: none

## Scenario

- Task: pure pull gather-min SpMM
- Workload: 64 deterministic finite float features per vertex
- Baseline: one Q64 kernel on the full GPU
- Candidate: two concurrent Q32 kernels on disjoint Green Context streams
- Synchronization: every stream synchronizes before and after each repetition
- Excluded: BFS state, frontier logic, compaction, and postprocessing

## Environment

- OS: Debian Linux, kernel 6.1.0-44-amd64
- GPU ordinal 0: NVIDIA Tesla V100-SXM2-32GB, 80 SM
- Driver: 570.211.01
- CUDA compiler: 12.8.93 at `/home/zyl/.conda/envs/torch2.8/bin/nvcc`
- Git commit: `a2281fb76228b81b2f1611747c4dd492ae665edb`

## Command

```bash
GPU_ID=0 WARMUP=5 ITERS=30 \
TILE_ROW=8 SEED=42 \
src/kernels/run_ge_spmm_group_slots.sh
```

Timeline validation:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o artifacts/twitter_green_timeline \
  src/kernels/ge_spmm_group_slots_bench \
  /home/zyl/data/csr_data/soc-twitter \
  --case=green_2x32 --gpu=0 --warmup=1 --iters=3 \
  --tile-row=8 --seed=42 --skip-verify
```

## Results

The primary metric is median per-round host makespan. A speedup above 1 means
the Green 2xQ32 candidate is faster.

| Dataset | Q64 (ms) | Green 2xQ32 (ms) | Speedup | Output |
|---|---:|---:|---:|---|
| cit-Patents | 13.286 | 16.802 | 0.791x | match |
| soc-orkut | 56.101 | 63.357 | 0.885x | match |
| soc-twitter | 182.128 | 167.546 | 1.087x | match |
| soc-sinaweibo | 181.720 | 241.433 | 0.753x | match |
| roadNet-CA | 2.011 | 3.087 | 0.651x | match |

Detailed distribution statistics are in `metrics.json`; all 30 per-round
measurements are retained in `artifacts/<dataset>.csv`.

## Observations

- Only soc-twitter benefits: candidate median latency is 8.0% lower and its
  run-to-run standard deviation is also lower (0.604 ms versus 1.390 ms).
- The other four datasets become slower under equal 40+40 SM partitioning.
- All five Q64/2xQ32 output fingerprints match, so no query work was removed or
  changed.
- The Nsight Systems trace contains four kernel pairs (one warmup and three
  measured repetitions). The two kernels in every pair start within 11 us on
  Green Context IDs 2 and 3; the following pair starts only after both kernels
  in the previous pair finish. This confirms concurrent groups and no
  cross-round drift.

## Conclusion

This isolated pull-kernel experiment reproduces a twitter-specific benefit
without BFS state, frontier handling, or asynchronous iteration progress.
Therefore pure pull execution is sufficient to explain at least a material
part of the previously observed twitter improvement. The result does not yet
establish that high-degree warp tails are the cause; it narrows the next test
to kernel scheduling and degree-tail behavior.

## Issues

- No dataset failed and soc-sinaweibo fit in device memory.
- Nsight Systems adds instrumentation overhead, so its durations are used only
  to validate ordering and overlap, not as performance measurements.

## Next Steps

Compare soc-twitter with soc-sinaweibo and soc-orkut using degree quantiles,
per-kernel SM active/eligible-warp metrics, and end-of-kernel timeline tails.
This is the discriminating test for the proposed extreme-degree load-imbalance
mechanism.
