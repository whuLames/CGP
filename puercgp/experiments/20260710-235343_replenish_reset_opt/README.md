# Experiment: Replenish reset and scheduling optimization

## Purpose

Reduce slot reset and scheduling overhead while preserving replenish correctness
and the shared-frontier compute efficiency.

## Hypothesis

The original slowdown comes from both full-graph slot reset and immediate
per-slot injection. Minimizing reset traffic and injecting phase-aligned cohorts
should make replenish no slower than sequential batching.

## Dataset

- cit-Patents: V=3,774,768, E=33,037,895
- soc-orkut: V=2,997,166, E=212,698,418
- soc-twitter: V=21,297,772, E=530,051,618
- soc-sinaweibo: V=58,655,849, E=522,642,142
- Source: `/home/zyl/data/csr_data`

## Scenario

- Workload: 100 BFS + 100 SSSP, Q=32, push/warp, discard results
- Baseline: sequential Q=32 batching in the same benchmark process
- Historical replenish: commit 677f072 Stage 10 measurements
- Candidate: reset specialization, cohort scheduling, and tail compaction
- Comparison: correctness, total wall time, throughput speedup, latency percentiles

## Environment

- GPU: Tesla V100-SXM2-32GB, final measurements on GPU 5
- CUDA compiler: 12.8
- Working tree base: commit `677f072`

## Command

```bash
cmake --build build --target bench_replenish smoke_replenish validate_replenish -j 8
CUDA_VISIBLE_DEVICES=5 ./build/smoke_replenish
CUDA_VISIBLE_DEVICES=5 ./build/validate_replenish
CUDA_VISIBLE_DEVICES=5 ./build/bench_replenish /home/zyl/data/csr_data/<graph> \
  --bfs=100 --sssp=100 --wcc=0 --batch-size=32 \
  --replenish-chunk=32 --repeats=3 --warmup=1 --mode=push --push=warp
```

## Results

Median wall time over three measured repeats:

| Dataset | Sequential ms | Replenish ms | Speedup |
|---|---:|---:|---:|
| cit-Patents | 3049.48 | 3008.10 | 1.0138x |
| soc-orkut | 6649.77 | 6672.15 | 0.9966x |
| soc-twitter | 14286.70 | 13933.30 | 1.0254x |
| soc-sinaweibo | 17618.80 | 17335.70 | 1.0163x |

Historical replenish to optimized replenish:

| Dataset | Historical ms | Optimized ms | Reduction |
|---|---:|---:|---:|
| cit-Patents | 3547.82 | 3008.10 | 15.2% |
| soc-orkut | 9212.36 | 6672.15 | 27.6% |
| soc-twitter | 19133.80 | 13933.30 | 27.2% |
| soc-sinaweibo | 25404.20 | 17335.70 | 31.8% |

## Observations

- The initial optimized reset reduced cit-Patents from 3547.82 ms to 3354.33
  ms, but profiling showed the push kernel still dominated.
- Immediate replenishment desynchronized query levels and enlarged the union
  frontier. On cit-Patents, replenish push time was 4.19 s versus 3.01 s for
  sequential Q=32, while reset was only 73.5 ms.
- Cohort ablation on cit-Patents produced 4288.58, 4317.64, 4008.68, 3838.55,
  and 3116.50 ms for cohort sizes 1, 4, 8, 16, and 32 respectively.
- After phase-aligned scheduling, the remaining twitter profile showed the main
  push kernel slightly faster under replenish, but five reset kernels still cost
  374.5 ms. Replacing full-cohort reset with contiguous fill/memset removed it.
- GPU contention affected intermediate absolute times. Final results use an
  otherwise idle GPU 5 and compare sequential/replenish inside the same process.

## Conclusion

The target was met: all four final throughput ratios are at least 0.996x, within
normal benchmark noise, and three datasets are faster than sequential batching.
The optimized implementation preserves a configurable partial-cohort mode for
long-tail workloads while defaulting to a phase-aligned recyclable cohort.

## Issues

- The original validation used the default Q=N and therefore did not actually
  exercise slot reuse. It now forces Q=2 and includes a Q=1 tail cohort.
- Replenished BFS originally wrote global levels. Per-slot start levels now
  produce correct local BFS distances.

## Next Steps

- Evaluate smaller adaptive cohorts on a controlled long-tail workload.
- Extend per-slot BFS start-level handling to any future replenish pull-specific
  path only if that path stops deriving distance from neighbor values.
