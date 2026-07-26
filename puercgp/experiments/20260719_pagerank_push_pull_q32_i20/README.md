# PageRank and PPR Push/Pull Performance (Q=32, 20 Iterations)

## Result First

### PageRank

| Dataset | Vertices | Edges | Push GPU (ms) | Pull GPU (ms) | Pull speedup | Push GEQ/s | Pull GEQ/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| cit-Patents | 3,774,768 | 33,037,895 | 2,166.82 | 484.86 | 4.47x | 9.76 | 43.61 |
| soc-orkut | 2,997,166 | 212,698,418 | 20,588.28 | 2,206.87 | 9.33x | 6.61 | 61.68 |
| soc-twitter | 21,297,772 | 530,051,618 | 26,744.42 | 11,309.67 | 2.36x | 12.68 | 30.00 |
| soc-sinaweibo | 58,655,849 | 522,642,142 | 25,188.76 | 8,335.09 | 3.02x | 13.28 | 40.13 |

### PPR

| Dataset | Vertices | Edges | Push GPU (ms) | Pull GPU (ms) | Pull speedup | Push GEQ/s | Pull GEQ/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| cit-Patents | 3,774,768 | 33,037,895 | 2,160.88 | 430.76 | 5.02x | 9.79 | 49.09 |
| soc-orkut | 2,997,166 | 212,698,418 | 20,577.71 | 1,906.90 | 10.79x | 6.62 | 71.39 |
| soc-twitter | 21,297,772 | 530,051,618 | 26,817.43 | 11,680.70 | 2.30x | 12.65 | 29.04 |
| soc-sinaweibo | 58,655,849 | 522,642,142 | 25,228.68 | 7,115.10 | 3.55x | 13.26 | 47.01 |

`GEQ/s` is the effective edge-query throughput calculated as
`20 * |E| * Q / gpu_time`; it includes engine overhead and is not a
kernel-only throughput measurement.

## Configuration

- GPU: NVIDIA Tesla V100-SXM2 32 GB (GPU 0)
- Algorithms: synchronous PageRank and source-personalized PageRank (PPR)
- Query count: 32
- Damping factor: 0.85 for every query
- PageRank queries use uniform personalization; PPR uses 32 distinct,
  deterministic sources distributed uniformly over the vertex ID range
- Iterations: exactly 20
- Repetitions: 3; the table reports the median
- Traversal modes: push-only and pull-only
- Profiling: disabled to avoid per-kernel profiling events
- Convergence epsilon: `1e-30`, used to prevent early convergence before the
  requested iteration count

The GPU timer includes rank initialization and all 20 iterations, including
dangling-mass computation, the push/pull backend, frontier postprocessing, and
iteration count synchronization. Graph loading, CSR transfer, and workspace
allocation are outside the timer.

All four binary CSR datasets passed a 100,000-edge symmetry sample and are
treated as undirected/symmetric graphs. The benchmark aliases the outgoing CSR
as incoming CSR, avoiding an unnecessary duplicate transpose on the GPU.

## Correctness Check

Both modes completed exactly 20 iterations in every repetition. The benchmark
compared all 32 query values at 16 evenly spaced vertices after the final
repetition. For PageRank, the maximum sampled relative differences were below
the original output's `5e-7` resolution on cit-Patents and soc-orkut, `6.2e-5`
on soc-twitter, and `3.0e-5` on soc-sinaweibo.

For PPR, the `(max absolute, max relative)` sampled differences were
`(4.55e-13, 4.32e-7)`, `(1.49e-8, 1.68e-6)`, `(1.17e-10, 3.95e-3)`, and
`(1.18e-6, 7.55e-4)` in table order. The larger relative value on soc-twitter
comes from rank values close to zero. These differences are consistent with
floating-point sum ordering between atomic scatter and pull gather.

A directed small-graph smoke test used a materialized transpose; after 20
iterations its sampled maximum absolute and relative differences were
`1.79e-7` and `2.86e-7`, respectively.

## Reproduction

```bash
cmake --build build --target bench_pagerank_push_pull -j4
CUDA_VISIBLE_DEVICES=0 ./build/bench_pagerank_push_pull \
  /home/zyl/data/csr_data/cit-Patents 32 20 3 pagerank
```

Replace the graph path with `soc-orkut`, `soc-twitter`, or `soc-sinaweibo` for
the other datasets. Replace the final argument with `ppr` to run PPR.

## Observation

Pull is faster for both algorithms on every tested graph. The advantage is
largest on soc-orkut, where its higher average degree amortizes vertex-level
setup and postprocessing while avoiding the push path's heavily contended
atomic scatter. On the two largest-vertex graphs, vertex-proportional scans and
rank-buffer traffic reduce the relative pull advantage, but pull remains 2.30x
to 3.55x faster.
