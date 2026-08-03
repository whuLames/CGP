# Gunrock Multi-Stream Results (N=256)

## Result Summary

Times are median end-to-end query-processing time, excluding graph loading. The
baseline is the previous Gunrock single-stream BFS result or sequential
single-query SSSP/PageRank result. A speedup above 1.0 is better.

| Algorithm | Dataset | Effective Q | Multi-stream (s) | Baseline (s) | Speedup | Status |
|---|---|---:|---:|---:|---:|---|
| BFS | cit-Patents | 64 | 4.512 | 8.984 | 1.991x | complete |
| BFS | soc-orkut | 32 | 13.497 | 15.822 | 1.172x | complete |
| BFS | soc-twitter | 8 | 54.460 | 45.178 | 0.830x | complete (clean rerun) |
| BFS | soc-sinaweibo | 8 | 60.785 | 64.223 | 1.057x | complete |
| SSSP | cit-Patents | 64 | 17.258 | 14.932 | 0.865x | complete |
| SSSP | soc-orkut | 2 | 51.780 | 38.333 | 0.740x | complete |
| SSSP | soc-twitter | 4 | 118.775 | 83.004 | 0.699x | complete (clean rerun) |
| SSSP | soc-sinaweibo | 2 | 205.217 | 185.912 | 0.906x | complete |
| PageRank | cit-Patents | 64 | 20.468 | 21.960 | 1.073x | complete |
| PageRank | soc-orkut | 64 | 49.295 | 49.906 | 1.012x | complete |
| PageRank | soc-twitter | 64 | 154.211 | 153.618 | 0.996x | complete |
| PageRank | soc-sinaweibo | 16 | 381.620 | 378.966 | 0.993x | complete |

## Main Findings

1. BFS benefits from stream-level concurrency on three datasets, with gains
   from 1.057x to 1.991x. On `soc-twitter`, Q=8 is 20.5% slower than the
   single-stream baseline.
2. SSSP does not benefit. It is 10.4% to 43.1% slower across all four datasets.
3. PageRank is nearly unchanged: 0.993x to 1.073x. A single dense PageRank query
   already consumes most available GPU execution and memory bandwidth, leaving
   little useful overlap for additional streams.
4. Memory capacity determines usable concurrency. On a clean V100,
   `soc-twitter` supports BFS at Q=8 but not Q=16, and SSSP at Q=4 but not Q=8.
5. The original `soc-twitter` BFS/SSSP `OOM at Q=2` entries were invalid. Two
   workers were scheduled on GPU 1 concurrently, and later attempts also
   observed reduced memory before graph/query initialization. A clean rerun on
   isolated GPUs produced the completed results above.

These results do not support treating naive Gunrock multi-stream execution as a
uniformly stronger baseline. It helps BFS on graphs where multiple independent
states fit in memory, but is neutral for PageRank and harmful for SSSP.

## Experimental Protocol

- GPU: Tesla V100-SXM2, 32 GiB.
- Total queries: `N=256` for every case.
- BFS/SSSP: exactly the same 256 source vertices and source order as the prior
  external-baseline experiment.
- PageRank: 256 independent executions, damping factor 0.85, fixed 10 iterations.
- Q fallback: `64 -> 32 -> 16 -> 8 -> 4 -> 2`; Q=1 is not reported as
  multi-stream and is already represented by the old baseline.
- Repetition: 2 warmups followed by 5 measured runs; median is reported.
- Timing excludes graph loading and includes per-batch output allocation,
  contexts, stream creation/destruction, and all `N=256` executions.
- Each concurrent query has an independent non-blocking stream, Gunrock context,
  algorithm state, and output buffer; the immutable graph is shared.
- The `soc-twitter` correction used the same source file and source order on
  otherwise idle GPUs 6 and 7. SSSP Q=4 reached about 29,692 MiB device memory.

The source files and SHA-256 values are:

| Dataset | SHA-256 |
|---|---|
| cit-Patents | `0fa9d653d9a7a8d01602440277ed3f4053b38a60644b8c0bedcd05c122caf259` |
| soc-orkut | `411210a61d09b2efd722703b9282fbbeaf5de1a611023f28d1db5b8d079ec071` |
| soc-twitter | `8095797d5d57b5ca3ba8fd6bbd4862b1f9e6829250a073ff645ccdd8e7a74e28` |
| soc-sinaweibo | `f1f0d19b13eb621a3ce8bea5695889b374bc08406b8f97342170615bc44522cb` |

## Correctness Checks

- BFS concurrent Q=4 was checked against Gunrock's CPU reference on a small
  symmetric graph with zero errors.
- SSSP Q=1 and Q=4 produced identical finite counts, sums, and weighted sums for
  every test source.
- PageRank Q=1 and Q=4 produced identical signatures for every execution; each
  rank vector summed to 1.0 after 10 iterations.

## Artifacts

- `results.csv`: complete machine-readable result and baseline comparison.
- `artifacts/<dataset>/<algorithm>/summary.json`: selected Q, all attempted Q
  values, median/min/max, and source hash.
- `artifacts/<dataset>/<algorithm>/q<Q>/runs.csv`: warmup and measured runs.
- `validation/`: correctness-check outputs.
- `run_case.py`: runner with OOM-only Q fallback.
- `summarize.py`: deterministic result and speedup generation.
- `../20260728_gunrock_twitter_oom_recheck/`: clean rerun and OOM diagnosis.
