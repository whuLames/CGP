# Experiment: Q32 phase-aware push scheduling

## Purpose

Compare lockstep baseline, heavy alignment, offset search, and greedy pause for
32 concurrent BFS queries using identical distinct random sources.

## Hypothesis

Cross-query, cross-iteration frontier alignment can reduce union edge work and
shared-push kernel time.

## Dataset

- cit-Patents
- soc-orkut
- soc-twitter
- soc-sinaweibo
- Source: `/home/zyl/data/csr_data`

## Scenario

- Algorithm: BFS
- Traversal: push
- N=Q=32
- Source seed: 42, sampled without replacement per graph
- Strategies: baseline, heavy alignment, offset search, greedy pause

## Environment

- GPU: Tesla V100-SXM2-32GB
- CUDA compiler: 12.8
- Git base: 677f072 plus current replenish optimization working tree

## Command

```bash
cmake --build build --target bench_phase_schedule -j 8
CUDA_VISIBLE_DEVICES=<gpu> ./build/bench_phase_schedule \
  /home/zyl/data/csr_data/<graph> \
  experiments/20260711-152345_phase_schedule_q32/artifacts/<graph> \
  --seed=42 --repeats=3
```

Each graph used one fixed set of 32 distinct random sources. All four strategies
within that graph replayed exactly the same source list from `sources.csv`.

### Scheduling policies

- `baseline`: all unfinished queries advance in lockstep every global step.
- `heavy_alignment`: for each query, find the local iteration with maximum exact
  frontier edge work. Delay its start by `latest_peak - query_peak` so all peaks
  are aligned.
- `offset_search`: construct a degree-weighted cross-query/cross-level overlap
  tensor from deterministic sampled vertices. Run multi-start coordinate ascent
  over offsets in `[0,16]`, maximizing cumulative pairwise overlap. This is an
  approximate Q=32 offset oracle, not an exhaustive global optimum.
- `greedy_pause`: start by advancing every unfinished query. Greedily pause a
  query when keeping its current frontier is predicted to increase pairwise
  overlap in the next step. At least 16 queries advance and no query pauses more
  than three consecutive steps.

The exact masks and local iteration vector for every global step are recorded in
`<strategy>_schedule.csv`.

## Results

Median GPU time over three untraced replays:

| Dataset | Strategy | GPU ms | Speedup | Union edges | Union reduction | Sharing |
|---|---|---:|---:|---:|---:|---:|
| cit-Patents | baseline | 272.983 | 1.000x | 198,126,703 | 0.0% | 5.334x |
| cit-Patents | heavy | 213.387 | 1.279x | 142,814,802 | 27.9% | 7.400x |
| cit-Patents | offset | 209.179 | 1.305x | 139,888,590 | 29.4% | 7.554x |
| cit-Patents | greedy | 353.143 | 0.773x | 276,108,587 | -39.4% | 3.827x |
| soc-orkut | baseline | 368.414 | 1.000x | 721,024,257 | 0.0% | 9.440x |
| soc-orkut | heavy | 364.272 | 1.011x | 681,265,145 | 5.5% | 9.991x |
| soc-orkut | offset | 355.933 | 1.035x | 670,425,149 | 7.0% | 10.152x |
| soc-orkut | greedy | 642.496 | 0.573x | 1,522,157,756 | -111.1% | 4.472x |
| soc-twitter | baseline | 1488.960 | 1.000x | 2,806,343,900 | 0.0% | 6.044x |
| soc-twitter | heavy | 725.704 | 2.052x | 1,225,668,913 | 56.3% | 13.839x |
| soc-twitter | offset | 726.682 | 2.049x | 1,225,668,913 | 56.3% | 13.839x |
| soc-twitter | greedy | 1311.910 | 1.135x | 2,396,652,464 | 14.6% | 7.077x |
| soc-sinaweibo | baseline | 1342.920 | 1.000x | 1,192,493,615 | 0.0% | 14.025x |
| soc-sinaweibo | heavy | 1376.110 | 0.976x | 1,229,843,790 | -3.1% | 13.599x |
| soc-sinaweibo | offset | 1278.220 | 1.051x | 1,116,714,451 | 6.4% | 14.977x |
| soc-sinaweibo | greedy | 1864.950 | 0.720x | 1,869,254,595 | -56.8% | 8.947x |

All 16 replays passed both per-query visited-count and BFS-distance checksum
comparison against baseline.

## Observations

- Union edge work strongly predicts shared-push time. Large reductions on
  cit-Patents and soc-twitter become large measured speedups.
- Heavy alignment is effective on three graphs but slightly harmful on
  soc-sinaweibo. Aligning only the maximum edge-work iteration is not universally
  sufficient.
- Offset search never increases union work in these four cases and improves GPU
  time by 3.5% to 51.2% relative to baseline.
- One-step greedy pause is unstable. It cannot see the long-term displacement of
  future frontiers; repeated locally attractive pauses can destroy global
  alignment. On twitter it reduces union work, but carry and extra steps consume
  part of the gain.
- Every traced replay records ready and scheduled union/virtual edge work,
  sharing factor, scheduled mask, local iterations, and push-kernel time in
  `<strategy>_steps.csv`.

## Conclusion

The experiment supports the phase-aware scheduling hypothesis. Cross-iteration
alignment can materially reduce actual shared-frontier traversal work at Q=32,
with up to 2.05x measured speedup for the tested sources. The robust policy in
this experiment is offset search; the current local greedy pause policy is not
suitable as an online scheduler.

## Issues

- The first replay version checked reachability only and used global BFS levels.
  The final version uses a phase-aware BFS kernel with per-query local levels and
  validates distance checksums.
- `offset_search` is sample-guided and approximate; it must not be reported as an
  exact global oracle.

## Next Steps

- Repeat with multiple source seeds and report confidence intervals.
- Evaluate exact small-Q pause oracles to quantify the approximation gap.
- Replace one-step greedy pause with a longer-horizon or target-offset online
  policy before integrating scheduling with replenish.
