# E0: replenish granularity sweep (roadNet-CA)

Protocol: identical to `20260712_roadnet_replenish` (BFS-only, Q=32,
all-push shared_node_warp, no-discard, warmup=1, repeats=3, median,
max_iterations=20000, same interleaved low/high eccentricity sources).

Reproduction check: chunk=1 gives 0.926x (N=400) / 0.889x (N=800) vs the
historical 0.925x / 0.892x — protocol reproduces.

## N=400 (static sequential = 3172.94 ms, 10602 steps, 0.2993 ms/step)

| chunk | total ms | vs static | global steps | ms/step | step-cost vs static |
|---:|---:|---:|---:|---:|---:|
| 1  | 3426.51 | 0.926x | 8915  | 0.3844 | +28.4% |
| 4  | 3423.60 | 0.927x | 9363  | 0.3657 | +22.2% |
| 8  | 3350.66 | 0.947x | 9464  | 0.3541 | +18.3% |
| 16 | 3336.96 | 0.951x | 10281 | 0.3246 | +8.5%  |
| 32 | 3234.43 | 0.981x | 10615 | 0.3047 | +1.8%  |

## N=800 (static sequential = 6151.32 ms, 19847 steps, 0.3099 ms/step)

| chunk | total ms | vs static | global steps | ms/step | step-cost vs static |
|---:|---:|---:|---:|---:|---:|
| 1  | 6920.48 | 0.889x | 17255 | 0.4011 | +29.4% |
| 4  | 6820.86 | 0.902x | 17908 | 0.3809 | +22.9% |
| 8  | 6608.96 | 0.931x | 18149 | 0.3642 | +17.5% |
| 16 | 6552.56 | 0.939x | 19828 | 0.3305 | +6.6%  |
| 32 | 6302.93 | 0.976x | 19872 | 0.3172 | +2.4%  |

## Reading

1. Monotone: coarser replenish granularity is strictly better on this
   workload; no interior optimum. chunk=32 (full-cohort) still trails static
   by ~2%, matching the earlier cohort-32 parity result on the social graphs.
2. Clean decomposition: immediate replenish (chunk=1) removes 13-16% of
   global steps but inflates per-step cost by ~29% (union-frontier phase
   mixing); the two effects cross over nowhere — step-cost inflation always
   wins. At chunk=32 the step count equals static (19872 vs 19847), i.e.
   full-cohort replenish saves no steps at all on an N>>Q FIFO queue; its
   only remaining value is buffer reuse and N>64 capability.
3. Consequence for the group design (E2): granularity alone cannot beat
   static batching. Any win must come from what refill-time scheduling adds
   on top: affinity-based batch formation and offset alignment. Group size
   should be as large as phase coherence allows (32, or 2x32 groups).
4. Latency: chunk=32 median completion 1765 ms (N=400) / 3357 ms (N=800) vs
   static 1775 / 3289 — no tail-latency penalty from coarse granularity.
