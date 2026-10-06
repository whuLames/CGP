# Wide-query Pull tile allocation and q-grouping microbenchmark

## Design

For D=128 and D=256 concurrent queries, this standalone SSSP Pull benchmark
compares two node-level query-tile mappings:

- `serial`: exactly one warp per destination vertex; that warp loops over
  `D/32` query tiles;
- `parallel`: `D/32` independent warps per destination vertex; every warp
  owns one 32-query tile.

Both mappings test q32, q16, and q8 warp-internal query grouping.  The fixed
baseline is serial-q32.  There is no degree bucketing or edge-chunk expansion.
Three paired warmups and 11 interleaved samples are used (7 for the largest
cases).  A two-word 64-bit full-output fingerprint is checked outside the
timed interval.

## Results relative to serial-q32

### D=128

| Graph | serial-q16 | serial-q8 | parallel-q32 | parallel-q16 | parallel-q8 | Best |
|---|---:|---:|---:|---:|---:|---|
| cit-Patents | 1.039x | 0.753x | **1.337x** | 1.238x | 1.173x | parallel-q32 |
| soc-LiveJournal1 | 0.951x | 0.603x | **1.619x** | 1.389x | 1.347x | parallel-q32 |
| indochina | 1.936x | 1.175x | **3.680x** | 3.490x | 3.574x | parallel-q32 |
| soc-orkut | 0.991x | 0.629x | **1.587x** | 1.498x | 1.479x | parallel-q32 |
| soc-twitter | 1.631x | 1.073x | 3.259x | **3.271x** | 3.201x | parallel-q16 (0.23% over q32) |

The five-graph geometric-mean speedup of parallel-q32 over serial-q32 is
2.104x.

### D=256

| Graph | serial-q16 | serial-q8 | parallel-q32 | parallel-q16 | parallel-q8 | Best |
|---|---:|---:|---:|---:|---:|---|
| cit-Patents | 1.026x | 0.743x | **1.330x** | 1.246x | 1.180x | parallel-q32 |
| soc-LiveJournal1 | 0.951x | 0.598x | **1.653x** | 1.526x | 1.474x | parallel-q32 |
| indochina | 1.799x | 1.126x | **4.310x** | 4.195x | 3.783x | parallel-q32 |
| soc-orkut | 0.977x | 0.621x | **1.596x** | 1.545x | 1.539x | parallel-q32 |

The four-graph geometric-mean speedup of parallel-q32 is 1.972x.  Twitter
D=256 is infeasible on a 32 GiB V100: old and new value matrices alone require
about 43.6 GB.

## Interpretation

At D>32, exposing query tiles as independent warp tasks dominates serial tile
looping.  It reduces the longest warp duration by up to D/32 and requires no
cross-warp reduction because tiles write disjoint queries.  The gain is largest
on indochina and twitter, where long vertex rows otherwise amplify the serial
loop tail.

Once tile-level warp parallelism is exposed, q32 is best in every tested case
except a 0.23% q16 advantage on twitter D=128, which is effectively a tie at
this sample count.  Smaller q remains useful for skewed graphs only when the
same warp must serialize all tiles; it cannot compensate for hiding tile-level
parallelism.

This is a dense kernel-only experiment and is not integrated into the engine.

## Reproduction

```bash
cmake --build build --target graphweft_pull_wide_qgroup -j4
CUDA_VISIBLE_DEVICES=0 ./build/graphweft_pull_wide_qgroup GRAPH D 0 11
```
