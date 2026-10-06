# Pull warp-internal query grouping microbenchmark

## Design

This standalone Q=32 SSSP Pull benchmark fixes node-level allocation at
exactly one warp per destination.  It varies only the warp-internal split
between query lanes and independent incoming-edge streams:

| Mapping | Edge streams/warp | Queries serially handled/lane |
|---|---:|---:|
| q32 | 1 | 1 |
| q16 | 2 | 2 |
| q8 | 4 | 4 |
| q4 | 8 | 8 |
| q2 | 16 | 16 |
| q1 | 32 | 32 |

WV-Q32 is the unchanged baseline.  Smaller-q candidates reduce edge-stream
partials in shared memory.  There is no node bucketing, multi-warp vertex,
edge-chunk expansion, or preprocessing.  Five paired warmups and 21
interleaved samples are used (11 for soc-twitter), and every result is checked
cell-by-cell against q32.

## V100 results

| Graph | q16 | q8 | q4 | q2 | q1 | Best |
|---|---:|---:|---:|---:|---:|---|
| cit-Patents | 0.951x | 0.824x | 0.794x | 0.452x | 0.287x | q32 |
| soc-LiveJournal1 | 0.833x | 0.900x | 0.857x | 0.497x | 0.314x | q32 |
| indochina | 1.035x | **1.187x** | 1.162x | 0.606x | 0.358x | q8 |
| soc-orkut | 0.929x | 0.969x | 0.929x | 0.526x | 0.350x | q32 |
| soc-twitter | 0.907x | 1.104x | **1.324x** | 0.755x | 0.499x | q4 |

Values are q32 time divided by candidate time, so values above one are
speedups.  The best absolute times are 40.786 ms for indochina q8 versus
48.399 ms q32, and 114.497 ms for soc-twitter q4 versus 151.549 ms q32.

## Interpretation

Warp-internal grouping matters only when enough incoming-edge work is
concentrated in long vertex rows.  Twitter benefits from eight edge streams
per warp and indochina from four.  The other three graphs do not recover the
cost of serializing queries and reducing edge-stream partials.  q1 and q2 are
uniformly poor because they over-rotate the warp toward edge parallelism,
increase per-lane query serialization and shared-memory pressure, and reduce
occupancy.

This is a kernel-only dense SSSP Pull step.  It does not yet include live-query
masks or captured iteration states and is not integrated into the engine.

## Reproduction

```bash
cmake --build build --target graphweft_pull_qgroup -j4
CUDA_VISIBLE_DEVICES=0 ./build/graphweft_pull_qgroup GRAPH_DIRECTORY 0 21
```
