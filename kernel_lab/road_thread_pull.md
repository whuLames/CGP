# Road-graph Thread-Vertex Pull microbenchmark

## Design

This standalone Q=32 SSSP Pull microbenchmark compares:

- `WV-Q32`: one warp per destination vertex and one query per lane;
- `TV-GK-QTT`: one thread owns `K` consecutive destination vertices and
  serially processes all 32 queries in tiles of `T` accumulators.

All candidates use the same incoming CSR and vertex-major Q32 values.  There
is no degree bucketing, edge-chunk expansion, partial buffer, or preprocessing.
Five paired warmups and 21 interleaved baseline/candidate samples are used;
the table reports CUDA-event medians.  Every candidate output is checked
cell-by-cell against WV-Q32.

## V100 results

| Graph | WV-Q32 | Best Thread-Vertex | TV time | WV/TV |
|---|---:|---|---:|---:|
| roadNet-CA | 1.278 ms | TV-G1-QT16 | 3.173 ms | 0.403x |
| roadNet-TX | 0.882 ms | TV-G1-QT16 | 2.237 ms | 0.394x |

Increasing vertex grouping consistently hurts.  On roadNet-CA, G2-QT16 is
3.538 ms, G4-QT8 is 8.039 ms, and G8-QT8 is 11.289 ms.  Query tiles smaller
than 16 reread each short adjacency list too often; QT32 does not improve on
QT16, consistent with increased accumulator/register pressure.

## Interpretation

Low degree alone does not make Thread-Vertex favorable for GraphWeft's Q32
layout.  WV-Q32 uses all 32 lanes for independent queries and accesses the 32
values of a source vertex contiguously.  Thread-Vertex serializes that query
parallelism inside one lane; neighboring lanes own different vertices, so
their value accesses are scattered.  The lost query parallelism and memory
coalescing dominate any reduction in warp-level edge work.

This is a kernel-level negative result and is not integrated into the engine.

## Reproduction

```bash
cmake --build build --target graphweft_road_thread_pull -j4
CUDA_VISIBLE_DEVICES=0 ./build/graphweft_road_thread_pull GRAPH_DIRECTORY 0 21
```
