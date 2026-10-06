# Strided single-query clear microbenchmark

## Question

For a vertex-major value array `[V][M]`, clearing one fixed query writes
`data[v*M+q]` for every vertex. This benchmark isolates the cost of that
stride. It also compares two ways of clearing a completed 32-query group:

- `vertex-major-clear-32`: `[V][M]` remains unchanged, but one warp writes the
  32 adjacent query values of each vertex;
- `tile32-clear-32`: the physically contiguous `V*32` tile is written linearly.

The operation is a write-only reset, matching query-state clearing rather than
a read-modify-write update. Tests used a V100-SXM2-32GB, five warmups, 21 timed
samples, and the median CUDA-event time. The written value changes on every
sample and the final strided value is copied back for validation.

## Single-query result, V=10,000,000

Each row logically writes 40 MB.

| M / element stride | Time (ms) | Logical GB/s | Slowdown vs contiguous |
|---:|---:|---:|---:|
| 1 | 0.071 | 566.1 | 1.0x |
| 2 | 0.196 | 204.5 | 2.8x |
| 4 | 0.389 | 102.8 | 5.5x |
| 8 | 0.778 | 51.4 | 11.0x |
| 16 | 1.047 | 38.2 | 14.8x |
| 32 | 2.084 | 19.2 | 29.5x |
| 64 | 3.209 | 12.5 | 45.4x |
| 128 | 3.727 | 10.7 | 52.8x |
| 256 | 5.183 | 7.7 | 73.4x |

The same trend appears at V=1M and V=4M. At M=32, the measured times are
0.236, 0.999, and 2.084 ms for V=1M, 4M, and 10M. At M=256 they are 0.522,
2.257, and 5.183 ms. The loss therefore persists after launch overhead is
amortized.

## Results at the five graph vertex counts

This microbenchmark does not traverse graph edges, so a dataset affects it
only through its vertex count. These runs use the exact `V` obtained from each
graph's CSR row-offset array. All use the same five-warmup, 21-sample protocol
and pass validation.

| Graph | V | Single q, M=32 | Single q, M=128 | Single q, M=256 |
|---|---:|---:|---:|---:|
| cit-Patents | 3,774,768 | 0.851 ms | 1.585 ms | 2.021 ms |
| LiveJournal | 4,847,571 | 1.207 ms | 1.847 ms | 2.738 ms |
| indochina | 7,414,768 | 1.654 ms | 2.900 ms | 3.882 ms |
| Orkut | 2,997,166 | 0.670 ms | 1.265 ms | 1.537 ms |
| Twitter | 21,297,772 | 4.394 ms | 7.525 ms | 11.054 ms |

For a simultaneous 32-query clear:

| Graph | tile32 contiguous | `[V][M]`, M=32 | `[V][M]`, M=128 | `[V][M]`, M=256 |
|---|---:|---:|---:|---:|
| cit-Patents | 0.555 ms | 0.559 ms | 0.584 ms | 0.680 ms |
| LiveJournal | 0.742 ms | 0.742 ms | 0.753 ms | 0.862 ms |
| indochina | 1.108 ms | 1.116 ms | 1.167 ms | 1.285 ms |
| Orkut | 0.436 ms | 0.436 ms | 0.446 ms | 0.520 ms |
| Twitter | 3.456 ms | 3.434 ms | 4.118 ms | 3.960 ms |

At M=32 the two group-clear layouts are equal within measurement noise. At
M=256, the vertex-major group kernel is 14.6--22.5% slower, with a 17.6%
geometric-mean penalty across the five graph sizes. This remains radically
different from issuing 32 single-query strided clears.

The reason is store-transaction utilization. At M>=32, adjacent warp lanes
write addresses at least 128 bytes apart. Although only 128 logical bytes are
written per warp, each lane touches a separate memory sector. Larger strides
also enlarge the page/TLB working set.

## Clearing 32 queries together, V=10,000,000

Each row logically writes 1.28 GB.

| Layout/method | M | Total time (ms) | Logical GB/s |
|---|---:|---:|---:|
| vertex-major, coalesced group kernel | 32 | 1.518 | 843.5 |
| vertex-major, coalesced group kernel | 64 | 1.741 | 735.3 |
| vertex-major, coalesced group kernel | 128 | 1.707 | 749.9 |
| vertex-major, coalesced group kernel | 256 | 1.759 | 727.6 |
| tile32, linear kernel | 32 | 1.518 | 843.5 |

The tile32 layout makes the whole operation a single contiguous interval and
therefore permits a direct linear clear (or `cudaMemset` for zero). However,
physical global contiguity is not required for efficient clearing: `[V][M]`
can retain 85--100% of the tile32 bandwidth by assigning one warp to the 32
adjacent values of each vertex. At M=256 the coalesced vertex-major group clear
is only 15.9% slower than the fully contiguous tile clear, not 73x slower.

Nsight Compute confirms that this is genuinely coalesced rather than a timing
artifact. At V=10M, the group kernel issues four 32-byte store sectors per warp
request and writes 1.28 GB to DRAM, exactly matching its 1.28-GB logical
payload. The M=32 single-query kernel issues 32 sectors per warp request and
generates about 321 MB of DRAM writes for only 40 MB of logical payload. The
group result assumes 32 consecutive, preferably 32-aligned, query IDs; an
arbitrary set of completed query IDs does not have this property.

Consequently:

- clearing one query at a time is very expensive in either `[V][M]` with large
  M or one lane of tile32 (whose effective vertex stride is 32);
- when 32 queries complete together, both layouts can clear them efficiently;
- tile32 primarily provides implementation simplicity and direct contiguous
  range operations, rather than an order-of-magnitude mandatory advantage for
  a custom GPU clear kernel.

## Reproduction

```bash
cmake --build build -j 8 --target graphweft_strided_query_clear
./build/graphweft_strided_query_clear 10000000 0 21
```
