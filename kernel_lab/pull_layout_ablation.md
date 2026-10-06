# Pull value-layout ablation

## Layouts

The same compile-time-specialized kernels were evaluated with two physical
value layouts:

```text
vertex-major [V][M]:       v * M + q
tile32 [M/32][V][32]:      (q/32) * V * 32 + v * 32 + (q%32)
```

Only address calculation changes. Graph data, arithmetic, launch geometry,
warmups, measurement count, and validation are identical. The optimized
`tile32` kernels replace `/32` and `%32` with shifts/masks and hoist the tile
stride/base (or the per-query offset in fused-serial) out of the hot edge loop.
Results below use three warmups, seven samples, and the median GPU-event time
on a V100-SXM2-32GB.

## Tile32, D=128: kernel time in ms

| Kernel | cit-Patents | LiveJournal | indochina | Orkut | Twitter |
|---|---:|---:|---:|---:|---:|
| Legacy serial-global-Q32 | 37.268 | 81.825 | 228.311 | 164.573 | 705.386 |
| Fused-serial-global-Q32 | **34.330** | 66.059 | 147.825 | 154.249 | 588.970 |
| Fused-serial-smem-Q32 | 34.500 | 64.028 | 129.423 | 145.823 | 491.952 |
| Fused-serial-smem-Q16 | 36.564 | 65.300 | 101.852 | 153.501 | 442.649 |
| Parallel-global-Q32 | 35.222 | **62.014** | **94.323** | **139.021** | **371.236** |
| Parallel-smem-Q16 | 37.223 | 66.360 | 123.631 | 145.102 | 420.277 |

Best tile32 choice:

- cit-Patents: fused-serial-global-Q32;
- LiveJournal, indochina, Orkut, and Twitter: parallel-global-Q32.

Across five graphs, parallel-global-Q32 is 1.500x faster than the tile32
legacy serial baseline. After address hoisting, adding smem-Q16 to parallel
has a 0.894x geometric mean relative to parallel-global-Q32 and is not the
winner on any of these graphs.

## Tile32, D=256: kernel time in ms

| Kernel | cit-Patents | LiveJournal | indochina | Orkut |
|---|---:|---:|---:|---:|
| Legacy serial-global-Q32 | 74.330 | 167.130 | 460.133 | 340.771 |
| Fused-serial-global-Q32 | **67.748** | 130.809 | 213.968 | 308.661 |
| Fused-serial-smem-Q32 | 70.702 | 137.996 | 339.063 | 319.189 |
| Fused-serial-smem-Q16 | 88.831 | 168.087 | 325.486 | 405.473 |
| Parallel-global-Q32 | 70.442 | **123.838** | **171.219** | **282.274** |
| Parallel-smem-Q16 | 74.745 | 133.695 | 208.067 | 296.418 |

Best tile32 choice:

- cit-Patents: fused-serial-global-Q32;
- LiveJournal, indochina, and Orkut: parallel-global-Q32.

Across four graphs, parallel-global-Q32 is 1.466x faster than the tile32
legacy serial baseline. Parallel-smem-Q16 is consistently slower, with a
0.909x geometric mean relative to parallel-global-Q32.

## Direct best-kernel layout comparison

| D | cit-Patents | LiveJournal | indochina | Orkut | Twitter | Geomean |
|---|---:|---:|---:|---:|---:|---:|
| 128: tile32 time / vertex-major time | 1.241 | 1.210 | 1.299 | 1.218 | 1.270 | **1.247** |
| 256: tile32 time / vertex-major time | 1.223 | 1.184 | 1.230 | 1.191 | n/a | **1.207** |

Even after independently selecting the fastest tested kernel for each layout,
tile32 is 24.7% slower at D=128 and 20.7% slower at D=256 in geometric mean.
These ratios use vertex-major and tile32 measurements produced by the same
layout-specialized binary. All 72 tile32 case/variant output fingerprints
match their tile32 serial-Q32 reference; the corresponding 72 vertex-major
fingerprints also match. Twitter D=256 is omitted because its value arrays do
not fit on a 32-GB V100.

## Interpretation

The layout materially changes both absolute performance and kernel selection.
Nsight Compute on cit-Patents D=128 confirms that the optimization reduces
address-generation work without changing DRAM traffic or cache behavior:

| Kernel/layout | Time | Executed warp instructions | DRAM read | L1 hit | L2 hit |
|---|---:|---:|---:|---:|---:|
| fused-smem-Q32, vertex-major | 27.48 ms | 0.988 B | 18.82 GB | 2.29% | 10.88% |
| fused-smem-Q32, tile32 before hoisting | 37.88 ms | 1.283 B | 18.82 GB | 2.27% | 10.90% |
| fused-smem-Q32, tile32 after hoisting | 34.29 ms | 1.222 B | 18.82 GB | - | 10.89% |
| parallel-global-Q32, vertex-major | 30.19 ms | 3.467 B | 18.81 GB | 29.94% | 11.83% |
| parallel-global-Q32, tile32 before hoisting | 35.19 ms | 3.981 B | 18.81 GB | 29.95% | 11.84% |
| parallel-global-Q32, tile32 after hoisting | 35.14 ms | 3.282 B | 18.81 GB | - | 11.82% |

For fused-smem, hoisting cuts executed instructions by 4.7% and kernel time by
9.5%. For parallel-global it cuts executed instructions by 17.6%, to slightly
fewer instructions than vertex-major, but changes time by less than 1%. Thus
the original repeated division/modulo-style indexing was a real cost, but it
was not the sole explanation for the layout gap. In fused-serial, one lane's
multiple outputs reside in widely separated query tiles, so the compiler must
maintain multiple bases rather than one contiguous vertex-major base. In the
parallel kernel, the remaining gap is not explained by instruction count,
DRAM bytes, or L2 hit rate and needs instruction-dependency/stall analysis
before attributing it to a single cause.

The result does not justify changing GraphWeft's system layout by itself:
tile32 is also used by Push, frontier construction, refill, and group-level
scheduling. It does show that Pull selection and any future layout decision
must be evaluated end-to-end rather than importing the vertex-major
microbenchmark result.

## Reproduction

```bash
cmake --build build -j 8 --target graphweft_pull_wide_smem_shuffle
./build/graphweft_pull_wide_smem_shuffle \
  /home/zyl/data/csr_data/indochina 128 0 7 fused tile32
```
