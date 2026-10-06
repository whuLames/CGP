# Wide-query Pull: serial/parallel, shuffle, and shared row cache

## Scope

This standalone weighted min-plus Pull experiment evaluates `D=128` and
`D=256` with:

- `serial`: one warp owns a vertex and processes all 32-query tiles in order;
- `parallel`: `D/32` query-tile warps process the same vertex concurrently;
- Q16/Q8 shared-memory reduction versus warp-shuffle reduction;
- GE-SpMM-style 32-edge shared-memory row caching.

For serial execution, the edge cache is warp-private. For parallel execution,
one block owns one vertex and all `D/32` query-tile warps reuse the same edge
cache. Thus the parallel cached kernel fetches each `(source, weight)` pair
once per vertex/tile instead of once per query-tile warp.

## Method

- GPU: Tesla V100-SXM2-32GB
- Value layout in this report: vertex-major `[V][D]`. These numbers are not
  GraphWeft's grouped/tile32 production layout; see `pull_layout_ablation.md`
  for the corrected layout comparison.
- Graphs: five non-road graphs for D=128; four for D=256
- Timing: three paired warmups and seven interleaved paired measurements;
  median GPU event time
- Validation: two-word full-output fingerprint outside the timed region
- Serial variants are relative to `serial-global-q32`
- Parallel variants are relative to `parallel-global-q32`, except the explicit
  `parallel-global-q32` row, which is relative to serial Q32

Twitter-D256 is infeasible on a 32 GiB V100 because the two dense value
matrices alone require about 43.6 GB.

## GE-SpMM-style fused serial correction

The original `serial` kernel below is a legacy tile-loop baseline: it traverses
the sparse row once for every 32-query tile. It is not the fully coarsened
GE-SpMM mapping. The corrected fused kernel assigns one warp to one vertex,
keeps `D/32` accumulators per lane, and traverses the sparse row exactly once.
Its smem variant cooperatively loads every 32-edge tile once before using that
tile for all 128 or 256 queries.

### D=128

| Graph | Fused-global / legacy serial | Fused-smem / legacy serial | Best parallel / legacy serial | Faster final kernel |
|---|---:|---:|---:|---|
| cit-Patents | 1.387 | **1.454** | 1.347 | fused-smem by 7.9% |
| LiveJournal | 1.625 | **1.744** | 1.712 | fused-smem by 1.9% |
| indochina | 3.205 | 3.848 | **4.862** | parallel by 26.3% |
| Orkut | 1.480 | 1.601 | **1.655** | parallel by 3.4% |
| Twitter | 2.311 | 2.807 | **3.663** | parallel by 30.5% |
| Geomean | 1.899 | 2.130 | **2.325** | parallel by 9.2% |

### D=256

| Graph | Fused-global / legacy serial | Fused-smem / legacy serial | Best parallel / legacy serial | Faster final kernel |
|---|---:|---:|---:|---|
| cit-Patents | 1.354 | **1.452** | 1.320 | fused-smem by 10.0% |
| LiveJournal | 1.482 | **1.751** | 1.709 | fused-smem by 2.5% |
| indochina | 2.978 | 3.861 | **5.188** | parallel by 34.4% |
| Orkut | 1.430 | 1.568 | **1.640** | parallel by 4.6% |
| Geomean | 1.710 | 1.981 | **2.093** | parallel by 5.7% |

Shared row caching adds 1.122x over fused-global at D=128 and 1.159x at
D=256. The fused-smem kernel uses 40 registers/thread and 2 KiB shared memory
per 256-thread block for both D values; no local-memory spill was generated.
All fused-suite case/variant fingerprints matched the reference.

This correction changes the main conclusion: parallel is not universally
superior. Fused serial wins on cit-Patents and LiveJournal, while parallel wins
on indochina, Orkut, and Twitter. The meaningful adaptive decision is therefore
between two kernel families—`fused-serial-smem-Q32` and
`parallel-smem-shuffle-Q16`—rather than the earlier Cartesian product.

### Fused-serial Q-width ablation

Q16 and Q8 split the warp into two or four disjoint edge groups. Each lane
holds `D/Q` query accumulators, and group partials are combined with shuffle
reduction after the single sparse-row traversal.

Smem results relative to the legacy serial-Q32 baseline:

| D=128 | cit-Patents | LiveJournal | indochina | Orkut | Twitter | Geomean |
|---|---:|---:|---:|---:|---:|---:|
| fused-smem-Q32 | **1.453** | **1.745** | **3.839** | **1.597** | **2.803** | **2.127** |
| fused-smem-Q16 | 1.381 | 1.697 | 3.765 | 1.567 | 2.798 | 2.077 |
| fused-smem-Q8 | 1.207 | 1.438 | 3.162 | 0.692 | 2.237 | 1.534 |

| D=256 | cit-Patents | LiveJournal | indochina | Orkut | Geomean |
|---|---:|---:|---:|---:|---:|
| fused-smem-Q32 | **1.453** | **1.752** | **3.862** | **1.569** | **1.982** |
| fused-smem-Q16 | 1.269 | 1.510 | 3.057 | 1.427 | 1.700 |
| fused-smem-Q8 | 1.074 | 1.216 | 2.750 | 1.258 | 1.458 |

Q32 wins every fused-smem case. At D=128, Q16 is close on Twitter (within
0.2%) but does not exceed Q32. The global-memory geomeans are respectively
1.898/1.427/1.480 for Q32/Q16/Q8 at D=128 and 1.706/1.697/1.342 at D=256.
Thus smaller Q can occasionally help a global kernel, but explicit row caching
removes its main motivation while retaining its shuffle and coalescing costs.

Register use explains part of the widening gap at D=256: fused-smem Q32/Q16/Q8
use 40/48/64 registers per thread. D=128 uses 40/40/48. None spills to local
memory, but Q8 reduces achievable occupancy and performs two shuffle-reduction
stages for 32 accumulators per lane. Across the nine cases and eight reported
variants, all 72 full-output fingerprints matched.

## D=128 results

| Variant | cit-Patents | LiveJournal | indochina | Orkut | Twitter | Geomean |
|---|---:|---:|---:|---:|---:|---:|
| serial-shared-Q16 | 0.988 | 0.873 | 1.831 | 0.945 | 1.659 | 1.199 |
| serial-shared-Q8 | 0.805 | 0.680 | 1.495 | 0.715 | 1.328 | 0.951 |
| serial-shuffle-Q16 | 1.099 | 1.250 | 2.277 | 1.184 | 2.017 | **1.495** |
| serial-shuffle-Q8 | 0.944 | 1.057 | 2.298 | 0.940 | 1.742 | 1.303 |
| serial-smem-Q32 | 1.058 | 1.068 | 1.577 | 1.114 | 1.565 | **1.254** |
| serial-smem-shuffle-Q16 | 0.913 | 1.084 | 1.908 | 1.002 | 1.622 | 1.251 |
| serial-smem-shuffle-Q8 | 0.858 | 0.923 | 1.937 | 1.034 | 1.897 | 1.246 |
| parallel-global-Q32 / serial-Q32 | 1.335 | 1.603 | 3.651 | 1.567 | 3.225 | **2.086** |
| parallel-shared-Q16 | 0.924 | 0.856 | 0.918 | 0.944 | 0.997 | 0.927 |
| parallel-shared-Q8 | 0.879 | 0.834 | 0.985 | 0.929 | 0.982 | 0.920 |
| parallel-shuffle-Q16 | 0.903 | 0.780 | 0.701 | 0.841 | 0.787 | 0.800 |
| parallel-shuffle-Q8 | 0.911 | 0.914 | 1.047 | 0.969 | 1.029 | 0.973 |
| parallel-smem-Q32 | 0.998 | 0.993 | 0.970 | 1.004 | 0.994 | 0.992 |
| parallel-smem-shuffle-Q16 | 1.009 | 1.068 | 1.332 | 1.052 | 1.134 | **1.114** |
| parallel-smem-shuffle-Q8 | 0.752 | 0.787 | 0.765 | 0.880 | 0.866 | 0.808 |

The best parallel cached-Q16 kernel has a 2.323x geometric-mean speedup over
serial-global-Q32 after composing the 2.086x tile-parallel gain and its 1.114x
gain over parallel-global-Q32.

## D=256 results

| Variant | cit-Patents | LiveJournal | indochina | Orkut | Geomean |
|---|---:|---:|---:|---:|---:|
| serial-shared-Q16 | 0.984 | 0.880 | 1.765 | 0.943 | 1.096 |
| serial-shared-Q8 | 0.796 | 0.677 | 1.433 | 0.707 | 0.859 |
| serial-shuffle-Q16 | 1.084 | 1.220 | 2.170 | 1.151 | **1.348** |
| serial-shuffle-Q8 | 0.929 | 1.014 | 2.147 | 0.902 | 1.162 |
| serial-smem-Q32 | 1.051 | 1.057 | 1.572 | 1.103 | **1.178** |
| serial-smem-shuffle-Q16 | 0.902 | 1.069 | 1.887 | 0.975 | 1.154 |
| serial-smem-shuffle-Q8 | 0.860 | 0.930 | 1.934 | 1.028 | 1.123 |
| parallel-global-Q32 / serial-Q32 | 1.331 | 1.638 | 4.298 | 1.581 | **1.962** |
| parallel-shared-Q16 | 0.932 | 0.921 | 0.946 | 0.968 | 0.942 |
| parallel-shared-Q8 | 0.891 | 0.895 | 0.877 | 0.967 | 0.907 |
| parallel-shuffle-Q16 | 0.910 | 0.814 | 0.719 | 0.867 | 0.824 |
| parallel-shuffle-Q8 | 0.892 | 0.904 | 0.903 | 0.965 | 0.915 |
| parallel-smem-Q32 | 0.992 | 0.983 | 1.018 | 0.994 | 0.996 |
| parallel-smem-shuffle-Q16 | 0.992 | 1.043 | 1.239 | 1.037 | **1.074** |
| parallel-smem-shuffle-Q8 | 0.775 | 0.802 | 0.761 | 0.889 | 0.805 |

The best parallel cached-Q16 kernel has a 2.106x geometric-mean speedup over
serial-global-Q32 after composing the 1.962x tile-parallel gain and its 1.074x
gain over parallel-global-Q32.

All 135 reported variant/case fingerprints matched the serial-Q32 reference.

## Conclusions

1. Relative to the legacy tile-loop baseline, query-tile parallelism reaches
   2.086x at D=128 and 1.962x at D=256, but fused serial closes nearly all of
   that gap and wins on two graphs.
2. Shuffle-Q16 is the best serial variant, reaching 1.495x and 1.348x. Removing
   a block-wide barrier becomes increasingly important when one warp repeats
   many query tiles.
3. Serial smem-Q32 independently provides 1.254x and 1.178x. The cache is
   rebuilt for each query tile, so this is specifically the benefit of
   cooperative metadata loading within a tile under the wider dense-value
   stride, not cross-tile cache persistence.
4. Parallel smem-Q32 alone is neutral because parallel Q32 already exposes
   enough work and same-address graph loads are efficiently served. However,
   cross-warp row caching plus Q16 is consistently useful on the larger graphs:
   up to 1.332x at D=128 and 1.239x at D=256 on indochina.
5. Shuffle alone is harmful in the parallel mapping. The useful design is the
   combination of cross-warp row sharing and Q16's two-query-per-thread reuse;
   Q8 incurs too much reduction and loses query-memory coalescing efficiency.

## Reproduction

```bash
cmake --build build -j 8 --target graphweft_pull_wide_smem_shuffle
./build/graphweft_pull_wide_smem_shuffle /home/zyl/data/csr_data/indochina 128 0 7
./build/graphweft_pull_wide_smem_shuffle /home/zyl/data/csr_data/indochina 256 0 7
./build/graphweft_pull_wide_smem_shuffle /home/zyl/data/csr_data/indochina 128 0 7 fused
```
