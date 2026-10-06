# Pull Q-group shuffle and row-cache ablation

## Question

For `D=32` and one warp per destination vertex, this experiment separates two
effects:

1. replace the existing Q16/Q8 shared-memory partial reduction and block
   barrier with warp-shuffle reduction;
2. cache 32 incoming `(source, weight)` pairs per warp in shared memory, using
   GE-SpMM's coalesced-row-caching pattern, before query lanes consume them.

The experiment is standalone and does not change GraphWeft's production pull
path. The implementation is `pull_smem_shuffle.cu`.

## Setup

- GPU: Tesla V100-SXM2-32GB
- Mapping: one warp per destination vertex, `D=32`, eight warps per block
- Operation: weighted min-plus pull
- Timing: five paired warmups, 21 paired measurements, median GPU event time
- Baseline: current global-memory Q32 kernel
- Correctness: device-side exact comparison of all `V * 32` output values
- Graphs: the five non-road graphs in local legacy CSR format

Each row reports `baseline_ms / candidate_ms`; values above one are faster than
global Q32.

## Results

| Variant | cit-Patents | LiveJournal | indochina | Orkut | Twitter | Geomean |
|---|---:|---:|---:|---:|---:|---:|
| global + shared reduction, Q16 | 0.953 | 0.847 | 1.086 | 0.937 | 0.972 | 0.956 |
| global + shuffle reduction, Q16 | 1.001 | 1.002 | 1.135 | 0.998 | 0.957 | 1.017 |
| global + shared reduction, Q8 | 0.825 | 0.916 | 1.248 | 0.977 | 1.181 | 1.017 |
| global + shuffle reduction, Q8 | 0.781 | 0.886 | 0.999 | 0.835 | 0.923 | 0.882 |
| shared row cache, Q32 | 0.961 | 0.949 | 1.008 | 0.980 | 1.058 | 0.991 |
| shared row cache + shuffle, Q16 | 0.725 | 0.693 | 0.657 | 0.710 | 0.717 | 0.700 |
| shared row cache + shuffle, Q8 | 0.857 | 0.972 | 1.364 | 0.937 | 1.307 | 1.068 |

All 35 candidate/graph combinations passed exact comparison.

## Interpretation

- Q16's old overhead is largely reduction-related on four graphs. Shuffle
  removes the shared partial array and block-wide barrier, raising its geomean
  from 0.956x to 1.017x and producing 1.135x on indochina.
- Q8 does not benefit from a direct shuffle substitution. It requires four
  query accumulators per lane and performs two shuffle/fmin stages for every
  accumulator in all 32 lanes. The old shared reduction lets only the first
  eight lanes perform the final reduction. Its 0.882x geomean shows that
  "shuffle is cheaper" is not generally true for this decomposition.
- GE-SpMM-style row caching alone is neutral on average for Q32 (0.991x). The
  baseline's warp lanes request the same edge metadata together, so hardware
  already coalesces/broadcasts much of this traffic; explicit caching adds two
  warp synchronizations and a shared-memory round trip per 32-edge tile.
- Row caching and Q8 are complementary on high-degree/skewed graphs:
  indochina reaches 1.364x and Twitter 1.307x. Q8 lets each lane use one cached
  edge for four queries, while cooperative loading ensures each edge pair is
  fetched once per tile. It regresses on cit-Patents and Orkut, so it should be
  selected adaptively rather than replace Q32 globally.
- The cached Q16 combination is consistently poor and should not enter the
  production candidate set in its current form.

CUDA resource inspection reports 32 registers/thread for global Q32,
global-shuffle Q16/Q8, cached Q32, and cached Q16. Cached Q8 uses 40 registers
per thread. The cached kernels use 2 KiB shared memory per block; therefore the
large cached-Q8 wins are not explained by a hidden occupancy advantage.

## Reproduction

```bash
cmake --build build -j 8 --target graphweft_pull_smem_shuffle
./build/graphweft_pull_smem_shuffle /home/zyl/data/csr_data/indochina 0 21
```

The shared-memory tiling follows the
[official GE-SpMM implementation](https://github.com/hgyhungry/ge-spmm) and its
[coalesced-row-caching description](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/publications/2020/arxiv_318.pdf):
a warp loads a 32-entry sparse-row tile into shared memory and then its lanes
consume that tile.
