# External Baseline Results

Values are `median ms / queries per second`; graph loading and reusable preprocessing are excluded. PuerCGP online-planner time is included where applicable. `[cold-1]` marks one measured cold run with no warmup; `[warm-1]` marks one measured run after warmup.

## BFS

| Dataset | Puer push+online | Puer hybrid+online | Puer pull | iBFS | Glign | ForkGraph | Gunrock s1 | Gunrock s64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cit-Patents | 1113.58 / 229.89 | 967.87 / 264.50 | 802.66 / 318.94 | 298.36 / 858.02 | 23641.40 / 10.83 | 72865.00 / 3.51 | 8983.57 / 28.50 | 4780.48 / 53.55 |
| soc-orkut | 1511.57 / 169.36 | 1679.54 / 152.42 | 1730.06 / 147.97 | 1030.02 / 248.54 | 47287.50 / 5.41 | 74200.00 / 3.45 | 15822.09 / 16.18 | OOM |
| soc-twitter | 3459.57 / 74.00 | 10268.30 / 24.93 | 11692.00 / 21.90 | 3463.27 / 73.92 | 349660.00 / 0.73 | 814854.00 / 0.31 [warm-1] | 45178.31 / 5.67 | OOM |
| soc-sinaweibo | 6219.76 / 41.16 | 5704.68 / 44.88 | 5672.51 / 45.13 | 5180.71 / 49.41 | 653204.00 / 0.39 | 1162425.00 / 0.22 [cold-1] | 64222.60 / 3.99 | OOM |

## SSSP

| Dataset | Puer push+online | Puer hybrid+online | Puer pull | Glign | ForkGraph | Gunrock seq |
|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | 3644.78 / 70.24 | 1448.76 / 176.70 | 1776.95 / 144.07 | 69724.90 / 3.67 | 443608.00 / 0.58 | 14932.16 / 17.14 |
| soc-orkut | 14641.20 / 17.48 | 3250.52 / 78.76 | 4549.76 / 56.27 | 155277.00 / 1.65 | 399121.00 / 0.64 | 38332.77 / 6.68 |
| soc-twitter | 20235.90 / 12.65 | 8085.19 / 31.66 | 19187.30 / 13.34 | 1012740.00 / 0.25 | 1053585.00 / 0.24 [cold-1] | 83004.49 / 3.08 |
| soc-sinaweibo | 37641.70 / 6.80 | 8383.39 / 30.54 | 10998.90 / 23.28 | 2100960.00 / 0.12 | OOM [cold-1] | 185911.94 / 1.38 |

## SSWP

| Dataset | Puer push+online | Puer hybrid+online | Puer pull | Glign |
|---|---:|---:|---:|---:|
| cit-Patents | 2651.16 / 96.56 | 1527.30 / 167.62 | 3145.82 / 81.38 | 79001.20 / 3.24 |
| soc-orkut | 10679.10 / 23.97 | 3346.51 / 76.50 | 5354.99 / 47.81 | 130333.00 / 1.96 |
| soc-twitter | 14373.90 / 17.81 | 6227.48 / 41.11 | 21495.90 / 11.91 | 602816.00 / 0.42 |
| soc-sinaweibo | 20962.70 / 12.21 | 7210.69 / 35.50 | 10721.80 / 23.88 | OOM |

## PAGERANK

| Dataset | Puer push | Puer hybrid | Puer pull | Gunrock seq |
|---|---:|---:|---:|---:|
| cit-Patents | 8620.04 / 29.70 | 2601.91 / 98.39 | 1948.42 / 131.39 | 21959.87 / 11.66 |
| soc-orkut | 82689.70 / 3.10 | 16218.60 / 15.78 | 8843.51 / 28.95 | 49905.63 / 5.13 |
| soc-twitter | 107328.00 / 2.39 | 51542.00 / 4.97 | 45254.60 / 5.66 | 153617.55 / 1.67 |
| soc-sinaweibo | 100977.00 / 2.54 | 40170.00 / 6.37 | 33506.90 / 7.64 | 378966.41 / 0.68 |

## PPR

| Dataset | Puer push | Puer hybrid | Puer pull |
|---|---:|---:|---:|
| cit-Patents | 8623.29 / 29.69 | 2445.76 / 104.67 | 1777.58 / 144.02 |
| soc-orkut | 82714.30 / 3.09 | 15226.00 / 16.81 | 7812.86 / 32.77 |
| soc-twitter | 107214.00 / 2.39 | 54284.80 / 4.72 | 48737.60 / 5.25 |
| soc-sinaweibo | 100677.00 / 2.54 | 35885.30 / 7.13 | 28998.90 / 8.83 |

## Best-Puer Speedup

| Baseline | Geomean | Cases |
|---|---:|---:|
| ForkGraph | 139.993x | 7 |
| Glign | 68.302x | 11 |
| Gunrock seq | 9.522x | 8 |
| Gunrock s1 | 11.472x | 4 |
| Gunrock s64 | 5.956x | 1 |
| iBFS | 0.694x | 4 |

## Validity Notes

- Weighted cross-system checks: `complete` (20 signatures).
- Corrected ForkGraph SinaWeibo SSSP single-query signature: `pass`.
- Gunrock `streams64` OOM failures remain failures under strict Q=64; concurrency is not reduced.
- ForkGraph uses 48 bound physical cores and 35.75 MiB edge-balanced range partitions; old P=8/OMP=96 diagnostics are excluded.
- ForkGraph cit-Patents and soc-Orkut use 2 warmups plus 5 repeats; Twitter BFS uses 1 warmup plus 1 measured run; Twitter SSSP and SinaWeibo BFS/SSSP use one cold measured run by request.
- ForkGraph SinaWeibo SSSP was killed by SIGKILL after at least 131 GB observed RSS and is reported as probable host OOM without reducing Q=64.
- ForkGraph timing on this shared host is provisional and should be repeated on an exclusive node for publication.
- Aggregate speedups that include `[cold-1]` or `[warm-1]` cases are preliminary because their sampling policy differs.
- PuerCGP PageRank/PPR use dense fixed 10-iteration execution; the frontier online planner is not applicable.
- Native ForkGraph/Gunrock PPR is residual PR-Nibble, not the same fixed-iteration dense algorithm, and is excluded from the primary table.
