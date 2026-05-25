# CGP BFS Hybrid Compare

Device: `CUDA_VISIBLE_DEVICES=2`. Median of 3 measured runs after 1 warmup. V2.2 values are from `build/cgp_bfs_optimization_results/v22_compare/summary.csv`.

| graph | q | push ms | hybrid ms | speedup vs push | V2.2 total ms | speedup vs V2.2 total | V2.2 batch ms | speedup vs V2.2 batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| soc-orkut | 2 | 89.436 | 8.741 | 10.232 | 116.763 | 13.358 | 114.957 | 13.151 |
| soc-orkut | 4 | 227.390 | 36.046 | 6.308 | 217.115 | 6.023 | 213.880 | 5.934 |
| soc-orkut | 8 | 526.094 | 95.199 | 5.526 | 407.534 | 4.281 | 402.749 | 4.231 |
| soc-orkut | 16 | 1143.424 | 310.275 | 3.685 | 800.693 | 2.581 | 789.279 | 2.544 |
| soc-twitter | 2 | 198.796 | 44.701 | 4.447 | 343.894 | 7.693 | 341.945 | 7.650 |
| soc-twitter | 4 | 425.688 | 136.563 | 3.117 | 651.855 | 4.773 | 644.757 | 4.721 |
| soc-twitter | 8 | 931.917 | 331.529 | 2.811 | 1287.616 | 3.884 | 1278.828 | 3.857 |
| soc-sinaweibo | 2 | 408.375 | 138.807 | 2.942 | 502.343 | 3.619 | 499.705 | 3.600 |
| soc-sinaweibo | 4 | 877.559 | 254.874 | 3.443 | 957.799 | 3.758 | 952.467 | 3.737 |

All tested hybrid runs improved over push and over the existing GunrockV2.2 total-wall baseline for the matching graph/query pairs.
