# BFS IPC/DRAM Comparison Summary

## L3 per-config comparison (median over repeats)

| Dataset | N | Framework | total DRAM (GB) | total time (ms) | IPC (median) | hot-iter IPC | BP_peak |
|---|---|---|---|---|---|---|---|
| soc-orkut | 2 | gunrock | 24.61 | 91.08 | 0.1118 | 0.1000 | 0.302 |
| soc-orkut | 2 | puercgp | 25.98 | 121.70 | 0.0108 | 0.0100 | 0.2371 |
| soc-orkut | 4 | gunrock | 47.66 | 174.76 | 0.1139 | 0.0950 | 0.3024 |
| soc-orkut | 4 | puercgp | 37.28 | 180.91 | 0.0102 | 0.0100 | 0.2289 |
| soc-orkut | 8 | gunrock | 92.66 | 331.44 | 0.1152 | 0.0900 | 0.3041 |
| soc-orkut | 8 | puercgp | 45.26 | 236.40 | 0.0101 | 0.0100 | 0.2127 |
| soc-orkut | 16 | gunrock | 183.96 | 648.48 | 0.1169 | 0.1000 | 0.3053 |
| soc-orkut | 16 | puercgp | 55.15 | 307.72 | 0.0102 | 0.0100 | 0.1992 |
| soc-orkut | 32 | gunrock | 368.22 | 1310.08 | 0.119 | 0.1000 | 0.3044 |
| soc-orkut | 32 | puercgp | 61.31 | 371.31 | 0.0102 | 0.0100 | 0.1834 |
| soc-sinaweibo | 2 | gunrock | 86.69 | 422.50 | 0.0597 | 0.0350 | 0.2273 |
| soc-sinaweibo | 2 | puercgp | 100.41 | 807.53 | 0.0164 | 0.0200 | 0.1382 |
| soc-sinaweibo | 4 | gunrock | 173.02 | 847.76 | 0.0602 | 0.0400 | 0.2272 |
| soc-sinaweibo | 4 | puercgp | 111.11 | 916.01 | 0.0199 | 0.0100 | 0.1348 |
| soc-sinaweibo | 8 | gunrock | 345.78 | 1692.48 | 0.0604 | 0.0400 | 0.2271 |
| soc-sinaweibo | 8 | puercgp | 126.69 | 1036.17 | 0.018 | 0.0100 | 0.136 |
| soc-sinaweibo | 16 | gunrock | 690.54 | 3385.44 | 0.0605 | 0.0400 | 0.2272 |
| soc-sinaweibo | 16 | puercgp | 143.96 | 1167.24 | 0.0218 | 0.0200 | 0.137 |
| soc-sinaweibo | 32 | gunrock | 1382.65 | 6782.40 | 0.0602 | 0.0400 | 0.2268 |
| soc-sinaweibo | 32 | puercgp | 177.71 | 1423.98 | 0.0232 | 0.0200 | 0.1387 |
| soc-twitter | 2 | gunrock | 51.83 | 289.70 | 0.1269 | 0.1150 | 0.1988 |
| soc-twitter | 2 | puercgp | 43.09 | 317.51 | 0.0215 | 0.0100 | 0.1508 |
| soc-twitter | 4 | gunrock | 103.35 | 584.80 | 0.1224 | 0.1150 | 0.1971 |
| soc-twitter | 4 | puercgp | 83.28 | 593.44 | 0.0222 | 0.0200 | 0.1559 |
| soc-twitter | 8 | gunrock | 207.31 | 1170.32 | 0.1217 | 0.1100 | 0.1962 |
| soc-twitter | 8 | puercgp | 113.94 | 823.10 | 0.0233 | 0.0200 | 0.1538 |
| soc-twitter | 16 | gunrock | 415.90 | 2200.48 | 0.1202 | 0.1100 | 0.2094 |
| soc-twitter | 16 | puercgp | 123.70 | 955.45 | 0.0274 | 0.0200 | 0.1439 |
| soc-twitter | 32 | gunrock | 833.13 | 4525.44 | 0.1214 | 0.1100 | 0.2043 |
| soc-twitter | 32 | puercgp | 167.82 | 1283.22 | 0.0307 | 0.0200 | 0.1453 |

## Notes

- gunrock total DRAM = (per-query median DRAM) × N (serial N queries)
- puercgp total DRAM = per-config median (already N-query shared)
- IPC is SMSP-level (V100 upper bound ≈ 1.0)
- hot-iter IPC = median over repeats of the longest iteration's IPC