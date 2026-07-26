# BFS Combined Summary with puercgp Hybrid GE-SpMM

Median algorithm time in milliseconds. puercgp uses `gpu_ms_median` from `validate_bfs`; `ok_runs/runs` is noted when not all three runs succeeded.

puercgp configuration: `traversal_mode=hybrid`, `push_strategy=shared_node_warp`, `pull_strategy=ge_spmm`.

## cit-Patents

| concurrency | ForkGraph ms | Glign ms | iBFS ms | puercgp hybrid GE-SpMM ms | notes |
|---:|---:|---:|---:|---:|---|
| 2 | 12643.360 | 1279.020 | 33.456 | 26.918 |  |
| 4 | 11399.090 | 2126.700 | 43.421 | 41.921 |  |
| 8 | 13978.930 | 2215.170 | 56.947 | 47.562 |  |
| 16 | 20223.745 | 2023.580 | 61.716 | 106.393 |  |
| 32 | 30089.350 | 3560.610 | 66.950 | 133.115 |  |
| 64 | 29312.353 | 5017.740 | 75.617 | 214.578 |  |

## soc-orkut

| concurrency | ForkGraph ms | Glign ms | iBFS ms | puercgp hybrid GE-SpMM ms | notes |
|---:|---:|---:|---:|---:|---|
| 2 | 5137.230 | 1716.220 | 155.023 | 45.165 |  |
| 4 | 6876.152 | 2392.960 | 188.988 | 110.815 |  |
| 8 | 9279.364 | 2989.260 | 220.917 | 87.094 |  |
| 16 | 12809.874 | 4436.100 | 222.659 | 99.045 |  |
| 32 | 16743.481 | 6322.270 | 241.240 | 127.906 |  |
| 64 | 21517.534 | 13813.200 | 252.086 | 276.831 |  |

## soc-sinaweibo

| concurrency | ForkGraph ms | Glign ms | iBFS ms | puercgp hybrid GE-SpMM ms | notes |
|---:|---:|---:|---:|---:|---|
| 2 | 45560.818 | 12770.900 | 746.684 | 209.274 |  |
| 4 | 69769.424 | 16424.400 | 806.075 | 235.690 |  |
| 8 | 87764.526 | 19167.300 | 304.894 | 286.768 |  |
| 16 | 578231.990 | 250025.000 | 856.864 | 354.333 |  |
| 32 | 158052.892 | 81643.800 | 909.144 | 468.547 |  |
| 64 | 1206285.227 | 185598.000 | 1269.020 | 1018.300 | forkgraph 2/3 ok |

## soc-twitter

| concurrency | ForkGraph ms | Glign ms | iBFS ms | puercgp hybrid GE-SpMM ms | notes |
|---:|---:|---:|---:|---:|---|
| 2 | 218198.801 | 13745.900 | 496.755 | 462.088 |  |
| 4 | 417579.312 | 10181.700 | 620.253 | 451.894 |  |
| 8 | 514627.631 | 8497.120 | 647.742 | 491.842 |  |
| 16 | 571067.158 | 24038.900 | 785.967 | 595.037 |  |
| 32 | 80561.799 | 50829.000 | 753.489 | 669.692 |  |
| 64 | 100981.960 | 90515.900 | 824.595 | 2156.820 |  |
