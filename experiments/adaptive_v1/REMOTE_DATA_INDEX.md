# Adaptive selector campaign data index

## Remote layout

| Alias | GPUs | Code | Campaign data |
|---|---:|---|---|
| `v100-a` | 2 × V100-SXM2-32GB | `/data/coding/GraphWeft` | `/data/graphweft-adaptive-v1` |
| `v100-b` | 2 × V100-SXM2-32GB | `/data/coding/GraphWeft` | `/data/graphweft-adaptive-v1` |

Credentials are deliberately not stored in this repository. Datasets are sharded rather than copied to both hosts. Each GPU runs one independent worker.

## Fixed experiment contract

- Branch: `experiment/dense-pull-integration`
- Queries: `N=1024`; `M=Q=128`, plus `M=Q=256` when the exact V100 allocation gate passes; `G=32`.
- Measurements: one complete warmup and two formal repetitions; the second Oracle repetition reverses candidate order.
- Oracle set: 30 static Push kernels, SharedPush, AdaptivePush, and 21 Dense Pull kernels (53 total).
- Baselines: threshold Hybrid with `push_mapping=adaptive` and with `push_mapping=iteration`; Pull remains `auto`.
- Storage safety: do not start an operation that would leave less than 30 GiB; pause at a safe output boundary below 25 GiB.

## Dataset policy

The machine-readable candidate list is [`dataset_catalog.json`](dataset_catalog.json). It is intentionally broader than a frozen shortlist. Real graphs are admitted first after conversion, canonical-hash deduplication, and the `M=128` memory gate. The four structural controls cover power-law, random-geometric, planar, and numerical-matrix structure. RGG and Delaunay are geometric controls, not road-network datasets.

The seven pre-existing graphs (`cit-Patents`, `indochina`, `roadNet-CA`, `roadNet-TX`, `soc-LiveJournal1`, `soc-orkut`, `soc-twitter`) are historical anchors and do not count toward the new-dataset target.

## Status

Dataset placement, canonical hashes, workload hashes, run IDs, and completion state are filled from the two remote `status/` trees after preparation. Large raw timing tables remain under `/data` on the owning server.
