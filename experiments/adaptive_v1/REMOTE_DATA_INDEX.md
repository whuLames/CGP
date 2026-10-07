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

The machine-readable candidate list is [`dataset_catalog.json`](dataset_catalog.json). A catalog entry is not download authorization. Before any new transfer, its `screening` object must record the original SNAP or Network Repository page, check time, webpage-reported V/E, estimated post-symmetry E, and an explicit `approved` decision. The downloader rejects unscreened entries.

Primary graphs require estimated post-symmetry `E >= 100M` and must fit at least one of `M=128` or `M=256` under the 80% V100-32GB gate. Download, extraction, conversion, exact post-dedup size validation, and workload generation happen only after that inexpensive metadata check. Smaller graphs already present are controls and do not count toward the primary 12-real-graph target. The structural controls cover power-law, random-geometric, planar, and numerical-matrix structure. RGG and Delaunay are geometric controls, not road-network datasets.

The seven pre-existing graphs (`cit-Patents`, `indochina`, `roadNet-CA`, `roadNet-TX`, `soc-LiveJournal1`, `soc-orkut`, `soc-twitter`) are historical anchors and do not count toward the new-dataset target.

## Status

The campaign is sharded as follows. These assignments are stable across resume; a failed or memory-gated graph is recorded and skipped rather than silently moved to another host.

| Host / GPU | Dataset queue |
|---|---|
| `v100-a / 0` | `tech-ip`, `delicious-ti`, `wb-edu`, `wikipedia-link-de`, `graph500-scale23-ef16-adj` |
| `v100-a / 1` | `socfb-A-anon`, `ljournal-2008`, `uk-2002`, `wikipedia-link-it`, `rgg-n-2-24-s0` |
| `v100-b / 0` | `soc-flickr-growth`, `soc-livejournal`, `soc-sinaweibo`, `wikipedia-link-fr`, `delaunay-n24` |
| `v100-b / 1` | `socfb-B-anon`, `livejournal-heter`, `wikipedia-growth`, `web-wikipedia-link-en13-all`, `nlpkkt160` |

Per-dataset preparation state and canonical hashes are in `/data/graphweft-adaptive-v1/status/`. Per-GPU campaign state is in `pipeline/gpu0.json` and `pipeline/gpu1.json`; detailed logs are in `pipeline/`. Generated graphs and workload bundles are in `datasets/` and `workloads/`, while raw and analyzed timing data remain in `results/` on the owning host. Temporary download archives and extracted edge lists are deleted after each successful conversion.

As of the initial 2026-10-07 launch, `tech-ip`, `delicious-ti`, and `socfb-A-anon` passed conversion on `v100-a`; `soc-flickr-growth`, `soc-livejournal`, and `socfb-B-anon` passed on `v100-b`. Preparation continues sequentially per host before the GPU campaign is resumed so conversion traffic cannot contaminate formal measurements.

`soc-sinaweibo` was imported directly from the pre-existing local symmetric CSR into `v100-b`; no transfer archive was created. Its sparse ID space expands the CSR row count to 58,655,849 despite the source page reporting 21M nodes. The measured allocation estimates are 62.95 GiB at M=128 and 120.63 GiB at M=256, so it is retained under `datasets/` but marked campaign-ineligible until IDs are compacted.
