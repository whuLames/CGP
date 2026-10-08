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
- Per-round coverage on every graph: the Iteration model's selected Push partition and
  `pull-dense-parallel-smem-q32` are measured from identical input state, regardless of
  the direction chosen by production Hybrid.
- Full Oracle set on representative graphs only: 30 static Push kernels, SharedPush,
  AdaptivePush, and 21 Dense Pull kernels (53 total). The representatives are
  `soc-orkut`, `uk-2002`, `graph500-scale23-ef16-adj`, and `delaunay-n24`.
- Baselines: threshold Hybrid with `push_mapping=adaptive` and with `push_mapping=iteration`; Pull remains `auto`.
- Refill policies: `none` and `eager_global`. `eager_group` is excluded from subsequent campaigns.
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

## Streaming handoff

Dataset preparation and V100 execution are asynchronous per dataset; there is no all-datasets barrier. The preparation host transfers `datasets/NAME`, `workloads/NAME`, and `status/NAME.json` completely, then runs `publish_ready_dataset.py` on the destination as the final atomic step. This writes a receipt under `inbox/gpuN/ready/` only after hashing every required CSR, workload, status, and campaign-manifest file.

One `watch_ready_campaign.py` process per GPU claims receipts with an atomic rename, revalidates size and SHA-256, and launches the fixed campaign immediately. `run_adaptive_remote_pipeline.py` also holds `pipeline/gpuN.lock` for its entire lifetime, so manual and READY-triggered work cannot overlap on the same GPU. Receipts move to `done/` or `failed/`; a partially transferred dataset has no READY receipt and is invisible to compute workers.

As of the initial 2026-10-07 launch, `tech-ip`, `delicious-ti`, and `socfb-A-anon` passed conversion on `v100-a`; `soc-flickr-growth`, `soc-livejournal`, and `socfb-B-anon` passed on `v100-b`. Preparation continues sequentially per host before the GPU campaign is resumed so conversion traffic cannot contaminate formal measurements.

`soc-sinaweibo` was imported directly from the pre-existing local symmetric CSR into `v100-b`; no transfer archive was created. Its sparse ID space expands the CSR row count to 58,655,849 despite the source page reporting 21M nodes. The measured allocation estimates are 62.95 GiB at M=128 and 120.63 GiB at M=256, so it is retained under `datasets/` but marked campaign-ineligible until IDs are compacted.

## Accelerated build queue (2026-10-07)

- All four V100 workers retain a historical-graph fallback queue, but newly prepared large graphs use a numeric READY prefix and run first.
- `graph500-scale23-ef16-adj` is preparing both M=128 and M=256 workloads on the data host using the V100 target memory budget; it is assigned to `v100-a / 0`.
- `uk-2002` was rebuilt from all 51 archive fragments: V=18,520,486, raw E=298,113,762, symmetric CSR E=529,444,615. It fits M=128 only and is queued at high priority on `v100-a / 1`.
- The data host continuously processes `rgg-n-2-24-s0`, `delaunay-n24`, and `nlpkkt160`. Converted CSR bundles are streamed immediately to their assigned V100; workloads that exceed the RTX 3060 preparation budget are generated on the destination V100 before measurement.
- Relay workers and dataset construction run in background processes. A completed conversion does not wait for the remainder of the build queue before transfer or execution.
