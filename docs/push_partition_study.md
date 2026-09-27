# Push partition study

The study treats every iteration of a fixed-Push execution as a separate workload. It compares 30 compiled variants with the original `shared_push`, using the same old values, source frontier, query mask, live slots and float32 atomic relaxation. Production execution always advances with the original Push. The diagnostic run's wall time includes extensive replay and validation and is not an end-to-end benchmark.

## Kernel mapping

`group_size` is one of 1, 2, 4, 8, 16, 32; `edge_groups = 32 / group_size`. Inside a warp, `lane / group_size` identifies the edge group and `lane % group_size` identifies its query lane. Each warp compacts active query IDs in ascending order within each 32-column tile. A group visits query ranks `query_lane + k * group_size`; inactive queries never relax state. A group leader loads an edge's destination and weight and broadcasts these within the group.

All new variants use this same enumeration. The original Push instead enumerates mask bits inside each edge thread. `q1_w1` is therefore the controlled mapping baseline, while `legacy_push` preserves the old implementation, including its V-sized launch and four vertices per block.

| Grain | Warps per block | Blocks per vertex | Total warps per vertex |
| --- | ---: | ---: | ---: |
| w1 | 1 | 1 | 1 |
| w2 | 2 | 1 | 2 |
| w4 | 4 | 1 | 4 |
| b2 | 4 | 2 | 8 |
| b4 | 4 | 4 | 16 |

For block part `b`, warp `w` and edge group `g`, an edge starts at `row[v] + (b * warps_per_block + w) * edge_groups + g`, then advances by `blocks_per_vertex * warps_per_block * edge_groups`. This covers each edge exactly once. All grains apply to every active vertex; there is no degree-based fallback. The w1/w2/w4 variants also change the block size (32/64/128 threads), so their comparison includes the scheduling and occupancy effects of that choice. It does not isolate edge splitting from block-size effects. Block grids are capped at 65532 and iterate over virtual tasks. The cap is divisible by four so a vertex's block parts remain on separate blocks.

Variant IDs are supplied by `push_partition_id(index)`; `push_partition(index)` exposes their mapping. The six group sizes are ordered ascending, with w1/w2/w4/b2/b4 inside each size. `launch` is shared by production and laboratory targets. `Context::frontier_size_hint` optionally sizes launches from a known frontier count; the device count remains authoritative. When a production replay selects a partition variant, the executor reads the frontier count for this hint and charges that transfer/synchronization to feature time. These variants are counted as Push rounds.

## Selecting a fixed candidate through the API

```cpp
// group_size=16 is index 4; w2 is grain index 1.
graphweft::Options options;
options.algorithm = graphweft::Algorithm::SSSP;
options.capacity = 32;
options.selector = graphweft::Options::Selector::Replay;
options.replay = {graphweft::push_partition_id(4 * 5 + 1)};
auto stats = graphweft::run(graph, queries, options);
```

A one-entry replay repeats that candidate every round. Multi-entry replay cycles through the supplied IDs. The new candidates are exposed through the C++ registry/API; the laboratory executable enumerates them automatically. This study does not add an online policy for choosing these IDs.

## Per-round replay

`run` accepts an optional final `DeviceRoundProbe` callback receiving the device context, batch and round number after activation and before production calculation. A probe may modify `new_values` and the error flag only. The executor restores `new <- old` and clears the error before its production launch. Default calls do not invoke a probe.

`graphweft_push_rounds` uses the hook to:

1. Snapshot the current inputs on the host and collect degree/query histograms outside timing.
2. Compute the original Push result and compare every candidate's entire output byte for byte, using a reusable 4 MiB pinned transfer buffer.
3. Warm up each candidate once, then measure ten samples per candidate with CUDA events. Each repetition shuffles candidate order using seed `0x4757`. If any candidate's IQR/median exceeds 10%, all receive twenty additional samples; remaining noise is reported.
4. Check the complete old values, mask, frontier, count and live slots have not changed, then return to the normal executor.

Every timed sample restores `new <- old` outside its event interval. Timings include query-list compaction inside the kernel. They exclude restoration, frontier comparison, feature gathering, validation and host transfers. The cache protocol is normal repeated replay after a full value copy; it does not reproduce an uninstrumented iteration's exact cache contents. Only one stream and one GPU are used. Frontier compaction uses the default unordered mode: all candidates see the identical list within a round, but a new whole-graph run can produce a different list order. Values, masks and frontier sets remain deterministic under the tested contract.

The host holds one old snapshot and one reference output for the current round, plus masks/frontier and the transfer chunk. There is no third value array on the GPU and no archive of full-state checkpoints. This permits Q32 on the existing V100 32 GiB workloads.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j8
CUDA_VISIBLE_DEVICES=0 ./build/graphweft_partition_validate
CUDA_VISIBLE_DEVICES=0 ./build/graphweft_push_rounds \
  --graph=/home/zyl/data/csr_data/cit-Patents \
  --queries=experiments/20260921-201450_sssp_n32_hybrid/cit-Patents/sources.csv \
  --output=experiments/my_push_study/cit-Patents --q=32
```

The laboratory entrypoint defaults to directed SSSP, original int32 CSR weights converted to float32, FIFO, zero offsets and vertex-major layout. Output contains raw `samples.csv`, per-candidate `rounds.csv`, `features.csv`, and degree/query `histogram.csv`. The degree bucket is zero for degree zero, otherwise `floor(log2(degree))+1`. `query_runs` counts contiguous runs of active query columns across the frontier. `completed.txt` is written only after all rounds finish.

The experiment harness records source/binary hashes, GPU information, source lists and exact commands. It skips completed graphs; an interrupted graph is restarted from round zero and its partial CSV files are replaced. Reuse the harness only with the recorded binary/source version, or create a new experiment directory.

The original engine's CPU-reference tests establish baseline correctness. The partition test compares all variants on every reached small-graph round, also reverses frontier order, covers both layouts, three algorithms, Q1 through Q256 (including non-word boundaries), delayed activation, multiple/tail batches, zero-degree vertices, duplicate/self edges and competing updates. The frontier test checks all new variants at the BFS float32 boundary and with empty input. Sanitizer runs exercise divergent degrees and query tails. A production replay test also executes all 30 IDs through the round executor and compares values, masks, stable frontier arrays and Push/Pull round counters against the original implementation.

Interpret per-round speedups alongside noise and absolute times. A sum of minimum replay kernel times is an offline kernel reference, not a measured adaptive or end-to-end result. Coalescing, bandwidth and atomic contention require additional profiler evidence; timing alone does not identify the cause.
