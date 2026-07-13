# roadNet-CA long-diameter replenish experiment

## Question

Does immediate slot replenishment become effective for a long-diameter graph,
large query populations, and BFS sources with deliberately different completion
levels?

This experiment was fixed before performance measurement:

- graph: roadNet-CA;
- algorithm: BFS;
- traversal: all-push;
- resident query slots: Q=32;
- total queries: N=400 and N=800;
- source order: deterministic offline low/high eccentricity interleaving;
- baseline: consecutive static batches;
- replenish: immediate FIFO replacement (`replenish_chunk=1`);
- no selective batching, offset alignment, or phase-aware admission.

## Dataset preparation

Downloaded:

```text
https://snap.stanford.edu/data/roadNet-CA.txt.gz
SHA256: 383f7b14424530a9e25c392969a9c19dace32ebcfd141f0967a1433498c2acdd
```

The SNAP header describes a directed edge-list representation. The existing
TCRGraph `el2csr` tool inserts both directions, removes self loops, renumbers
vertices densely, sorts edges, and removes duplicates.

Conversion result:

```text
input edge-list rows:       5,533,214
temporary directed edges:  11,066,428
duplicates removed:         5,533,214
final vertices:             1,965,206
final stored CSR edges:      5,533,214
row-offset type:             int32
```

The final files are:

```text
/home/zyl/data/csr_data/roadNet-CA/csr_vlist.bin
/home/zyl/data/csr_data/roadNet-CA/csr_elist.bin
```

## Offline source selection

Source selection is deterministic and restricted to the largest connected
component, avoiding small disconnected components as an artificial source of
short queries.

1. Run a two-sweep landmark BFS.
2. Rank all giant-component vertices by
   `max(distance_from_A, distance_from_B)`.
3. Select 2,048 candidates uniformly across that ranking.
4. Run exact CPU BFS from every candidate.
5. Select the 400 lowest and 400 highest exact-eccentricity candidates.
6. Interleave low/high sources so every static batch contains both groups.

Characterization:

```text
largest component vertices: 1,957,027
double-sweep distance:       857
selected low range:          508-580 levels
selected high range:         755-857 levels
```

N=400 contains the first 200 low/high pairs. N=800 contains all 400 pairs, so
the N=400 workload is a prefix of N=800.

| Workload | Min level | Max level | Mean level |
|---|---:|---:|---:|
| N=400 | 508 | 857 | 676.44 |
| N=800 | 508 | 857 | 673.13 |

Source metadata and exact CPU fingerprints are in `sources/`.

## Primary performance results

Configuration:

```text
batch_size=32
replenish_chunk=1
mode=push
push=shared_node_warp
discard_results=false
warmup=1
repeats=3
reported value=median
```

| N | Static ms | Immediate replenish ms | Speedup | Static query/s | Replenish query/s |
|---:|---:|---:|---:|---:|---:|
| 400 | 3,191.01 | 3,450.39 | 0.925x | 125.35 | 115.93 |
| 800 | 6,178.18 | 6,924.43 | 0.892x | 129.49 | 115.53 |

Immediate replenish regresses throughput by 7.5% for N=400 and 10.8% for
N=800. The original hypothesis is therefore not supported.

The first cold N=400 run showed 1.074x, but this disappeared after warmup and
three repetitions. It is retained in `runs/n400_smoke.log` and is not used as
the reported result.

## Slot utilization

Offline exact BFS levels allow an iteration-level occupancy calculation.

| N | Sum query levels | Static batch steps | Replenish steps | Static utilization | Replenish utilization |
|---:|---:|---:|---:|---:|---:|
| 400 | 270,574 | 10,602 | 8,915 | 79.75% | 94.85% |
| 800 | 538,501 | 19,847 | 17,255 | 84.79% | 97.53% |

Replenish does what it is intended to do: it substantially reduces empty-slot
area. Higher logical occupancy nevertheless does not imply higher GPU
throughput. Average wall time per global step rises by roughly 25%, more than
offsetting the 13%-16% reduction in step count.

The likely contributors are:

- immediate per-slot value reset and visited-mask clearing;
- more frequent host/device scheduling and slot metadata updates;
- newly admitted queries running at different BFS phases from resident queries;
- lower shared-frontier mask density and less edge sharing;
- a larger union frontier and memory traffic per global step.

## Diagnostic experiments

### Discard final results

This removes per-query final snapshots but keeps immediate replacement.

| N | Static ms | Replenish ms | Speedup |
|---:|---:|---:|---:|
| 400 | 3,170.55 | 3,333.52 | 0.951x |
| 800 | 6,156.00 | 6,725.16 | 0.915x |

Snapshot removal recovers only 2%-3%. Snapshot cost is not the primary cause.

### Full-cohort replacement

`replenish_chunk=32` is a diagnostic only; it disables immediate filling and
waits for a complete cohort. Results use `discard_results=true`.

| N | Static ms | Cohort-32 replenish ms | Speedup |
|---:|---:|---:|---:|
| 400 | 3,170.55 | 3,135.22 | 1.011x |
| 800 | 6,156.00 | 6,033.83 | 1.020x |

The engine and persistent buffer reuse are not inherently slower. Regression
appears when immediate replacement mixes phases and performs frequent slot
maintenance.

## Correctness

The N=800 workload was run with final result storage enabled. For every query,
GPU output was compared with exact offline CPU BFS using:

- reachable vertex count;
- sum of all finite BFS distances.

Result:

```text
queries=800
iterations=17255
mismatches=0
```

N=400 is a prefix of the validated N=800 source list.

## Conclusion

Long diameter, large N, and completion-level skew are not sufficient for
immediate replenish to improve throughput. They create real empty-slot
opportunity, but immediate admission changes the shared execution workload.

The result suggests that replenish should not be optimized only for occupancy.
A useful admission policy must compare:

```text
saved empty-slot work
versus
slot-reset cost + added union-frontier work + lost sharing
```

For the current implementation, full-cohort reuse is slightly beneficial while
immediate replacement is not. Any future phase-aware replenish policy should
be evaluated against these fixed source lists and must include its scheduling
overhead.

## Reproduction

Primary run:

```bash
CUDA_VISIBLE_DEVICES=2 ./build/bench_replenish \
  /home/zyl/data/csr_data/roadNet-CA \
  --sources-file=experiments/20260712_roadnet_replenish/sources/sources_n800.csv \
  --bfs=0 --sssp=0 --wcc=0 \
  --batch-size=32 --replenish-chunk=1 \
  --max-iterations=30000 --mode=push --push=warp \
  --no-discard --warmup=1 --repeats=3 --run=both
```

Artifacts:

- `sources/candidate_eccentricities.csv`
- `sources/sources_n400.csv`
- `sources/sources_n800.csv`
- `runs/n400_formal.log`
- `runs/n800_formal.log`
- `runs/n400_discard.log`
- `runs/n800_discard.log`
- `runs/n400_cohort32_diagnostic.log`
- `runs/n800_cohort32_diagnostic.log`
- `runs/n800_correctness.log`
