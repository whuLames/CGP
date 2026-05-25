# CGP BFS v2 Hybrid Push/Pull Summary

Implemented a hybrid traversal mode for `cgp_bfs` with a BFS pull kernel based on
the GE-SpMM row/feature mapping idea:

- rows are destination vertices;
- feature columns are concurrent BFS queries;
- pull uses a dense vertex-major current frontier bitmap;
- distances remain query-major for compatibility with validation and JSON;
- push/list and pull/bitmap state conversions are explicit.

Build and smoke validation:

```bash
cmake -S . -B build -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build build --target cgp_bfs -j 8
./build/bin/cgp_bfs --help
```

`chesapeake q3` validation results:

| mode | thresholds | profile | wall ms | gpu ms | level modes | errors |
| --- | --- | --- | ---: | ---: | --- | ---: |
| push | default | no | 0.633 | 0.609 | push,push,push,push | 0 |
| pull | default | yes | 0.793 | 0.770 | pull,pull,pull,pull | 0 |
| hybrid | default | no | 0.480 | 0.461 | push,pull,pull,push | 0 |
| hybrid | 0.01 / 0.01 | yes | 0.538 | 0.519 | pull,pull,pull,pull | 0 |
| hybrid | 0.30 / 1.00 | yes | 1.335 | 1.317 | push,push,pull,push | 0 |

Raw JSON:

- `build/cgp_bfs_optimization_results/cgp_bfs_chesapeake_q3_push.json`
- `build/cgp_bfs_optimization_results/cgp_bfs_chesapeake_q3_pull.json`
- `build/cgp_bfs_optimization_results/cgp_bfs_chesapeake_q3_hybrid_default.json`
- `build/cgp_bfs_optimization_results/cgp_bfs_chesapeake_q3_hybrid.json`
- `build/cgp_bfs_optimization_results/cgp_bfs_chesapeake_q3_hybrid_transitions.json`

Large-graph benchmarking remains pending. The next pass should collect three
run medians for `cit-Patents`, `soc-orkut`, `soc-twitter`, and
`soc-sinaweibo`, comparing `push`, `pull`, and `hybrid` against the previous CGP
v1 numbers and the GunrockV2.2 multi-stream baseline.
