# Stage 4 legacy cleanup report

Branch:

```text
refactor/puercgp-structure
```

## Scope

Completed Stage 4:

- Removed homogeneous legacy list / bitmap / dense / GE-SpMM execution paths from
  `frontier_engine.hxx`.
- Kept only shared frontier + fused pull in the homogeneous engine.
- Reduced public strategy enums:
  - `push_strategy_t`: `shared_node`, `shared_node_query_parallel`,
    `shared_node_warp`.
  - `pull_strategy_t`: `fused`.
  - `frontier_repr_t`: `shared`.
- Removed obsolete `state/frontier_storage.hxx`.
- Removed old bitmap / dense / GE-SpMM profile fields.
- Migrated examples to the new `fused` pull option.

## Pull adjacency fix

During Stage 4 validation, directed toy graphs exposed that fused pull was
reading outgoing CSR adjacency. That computes reverse propagation on directed
graphs.

Fix:

- Added optional incoming adjacency to `csr_graph_view` / `csr_graph_storage`.
- Added `backend/pull_graph_access.hxx` to centralize pull-edge access.
- Updated homogeneous and hybrid fused pull kernels to read incoming adjacency
  when present.
- Examples build incoming adjacency only for `pull` / `hybrid` modes, so
  push-only runs do not pay the extra graph-memory cost.
- Explicit `pull` mode now rejects graph views without incoming adjacency.
  `hybrid` falls back to push until incoming adjacency is available.

## Build

Command:

```bash
cmake --build /home/zyl/Projects/ocgp/puercgp/build -j 8
```

Result: PASS.

Known warnings match previous stages:

- nvcc deprecated offline compilation target warning.
- `validate_hybrid.cu` unused `INF` warning.

## Correctness

Gate command group:

- `smoke_frontier`
- `validate_bfs`
- `validate_sssp`
- `validate_wcc`
- `validate_hybrid`
- `smoke_hybrid_init`
- `smoke_hybrid_push`
- `smoke_replenish`
- `validate_replenish`

Additional directed pull/hybrid checks:

- `validate_bfs ... pull shared_node_warp fused`
- `validate_bfs ... hybrid shared_node_warp fused`
- `validate_sssp ... pull shared_node_warp fused`
- `validate_sssp ... hybrid shared_node_warp fused`

Result: PASS. Directed pull/hybrid BFS and SSSP report `distance_mismatches=0`.

## Symbol cleanup

`rg` confirms removed strategy/path symbols are no longer referenced in
`include/` or `examples/`:

- `frontier_storage`
- `frontier_item_t`
- `edge_balanced`
- `shared_node_degree`
- `pull_strategy_t::bitmap`
- `pull_strategy_t::ge_spmm`
- old bitmap / dense / GE-SpMM profile field names

## Performance sanity

Command output saved in:

```text
puercgp/experiments/refactor_reports/stage4_perf_gate_b.log
```

Sanity runs:

```text
cit-Patents Q=16 pull:   hybrid wall_ms 261.913
cit-Patents Q=16 hybrid: hybrid wall_ms 140.703
soc-orkut   Q=16 pull:   hybrid wall_ms 743.734
soc-orkut   Q=16 hybrid: hybrid wall_ms 413.858
```

Notes:

- These are sanity results, not a strict regression gate, because the command
  used simple sources `0..7`; Stage 3 gate logs do not record a fully
  reproducible source set for these Q=16 runs.
- The absolute hybrid runtimes are close to Stage 3 gate A numbers for the same
  graph/mode categories.
- Full reproducible performance comparison remains assigned to Stage 8.
