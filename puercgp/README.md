# puercgp

`puercgp` is a header-only C++/CUDA framework scaffold for concurrent GPU graph processing. It is now standalone and does not depend on the `cgp/` tree.

## Build

```bash
cmake -S /home/zyl/Projects/ocgp/puercgp -B /home/zyl/Projects/ocgp/puercgp/build
cmake --build /home/zyl/Projects/ocgp/puercgp/build
```

## API Shape

```cpp
#include <puercgp/puercgp.hxx>

puercgp::query_batch<int> queries({0, 1, 2, 3});

puercgp::run_options options;
options.traversal_mode = puercgp::traversal_mode_t::hybrid;
options.push_strategy = puercgp::push_strategy_t::shared_node_warp;
options.pull_strategy = puercgp::pull_strategy_t::fused;
options.profile_iterations = true;

puercgp::execution_context context;

auto result = puercgp::run<puercgp::algorithms::bfs_policy>(
    G, queries, context, options);
```

## Current Status

The framework currently provides:

- public API and umbrella header
- runtime context wrapper for CUDA streams
- query batch validation
- structured result/profile types
- value matrix and frontier storage resources
- shared frontier engine for BFS, SSSP, SSWP, and WCC
- synchronous rank engine for PageRank and PPR
- compile-time policies plus runtime heterogeneous frontier dispatch
- smoke and validation examples

The BFS, SSSP, and SSWP policies use the shared frontier engine. BFS keeps its
visited-mask fast path; SSSP uses atomic min; SSWP uses atomic max with
`min(path_width, edge_capacity)` relaxation. All three support push, fused
pull, and hybrid traversal.

PageRank and PPR share a standard synchronous power-iteration engine with
`old_rank` and `next_rank` buffers. Push mode rebuilds `next_rank` by scattering
every active query over all outgoing edges. Pull mode performs sum reduction
over incoming neighbors. Both modes mark
`abs(next_rank-old_rank) > epsilon`, run the same convergence post-processing,
and then swap rank buffers. Hybrid mode samples one push and one pull iteration,
then selects the lower measured cost per active query. PageRank uses uniform
personalization and PPR uses a source-specific one-hot vector. Dangling mass
returns to the same personalization distribution. Edge weights are ignored
when constructing transition probabilities.

```cpp
puercgp::algorithms::pagerank_query_batch rank_queries({
    {0.85f, 1.0e-8f},
    {0.95f, 1.0e-8f},
});
auto rank_result =
    puercgp::run<puercgp::algorithms::pagerank_policy>(
        G, rank_queries, context, options);

puercgp::algorithms::ppr_query_batch ppr_queries({
    {0, 0.85f, 1.0e-8f},
    {17, 0.90f, 1.0e-8f},
});
auto ppr_result = puercgp::run<puercgp::algorithms::ppr_policy>(
    G, ppr_queries, context, options);
```

All result matrices use vertex-major `V*Q` layout. The rank policies
support at most 64 concurrent queries, matching `query_mask_t`.

## Validation

```bash
/home/zyl/Projects/ocgp/puercgp/build/validate_bfs \
  /path/to/matrix.mtx \
  0,997,1994,2991,3988,4985,5982,6979 \
  7

/home/zyl/Projects/ocgp/puercgp/build/validate_sssp \
  /path/to/weighted.mtx \
  0,1,2,3 \
  7

/home/zyl/Projects/ocgp/puercgp/build/validate_sswp
/home/zyl/Projects/ocgp/puercgp/build/validate_pagerank
/home/zyl/Projects/ocgp/puercgp/build/validate_ppr
```

The validators compare full GPU result matrices against CPU references and
report numerical error and median GPU time.

## Constraints

- Shared-frontier and rank paths support at most 64 concurrent queries.
- The current implementation requires `int` query vertices.
- PageRank and PPR support full-rank push, fused pull, and adaptive hybrid.
- Replenishment currently supports BFS/SSSP/WCC, not SSWP or rank policies.
- `puercgp` does not use Gunrock or cgp namespaces.
- This project should not modify any external cgp checkout or baseline implementation.
