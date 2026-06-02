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
options.push_strategy = puercgp::push_strategy_t::edge_balanced;
options.pull_strategy = puercgp::pull_strategy_t::bitmap;
options.profile_iterations = true;

puercgp::execution_context context;

auto result = puercgp::run<puercgp::algorithms::bfs_policy>(
    G, queries, context, options);
```

## Current Status

This first version establishes the framework boundary:

- public API and umbrella header
- runtime context wrapper for CUDA streams
- query batch validation
- structured result/profile types
- value matrix and frontier storage resources
- standalone frontier engine for BFS and SSSP
- dense engine skeleton for PageRank-style algorithms
- BFS, SSSP, and PageRank policy declarations
- smoke and validation examples

The BFS and SSSP kernels are implemented inside `puercgp` and operate on the
project-local CSR graph view.

The dense engine still returns a fixed-iteration placeholder result for PageRank-style algorithms. Convergence and PageRank kernels are still TODO.

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
```

Both validators run one warmup, then repeat the standalone `puercgp` kernel
path and compare against CPU references. They report median wall/GPU times.

## Migration Route

1. Split the current standalone frontier bitmap kernels into push/pull backend modules.
2. Add edge-balanced push, shared-node push, bitmap pull, and GE-SpMM pull implementations.
3. Keep BFS/SSSP differences expressed as policy hooks: init value, source value, relax/update, and dense postprocess.

## Constraints

- Optimized concurrent paths default to `max_queries = 32`.
- The first version requires `int` query vertices.
- `puercgp` does not use Gunrock or cgp namespaces.
- This project should not modify any external cgp checkout or baseline implementation.
