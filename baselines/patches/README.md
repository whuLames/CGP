# Local baseline adaptations

This directory records the source-level changes used by the PuerCGP baseline
experiments without vendoring local baseline checkouts, build products, or
datasets into this repository.

## ForkGraph

- Upstream: `git@github.com:Xtra-Computing/ForkGraph.git`
- Base commit: `9177b86bdb6f3477a5f50f37e56e6de7495f27c9`
- Patch: `ForkGraph-puercgp.patch`

```bash
git clone git@github.com:Xtra-Computing/ForkGraph.git
cd ForkGraph
git checkout 9177b86bdb6f3477a5f50f37e56e6de7495f27c9
git apply /path/to/CGP/baselines/patches/ForkGraph-puercgp.patch
```

## iBFS

- Upstream: `git@github.com:iHeartGraph/iBFS.git`
- Base commit: `08455983024543f04e1c4323b1bcc379bbd9930c`
- Patch: `iBFS-puercgp.patch`

```bash
git clone git@github.com:iHeartGraph/iBFS.git
cd iBFS
git checkout 08455983024543f04e1c4323b1bcc379bbd9930c
git apply /path/to/CGP/baselines/patches/iBFS-puercgp.patch
```

The patches include the benchmark-facing source changes, conversion utilities,
run scripts, and experiment notes. Generated binaries, backup files, datasets,
and build directories are intentionally excluded.
