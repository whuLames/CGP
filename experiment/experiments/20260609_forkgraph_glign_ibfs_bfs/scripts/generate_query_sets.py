#!/usr/bin/env python3
import argparse
import json
import random
import struct
from pathlib import Path


DATASETS = {
    "cit-Patents": "/home/zyl/data/ggr_data/singlegpu/cit-Patents.gr",
    "soc-sinaweibo": "/home/zyl/data/ggr_data/singlegpu/soc-sinaweibo.gr",
    "soc-twitter": "/home/zyl/data/ggr_data/singlegpu/soc-twitter.gr",
    "soc-orkut": "/home/zyl/data/ggr_data/singlegpu/soc-orkut.gr",
}
CONCURRENCIES = [2, 4, 8, 16, 32, 64]
BASE_SEED = 20260609


def read_gr_csr(path: Path):
    with path.open("rb") as f:
        version, size_edge_ty, n, m = struct.unpack("<4Q", f.read(32))
        if version != 1:
            raise ValueError(f"{path} has unsupported version {version}")
        row = list(struct.unpack(f"<{n}q", f.read(8 * n)))
        offsets = [0] + row
        edges = list(struct.unpack(f"<{m}i", f.read(4 * m)))
    if offsets[-1] != m:
        raise ValueError(f"{path} row_start[-1] {offsets[-1]} != nedges {m}")
    return n, m, offsets, edges, size_edge_ty


def valid_source(src: int, offsets, edges, n: int) -> bool:
    if offsets[src] == offsets[src + 1]:
        return False
    seen = {src}
    level1 = []
    for i in range(offsets[src], offsets[src + 1]):
        dst = edges[i]
        if 0 <= dst < n and dst not in seen:
            seen.add(dst)
            level1.append(dst)
    if not level1:
        return False
    for v in level1:
        for i in range(offsets[v], offsets[v + 1]):
            dst = edges[i]
            if 0 <= dst < n and dst not in seen:
                return True
    return False


def choose_sources(dataset: str, concurrency: int, n: int, offsets, edges):
    seed = BASE_SEED + sum(ord(c) for c in dataset) * 1000 + concurrency
    rng = random.Random(seed)
    sources = []
    seen = set()
    attempts = 0
    max_attempts = max(100000, concurrency * 10000)
    while len(sources) < concurrency and attempts < max_attempts:
        attempts += 1
        src = rng.randrange(n)
        if src in seen:
            continue
        if valid_source(src, offsets, edges, n):
            sources.append(src)
            seen.add(src)
    if len(sources) < concurrency:
        for src in range(n):
            if src in seen:
                continue
            if valid_source(src, offsets, edges, n):
                sources.append(src)
                seen.add(src)
                if len(sources) >= concurrency:
                    break
    if len(sources) != concurrency:
        raise RuntimeError(f"only found {len(sources)} valid sources for {dataset} q{concurrency}")
    return seed, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for dataset, graph_path in DATASETS.items():
        graph_path = Path(graph_path)
        print(f"Loading {dataset}: {graph_path}")
        n, m, offsets, edges, size_edge_ty = read_gr_csr(graph_path)
        ds_dir = args.out_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        for concurrency in CONCURRENCIES:
            seed, sources = choose_sources(dataset, concurrency, n, offsets, edges)
            src_path = ds_dir / f"q{concurrency}.sources"
            meta_path = ds_dir / f"q{concurrency}.json"
            src_path.write_text("".join(f"{s}\n" for s in sources))
            meta = {
                "dataset": dataset,
                "graph": str(graph_path),
                "nvtxs": n,
                "nedges": m,
                "sizeEdgeTy": size_edge_ty,
                "concurrency": concurrency,
                "seed": seed,
                "sources": sources,
            }
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            manifest.append({
                "dataset": dataset,
                "concurrency": concurrency,
                "sources_file": str(src_path),
                "metadata_file": str(meta_path),
                "seed": seed,
                "sources": sources,
            })
            print(f"  q{concurrency}: {sources}")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
