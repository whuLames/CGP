#!/usr/bin/env python3

import csv
import hashlib
import json
import struct
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
PUER_ROOT = Path("/home/zyl/Projects/ocgp/puercgp")
CSR_ROOT = Path("/home/zyl/data/csr_data")
DATASETS = ("cit-Patents", "soc-orkut", "soc-twitter", "soc-sinaweibo")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sources(path: Path) -> list[int]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        sources = [int(row["source"]) for row in rows]
    if len(sources) != 256:
        raise RuntimeError(f"{path} contains {len(sources)} sources, expected 256")
    if len(set(sources)) != len(sources):
        raise RuntimeError(f"{path} contains duplicate sources")
    return sources


def read_i32_offsets(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    if len(data) % 4:
        raise RuntimeError(f"invalid int32 offset file: {path}")
    return struct.unpack(f"<{len(data) // 4}i", data)


def write_sources(path: Path, sources: list[int]) -> None:
    path.write_text("".join(f"{source}\n" for source in sources))


def main() -> None:
    manifest = {
        "source_seed": 42,
        "total_queries": 256,
        "datasets": {},
    }
    for dataset in DATASETS:
        source_path = (
            PUER_ROOT
            / "experiments/20260719_online_runner_n256_q64"
            / f"final-{dataset}-bfs/sources.csv"
        )
        sources = read_sources(source_path)
        offsets = read_i32_offsets(CSR_ROOT / dataset / "csr_vlist.bin")
        vertices = len(offsets) - 1
        for source in sources:
            if source < 0 or source >= vertices:
                raise RuntimeError(f"{dataset}: source {source} is out of range")
            if offsets[source] == offsets[source + 1]:
                raise RuntimeError(f"{dataset}: source {source} is isolated")

        output_dir = EXP_ROOT / "query_sets" / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical_csv = output_dir / "sources.csv"
        canonical_csv.write_text(source_path.read_text())
        plain_path = output_dir / "sources.txt"
        write_sources(plain_path, sources)

        batch_files = {"q64": [], "q32": []}
        for batch_size in (64, 32):
            key = f"q{batch_size}"
            for first in range(0, len(sources), batch_size):
                batch_path = output_dir / f"{key}_batch{first // batch_size}.sources"
                write_sources(batch_path, sources[first : first + batch_size])
                batch_files[key].append(str(batch_path))

        manifest["datasets"][dataset] = {
            "vertices": vertices,
            "sources_csv": str(canonical_csv),
            "sources_txt": str(plain_path),
            "sources_sha256": sha256(plain_path),
            "batch_files": batch_files,
            "sources": sources,
        }

    (EXP_ROOT / "query_sets/manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
