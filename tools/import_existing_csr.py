#!/usr/bin/env python3
"""Import a trusted legacy int32 CSR into a preparation root without duplicating data."""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GIB = 1 << 30


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def allocation_bytes(vertices, edges, capacity):
    words = (capacity + 63) // 64
    return (8 * (vertices + 1) + 8 * edges + 8 * vertices * capacity +
            16 * vertices * words + 12 * vertices + max(4 * capacity, 8 * words) +
            4 * capacity + 2 * capacity + capacity + 4 * capacity +
            ((capacity + 31) // 32) * 32 + vertices + 4 * vertices + 40)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def link_or_copy(source, target):
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--compact-ids", action="store_true",
                        help="renumber non-isolated CSR rows into a contiguous [0,V) range")
    parser.add_argument("--compactor", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build/graphweft_compact_csr_ids")
    args = parser.parse_args()
    source = args.source.resolve(); root = args.root.resolve()
    names = ("csr_vlist.bin", "csr_elist.bin", "csr_weightlist.bin")
    paths = [source / name for name in names]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("source CSR is incomplete")
    if paths[0].stat().st_size % 4 or paths[1].stat().st_size % 4:
        raise RuntimeError("legacy CSR files must contain int32 values")
    vertices = paths[0].stat().st_size // 4 - 1
    edges = paths[1].stat().st_size // 4
    if paths[2].stat().st_size != edges * 4:
        raise RuntimeError("weight and edge counts differ")
    with paths[0].open("rb") as handle:
        first = struct.unpack("<i", handle.read(4))[0]
        handle.seek(-4, 2); last = struct.unpack("<i", handle.read(4))[0]
    if first != 0 or last != edges:
        raise RuntimeError(f"invalid CSR row offsets: first={first}, last={last}, edges={edges}")
    target = root / "datasets" / args.name
    if target.exists():
        raise RuntimeError(f"target already exists: {target}")
    if args.compact_ids:
        result = subprocess.run([str(args.compactor.resolve()), str(source), str(target)], text=True)
        if result.returncode:
            raise RuntimeError("CSR ID compaction failed")
        methods = {name: "compacted" for name in names}
    else:
        target.mkdir(parents=True)
        methods = {name: link_or_copy(path, target / name) for name, path in zip(names, paths)}
    paths = [target / name for name in names]
    vertices = paths[0].stat().st_size // 4 - 1
    edges = paths[1].stat().st_size // 4
    hashes = {name: digest(target / name) for name in names}
    allocation = {str(capacity): allocation_bytes(vertices, edges, capacity)
                  for capacity in (128, 256)}
    budget = int(.8 * 32 * GIB)
    generated = json.loads((target / "conversion_manifest.json").read_text()) if args.compact_ids else {}
    manifest = {**generated, "schema": 1, "construction": "compact_existing_symmetric_csr" if args.compact_ids else "trusted_legacy_csr_import",
                "source": str(source), "vertices": vertices, "symmetric_edges": edges,
                "link_methods": methods, "files_sha256": hashes}
    if args.compact_ids:
        manifest["external_ids_compacted"] = True
        manifest["external_vertex_map"] = "external_vertex_ids.bin"
    atomic_json(target / "conversion_manifest.json", manifest)
    status = {"status": "success", "dataset": args.name, "campaign_eligible": allocation["128"] <= budget,
              "conversion": manifest, "files_sha256": hashes, "allocation_bytes": allocation,
              "fits_v100_32g_80pct": {key: value <= budget for key, value in allocation.items()},
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    status_dir = root / "status"; status_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(status_dir / f"{args.name}.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
