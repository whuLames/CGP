#!/usr/bin/env python3
"""Convert a little-endian Galois .gr graph to GraphWeft legacy CSR."""
import argparse
import array
import json
import shutil
import struct
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw_size = args.input.stat().st_size
    with args.input.open("rb") as source:
        header = source.read(32)
        if len(header) != 32:
            raise SystemExit("truncated .gr header")
        version, edge_bytes, vertices, edges = struct.unpack("<4Q", header)
        if version != 1 or vertices >= 2**31 or edges >= 2**31:
            raise SystemExit("unsupported .gr header")
        ends = array.array("Q")
        ends.fromfile(source, vertices)
        if sys.byteorder != "little":
            ends.byteswap()
        if not ends or ends[-1] != edges:
            raise SystemExit("invalid cumulative edge index")
        args.output.mkdir(parents=True, exist_ok=False)
        rows = array.array("i", [0])
        rows.extend(int(value) for value in ends)
        if sys.byteorder != "little":
            rows.byteswap()
        with (args.output / "csr_vlist.bin").open("wb") as target:
            rows.tofile(target)
        with (args.output / "csr_elist.bin").open("wb") as target:
            shutil.copyfileobj(source, target, 8 << 20)
        # Trim alignment and optional edge payload from the destination file.
        edge_path = args.output / "csr_elist.bin"
        with edge_path.open("r+b") as target:
            target.truncate(edges * 4)
        manifest = {"source": str(args.input.resolve()), "version": version,
                    "edge_data_bytes": edge_bytes, "vertices": vertices,
                    "edges": edges, "input_bytes": raw_size,
                    "weights": "implicit_unit"}
        (args.output / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
