#!/usr/bin/env python3
"""Convert GraphWeft's int32 CSR triplet to the legacy weighted Galois GR layout."""
import argparse
import os
import struct
from array import array
from pathlib import Path


def copy_file(source, output, chunk_bytes=64 << 20):
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            output.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    offsets_path = args.csr / "csr_vlist.bin"
    edges_path = args.csr / "csr_elist.bin"
    weights_path = args.csr / "csr_weightlist.bin"
    offset_bytes = offsets_path.stat().st_size
    edge_bytes = edges_path.stat().st_size
    if offset_bytes % 4 or edge_bytes % 4:
        raise ValueError("expected int32 offsets and destinations")
    if weights_path.exists() and weights_path.stat().st_size != edge_bytes:
        raise ValueError("CSR weight count mismatch")
    vertices = offset_bytes // 4 - 1
    edges = edge_bytes // 4

    with offsets_path.open("rb") as handle:
        first = struct.unpack("<i", handle.read(4))[0]
        handle.seek(-4, os.SEEK_END)
        last = struct.unpack("<i", handle.read(4))[0]
    if first != 0 or last != edges:
        raise ValueError(f"invalid CSR endpoints: first={first}, last={last}, E={edges}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as output:
        # version=1, logical one-byte weights.  Existing benchmark GR files
        # use this declaration while physically storing zero-extended uint32
        # weights, which both modified baseline loaders recognize exactly.
        output.write(struct.pack("<QQQQ", 1, 1, vertices, edges))
        with offsets_path.open("rb") as offsets:
            offsets.read(4)  # GR stores xadj[1:] as V uint64 end offsets.
            while chunk := offsets.read(16 << 20):
                values = array("i")
                values.frombytes(chunk)
                widened = array("Q", values)
                output.write(widened.tobytes())
        copy_file(edges_path, output)
        padding = (8 - (edge_bytes % 8)) % 8
        if padding:
            output.write(bytes(padding))
        if weights_path.exists():
            copy_file(weights_path, output)
        else:
            unit = struct.pack("<I", 1) * (1 << 20)
            remaining = edges
            while remaining:
                count = min(remaining, 1 << 20)
                output.write(unit[:count * 4])
                remaining -= count
    os.replace(temporary, args.output)
    print(f"wrote {args.output}: V={vertices} E={edges}")


if __name__ == "__main__":
    main()
