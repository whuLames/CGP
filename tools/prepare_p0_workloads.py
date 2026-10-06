#!/usr/bin/env python3
"""Freeze the seven-graph seed-42 BFS/SSSP/SSWP inputs for P0 campaigns."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
from array import array
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MAIN_INPUTS = PROJECT / "experiments/20260929-175832_adaptive_push_g32/inputs"
ROADS = {
    "roadNet-CA": (Path("/home/zyl/data/csr_data/roadNet-CA"), "145d40a3a1d44d6a"),
    "roadNet-TX": (Path("/home/zyl/data/csr_data/roadNet-TX"), "4a7afb88fd7b5c01"),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def eligible_sources(graph):
    offsets = array("i")
    path = graph / "csr_vlist.bin"
    with path.open("rb") as handle:
        offsets.fromfile(handle, path.stat().st_size // 4)
    return [vertex for vertex in range(len(offsets) - 1)
            if offsets[vertex + 1] > offsets[vertex]]


def write_base(path, identity, sources, algorithm):
    with path.open("w", newline="") as handle:
        handle.write(f"# graph_identity={identity}\n# capacity=64\n")
        handle.write("# seed=42; uniform positive-outdegree sources without replacement\n")
        handle.write("# id,source,score,offset,feature_key,algorithm,reference_rounds\n")
        writer = csv.writer(handle)
        offset = algorithm * 100_000
        for index, source in enumerate(sources):
            writer.writerow((offset + index, source, 0, 0, 0, algorithm, 0))


def freeze(cli, graph, source, output, algorithm, predictor):
    command = [str(cli), f"--graph={graph}", f"--queries={source}",
               f"--algorithm={algorithm}", "--n=1024", "--q=64",
               "--layout=grouped", "--group_width=32", "--planner=length",
               f"--predictor={predictor}", "--plan_only", f"--plan_output={output}",
               "--legacy_int_weights", "--log_level=warn"]
    run = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    output.with_suffix(".stdout.log").write_text(run.stdout)
    output.with_suffix(".stderr.log").write_text(run.stderr)
    if run.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{run.stderr[-3000:]}")
    return command


def convert_sswp(source, output):
    lines = []
    for line in source.read_text().splitlines():
        if not line or line.startswith("#"):
            lines.append(line)
        else:
            fields = line.split(","); fields[5] = "2"; lines.append(",".join(fields))
    output.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source_dir in sorted(path for path in MAIN_INPUTS.iterdir() if path.is_dir()):
        target = args.output / source_dir.name; target.mkdir(exist_ok=True)
        for workload in ("bfs", "sssp", "sswp"):
            source = source_dir / f"{workload}.csv"; destination = target / source.name
            shutil.copyfile(source, destination)
            records.append({"dataset": source_dir.name, "workload": workload,
                            "construction": "frozen_existing_seed42",
                            "path": str(destination), "sha256": digest(destination)})
    for dataset, (graph, identity) in ROADS.items():
        target = args.output / dataset; target.mkdir(exist_ok=True)
        if dataset == "roadNet-CA":
            existing = PROJECT / "experiments/20260927_fixed_n1024_m64/roadNet-CA/workloads/bfs.csv"
            sources = []
            with existing.open() as handle:
                for line in handle:
                    if line.strip() and not line.startswith("#"):
                        sources.append(int(next(csv.reader([line]))[1]))
        else:
            sources = random.Random(42).sample(eligible_sources(graph), 1024)
        base_bfs, base_sssp = target / "bfs_base.csv", target / "sssp_base.csv"
        write_base(base_bfs, identity, sources, 0); write_base(base_sssp, identity, sources, 1)
        commands = [freeze(args.cli.resolve(), graph, base_bfs, target / "bfs.csv", "bfs", "core_distance"),
                    freeze(args.cli.resolve(), graph, base_sssp, target / "sssp.csv", "sssp", "weighted_boundary")]
        convert_sswp(target / "sssp.csv", target / "sswp.csv")
        for workload in ("bfs", "sssp", "sswp"):
            destination = target / f"{workload}.csv"
            records.append({"dataset": dataset, "workload": workload,
                            "construction": "seed42_graph_predictor",
                            "path": str(destination), "sha256": digest(destination)})
        (target / "preparation_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps({"N": 1024, "Q": 64,
        "G": 32, "seed": 42, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
