#!/usr/bin/env python3
"""Calibrate a static Push mapping, then compare frozen production mappings.

Calibration sources must be disjoint from all reported workloads.  Every run
uses stable frontier compaction so separate executions receive the same
deterministic round inputs.  The selected static mapping is persisted before
the formal workload is accepted by the ``compare`` command.
"""
import argparse
import csv
import json
import re
import statistics
import subprocess
from pathlib import Path


LANES = (1, 2, 4, 8, 16, 32)
GRAINS = range(5)


def parse_metrics(text):
    result = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try:
            result[key] = float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError:
            result[key] = value
    return result


def common(args, queries):
    command = [str(args.cli), f"--graph={args.graph}", f"--queries={queries}",
               "--n=1024", "--q=64", "--layout=grouped", "--group_width=8",
               "--algorithm=bfs", "--same_algorithm_groups", "--planner=fifo",
               "--selector=push", "--frontier=stable", "--profile_kernel",
               "--log_level=warn", f"--device={args.device}"]
    if args.directed:
        command.append("--directed")
    if args.legacy_int_weights:
        command.append("--legacy_int_weights")
    return command


def execute(command):
    run = subprocess.run(command, check=True, text=True, capture_output=True)
    return parse_metrics(run.stdout), run.stdout + run.stderr


def write_rows(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def calibrate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = [("shared", ["--push_mapping=shared"])]
    candidates += [(f"q{lanes}_g{grain}", ["--push_mapping=static",
                    f"--push_query_lanes={lanes}", f"--push_grain={grain}"])
                   for lanes in LANES for grain in GRAINS]
    # Warm each candidate once, then rotate its position in every repetition.
    base = common(args, args.queries)
    for _, flags in candidates:
        subprocess.run(base + flags, check=True, stdout=subprocess.DEVNULL)
    rows = []
    for repetition in range(args.repetitions):
        order = candidates[repetition % len(candidates):] + candidates[:repetition % len(candidates)]
        for name, flags in order:
            metrics, log = execute(base + flags)
            (args.output / f"rep{repetition}_{name}.log").write_text(log)
            rows.append({"repetition": repetition, "candidate": name, **metrics})
    write_rows(args.output / "calibration_timings.csv", rows)
    medians = {name: statistics.median(float(row["kernel_gpu_ms"]) for row in rows
               if row["candidate"] == name) for name, _ in candidates}
    # The shared path is a baseline, not a selectable static mapping.
    selected = min((name for name in medians if name != "shared"), key=medians.get)
    match = re.fullmatch(r"q(\d+)_g(\d+)", selected)
    frozen = {"candidate": selected, "query_lanes": int(match.group(1)),
              "grain": int(match.group(2)), "selection_metric": "median_kernel_gpu_ms",
              "samples": args.repetitions, "medians_ms": medians}
    (args.output / "frozen_mapping.json").write_text(json.dumps(frozen, indent=2) + "\n")


def compare(args):
    frozen = json.loads(args.mapping.read_text())
    configs = [
        ("shared", ["--push_mapping=shared"]),
        ("static_frozen", ["--push_mapping=static", f"--push_query_lanes={frozen['query_lanes']}",
                           f"--push_grain={frozen['grain']}"]),
        ("degree", ["--push_mapping=degree", f"--push_grain={frozen['grain']}"]),
        ("density", ["--push_mapping=density", f"--push_grain={frozen['grain']}"]),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    base = common(args, args.queries)
    for _, flags in configs:
        subprocess.run(base + flags, check=True, stdout=subprocess.DEVNULL)
    rows = []
    for repetition in range(args.repetitions):
        order = configs[repetition % len(configs):] + configs[:repetition % len(configs)]
        for name, flags in order:
            metrics, log = execute(base + flags)
            (args.output / f"rep{repetition}_{name}.log").write_text(log)
            rows.append({"repetition": repetition, "mapping": name, **metrics})
    write_rows(args.output / "timings.csv", rows)
    summary = []
    for name, _ in configs:
        selected = [row for row in rows if row["mapping"] == name]
        item = {"mapping": name, "samples": len(selected)}
        for metric in ("kernel_gpu_ms", "kernel_ms", "feature_ms", "planning_ms", "task_wall_ms", "workload_ms"):
            values = [float(row[metric]) for row in selected if metric in row]
            if values:
                item[metric + "_median"] = statistics.median(values)
                item[metric + "_min"] = min(values)
                item[metric + "_max"] = max(values)
        summary.append(item)
    write_rows(args.output / "summary.csv", summary)
    (args.output / "frozen_mapping.json").write_text(json.dumps(frozen, indent=2) + "\n")


def add_common(parser):
    parser.add_argument("--cli", type=Path, default=Path("build/graphweft_cli"))
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--legacy-int-weights", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    calibration = commands.add_parser("calibrate")
    add_common(calibration)
    calibration.set_defaults(func=calibrate)
    comparison = commands.add_parser("compare")
    add_common(comparison)
    comparison.add_argument("--mapping", type=Path, required=True)
    comparison.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
