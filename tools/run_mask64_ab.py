#!/usr/bin/env python3
"""Interleaved single-bit versus 64-bit update-driven frontier A/B."""

import argparse
import csv
import re
import statistics
import subprocess
from pathlib import Path


METRICS = ("workload_ms", "throughput_qps", "kernel_gpu_ms", "kernel_ms",
           "frontier_ms", "copy_ms", "feature_ms")


def parse_metrics(text):
    parsed = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try:
            parsed[key] = float(value)
        except ValueError:
            pass
    return parsed


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=Path("build/graphweft_cli"))
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("bfs", "sssp", "sswp"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--legacy-int-weights", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base = [str(args.cli), f"--graph={args.graph}", f"--queries={args.queries}",
            "--n=1024", "--q=64", "--layout=grouped", "--group_width=8",
            f"--algorithm={args.algorithm}", "--same_algorithm_groups",
            "--planner=fifo", "--predictor=import", "--selector=push",
            "--frontier=unordered", "--frontier_build=direct",
            "--push_mapping=shared", "--profile_kernel", "--profile_compare",
            "--log_level=warn", f"--device={args.device}"]
    if args.directed:
        base.append("--directed")
    if args.legacy_int_weights:
        base.append("--legacy_int_weights")

    def execute(mask64, label):
        command = base + [f"--frontier_mask64={'true' if mask64 else 'false'}"]
        run = subprocess.run(command, check=True, text=True, capture_output=True)
        (args.output / f"{label}.log").write_text(run.stdout + run.stderr)
        return parse_metrics(run.stdout)

    execute(False, "warmup_legacy")
    execute(True, "warmup_mask64")
    rows = []
    for repetition in range(args.repetitions):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        for mask64 in order:
            name = "mask64" if mask64 else "legacy"
            metrics = execute(mask64, f"rep{repetition}_{name}")
            rows.append({"repetition": repetition, "variant": name, **metrics})
    write_csv(args.output / "raw.csv", rows)

    summary = []
    for name in ("legacy", "mask64"):
        selected = [row for row in rows if row["variant"] == name]
        item = {"variant": name, "samples": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected if metric in row]
            if values:
                item[f"{metric}_median"] = statistics.median(values)
                item[f"{metric}_min"] = min(values)
                item[f"{metric}_max"] = max(values)
        summary.append(item)
    write_csv(args.output / "summary.csv", summary)


if __name__ == "__main__":
    main()
