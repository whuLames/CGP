#!/usr/bin/env python3
"""Compare scan and fused frontier construction on one frozen workload."""
import argparse
import csv
import re
import statistics
import subprocess
from pathlib import Path


METRICS = ("workload_ms", "throughput_qps", "task_wall_ms", "kernel_ms",
           "kernel_gpu_ms", "frontier_ms", "compare_ms", "copy_ms",
           "feature_ms", "recycle_ms")


def parse_metrics(text):
    result = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try:
            result[key] = float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError:
            result[key] = value
    return result


def execute(command):
    run = subprocess.run(command, check=True, text=True, capture_output=True)
    return parse_metrics(run.stdout), run.stdout + run.stderr


def write_rows(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=Path("build/graphweft_cli"))
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--legacy-int-weights", action="store_true")
    parser.add_argument("--selector", choices=("push", "threshold"), default="push")
    parser.add_argument("--frontier", choices=("unordered", "stable"), default="unordered")
    parser.add_argument("--push-mapping", choices=("shared", "static", "degree", "density"), default="shared")
    parser.add_argument("--push-query-lanes", type=int, default=8)
    parser.add_argument("--push-grain", type=int, default=0)
    parser.add_argument("--planner", choices=("fifo", "length"), default="fifo")
    parser.add_argument("--predictor", default="import")
    parser.add_argument("--group-refill", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base = [str(args.cli), f"--graph={args.graph}", f"--queries={args.queries}",
            "--n=1024", "--q=64", "--layout=grouped", "--group_width=8",
            "--algorithm=bfs", "--same_algorithm_groups", f"--planner={args.planner}",
            f"--predictor={args.predictor}", f"--selector={args.selector}",
            f"--frontier={args.frontier}", f"--push_mapping={args.push_mapping}",
            f"--push_query_lanes={args.push_query_lanes}", f"--push_grain={args.push_grain}",
            "--profile_kernel", "--profile_compare", "--log_level=warn", f"--device={args.device}"]
    if args.group_refill:
        base.append("--group_refill")
    if args.directed:
        base.append("--directed")
    if args.legacy_int_weights:
        base.append("--legacy_int_weights")

    if args.frontier != "unordered":
        raise ValueError("direct frontier comparison requires --frontier=unordered")
    modes = ("scan", "fused", "direct")
    for mode in modes:
        subprocess.run(base + [f"--frontier_build={mode}"], check=True, stdout=subprocess.DEVNULL)
    rows = []
    for repetition in range(args.repetitions):
        order = modes if repetition % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            metrics, log = execute(base + [f"--frontier_build={mode}"])
            if mode in ("fused", "direct") and float(metrics.get("compare_ms", -1)) != 0:
                raise RuntimeError(f"{mode} mode unexpectedly executed compare_kernel")
            (args.output / f"rep{repetition}_{mode}.log").write_text(log)
            rows.append({"repetition": repetition, "frontier_build": mode, **metrics})
    write_rows(args.output / "timings.csv", rows)

    summary = []
    for mode in modes:
        selected = [row for row in rows if row["frontier_build"] == mode]
        item = {"frontier_build": mode, "samples": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected if metric in row]
            if values:
                item[metric + "_median"] = statistics.median(values)
                item[metric + "_min"] = min(values)
                item[metric + "_max"] = max(values)
        summary.append(item)
    write_rows(args.output / "summary.csv", summary)


if __name__ == "__main__":
    main()
