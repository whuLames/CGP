#!/usr/bin/env python3
"""Collect and merge Gunrock per-iteration CSVs with Nsight Compute CSV output."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_NCU = "/home/zyl/.conda/envs/torch2.8/bin/ncu"
DEFAULT_DATA_DIR = Path("/home/zyl/data/ggr_data/singlegpu")
DEFAULT_BUILD_DIR = Path("build_cuda")
DEFAULT_OUT_DIR = Path("iteration_profiles")
METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_issued.avg.per_cycle_active",
]
DURATION_METRIC = "gpu__time_duration.sum"
DATASETS = ["cit-Patents", "soc-orkut", "soc-sinaweibo", "soc-twitter"]
BFS_SOURCES = {
    "cit-Patents": 0,
    "soc-orkut": 1506298,
    "soc-sinaweibo": 53297474,
    "soc-twitter": 11702603,
}
RANGE_RE = re.compile(r"gunrock:(bfs|pr):iter:(\d+)(?::edge_spread)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", default=DEFAULT_NCU, help="Path to ncu")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--algorithms", nargs="+", choices=["bfs", "pr"], default=["bfs", "pr"])
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: List[str], env: Dict[str, str], dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
      return 0
    completed = subprocess.run(command, env=env)
    return completed.returncode


def run_ncu_command(command: List[str], env: Dict[str, str], raw_csv: Path, dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    raw_csv.write_text(completed.stdout)
    if completed.returncode != 0 and completed.stdout:
        sys.stderr.write(completed.stdout)
    return completed.returncode


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    start = 0
    for index, line in enumerate(lines):
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        normalized = {field.strip().lower() for field in fields}
        if (
            "algorithm" in normalized
            or "metric name" in normalized
            or any(metric.lower() in normalized for metric in METRICS)
        ):
            start = index
            break
    return list(csv.DictReader(lines[start:]))


def find_field(row: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {key.lower().strip(): key for key in row}
    for candidate in candidates:
        key = normalized.get(candidate.lower())
        if key is not None:
            return row.get(key)
    for key, value in row.items():
        lowered = key.lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return value
    return None


def extract_ranges(row: Dict[str, str]) -> List[str]:
    ranges: List[str] = []
    for value in row.values():
        if value and "gunrock:" in value:
            for match in RANGE_RE.finditer(value):
                range_name = match.group(0)
                if range_name not in ranges:
                    ranges.append(range_name)
    return ranges


def extract_range(row: Dict[str, str]) -> Optional[str]:
    ranges = extract_ranges(row)
    if not ranges:
        return None
    return ranges[-1]


def extract_metric(row: Dict[str, str], metric: str) -> Optional[str]:
    metric_name = find_field(row, ["Metric Name", "Metric Name"])
    if metric_name == metric:
        return find_field(row, ["Metric Value", "Value"])

    for key, value in row.items():
        if key == metric or metric in key:
            return value
    return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def parse_ncu_raw(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path)
    totals: Dict[str, Dict[str, float]] = {}
    weights: Dict[str, Dict[str, float]] = {}
    for row in rows:
        range_names = extract_ranges(row)
        if not range_names:
            continue
        duration = parse_float(row.get(DURATION_METRIC)) or 1.0
        if duration <= 0:
            duration = 1.0
        for metric in METRICS:
            value = parse_float(extract_metric(row, metric))
            if value is None:
                continue
            for range_name in range_names:
                totals.setdefault(range_name, {}).setdefault(metric, 0.0)
                weights.setdefault(range_name, {}).setdefault(metric, 0.0)
                totals[range_name][metric] += value * duration
                weights[range_name][metric] += duration

    by_range: Dict[str, Dict[str, str]] = {}
    for range_name, metric_totals in totals.items():
        by_range[range_name] = {}
        for metric, total in metric_totals.items():
            weight = weights[range_name].get(metric, 0.0)
            if weight > 0:
                by_range[range_name][metric] = f"{total / weight:.6g}"
    return by_range


def merge(iter_csv: Path, raw_csv: Path, merged_csv: Path) -> None:
    iter_rows = read_csv_rows(iter_csv)
    ncu_by_range = parse_ncu_raw(raw_csv)
    fieldnames = [
        "algorithm",
        "dataset",
        "source",
        "run",
        "iteration",
        "range_name",
        "active_vertices",
        "active_edges",
        "input_frontier",
        "output_frontier",
        "elapsed_ms",
        "achieved_occupancy",
        "ipc",
    ]
    with merged_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in iter_rows:
            metrics = ncu_by_range.get(row["range_name"], {})
            out = dict(row)
            out["achieved_occupancy"] = metrics.get(METRICS[0], "")
            out["ipc"] = metrics.get(METRICS[1], "")
            writer.writerow(out)


def raw_csv_path(report_path: Path) -> Path:
    return report_path.with_name(report_path.name + "_raw.csv")


def build_app_command(
    args: argparse.Namespace,
    algorithm: str,
    dataset: str,
    iter_csv: Path,
) -> List[str]:
    graph = args.data_dir / f"{dataset}.gr"
    binary = args.build_dir / "bin" / algorithm
    command = [
        str(binary),
        "-m",
        str(graph),
        "-n",
        "1",
    ]
    if algorithm == "bfs":
        command.extend(["-s", str(BFS_SOURCES[dataset])])
    else:
        command.extend(["--max_iterations", str(args.max_iterations)])
    command.extend(["--iter_profile", str(iter_csv)])
    return command


def build_experiment_command(
    args: argparse.Namespace,
    algorithm: str,
    dataset: str,
    iter_csv: Path,
    report_path: Path,
) -> List[str]:
    return [
        args.ncu,
        "--nvtx",
        "--metrics",
        ",".join(METRICS),
        "--csv",
        "--page",
        "raw",
        "-o",
        str(report_path),
    ] + build_app_command(args, algorithm, dataset, iter_csv)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{Path(args.ncu).parent}:{env.get('PATH', '')}"

    failures = 0
    for dataset in args.datasets:
        graph = args.data_dir / f"{dataset}.gr"
        if not graph.exists():
            print(f"missing graph: {graph}", file=sys.stderr)
            failures += 1
            continue
        for algorithm in args.algorithms:
            stem = f"{algorithm}_{dataset}"
            iter_csv = args.out_dir / f"{stem}_iterations.csv"
            ncu_iter_csv = args.out_dir / f"{stem}_ncu_iterations.csv"
            report_path = args.out_dir / f"{stem}_ncu"
            raw_csv = raw_csv_path(report_path)
            merged_csv = args.out_dir / f"{stem}_merged.csv"

            plain_command = build_app_command(args, algorithm, dataset, iter_csv)
            plain_status = run_command(plain_command, env, args.dry_run)
            if plain_status != 0:
                print(f"plain run failed for {algorithm}/{dataset}", file=sys.stderr)
                failures += 1
                continue

            command = build_experiment_command(args, algorithm, dataset, ncu_iter_csv, report_path)
            status = run_ncu_command(command, env, raw_csv, args.dry_run)
            if args.dry_run:
                continue
            if status != 0:
                if not iter_csv.exists():
                    print(
                        f"rerunning without ncu to preserve iteration CSV for {algorithm}/{dataset}",
                        file=sys.stderr,
                    )
                    run_command(build_app_command(args, algorithm, dataset, iter_csv), env, False)
                print(
                    f"ncu failed for {algorithm}/{dataset}; iteration CSV left at {iter_csv}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            if not raw_csv.exists():
                print(f"ncu CSV not found for {algorithm}/{dataset}: {raw_csv}", file=sys.stderr)
                failures += 1
                continue
            merge(iter_csv, raw_csv, merged_csv)
            print(f"merged: {merged_csv}", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
