#!/usr/bin/env python3
"""Calibrate ForkGraph OpenMP placement with one fixed Q=64 BFS batch."""

from __future__ import annotations

import csv
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
FORKGRAPH_ROOT = Path("/home/zyl/Projects/ocgp/baselines/ForkGraph")
DATASET = "cit-Patents"
OUTPUT = EXP_ROOT / "artifacts" / "forkgraph_thread_calibration"
WARMUPS = 1
REPEATS = 3

CONFIGS = {
    "physical24_bound": {
        "OMP_NUM_THREADS": "24",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "spread",
    },
    "physical32_bound": {
        "OMP_NUM_THREADS": "32",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "spread",
    },
    "physical40_bound": {
        "OMP_NUM_THREADS": "40",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "spread",
    },
    "physical48_bound": {
        "OMP_NUM_THREADS": "48",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "spread",
    },
    "q64_bound": {
        "OMP_NUM_THREADS": "64",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "spread",
    },
    "logical96_bound": {
        "OMP_NUM_THREADS": "96",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "spread",
    },
    "logical96_unbound": {
        "OMP_NUM_THREADS": "96",
    },
}


def command() -> list[str]:
    data = FORKGRAPH_ROOT / "inputs_gr_partitioned"
    query_file = EXP_ROOT / "query_sets" / DATASET / "q64_batch0.sources"
    return [
        str(FORKGRAPH_ROOT / "build" / "bfs"),
        "-p",
        "8",
        "-gr",
        str(query_file),
        str(data / f"{DATASET}.intra.gr"),
        str(data / f"{DATASET}.inter.gr"),
        str(data / f"{DATASET}.part"),
    ]


def run_once(config_name: str, run_kind: str, run_index: int) -> dict:
    env = os.environ.copy()
    for key in ("OMP_PLACES", "OMP_PROC_BIND"):
        env.pop(key, None)
    env["OMP_DYNAMIC"] = "FALSE"
    env.update(CONFIGS[config_name])

    run_dir = OUTPUT / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    proc = subprocess.run(
        command(),
        cwd=FORKGRAPH_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1800,
        check=False,
    )
    wall_ms = (time.monotonic() - start) * 1000.0
    stem = f"{run_kind}-{run_index}"
    (run_dir / f"{stem}.stdout.log").write_text(proc.stdout)
    (run_dir / f"{stem}.stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{config_name} {stem} exited with {proc.returncode}")

    match = re.search(r"^exuection time:\s+([0-9.]+)\s+ms$", proc.stdout, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{config_name} {stem} did not report execution time")
    compute_ms = float(match.group(1))
    print(
        f"PROGRESS {config_name} {stem} compute_ms={compute_ms:.3f} "
        f"wall_ms={wall_ms:.3f}",
        flush=True,
    )
    return {
        "config": config_name,
        "kind": run_kind,
        "index": run_index,
        "compute_ms": compute_ms,
        "wall_ms": wall_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIGS,
        default=list(CONFIGS),
    )
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def main() -> int:
    global OUTPUT
    args = parse_args()
    output = OUTPUT if not args.output_tag else OUTPUT.parent / args.output_tag
    OUTPUT = output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Rotate configuration order between rounds to reduce temporal bias.
    names = args.configs
    for warmup in range(args.warmups):
        for name in names:
            rows.append(run_once(name, "warmup", warmup))
    for repeat in range(args.repeats):
        order = names[repeat:] + names[:repeat]
        for name in order:
            rows.append(run_once(name, "repeat", repeat))

    with (OUTPUT / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for name in names:
        samples = [
            row["compute_ms"]
            for row in rows
            if row["config"] == name and row["kind"] == "repeat"
        ]
        summary[name] = {
            "environment": {"OMP_DYNAMIC": "FALSE", **CONFIGS[name]},
            "median_compute_ms": statistics.median(samples),
            "samples_compute_ms": samples,
        }
    best = min(summary, key=lambda name: summary[name]["median_compute_ms"])
    summary["best"] = best
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
