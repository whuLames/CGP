#!/usr/bin/env python3
"""Summarize GE-SpMM group-slots CSV artifacts."""

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


DATASETS = [
    "cit-Patents",
    "soc-orkut",
    "soc-twitter",
    "soc-sinaweibo",
    "roadNet-CA",
]


def initialize_experiment(args):
    experiment_dir = args.experiment_dir.resolve()
    (experiment_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    start_time = datetime.now().astimezone().isoformat()
    config = {
        "name": "ge_spmm_pull_group_slots",
        "purpose": (
            "Compare full-GPU Q64 pull GE-SpMM with Green Context 2xQ32"
        ),
        "start_time": start_time,
        "git_commit": args.git_commit,
        "gpu": args.gpu,
        "data_root": args.data_root,
        "datasets": DATASETS,
        "parameters": {
            "queries": 64,
            "groups": 2,
            "queries_per_group": 32,
            "tile_row": args.tile_row,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "reduction": "min",
            "round_barrier": True,
        },
        "baseline": "one Q64 gather-min kernel on the full GPU",
        "candidate": (
            "two Q32 gather-min kernels on disjoint Green Context streams"
        ),
        "binary": "src/kernels/ge_spmm_group_slots_bench",
    }
    with (experiment_dir / "config.json").open("w") as destination:
        json.dump(config, destination, indent=2)
        destination.write("\n")
    with (experiment_dir / "metrics.json").open("w") as destination:
        json.dump(
            {
                "status": "running",
                "metrics": {},
                "notes": "Benchmark is in progress.",
            },
            destination,
            indent=2,
        )
        destination.write("\n")

    readme = f"""# Experiment: GE-SpMM pull group-slots validation

## Purpose

Compare a single 64-query gather-min GE-SpMM kernel with two concurrent
32-query kernels assigned to disjoint Green Context SM partitions.

## Hypothesis

Splitting the independent query dimension can reduce pull-kernel makespan on
graphs whose degree distribution creates a long execution tail.

## Dataset

- Names: {", ".join(DATASETS)}
- Source: `{args.data_root}`
- Representation: original CSR; int32 row offsets are promoted to int64
- Split and preprocessing: none

## Scenario

- Task: pure pull gather-min SpMM
- Workload: 64 deterministic finite float features per vertex
- Baseline: one Q64 kernel on the full GPU
- Candidate: two concurrent Q32 kernels on disjoint Green Context streams
- Synchronization: every stream synchronizes before and after each repetition
- Excluded: BFS state, frontier logic, compaction, and postprocessing

## Environment

- GPU ordinal: {args.gpu}
- CUDA compiler: `{args.nvcc}`
- Git commit: `{args.git_commit}`

## Command

```bash
GPU_ID={args.gpu} WARMUP={args.warmup} ITERS={args.iterations} \\
TILE_ROW={args.tile_row} SEED={args.seed} \\
src/kernels/run_ge_spmm_group_slots.sh
```

## Results

See `result.csv` and `metrics.json`.

## Observations

Inspect `stdout.log`, `stderr.log`, and per-dataset CSV files under
`artifacts/`.

## Conclusion

Pending interpretation of the generated metrics.

## Issues

Failed datasets, if any, are recorded in `artifacts/status.tsv`.

## Next Steps

Use the cross-dataset pattern to decide whether degree-tail profiling is
warranted.
"""
    (experiment_dir / "README.md").write_text(readme)
    (experiment_dir / "stdout.log").touch()
    (experiment_dir / "stderr.log").touch()


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values):
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "stddev_ms": statistics.pstdev(values),
    }


def load_statuses(path):
    statuses = {}
    if not path.exists():
        return statuses
    with path.open(newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            statuses[row["dataset"]] = {
                "status": row["status"],
                "exit_code": int(row["exit_code"]),
            }
    return statuses


def load_dataset(path):
    rows = []
    with path.open(newline="") as source:
        rows.extend(csv.DictReader(source))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(float(row["wall_ms"]))
    if set(grouped) != {"q64", "green_2x32"}:
        raise ValueError(f"{path}: expected q64 and green_2x32 rows")
    first = rows[0]
    q64 = describe(grouped["q64"])
    green = describe(grouped["green_2x32"])
    return {
        "V": int(first["V"]),
        "E": int(first["E"]),
        "q64": q64,
        "green_2x32": green,
        "speedup": q64["median_ms"] / green["median_ms"],
        "fingerprint_match": all(
            row["fingerprint_match"] == "1" for row in rows
        ),
    }


def finalize_readme(experiment_dir, datasets):
    readme_path = experiment_dir / "README.md"
    if not readme_path.exists():
        return
    successful = {
        name: value
        for name, value in datasets.items()
        if value["status"] == "success"
    }
    table = [
        "The primary metric is median per-round host makespan. A speedup above "
        "1 means Green 2xQ32 is faster.",
        "",
        "| Dataset | Q64 (ms) | Green 2xQ32 (ms) | Speedup | Output |",
        "|---|---:|---:|---:|---|",
    ]
    for name in DATASETS:
        value = datasets[name]
        if value["status"] == "success":
            table.append(
                f"| {name} | {value['q64']['median_ms']:.3f} | "
                f"{value['green_2x32']['median_ms']:.3f} | "
                f"{value['speedup']:.3f}x | "
                f"{'match' if value['fingerprint_match'] else 'mismatch'} |"
            )
        else:
            table.append(f"| {name} | - | - | - | {value['status']} |")
    faster = [
        name for name, value in successful.items() if value["speedup"] > 1.0
    ]
    observations = (
        f"Green 2xQ32 is faster on {', '.join(faster)}. "
        if faster
        else "Green 2xQ32 is not faster on any successful dataset. "
    )
    observations += (
        f"{len(successful)}/{len(DATASETS)} datasets completed successfully; "
        "per-round distributions are retained under `artifacts/`."
    )
    conclusion = (
        "The isolated pull-kernel comparison reproduces the cross-dataset "
        "performance pattern without BFS state or asynchronous iteration "
        "progress. Further profiling is required to attribute the difference "
        "to degree-tail load imbalance."
    )

    readme = readme_path.read_text()
    readme = readme.replace(
        "See `result.csv` and `metrics.json`.", "\n".join(table)
    )
    readme = readme.replace(
        "Inspect `stdout.log`, `stderr.log`, and per-dataset CSV files under\n"
        "`artifacts/`.",
        observations,
    )
    readme = readme.replace(
        "Pending interpretation of the generated metrics.", conclusion
    )
    readme_path.write_text(readme)


def format_value(value):
    return "" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--tile-row", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="/home/zyl/data/csr_data")
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()

    if args.initialize:
        initialize_experiment(args)
        return 0

    experiment_dir = args.experiment_dir.resolve()
    artifacts = experiment_dir / "artifacts"
    statuses = load_statuses(artifacts / "status.tsv")
    datasets = {}
    error_count = 0

    for name in DATASETS:
        artifact = artifacts / f"{name}.csv"
        status = statuses.get(
            name, {"status": "missing", "exit_code": -1}
        )
        if status["exit_code"] == 0 and artifact.exists():
            try:
                datasets[name] = {
                    **load_dataset(artifact),
                    "status": "success",
                }
            except (OSError, ValueError) as error:
                datasets[name] = {
                    "status": "summary_error",
                    "error": str(error),
                }
                error_count += 1
        else:
            datasets[name] = {
                "status": status["status"],
                "exit_code": status["exit_code"],
            }
            error_count += 1

    result_path = experiment_dir / "result.csv"
    with result_path.open("w", newline="") as destination:
        fields = [
            "dataset",
            "V",
            "E",
            "q64_median_ms",
            "green_2x32_median_ms",
            "speedup",
            "fingerprint_match",
            "status",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for name in DATASETS:
            result = datasets[name]
            writer.writerow(
                {
                    "dataset": name,
                    "V": result.get("V"),
                    "E": result.get("E"),
                    "q64_median_ms": format_value(
                        result.get("q64", {}).get("median_ms")
                    ),
                    "green_2x32_median_ms": format_value(
                        result.get("green_2x32", {}).get("median_ms")
                    ),
                    "speedup": format_value(result.get("speedup")),
                    "fingerprint_match": result.get(
                        "fingerprint_match", ""
                    ),
                    "status": result["status"],
                }
            )

    successful = [
        value
        for value in datasets.values()
        if value["status"] == "success"
    ]
    metrics = {
        "status": "success" if error_count == 0 else "partial_failure",
        "completed_at": datetime.now().astimezone().isoformat(),
        "metrics": {
            "dataset_count": len(DATASETS),
            "successful_dataset_count": len(successful),
            "error_count": error_count,
            "median_speedup_across_successes": (
                statistics.median(item["speedup"] for item in successful)
                if successful
                else None
            ),
        },
        "datasets": datasets,
        "notes": (
            "Primary latency is per-round host wall makespan with a barrier "
            "before and after every repetition."
        ),
    }
    with (experiment_dir / "metrics.json").open("w") as destination:
        json.dump(metrics, destination, indent=2)
        destination.write("\n")
    finalize_readme(experiment_dir, datasets)

    print(result_path.read_text(), end="")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
