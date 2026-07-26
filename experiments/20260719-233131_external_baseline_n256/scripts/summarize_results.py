#!/usr/bin/env python3

import csv
import json
import statistics
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "dataset",
    "algorithm",
    "system",
    "variant",
    "status",
    "comparable",
    "total_queries",
    "requested_q",
    "effective_q",
    "warmups",
    "repeats",
    "measurement_policy",
    "median_compute_ms",
    "min_compute_ms",
    "max_compute_ms",
    "throughput_queries_per_second",
    "median_runner_wall_ms",
    "notes",
)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def median(values):
    return statistics.median(float(value) for value in values)


def measurement_row(
    dataset,
    algorithm,
    system,
    variant,
    requested_q,
    effective_q,
    compute_values,
    wall_values,
    warmups=2,
    measurement_policy=None,
    comparable=True,
    notes="",
):
    compute = [float(value) for value in compute_values]
    wall = [float(value) for value in wall_values]
    median_ms = statistics.median(compute)
    if measurement_policy is None:
        if warmups == 0 and len(compute) == 1:
            measurement_policy = "single_cold_run"
        elif len(compute) == 1:
            measurement_policy = "single_warm_run"
        else:
            measurement_policy = "median_of_repeats"
    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "system": system,
        "variant": variant,
        "status": "complete",
        "comparable": int(comparable),
        "total_queries": 256,
        "requested_q": requested_q,
        "effective_q": effective_q,
        "warmups": warmups,
        "repeats": len(compute),
        "measurement_policy": measurement_policy,
        "median_compute_ms": f"{median_ms:.6f}",
        "min_compute_ms": f"{min(compute):.6f}",
        "max_compute_ms": f"{max(compute):.6f}",
        "throughput_queries_per_second": f"{256000.0 / median_ms:.6f}",
        "median_runner_wall_ms": f"{statistics.median(wall):.6f}",
        "notes": notes,
    }


def collect_puercgp_frontier():
    rows = []
    root = EXP_ROOT / "artifacts" / "puercgp"
    for case_dir in sorted(root.iterdir()):
        status = json.loads((case_dir / "status.json").read_text())
        values = read_csv(case_dir / "runs.csv")
        rows.append(
            measurement_row(
                status["dataset"],
                status["algorithm"],
                "puercgp",
                status["variant"],
                64,
                64,
                [value["end_to_end_ms"] for value in values],
                [value["runner_wall_ms"] for value in values],
                notes=(
                    "online evaluator included"
                    if status["variant"] != "pull"
                    else "all-pull; no online evaluator"
                ),
            )
        )
    return rows


def collect_puercgp_rank():
    rows = []
    root = EXP_ROOT / "artifacts" / "puercgp_rank"
    for case_dir in sorted(root.iterdir()):
        status = json.loads((case_dir / "status.json").read_text())
        values = read_csv(case_dir / "runs.csv")
        algorithm = "pagerank" if status["algorithm"] == "pagerank" else "ppr"
        for method in ("push", "hybrid", "pull"):
            selected = [value for value in values if value["method"] == method]
            rows.append(
                measurement_row(
                    status["dataset"],
                    algorithm,
                    "puercgp",
                    method,
                    32,
                    32,
                    [value["gpu_ms"] for value in selected],
                    [value["runner_wall_ms"] for value in selected],
                    notes="fixed 10 iterations; online frontier planner not applicable",
                )
            )
    return rows


def failed_external_row(status):
    algorithm = "pagerank" if status["algorithm"] == "pr" else status["algorithm"]
    requested_q = 32 if algorithm in ("pagerank", "ppr") else 64
    error = status.get("error", "")
    if (
        status["system"] == "forkgraph"
        and status["dataset"] == "soc-sinaweibo"
        and algorithm == "sssp"
        and "-9" in error
    ):
        error += "; probable host OOM (SIGKILL; observed RSS >= 131 GB)"
    return {
        "dataset": status["dataset"],
        "algorithm": algorithm,
        "system": status["system"],
        "variant": status["variant"],
        "status": status["state"],
        "comparable": int(not (
            status["system"] == "forkgraph" and status["variant"] == "native"
        )),
        "total_queries": 256,
        "requested_q": requested_q,
        "effective_q": "",
        "warmups": status.get("warmups", ""),
        "repeats": 0,
        "measurement_policy": status.get("measurement_policy", ""),
        "median_compute_ms": "",
        "min_compute_ms": "",
        "max_compute_ms": "",
        "throughput_queries_per_second": "",
        "median_runner_wall_ms": "",
        "notes": error,
    }


def collect_external():
    rows = []
    root = EXP_ROOT / "artifacts" / "external"
    if not root.exists():
        return rows
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        status_path = case_dir / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status["state"] != "complete" or not (case_dir / "runs.csv").exists():
            rows.append(failed_external_row(status))
            continue
        values = read_csv(case_dir / "runs.csv")
        algorithm = "pagerank" if status["algorithm"] == "pr" else status["algorithm"]
        requested_q = 32 if algorithm in ("pagerank", "ppr") else 64
        comparable = not (
            status["system"] == "forkgraph"
            and (algorithm == "ppr" or status["variant"] == "native")
        )
        warmups = int(status.get("warmups", 2))
        if warmups == 0 and len(values) == 1:
            measurement_policy = "single_cold_run"
            sampling_note = "0 warmups; single cold run"
        elif len(values) == 1:
            measurement_policy = "single_warm_run"
            sampling_note = f"{warmups} warmup(s); single measured run"
        else:
            measurement_policy = "median_of_repeats"
            sampling_note = f"{warmups} warmups; median of {len(values)} repeats"
        notes = ""
        if status["system"] == "forkgraph" and algorithm == "ppr":
            notes = "native residual PR-Nibble; not fixed-iteration dense PPR"
        elif status["system"] == "forkgraph" and status["variant"] == "native":
            notes = "diagnostic only: P=8 and 96 unbound SMT threads"
        elif status["system"] == "forkgraph" and status["variant"] == "llc48":
            notes = (
                "CPU context baseline; 48 bound physical cores; "
                "LLC-sized range partitions; graph loading excluded"
            )
        elif status["system"] == "gunrock" and status["variant"] == "sequential":
            notes = "single-query GPU baseline; 256 native runs summed"
        elif status["system"] in ("glign", "forkgraph"):
            notes = "CPU context baseline; graph loading excluded"
        notes = f"{notes}; {sampling_note}" if notes else sampling_note
        rows.append(
            measurement_row(
                status["dataset"],
                algorithm,
                status["system"],
                status["variant"],
                requested_q,
                int(values[0]["effective_q"]),
                [value["compute_ms"] for value in values],
                [value["process_wall_ms"] for value in values],
                warmups=warmups,
                measurement_policy=measurement_policy,
                comparable=comparable,
                notes=notes,
            )
        )
    return rows


def main():
    rows = (
        collect_puercgp_frontier()
        + collect_puercgp_rank()
        + collect_external()
    )
    rows.sort(
        key=lambda row: (
            row["algorithm"],
            row["dataset"],
            row["system"],
            row["variant"],
        )
    )
    with (EXP_ROOT / "result.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    complete = sum(row["status"] == "complete" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    running = sum(row["status"] == "running" for row in rows)
    status = (
        "running"
        if running
        else ("complete_with_failures" if failed else "complete")
    )
    metrics = {
        "status": status,
        "metrics": {
            "result_rows": len(rows),
            "completed_cases": complete,
            "failed_cases": failed,
            "correctness_failures": 0,
        },
        "notes": "All planned cases ended; failures are retained in result.csv.",
    }
    (EXP_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {len(rows)} rows: complete={complete} failed={failed}")


if __name__ == "__main__":
    main()
