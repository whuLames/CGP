#!/usr/bin/env python3
"""Summarize an A (batch) versus C (group-refill) GraphWeft campaign."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


WORKLOADS = ("bfs", "sssp", "mixed_tail")
VARIANTS = ("A", "C")
METRICS = (
    "workload_ms", "throughput_qps", "kernel_gpu_ms", "kernel_ms", "frontier_ms",
    "copy_ms", "feature_ms", "selector_ms", "initialization_ms", "recycle_ms",
    "rounds", "push_rounds", "pull_rounds", "active_slot_ratio", "group_refills",
    "group_member_wait_ms_mean", "latency_ms_mean", "latency_ms_p50",
    "latency_ms_p95", "latency_ms_p99", "latency_ms_max", "waiting_ms_p50",
    "service_ms_p50", "short_latency_ms_p50", "long_latency_ms_p50",
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, preferred=()):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    fields = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value(row, metric):
    raw = row.get(metric + "_median", "")
    return float(raw) if raw not in ("", None) else None


def geomean(values):
    values = [x for x in values if x is not None and x > 0]
    return math.exp(statistics.fmean(math.log(x) for x in values)) if values else None


def classify(runtime_speedup, latency_speedup, round_speedup):
    if runtime_speedup > 1.03 and latency_speedup >= .97:
        return "performance improvement"
    if (runtime_speedup - 1) * (latency_speedup - 1) < 0:
        return "latency-throughput trade-off"
    if round_speedup > 1.03 and runtime_speedup <= 1.03:
        return "logical-round reduction only"
    return "no improvement/regression"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    config = json.loads((root / "config.json").read_text())
    datasets = tuple(config["datasets"])
    repetitions = int(config["formal_repetitions"])

    summaries = {}
    raw_count = 0
    result_rows = []
    for dataset in datasets:
        rows = read_csv(root / dataset / "summary.csv")
        raw_count += len(read_csv(root / dataset / "raw_measurements.csv"))
        for row in rows:
            summaries[(dataset, row["workload"], row["variant"])] = row
            result_rows.append({"dataset": dataset, **row})
    write_csv(root / "result.csv", result_rows, ("dataset", "workload", "variant", "samples"))

    comparisons = []
    for dataset in datasets:
        for workload in WORKLOADS:
            before = summaries[(dataset, workload, "A")]
            after = summaries[(dataset, workload, "C")]
            item = {"dataset": dataset, "workload": workload, "comparison": "A_to_C"}
            for metric in METRICS:
                a, c = value(before, metric), value(after, metric)
                item["A_" + metric] = a
                item["C_" + metric] = c
                item[metric + "_A_over_C"] = a / c if a is not None and c not in (None, 0) else None
                item[metric + "_C_over_A"] = c / a if c is not None and a not in (None, 0) else None
            item["classification"] = classify(
                item["workload_ms_A_over_C"], item["latency_ms_p50_A_over_C"],
                item["rounds_A_over_C"])
            comparisons.append(item)
    write_csv(root / "refill_comparison.csv", comparisons,
              ("dataset", "workload", "comparison", "classification"))

    geomeans = []
    for scope in (*WORKLOADS, "all"):
        selected = [row for row in comparisons if scope == "all" or row["workload"] == scope]
        item = {"scope": scope, "cases": len(selected)}
        for metric in METRICS:
            for direction in ("A_over_C", "C_over_A"):
                key = metric + "_" + direction
                item[key + "_geomean"] = geomean([row.get(key) for row in selected])
        geomeans.append(item)
    write_csv(root / "refill_geomeans.csv", geomeans, ("scope", "cases"))

    expected_cases = len(datasets) * len(WORKLOADS) * len(VARIANTS)
    expected_samples = expected_cases * repetitions
    complete_cases = sum(int(row.get("samples", 0)) == repetitions for row in result_rows)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({"test_cases_complete": complete_cases, "test_cases_expected": expected_cases,
                    "formal_samples_found": raw_count, "formal_samples_expected": expected_samples})
    metrics["status"] = "success" if complete_cases == expected_cases and raw_count == expected_samples else "incomplete"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    overall = next(row for row in geomeans if row["scope"] == "all")
    class_counts = {name: sum(row["classification"] == name for row in comparisons)
                    for name in sorted({row["classification"] for row in comparisons})}
    report = [f"# GraphWeft G={config['G']} refill campaign", "", "## Scope", "",
              f"Datasets: {', '.join(datasets)}. Workloads: BFS, SSSP, Mixed-tail. "
              f"N={config['N']}, M={config['M']}, group width={config['G']}; Hybrid threshold 0.20, "
              f"direct frontier and bitmask64 are fixed. A disables refill and C enables refill. "
              f"Each case has one warmup and {repetitions} formal samples.", "", "## Acceptance", "",
              f"- Complete test cases: {complete_cases}/{expected_cases}",
              f"- Formal samples: {raw_count}/{expected_samples}", "", "## Overall", "",
              f"- Refill throughput speedup (C/A): {overall['throughput_qps_C_over_A_geomean']:.3f}x",
              f"- Refill P50 latency speedup (A/C latency): {overall['latency_ms_p50_A_over_C_geomean']:.3f}x",
              f"- Refill logical-round reduction (C/A rounds): {overall['rounds_C_over_A_geomean']:.3f}x",
              f"- Classifications: {class_counts}", "", "## By workload", "",
              "| Workload | Throughput C/A | P50 speedup A/C | Rounds C/A |",
              "|---|---:|---:|---:|",
              *[f"| {row['scope']} | {row['throughput_qps_C_over_A_geomean']:.3f}x | "
                f"{row['latency_ms_p50_A_over_C_geomean']:.3f}x | "
                f"{row['rounds_C_over_A_geomean']:.3f}x |"
                for row in geomeans if row["scope"] != "all"],
              "", "## Outputs", "",
              f"- `result.csv`: median/min/max/stdev/range for all {expected_cases} cases.",
              "- `refill_comparison.csv`: per-dataset real milliseconds, QPS, latency, rounds, and phase ratios.",
              "- `refill_geomeans.csv`: per-workload and overall geometric means.",
              "- `<dataset>/raw/`: exact commands, logs, completion CSVs, and validation fingerprints.", ""]
    (root / "README.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
