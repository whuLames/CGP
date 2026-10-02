#!/usr/bin/env python3
"""Compare FIFO, predicted-grouping, and optimized strong-tail refill runs."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def geomean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def median_by_query(latencies, variant, metric):
    grouped = {}
    for row in latencies:
        if row["variant"] != variant:
            continue
        key = (row["dataset"], row["workload"], row["algorithm"],
               row["tail_label"], int(row["query_id"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: statistics.median(values) for key, values in grouped.items()}


def summarize(label, root):
    end_to_end = rows(root / "end_to_end_summary.csv")
    latencies = []
    for path in sorted(root.glob("*/query_latency.csv")):
        latencies += rows(path)
    if not latencies and (root / "query_latency.csv").exists():
        latencies = rows(root / "query_latency.csv")
    no_refill = median_by_query(latencies, "A_no_refill", "submit_to_completion_ms")
    refill = median_by_query(latencies, "C_refill", "submit_to_completion_ms")
    if no_refill.keys() != refill.keys():
        raise RuntimeError(f"{label}: latency query sets differ")
    speedups = {key: no_refill[key] / refill[key] for key in no_refill}
    long_speedups = [value for key, value in speedups.items() if key[3] == "long"]
    measurement_rows = []
    for path in sorted(root.glob("*/raw_measurements.csv")):
        measurement_rows += rows(path)
    recycle = {variant: statistics.median(
        float(row.get("recycle_ms", 0)) for row in measurement_rows if row["variant"] == variant)
        for variant in ("A_no_refill", "C_refill")}
    return {
        "label": label, "root": str(root),
        "end_to_end_geomean_refill_speedup": geomean(
            [float(row["refill_speedup"]) for row in end_to_end]),
        "all_query_submit_latency_geomean_speedup": geomean(speedups.values()),
        "long_query_submit_latency_geomean_speedup": geomean(long_speedups),
        "all_query_improved_fraction": sum(value > 1 for value in speedups.values()) / len(speedups),
        "long_query_improved_fraction": sum(value > 1 for value in long_speedups) / len(long_speedups),
        "refill_recycle_ms_median": recycle["C_refill"],
        "no_refill_recycle_ms_median": recycle["A_no_refill"],
        "fingerprints_match": json.loads((root / "metrics.json").read_text())["all_fingerprints_match"],
        "cases": [{"dataset": row["dataset"], "workload": row["workload"],
                   "refill_speedup": float(row["refill_speedup"])} for row in end_to_end],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = [summarize(label, Path(path).resolve()) for label, path in args.campaign]
    (args.output / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
    case_rows = []
    for summary in summaries:
        for case in summary.pop("cases"):
            case_rows.append({"campaign": summary["label"], **case})
    with (args.output / "case_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("campaign", "dataset", "workload", "refill_speedup"))
        writer.writeheader(); writer.writerows(case_rows)
    lines = ["# Strong-tail refill comparison", "",
             "| Campaign | E2E refill speedup | All-query latency | Long-query latency | "
             "All improved | Long improved | Recycle ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['label']} | {item['end_to_end_geomean_refill_speedup']:.3f}x | "
                     f"{item['all_query_submit_latency_geomean_speedup']:.3f}x | "
                     f"{item['long_query_submit_latency_geomean_speedup']:.3f}x | "
                     f"{item['all_query_improved_fraction']:.1%} | "
                     f"{item['long_query_improved_fraction']:.1%} | "
                     f"{item['refill_recycle_ms_median']:.3f} |")
    lines += ["", "All latency values are paired per-query medians across formal repetitions. "
              "Speedups above one favor refill."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
