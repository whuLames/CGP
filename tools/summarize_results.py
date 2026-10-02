#!/usr/bin/env python3
"""Rebuild timing summaries and audit group-wait slot-rounds from raw CSVs."""
import argparse
import csv
import statistics
from pathlib import Path


METRICS = ("workload_ms", "throughput_qps", "kernel_ms", "copy_ms", "frontier_ms",
           "planning_ms", "recycle_ms", "completed_slot_rounds", "active_slot_ratio",
           "final_drain_rounds")


def group_wait(path):
    groups = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault((int(row["group"]), int(row["activation_round"])), []).append(int(row["completion_round"]))
    return sum(max(values) * len(values) - sum(values) for values in groups.values())


def main():
    p = argparse.ArgumentParser(); p.add_argument("directory", type=Path); args = p.parse_args()
    timing = args.directory / "timings.csv"
    with timing.open(newline="") as f: rows = list(csv.DictReader(f))
    for row in rows:
        completion = args.directory / f"rep{row['repetition']}_{row['variant']}_completion.csv"
        row["completed_slot_rounds"] = str(group_wait(completion))
    keys = list(rows[0])
    with timing.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        selected = [row for row in rows if row["variant"] == variant]
        item = {"variant": variant, "samples": len(selected)}
        for metric in METRICS:
            values = sorted(float(row[metric]) for row in selected if row.get(metric, "") != "")
            if values:
                item[metric + "_median"] = statistics.median(values)
                item[metric + "_min"] = values[0]; item[metric + "_max"] = values[-1]
        summary.append(item)
    fields = ["variant", "samples"] + sorted(set().union(*(x.keys() for x in summary)) - {"variant", "samples"})
    with (args.directory / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)


if __name__ == "__main__": main()
