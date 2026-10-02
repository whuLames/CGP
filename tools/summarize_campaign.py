#!/usr/bin/env python3
"""Create a road-dataset-free audit report from the fixed experiment tree."""
import argparse
import csv
import json
from pathlib import Path


DATASETS = ("cit-Patents", "soc-LiveJournal1", "soc-orkut", "indochina", "soc-twitter")
EXPECTED = {
    "push_abcd": 20, "bfs_push_abcd": 20, "sssp_push_abcd": 20,
    "hybrid_key": 10, "oracle": 5, "g_sensitivity": 30,
    "kernel_mixed": 20, "kernel_bfs": 20, "kernel_sssp": 20,
    "kernel_calibration": 155,
}


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row, key):
    raw = row.get(key, "")
    return "-" if raw == "" else f"{float(raw):.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = ["# GraphWeft fixed N=1024, M=64 campaign", "",
           "Scope: five non-road datasets; roadNet and other road graphs are excluded.", "",
           "## Natural-tail audit", "",
           "| Dataset | Algorithm | short median rounds | long median rounds | R_tail | Strong tail |",
           "|---|---|---:|---:|---:|---|"]
    for dataset in DATASETS:
        report = json.loads((args.root / dataset / "workloads/mixed_tail_report.json").read_text())
        for algorithm in ("bfs", "sssp"):
            item = report[algorithm]
            out.append(f"| {dataset} | {algorithm.upper()} | {item['short_median']} | "
                       f"{item['long_median']} | {item['R_tail']:.3f} | {item['strong_tail']} |")
    out += ["", "`R_tail < 2` is reported as insufficient natural separation; selected longest queries are not relabeled as a strong tail.",
            "", "## Completed timing summaries", "",
            "| Dataset | Suite | Variant | samples | workload median ms | throughput median q/s | wait slot-rounds | active ratio |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    status = []
    for dataset in DATASETS:
        results = args.root / dataset / "results"
        for suite, expected in EXPECTED.items():
            if suite == "kernel_calibration" and dataset != "cit-Patents":
                continue  # One globally frozen mapping; no per-dataset tuning.
            directory = results / suite
            measured = len(list(directory.glob("rep*.log"))) if directory.exists() else 0
            status.append((dataset, suite, measured, expected, measured == expected))
            summary = directory / "summary.csv"
            if not summary.exists() or suite == "kernel_calibration":
                continue
            for row in rows(summary):
                out.append(f"| {dataset} | {suite} | {row.get('variant', row.get('mapping', '-'))} | "
                           f"{row.get('samples', '-')} | {value(row, 'workload_ms_median')} | "
                           f"{value(row, 'throughput_qps_median')} | {value(row, 'completed_slot_rounds_median')} | "
                           f"{value(row, 'active_slot_ratio_median')} |")
    out += ["", "## Campaign status", "",
            "| Dataset | Suite | measured/expected | Complete |", "|---|---|---:|---|"]
    for dataset, suite, measured, expected, complete in status:
        out.append(f"| {dataset} | {suite} | {measured}/{expected} | {complete} |")
    out += ["", "Directories containing `.invalid_` are retained for audit but are never read by this report.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out))


if __name__ == "__main__":
    main()
