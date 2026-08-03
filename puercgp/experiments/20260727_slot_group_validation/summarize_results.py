#!/usr/bin/env python3
"""Collect slot-group benchmark logs into reproducible CSV tables and plots."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def parse_key_values(line: str, separator: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.strip().split(separator):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def config_name(path: Path, method: str) -> str:
    name = path.name
    if "group4_green" in name:
        return "group_green_4x16"
    if method == "group_green":
        return "group_green_2x32"
    if method == "group_stream":
        return "group_stream_2x32"
    if method == "group_serial":
        return "group_serial_2x32"
    return method


def parse_benchmark(path: Path) -> dict[str, object]:
    lines = path.read_text().splitlines()
    header = next(parse_key_values(line, " ") for line in lines
                  if line.startswith("graph="))
    result = next(parse_key_values(line, ",") for line in lines
                  if line.startswith("RESULT,"))
    method = result["method"]
    return {
        "dataset": Path(header["graph"]).name,
        "n": int(header["N"]),
        "mode": header["mode"],
        "config": config_name(path, method),
        "median_host_ms": float(result["median_host_ms"]),
        "median_engine_ms": float(result["median_engine_ms"]),
        "qps": float(result["qps"]),
        "p50_ms": float(result["p50_ms"]),
        "p95_ms": float(result["p95_ms"]),
        "p99_ms": float(result["p99_ms"]),
        "measured_repeats": sum(line.startswith("run,") for line in lines),
        "log": str(path.relative_to(ROOT)),
    }


def add_speedups(records: list[dict[str, object]]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["n"], record["mode"])].append(record)
    for group in groups.values():
        by_config = {str(row["config"]): row for row in group}
        cohort = by_config.get("cohort64")
        large_batches = [by_config[key] for key in ("static64", "cohort64")
                         if key in by_config]
        best_large_ms = min(float(row["median_host_ms"])
                            for row in large_batches)
        for row in group:
            elapsed = float(row["median_host_ms"])
            row["speedup_vs_cohort64"] = (
                float(cohort["median_host_ms"]) / elapsed if cohort else ""
            )
            row["speedup_vs_best_large_batch"] = best_large_ms / elapsed


def write_benchmark_csv(name: str, records: list[dict[str, object]]) -> None:
    add_speedups(records)
    records.sort(key=lambda row: (
        str(row["dataset"]), int(row["n"]), str(row["mode"]),
        str(row["config"])))
    fields = [
        "dataset", "n", "mode", "config", "median_host_ms",
        "median_engine_ms", "qps", "p50_ms", "p95_ms", "p99_ms",
        "speedup_vs_cohort64", "speedup_vs_best_large_batch",
        "measured_repeats", "log",
    ]
    with (RESULTS / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_ncu() -> list[dict[str, object]]:
    selected = {
        "cit": ROOT / "ncu/cit_pull_q32_80sm_v2.csv",
        "orkut": ROOT / "ncu/orkut_pull_q32_80sm.csv",
        "twitter": ROOT / "ncu/twitter_pull_q32_80sm.csv",
        "sinaweibo": ROOT / "ncu/sina_pull_q32_80sm.csv",
    }
    steps = {"cit": 8, "orkut": 4, "twitter": 6, "sinaweibo": 4}
    records = []
    for dataset, path in selected.items():
        rows = [row for row in csv.reader(path.read_text().splitlines())
                if len(row) >= 15 and row[0] == "0"]
        metrics = {row[-3]: float(row[-1]) for row in rows}
        duration_ns = metrics["gpu__time_duration.sum"]
        read_bytes = metrics["dram__bytes_read.sum"]
        write_bytes = metrics["dram__bytes_write.sum"]
        records.append({
            "dataset": dataset,
            "bfs_step": steps[dataset],
            "query_count": 32,
            "sm_count": 80,
            "kernel_ms": duration_ns / 1.0e6,
            "dram_read_gb": read_bytes / 1.0e9,
            "dram_write_gb": write_bytes / 1.0e9,
            "effective_dram_gbps": (read_bytes + write_bytes) / duration_ns,
            "dram_peak_pct": metrics[
                "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
            "l2_peak_pct": metrics[
                "lts__throughput.avg.pct_of_peak_sustained_elapsed"],
            "sm_peak_pct": metrics[
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"],
            "long_scoreboard_stall_pct": metrics[
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"],
            "raw_profile": str(path.relative_to(ROOT)),
        })
    return records


def write_ncu_csv(records: list[dict[str, object]]) -> None:
    with (RESULTS / "ncu_pull_q32_80sm.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_plot(e2: list[dict[str, object]], e3: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]

    social = ["cit-Patents", "soc-orkut", "soc-twitter", "soc-sinaweibo"]
    configs = ["immediate64", "group_stream_2x32", "group_green_2x32"]
    labels = ["Immediate-64", "2x32 streams", "2x32 Green"]
    x = np.arange(len(social))
    width = 0.24
    for i, (config, label) in enumerate(zip(configs, labels)):
        values = [next(float(row["speedup_vs_cohort64"]) for row in e2
                       if row["dataset"] == dataset and
                       row["config"] == config) for dataset in social]
        axes[0].bar(x + (i - 1) * width, values, width, label=label,
                    color=colors[i])
    axes[0].set_xticks(x, ["cit", "orkut", "twitter", "sina"])
    axes[0].set_title("All-push, N=256")
    axes[0].set_ylabel("Speedup over cohort-64")
    axes[0].legend(fontsize=8)

    road_configs = configs + ["group_green_4x16"]
    road_labels = labels + ["4x16 Green"]
    road_x = np.arange(3)
    road_width = 0.19
    for i, (config, label) in enumerate(zip(road_configs, road_labels)):
        values = [next(float(row["speedup_vs_cohort64"]) for row in e2
                       if row["dataset"] == "roadNet-CA" and
                       row["n"] == n and row["config"] == config)
                  for n in (256, 400, 800)]
        axes[1].bar(road_x + (i - 1.5) * road_width, values, road_width,
                    label=label, color=colors[i])
    axes[1].set_xticks(road_x, ["N=256", "N=400", "N=800"])
    axes[1].set_title("roadNet-CA all-push")
    axes[1].legend(fontsize=8)

    mode_dataset = [(mode, dataset) for mode in ("pull", "hybrid")
                    for dataset in social]
    values = [next(float(row["speedup_vs_cohort64"]) for row in e3
                   if row["mode"] == mode and row["dataset"] == dataset and
                   row["config"] == "group_green_2x32")
              for mode, dataset in mode_dataset]
    bar_colors = [colors[0]] * 4 + [colors[1]] * 4
    axes[2].bar(np.arange(len(values)), values, color=bar_colors)
    axes[2].set_xticks(np.arange(len(values)),
                       ["cit", "ork", "twi", "sin"] * 2)
    axes[2].set_title("2x32 Green, pull then hybrid")

    for axis in axes:
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "slot_group_speedups.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    e2 = [parse_benchmark(path) for path in RUNS.glob("e2_*_formal.log")]
    e3 = [parse_benchmark(path) for path in RUNS.glob("e3_*_screen.log")]
    e3_formal = [parse_benchmark(path)
                 for path in RUNS.glob("e3_*_formal.log")]
    write_benchmark_csv("e2_push_formal.csv", e2)
    write_benchmark_csv("e3_pull_hybrid_screen.csv", e3)
    write_benchmark_csv("e3_twitter_formal.csv", e3_formal)
    ncu = parse_ncu()
    write_ncu_csv(ncu)
    make_plot(e2, e3)


if __name__ == "__main__":
    main()
