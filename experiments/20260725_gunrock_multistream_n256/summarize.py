#!/usr/bin/env python3

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_CSV = ROOT.parent / "20260719-233131_external_baseline_n256" / "result.csv"
DATASETS = ("cit-Patents", "soc-orkut", "soc-twitter", "soc-sinaweibo")
ALGORITHMS = ("bfs", "sssp", "pagerank")


baseline_variants = {"bfs": "streams1", "sssp": "sequential", "pagerank": "sequential"}
baselines = {}
with BASELINE_CSV.open(newline="") as stream:
    for row in csv.DictReader(stream):
        algorithm = row["algorithm"]
        if (
            row["system"] == "gunrock"
            and algorithm in baseline_variants
            and row["variant"] == baseline_variants[algorithm]
            and row["status"] == "complete"
        ):
            baselines[(algorithm, row["dataset"])] = row


rows = []
for algorithm in ALGORITHMS:
    for dataset in DATASETS:
        path = ROOT / "artifacts" / dataset / algorithm / "summary.json"
        if not path.exists():
            rows.append(
                {"algorithm": algorithm, "dataset": dataset, "status": "missing"}
            )
            continue
        summary = json.loads(path.read_text())
        baseline = baselines.get((algorithm, dataset), {})
        baseline_ms = baseline.get("median_compute_ms", "")
        median_ms = summary.get("median_ms", "")
        speedup = ""
        if baseline_ms != "" and median_ms != "":
            speedup = float(baseline_ms) / float(median_ms)
        rows.append(
            {
                "algorithm": algorithm,
                "dataset": dataset,
                "status": summary["status"],
                "N": summary.get("total_queries", 256),
                "Q": summary.get("effective_q", ""),
                "attempted_Q": ";".join(str(item["q"]) for item in summary.get("attempts", [])),
                "iterations": summary.get("pagerank_iterations") or "",
                "median_ms": median_ms,
                "min_ms": summary.get("min_ms", ""),
                "max_ms": summary.get("max_ms", ""),
                "baseline_variant": baseline.get("variant", ""),
                "baseline_ms": baseline_ms,
                "speedup_vs_baseline": speedup,
            }
        )

fields = (
    "algorithm",
    "dataset",
    "status",
    "N",
    "Q",
    "attempted_Q",
    "iterations",
    "median_ms",
    "min_ms",
    "max_ms",
    "baseline_variant",
    "baseline_ms",
    "speedup_vs_baseline",
)
with (ROOT / "results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(",".join(fields))
for row in rows:
    print(",".join(str(row.get(field, "")) for field in fields))
