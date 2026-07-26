#!/usr/bin/env python3

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cit-Patents", "soc-orkut", "soc-twitter", "soc-sinaweibo")
ALGORITHMS = ("bfs", "sssp", "sswp", "pagerank", "ppr")
ORDER = {
    "bfs": (
        ("puercgp", "push_online"),
        ("puercgp", "hybrid_online"),
        ("puercgp", "pull"),
        ("ibfs", "native"),
        ("glign", "native"),
        ("forkgraph", "llc48"),
        ("gunrock", "streams1"),
        ("gunrock", "streams64"),
    ),
    "sssp": (
        ("puercgp", "push_online"),
        ("puercgp", "hybrid_online"),
        ("puercgp", "pull"),
        ("glign", "native"),
        ("forkgraph", "llc48"),
        ("gunrock", "sequential"),
    ),
    "sswp": (
        ("puercgp", "push_online"),
        ("puercgp", "hybrid_online"),
        ("puercgp", "pull"),
        ("glign", "native"),
    ),
    "pagerank": (
        ("puercgp", "push"),
        ("puercgp", "hybrid"),
        ("puercgp", "pull"),
        ("gunrock", "sequential"),
    ),
    "ppr": (
        ("puercgp", "push"),
        ("puercgp", "hybrid"),
        ("puercgp", "pull"),
    ),
}


def label(key: tuple[str, str]) -> str:
    system, variant = key
    aliases = {
        ("puercgp", "push_online"): "Puer push+online",
        ("puercgp", "hybrid_online"): "Puer hybrid+online",
        ("puercgp", "pull"): "Puer pull",
        ("puercgp", "push"): "Puer push",
        ("puercgp", "hybrid"): "Puer hybrid",
        ("ibfs", "native"): "iBFS",
        ("glign", "native"): "Glign",
        ("forkgraph", "native"): "ForkGraph",
        ("forkgraph", "llc48"): "ForkGraph",
        ("gunrock", "streams1"): "Gunrock s1",
        ("gunrock", "streams64"): "Gunrock s64",
        ("gunrock", "sequential"): "Gunrock seq",
    }
    return aliases.get(key, f"{system} {variant}")


def format_result(row: dict | None) -> str:
    if row is None:
        return "-"
    policy = row.get("measurement_policy", "")
    suffix = ""
    if policy == "single_cold_run":
        suffix = " [cold-1]"
    elif policy == "single_warm_run":
        suffix = " [warm-1]"
    if row["status"] != "complete":
        notes = row["notes"].lower()
        value = "OOM" if "-6" in notes or "-9" in notes or "oom" in notes else row["status"]
        return value + suffix
    milliseconds = float(row["median_compute_ms"])
    throughput = float(row["throughput_queries_per_second"])
    return f"{milliseconds:.2f} / {throughput:.2f}{suffix}"


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    with (ROOT / "result.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {
        (row["dataset"], row["algorithm"], row["system"], row["variant"]): row
        for row in rows
    }

    lines = [
        "# External Baseline Results",
        "",
        "Values are `median ms / queries per second`; graph loading and reusable preprocessing are excluded. PuerCGP online-planner time is included where applicable. `[cold-1]` marks one measured cold run with no warmup; `[warm-1]` marks one measured run after warmup.",
        "",
    ]
    for algorithm in ALGORITHMS:
        columns = ORDER[algorithm]
        lines.extend(
            [
                f"## {algorithm.upper()}",
                "",
                "| Dataset | " + " | ".join(label(column) for column in columns) + " |",
                "|---|" + "---:|" * len(columns),
            ]
        )
        for dataset in DATASETS:
            values = [
                format_result(indexed.get((dataset, algorithm, *column)))
                for column in columns
            ]
            lines.append(f"| {dataset} | " + " | ".join(values) + " |")
        lines.append("")

    speedups = defaultdict(list)
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            candidates = [
                row
                for (d, a, system, _), row in indexed.items()
                if d == dataset
                and a == algorithm
                and system == "puercgp"
                and row["status"] == "complete"
                and row["comparable"] == "1"
            ]
            if not candidates:
                continue
            best = min(float(row["median_compute_ms"]) for row in candidates)
            for (d, a, system, variant), row in indexed.items():
                if d != dataset or a != algorithm or system == "puercgp":
                    continue
                if row["status"] != "complete" or row["comparable"] != "1":
                    continue
                speedups[(system, variant)].append(float(row["median_compute_ms"]) / best)

    lines.extend(["## Best-Puer Speedup", "", "| Baseline | Geomean | Cases |", "|---|---:|---:|"])
    for key, values in sorted(speedups.items()):
        lines.append(f"| {label(key)} | {geometric_mean(values):.3f}x | {len(values)} |")
    lines.append("")

    correctness_path = ROOT / "artifacts" / "correctness" / "weighted_cross_system" / "status.json"
    correctness = json.loads(correctness_path.read_text()) if correctness_path.exists() else None
    forkgraph_sina_path = (
        ROOT
        / "artifacts"
        / "correctness"
        / "forkgraph_llc48_sinaweibo_sssp_q1.json"
    )
    forkgraph_sina = (
        json.loads(forkgraph_sina_path.read_text())
        if forkgraph_sina_path.exists()
        else None
    )
    lines.extend(
        [
            "## Validity Notes",
            "",
            f"- Weighted cross-system checks: `{correctness['state']}` ({correctness['checks']} signatures)." if correctness else "- Weighted cross-system checks: pending.",
            f"- Corrected ForkGraph SinaWeibo SSSP single-query signature: `{forkgraph_sina['result']}`." if forkgraph_sina else "- Corrected ForkGraph SinaWeibo SSSP single-query signature: pending.",
            "- Gunrock `streams64` OOM failures remain failures under strict Q=64; concurrency is not reduced.",
            "- ForkGraph uses 48 bound physical cores and 35.75 MiB edge-balanced range partitions; old P=8/OMP=96 diagnostics are excluded.",
            "- ForkGraph cit-Patents and soc-Orkut use 2 warmups plus 5 repeats; Twitter BFS uses 1 warmup plus 1 measured run; Twitter SSSP and SinaWeibo BFS/SSSP use one cold measured run by request.",
            "- ForkGraph SinaWeibo SSSP was killed by SIGKILL after at least 131 GB observed RSS and is reported as probable host OOM without reducing Q=64.",
            "- ForkGraph timing on this shared host is provisional and should be repeated on an exclusive node for publication.",
            "- Aggregate speedups that include `[cold-1]` or `[warm-1]` cases are preliminary because their sampling policy differs.",
            "- PuerCGP PageRank/PPR use dense fixed 10-iteration execution; the frontier online planner is not applicable.",
            "- Native ForkGraph/Gunrock PPR is residual PR-Nibble, not the same fixed-iteration dense algorithm, and is excluded from the primary table.",
            "",
        ]
    )
    (ROOT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
