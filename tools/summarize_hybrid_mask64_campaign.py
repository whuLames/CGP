#!/usr/bin/env python3
"""Audit and summarize the fixed Hybrid/direct/mask64 campaign."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("cit-Patents", "soc-LiveJournal1", "indochina", "soc-orkut", "soc-twitter")
WORKLOADS = ("bfs", "sssp", "mixed_tail")
VARIANTS = ("P", "H0", "A", "B", "C", "D")
COMPARISONS = (("P_to_A", "P", "A"), ("H0_to_A", "H0", "A"),
               ("A_to_B", "A", "B"), ("A_to_C", "A", "C"),
               ("C_to_D", "C", "D"), ("A_to_D", "A", "D"))
PHASES = ("kernel_gpu_ms", "frontier_ms", "copy_ms", "feature_ms", "selector_ms",
          "initialization_ms", "recycle_ms")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, preferred=()):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    ordered = [field for field in preferred if field in fields]
    fields = ordered + [field for field in fields if field not in ordered]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def geomean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values)) if values else None


def f(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def baseline_raw_paths(root):
    main = root / "20260928_external_baselines/raw.csv"
    matched = root / "20260928_external_baselines_matched_full"
    result = {"cit-Patents": main}
    result.update({dataset: matched / dataset / "raw.csv" for dataset in DATASETS[1:]})
    return result


def load_baselines(root):
    output = {}
    for dataset, path in baseline_raw_paths(root).items():
        if not path.exists():
            continue
        rows = [row for row in read_csv(path) if row.get("run_kind") == "formal"]
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["system"], row["workload"], row["mode"], int(row["repeat"]))].append(row)
        samples = defaultdict(list)
        for (system, workload, mode, _), parts in grouped.items():
            expected = 2 if workload == "mixed_tail" else 1
            algorithms = {part["algorithm"] for part in parts}
            if len(parts) != expected or (expected == 2 and algorithms != {"bfs", "sssp"}):
                continue
            samples[(dataset, workload, system, mode)].append(sum(float(part["runtime_ms"]) for part in parts))
        for key, values in samples.items():
            if len(values) == 5:
                output[key] = statistics.median(values)
    return output


def refill_class(runtime_speedup, latency_speedup, round_speedup):
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
    parser.add_argument("--baseline-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments")
    args = parser.parse_args(); root = args.campaign.resolve()
    summaries, raw_rows = {}, []
    for dataset in DATASETS:
        summary_path = root / dataset / "summary.csv"
        raw_path = root / dataset / "raw_measurements.csv"
        if not summary_path.exists() or not raw_path.exists():
            continue
        raw_rows.extend(read_csv(raw_path))
        for row in read_csv(summary_path):
            summaries[(dataset, row["workload"], row["variant"])] = row

    case_rows = []
    for dataset in DATASETS:
        for workload in WORKLOADS:
            for variant in VARIANTS:
                row = summaries.get((dataset, workload, variant), {})
                case_rows.append({"dataset": dataset, "workload": workload, "variant": variant,
                                  "samples": row.get("samples", 0),
                                  **{key: value for key, value in row.items()
                                     if key not in {"workload", "variant", "samples"}}})
    write_csv(root / "result.csv", case_rows, ("dataset", "workload", "variant", "samples"))

    comparisons = []
    for dataset in DATASETS:
        for workload in WORKLOADS:
            for name, before, after in COMPARISONS:
                left = summaries.get((dataset, workload, before), {})
                right = summaries.get((dataset, workload, after), {})
                left_ms, right_ms = f(left, "workload_ms_median"), f(right, "workload_ms_median")
                if left_ms is None or right_ms is None:
                    continue
                item = {"dataset": dataset, "workload": workload, "comparison": name,
                        "before": before, "after": after, "workload_speedup": left_ms / right_ms}
                for metric in ("throughput_qps", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99",
                               "latency_ms_max", "waiting_ms_p50", "service_ms_p50", "rounds"):
                    a, b = f(left, metric + "_median"), f(right, metric + "_median")
                    if a is not None and b not in (None, 0):
                        item[metric + "_before_over_after"] = a / b
                comparisons.append(item)
    write_csv(root / "comparisons.csv", comparisons,
              ("dataset", "workload", "comparison", "before", "after"))

    comparison_geomeans = []
    for name, _, _ in COMPARISONS:
        for scope in (*WORKLOADS, "all"):
            selected = [row for row in comparisons if row["comparison"] == name and
                        (scope == "all" or row["workload"] == scope)]
            item = {"comparison": name, "scope": scope, "cases": len(selected)}
            metrics = sorted(set().union(*(row.keys() for row in selected)) -
                             {"dataset", "workload", "comparison", "before", "after"})
            for metric in metrics:
                values = [row[metric] for row in selected if row.get(metric) not in (None, 0)]
                item[metric + "_geomean"] = geomean(values)
            comparison_geomeans.append(item)
    write_csv(root / "comparison_geomeans.csv", comparison_geomeans,
              ("comparison", "scope", "cases"))

    baselines = load_baselines(args.baseline_root)
    speedups = []
    for dataset in DATASETS:
        for workload in WORKLOADS:
            graphweft = summaries.get((dataset, workload, "A"), {})
            runtime = f(graphweft, "workload_ms_median")
            for system in ("gunrock", "groute"):
                sequential = baselines.get((dataset, workload, system, "sequential"))
                concurrent = baselines.get((dataset, workload, system, "concurrent"))
                speedups.append({"dataset": dataset, "workload": workload, "system": system,
                                 "graphweft_variant": "A", "graphweft_ms": runtime,
                                 "sequential_ms": sequential, "concurrent_ms": concurrent,
                                 "faster_baseline_ms": min(x for x in (sequential, concurrent) if x is not None)
                                     if sequential is not None or concurrent is not None else None,
                                 "speedup_vs_sequential": sequential / runtime if sequential and runtime else None,
                                 "speedup_vs_concurrent": concurrent / runtime if concurrent and runtime else None,
                                 "speedup_vs_faster": min(x for x in (sequential, concurrent) if x is not None) / runtime
                                     if runtime and (sequential is not None or concurrent is not None) else None,
                                 "status": "available" if sequential is not None or concurrent is not None else "missing"})
    write_csv(root / "baseline_speedups.csv", speedups,
              ("dataset", "workload", "system", "status"))

    geo_rows = []
    for system in ("gunrock", "groute"):
        for scope, selected in [(workload, [row for row in speedups if row["system"] == system and
                                            row["workload"] == workload]) for workload in WORKLOADS] + [
                                                ("all", [row for row in speedups if row["system"] == system])]:
            available = [row for row in selected if row["status"] == "available"]
            geo_rows.append({"system": system, "scope": scope, "available_cases": len(available),
                             "expected_cases": len(selected),
                             "geomean_vs_sequential": geomean([row["speedup_vs_sequential"] for row in available
                                                                 if row["speedup_vs_sequential"]]),
                             "geomean_vs_concurrent": geomean([row["speedup_vs_concurrent"] for row in available
                                                                 if row["speedup_vs_concurrent"]]),
                             "geomean_vs_faster": geomean([row["speedup_vs_faster"] for row in available
                                                             if row["speedup_vs_faster"]])})
    write_csv(root / "geomean_speedups.csv", geo_rows, ("system", "scope"))

    stage_rows = []
    for dataset in DATASETS:
        for workload in WORKLOADS:
            row = summaries.get((dataset, workload, "A"), {})
            runtime = f(row, "workload_ms_median")
            if not runtime:
                continue
            item = {"dataset": dataset, "workload": workload, "variant": "A",
                    "workload_ms": runtime}
            phase_values = {}
            for phase in PHASES:
                value = f(row, phase + "_median")
                if value is not None:
                    item[phase] = value
                    item[phase + "_share"] = value / runtime
                    phase_values[phase] = value
            item["dominant_phase"] = max(phase_values, key=phase_values.get) if phase_values else ""
            item["gunrock_speedup_vs_faster"] = next(
                (entry["speedup_vs_faster"] for entry in speedups
                 if entry["dataset"] == dataset and entry["workload"] == workload and
                 entry["system"] == "gunrock"), None)
            item["below_3x_gunrock"] = bool(item["gunrock_speedup_vs_faster"] is not None and
                                               item["gunrock_speedup_vs_faster"] < 3.0)
            stage_rows.append(item)
    write_csv(root / "stage_breakdown.csv", stage_rows,
              ("dataset", "workload", "variant", "workload_ms", "dominant_phase",
               "gunrock_speedup_vs_faster", "below_3x_gunrock"))

    refill = []
    for dataset in DATASETS:
        for workload in WORKLOADS:
            row = next((item for item in comparisons if item["dataset"] == dataset and
                        item["workload"] == workload and item["comparison"] == "A_to_C"), None)
            if not row:
                continue
            runtime = row["workload_speedup"]
            latency = row.get("latency_ms_p50_before_over_after", 1.0)
            rounds = row.get("rounds_before_over_after", 1.0)
            before = summaries[(dataset, workload, "A")]
            after = summaries[(dataset, workload, "C")]
            item = {"dataset": dataset, "workload": workload,
                    "throughput_runtime_speedup": runtime, "latency_p50_speedup": latency,
                    "logical_round_speedup": rounds,
                    "classification": refill_class(runtime, latency, rounds)}
            for metric in ("workload_ms", "throughput_qps", "latency_ms_mean", "latency_ms_p50",
                           "latency_ms_p95", "latency_ms_p99", "latency_ms_max", "rounds"):
                item["A_" + metric] = f(before, metric + "_median")
                item["C_" + metric] = f(after, metric + "_median")
            refill.append(item)
    write_csv(root / "refill_assessment.csv", refill, ("dataset", "workload", "classification"))

    formal_counts = defaultdict(int)
    for row in raw_rows:
        formal_counts[(row["dataset"], row["workload"], row["variant"])] += 1
    complete_cases = sum(formal_counts[key] == 5 for key in
                         ((d, w, v) for d in DATASETS for w in WORKLOADS for v in VARIANTS))
    gunrock_all = next(row for row in geo_rows if row["system"] == "gunrock" and row["scope"] == "all")
    metrics = json.loads((root / "metrics.json").read_text()) if (root / "metrics.json").exists() else {}
    metrics.update({"test_cases_complete": complete_cases, "test_cases_expected": 90,
                    "formal_samples_found": len(raw_rows), "formal_samples_expected": 450,
                    "gunrock_geomean_vs_faster": gunrock_all["geomean_vs_faster"],
                    "gunrock_target_3x_met": bool(gunrock_all["available_cases"] == 15 and
                                                   gunrock_all["geomean_vs_faster"] and
                                                   gunrock_all["geomean_vs_faster"] >= 3.0)})
    metrics["status"] = "success" if complete_cases == 90 and len(raw_rows) == 450 else "incomplete"
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    gunrock_geo = gunrock_all["geomean_vs_faster"]
    workload_gap_lines = []
    for workload in WORKLOADS:
        geo = next(row for row in geo_rows if row["system"] == "gunrock" and
                   row["scope"] == workload)
        selected = [row for row in stage_rows if row["workload"] == workload]
        shares = {phase: statistics.median(row.get(phase + "_share", 0.0)
                                            for row in selected) for phase in PHASES}
        dominant = max(shares, key=shares.get)
        workload_gap_lines.append(
            f"- {workload}: {geo['geomean_vs_faster']:.3f}x vs faster Gunrock; "
            f"dominant measured phase is {dominant} (median {shares[dominant] * 100:.1f}% of workload time).")
    report = ["# GraphWeft Hybrid + bitmask64 campaign", "",
              "## Scope", "",
              "Five non-road graphs, BFS/SSSP/Mixed-tail, fixed N=1024, M=64, G=8 and Hybrid threshold 0.20. "
              "roadNet and sinaweibo are excluded. Each case has one warmup and five formal samples.", "",
              "## Acceptance", "",
              f"- Complete test cases: {complete_cases}/90",
              f"- Formal samples: {len(raw_rows)}/450",
              f"- Gunrock faster-mode geometric mean: {gunrock_geo:.3f}x" if gunrock_geo else
                  "- Gunrock faster-mode geometric mean: unavailable",
              f"- Approximate 3x Gunrock target: {'met' if metrics['gunrock_target_3x_met'] else 'not met'}", "",
              "No unavailable Gunrock/Groute result is replaced by a capacity-probe runtime. Missing entries are marked in `baseline_speedups.csv`.", "",
              "## Gunrock target gap", "",
              *(workload_gap_lines if not metrics["gunrock_target_3x_met"] else
                ["The all-workload 3x target was met."]), "",
              "The phase percentages are diagnostic timer shares and may overlap; they identify where measured time is concentrated rather than forming an additive decomposition.", "",
              "## Outputs", "",
              "- `result.csv`: all 90 case summaries, including median/min/max/stdev/range.",
              "- `comparisons.csv`: P→A, H0→A, A→B, A→C, C→D, and A→D.",
              "- `comparison_geomeans.csv`: the same comparisons aggregated across graphs and overall.",
              "- `baseline_speedups.csv` and `geomean_speedups.csv`: reused baseline comparisons.",
              "- `refill_assessment.csv`: real-ms latency/throughput/round classification.",
              "- `stage_breakdown.csv`: per-case A timing phases and Gunrock-target gap localization.",
              "- `<dataset>/raw/`: commands, logs, completion CSVs, and validation hashes.", "",
              "## Reproduction", "",
              "The exact command, code identity, input roots, GPU binding, and variant flags are in `config.json`; every subprocess command is in each dataset's `commands.jsonl`.", ""]
    (root / "README.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
