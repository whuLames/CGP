#!/usr/bin/env python3
"""Measure group refill on frozen strong-tail workloads.

The campaign compares batch-barrier reclamation (A) with group refill (C),
retains every formal per-query completion record, and joins those records with
the frozen short/long labels.  It can wait for occupied GPUs before starting.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "experiments/20260929_strong_tail_workloads"
DATASETS = {
    "soc-orkut": {"directed": False},
    "soc-twitter": {"directed": True},
}
WORKLOADS = {"bfs": "bfs", "sssp": "sssp", "mixed": "bfs"}
VARIANTS = {
    "A_no_refill": ["--same_algorithm_groups"],
    "C_refill": ["--group_refill"],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_metrics(text):
    result = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def write_csv(path, rows, preferred=()):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    fields = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gpu_memory_used():
    run = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    return [int(line.strip()) for line in run.stdout.splitlines()]


def wait_for_devices(devices, threshold, poll, log):
    while True:
        used = gpu_memory_used()
        ready = all(device < len(used) and used[device] <= threshold for device in devices)
        with log.open("a") as handle:
            stamp = datetime.now().astimezone().isoformat()
            handle.write(f"{stamp} used_mib={used} devices={devices} ready={int(ready)}\n")
        if ready:
            return
        time.sleep(poll)


def read_audit(dataset_root):
    result = {}
    with (dataset_root / "workloads/workload_audit.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[(row["workload"], int(row["id"]))] = row
    return result


def read_fingerprints(path):
    with path.open(newline="") as handle:
        return {int(row["query_id"]): tuple((k, row[k]) for k in row if k != "slot")
                for row in csv.DictReader(handle)}


def common_command(cli, dataset, workload, device, workload_dirname, length_predicted):
    root = DATA_ROOT / dataset
    command = [
        str(cli), f"--graph={root/'derived_graph'}",
        f"--queries={root/workload_dirname/f'{workload}_strong_tail.csv'}",
        "--n=1024", "--q=64", "--layout=grouped", "--group_width=32",
        f"--algorithm={WORKLOADS[workload]}", "--selector=threshold",
        "--pull_threshold=0.20", "--frontier=unordered", "--frontier_build=direct",
        "--frontier_mask64=true", "--push_mapping=shared", "--legacy_int_weights",
        "--profile_kernel", "--log_level=warn", f"--device={device}",
    ]
    if length_predicted:
        command += ["--planner=length", "--predictor=import_key"]
    if DATASETS[dataset]["directed"]:
        command.append("--directed")
    return command


def run_one(command, stdout_path, stderr_path, command_log, metadata):
    started = datetime.now().astimezone().isoformat()
    run = subprocess.run(command, text=True, capture_output=True)
    stdout_path.write_text(run.stdout)
    stderr_path.write_text(run.stderr)
    with command_log.open("a") as handle:
        handle.write(json.dumps({**metadata, "started": started, "returncode": run.returncode,
                                 "command": command}) + "\n")
    if run.returncode:
        raise RuntimeError(f"command failed ({run.returncode}): {run.stderr[-1000:]}")
    return parse_metrics(run.stdout)


def run_dataset(dataset, device, output, repetitions, warmups, workload_dirname,
                length_predicted, cli):
    target = output / dataset
    raw = target / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    command_log = target / "commands.jsonl"
    audit = read_audit(DATA_ROOT / dataset)

    # Validate final values before timing.  Refill may change global completion
    # rounds, so fingerprints compare only final-result fields and local rounds.
    for workload in WORKLOADS:
        fingerprints = {}
        for variant, flags in VARIANTS.items():
            path = raw / f"validate_{workload}_{variant}_fingerprints.csv"
            command = common_command(cli, dataset, workload, device, workload_dirname,
                                     length_predicted) + flags + [
                f"--result_fingerprints={path}"]
            run_one(command, raw / f"validate_{workload}_{variant}.stdout.log",
                    raw / f"validate_{workload}_{variant}.stderr.log", command_log,
                    {"phase": "validate", "workload": workload, "variant": variant})
            fingerprints[variant] = read_fingerprints(path)
        if fingerprints["A_no_refill"] != fingerprints["C_refill"]:
            raise RuntimeError(f"{dataset}/{workload}: refill fingerprint mismatch")

    measurements, latency_rows = [], []
    for workload in WORKLOADS:
        # Alternate the first variant by repetition to reduce temporal bias.
        for repetition in range(-warmups, repetitions):
            order = list(VARIANTS)
            if repetition % 2:
                order.reverse()
            for variant in order:
                phase = "warmup" if repetition < 0 else "formal"
                tag = f"{phase}_{workload}_{variant}_r{repetition if repetition >= 0 else repetition+warmups}"
                command = common_command(cli, dataset, workload, device, workload_dirname,
                                         length_predicted) + VARIANTS[variant]
                completion = raw / f"{tag}_completion.csv"
                if phase == "formal":
                    command.append(f"--completion_output={completion}")
                metrics = run_one(command, raw / f"{tag}.stdout.log", raw / f"{tag}.stderr.log",
                                  command_log, {"phase": phase, "workload": workload,
                                                "variant": variant, "repetition": repetition})
                if phase == "warmup":
                    continue
                measurements.append({"dataset": dataset, "workload": workload,
                                     "variant": variant, "repetition": repetition, **metrics})
                with completion.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        qid = int(row["query_id"])
                        frozen = audit[(workload, qid)]
                        if int(row["service_rounds"]) != int(frozen["reference_rounds"]):
                            raise RuntimeError(
                                f"{dataset}/{workload}/{variant}/q{qid}: service-round mismatch")
                        latency_rows.append({
                            "dataset": dataset, "workload": workload, "variant": variant,
                            "repetition": repetition, "query_id": qid,
                            "algorithm": "bfs" if int(row["algorithm"]) == 0 else "sssp",
                            "tail_label": frozen["label"],
                            "reference_rounds": int(frozen["reference_rounds"]),
                            **{key: row[key] for key in (
                                "activation_round", "completion_round", "waiting_rounds",
                                "service_rounds", "submit_to_completion_rounds", "activation_ms",
                                "completion_ms", "waiting_ms", "service_ms",
                                "submit_to_completion_ms")},
                        })

    write_csv(target / "raw_measurements.csv", measurements,
              ("dataset", "workload", "variant", "repetition"))
    write_csv(target / "query_latency.csv", latency_rows,
              ("dataset", "workload", "variant", "repetition", "query_id",
               "algorithm", "tail_label"))
    return measurements, latency_rows


def summarize(measurements, latencies, output):
    end_to_end = []
    keys = sorted({(r["dataset"], r["workload"]) for r in measurements})
    for dataset, workload in keys:
        row = {"dataset": dataset, "workload": workload}
        for variant in VARIANTS:
            values = [float(r["workload_ms"]) for r in measurements
                      if r["dataset"] == dataset and r["workload"] == workload and
                      r["variant"] == variant]
            row[f"{variant}_workload_ms_median"] = statistics.median(values)
            row[f"{variant}_workload_ms_min"] = min(values)
            row[f"{variant}_workload_ms_max"] = max(values)
        row["refill_speedup"] = (row["A_no_refill_workload_ms_median"] /
                                 row["C_refill_workload_ms_median"])
        end_to_end.append(row)

    latency_summary = []
    group_keys = sorted({(r["dataset"], r["workload"], r["algorithm"], r["tail_label"],
                          r["variant"]) for r in latencies})
    for dataset, workload, algorithm, label, variant in group_keys:
        selected = [r for r in latencies if (r["dataset"], r["workload"], r["algorithm"],
                    r["tail_label"], r["variant"]) ==
                    (dataset, workload, algorithm, label, variant)]
        row = {"dataset": dataset, "workload": workload, "algorithm": algorithm,
               "tail_label": label, "variant": variant, "samples": len(selected),
               "queries_per_repetition": len({r["query_id"] for r in selected})}
        for metric in ("service_ms", "submit_to_completion_ms", "waiting_ms",
                       "service_rounds", "submit_to_completion_rounds"):
            values = [float(r[metric]) for r in selected]
            row[f"{metric}_mean"] = statistics.fmean(values)
            for name, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
                row[f"{metric}_{name}"] = percentile(values, q)
        latency_summary.append(row)

    per_query = []
    query_keys = sorted({(r["dataset"], r["workload"], r["algorithm"],
                          r["tail_label"], r["query_id"]) for r in latencies})
    for dataset, workload, algorithm, label, qid in query_keys:
        row = {"dataset": dataset, "workload": workload, "algorithm": algorithm,
               "tail_label": label, "query_id": qid}
        for metric in ("service_ms", "submit_to_completion_ms"):
            for variant in VARIANTS:
                values = [float(r[metric]) for r in latencies if
                          (r["dataset"], r["workload"], r["algorithm"], r["tail_label"],
                           r["query_id"], r["variant"]) ==
                          (dataset, workload, algorithm, label, qid, variant)]
                row[f"{variant}_{metric}_median"] = statistics.median(values)
            row[f"refill_{metric}_speedup"] = (row[f"A_no_refill_{metric}_median"] /
                                                row[f"C_refill_{metric}_median"])
        per_query.append(row)

    latency_comparison = []
    comparison_keys = sorted({(r["dataset"], r["workload"], r["algorithm"],
                               r["tail_label"]) for r in latencies})
    for dataset, workload, algorithm, label in comparison_keys:
        selected = [r for r in latencies if
                    (r["dataset"], r["workload"], r["algorithm"], r["tail_label"]) ==
                    (dataset, workload, algorithm, label)]
        row = {"dataset": dataset, "workload": workload, "algorithm": algorithm,
               "tail_label": label}
        for metric in ("service_ms", "submit_to_completion_ms", "waiting_ms"):
            for variant in VARIANTS:
                values = [float(r[metric]) for r in selected if r["variant"] == variant]
                row[f"{variant}_{metric}_mean"] = statistics.fmean(values)
                row[f"{variant}_{metric}_p50"] = percentile(values, .5)
                row[f"{variant}_{metric}_p95"] = percentile(values, .95)
            for statistic in ("mean", "p50", "p95"):
                numerator = row[f"A_no_refill_{metric}_{statistic}"]
                denominator = row[f"C_refill_{metric}_{statistic}"]
                row[f"refill_{metric}_{statistic}_speedup"] = (
                    numerator / denominator if denominator else
                    (1.0 if numerator == 0 else math.inf))
        paired = [r for r in per_query if
                  (r["dataset"], r["workload"], r["algorithm"], r["tail_label"]) ==
                  (dataset, workload, algorithm, label)]
        for metric in ("service_ms", "submit_to_completion_ms"):
            speedups = [r[f"refill_{metric}_speedup"] for r in paired]
            row[f"per_query_{metric}_geomean_speedup"] = math.exp(
                statistics.fmean(math.log(value) for value in speedups))
            row[f"per_query_{metric}_improved_fraction"] = sum(
                value > 1 for value in speedups) / len(speedups)
        latency_comparison.append(row)

    write_csv(output / "end_to_end_summary.csv", end_to_end, ("dataset", "workload"))
    write_csv(output / "latency_summary.csv", latency_summary,
              ("dataset", "workload", "algorithm", "tail_label", "variant"))
    write_csv(output / "per_query_comparison.csv", per_query,
              ("dataset", "workload", "algorithm", "tail_label", "query_id"))
    write_csv(output / "latency_comparison_summary.csv", latency_comparison,
              ("dataset", "workload", "algorithm", "tail_label"))
    geomean = math.exp(statistics.fmean(math.log(r["refill_speedup"]) for r in end_to_end))
    all_submit = [r["refill_submit_to_completion_ms_speedup"] for r in per_query]
    long_submit = [r["refill_submit_to_completion_ms_speedup"] for r in per_query
                   if r["tail_label"] == "long"]
    all_submit_geomean = math.exp(statistics.fmean(math.log(value) for value in all_submit))
    long_submit_geomean = math.exp(statistics.fmean(math.log(value) for value in long_submit))
    metrics = {"end_to_end_geomean_refill_speedup": geomean,
               "all_query_submit_latency_geomean_speedup": all_submit_geomean,
               "long_query_submit_latency_geomean_speedup": long_submit_geomean,
               "accepted": geomean > 1 and all_submit_geomean > 1 and long_submit_geomean > 1,
               "all_fingerprints_match": True, "formal_cases": len(end_to_end) * 2,
               "formal_measurements": len(measurements), "query_latency_rows": len(latencies)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--gpu-memory-threshold", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--workload-dirname", default="workloads")
    parser.add_argument("--length-predicted", action="store_true")
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    args = parser.parse_args()
    output = args.output.resolve()
    cli = args.cli.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "started_at": datetime.now().astimezone().isoformat(), "datasets": list(DATASETS),
        "devices": dict(zip(DATASETS, args.devices)), "workloads": list(WORKLOADS),
        "variants": VARIANTS, "N": 1024, "M": 64, "G": 32,
        "pull_threshold": .2, "warmups": args.warmups, "formal_repetitions": args.repetitions,
        "workload_dirname": args.workload_dirname,
        "length_predicted": args.length_predicted,
        "binary": str(cli), "binary_sha256": sha256(cli),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.wait_for_gpus:
        wait_for_devices(args.devices, args.gpu_memory_threshold, args.poll_seconds,
                         output / "wait.log")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_dataset, dataset, device, output, args.repetitions, args.warmups,
                               args.workload_dirname, args.length_predicted, cli)
                   for dataset, device in zip(DATASETS, args.devices)]
        measurements, latencies = [], []
        for future in futures:
            m, l = future.result(); measurements += m; latencies += l
    summarize(measurements, latencies, output)
    print((output / "metrics.json").read_text(), end="")


if __name__ == "__main__":
    main()
