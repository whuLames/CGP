#!/usr/bin/env python3
"""Run no-refill versus LSSS bridge refill on prepared sparse-tail SSSP traces."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DIRECTED = {"cit-Patents", "soc-LiveJournal1", "indochina"}
VARIANTS = {
    "A_no_refill": ["--same_algorithm_groups"],
    "B_lsss_bridge": ["--group_refill", "--interference_bridge_refill"],
}


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def parse_metrics(text):
    result = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def write_csv(path, rows, preferred=()):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    fields = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] * (high - position) + values[high] * (position - low)


def run(command, stdout, stderr, log, metadata):
    started = datetime.now().astimezone().isoformat()
    result = subprocess.run(command, text=True, capture_output=True)
    stdout.write_text(result.stdout); stderr.write_text(result.stderr)
    with log.open("a") as handle:
        handle.write(json.dumps({**metadata, "started": started,
                                 "returncode": result.returncode,
                                 "command": command}) + "\n")
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stderr[-3000:]}")
    return parse_metrics(result.stdout)


def fingerprints(path):
    with path.open(newline="") as handle:
        return {int(row["query_id"]): (row["source"], row["vertices"], row["sum64"], row["xor64"])
                for row in csv.DictReader(handle)}


def run_dataset(dataset, device, prepared, output, cli, warmups, repetitions):
    source = prepared / dataset
    manifest = json.loads((source / "manifest.json").read_text())
    with (source / "workload_audit.csv").open(newline="") as handle:
        audit = {int(row["query_id"]): row for row in csv.DictReader(handle)}
    case = output / dataset; raw = case / "raw"; raw.mkdir(parents=True)
    command_log = case / "commands.jsonl"
    base = [str(cli), f"--graph={source/'derived_graph'}",
            f"--queries={source/'sssp_lsss_tail256.csv'}", "--n=1024", "--q=128",
            "--layout=grouped", "--group_width=32", "--algorithm=sssp",
            "--selector=threshold", "--pull_threshold=0.20", "--frontier=unordered",
            "--frontier_build=direct", "--frontier_mask64=true",
            "--push_mapping=iteration", "--legacy_int_weights", "--planner=fifo",
            "--predictor=import_key", "--profile_kernel", "--log_level=warn",
            "--memory_fraction=0.95", f"--device={device}"]
    if dataset in DIRECTED:
        base.append("--directed")
    reference = None
    for variant, flags in VARIANTS.items():
        target = raw / f"validate_{variant}_fingerprints.csv"
        run(base + flags + [f"--result_fingerprints={target}"],
            raw / f"validate_{variant}.stdout.log", raw / f"validate_{variant}.stderr.log",
            command_log, {"phase": "validate", "dataset": dataset, "variant": variant})
        current = fingerprints(target)
        if len(current) != 1024:
            raise RuntimeError(f"{dataset}: incomplete fingerprints")
        if reference is None:
            reference = current
        elif current != reference:
            raise RuntimeError(f"{dataset}: fingerprint mismatch")

    measurements, latencies = [], []
    for repetition in range(-warmups, repetitions):
        order = list(VARIANTS) if repetition % 2 == 0 else list(reversed(VARIANTS))
        for variant in order:
            phase = "warmup" if repetition < 0 else "formal"
            shown_rep = repetition if repetition >= 0 else repetition + warmups
            tag = f"{phase}_{variant}_r{shown_rep}"
            completion = raw / f"{tag}_completion.csv"
            command = base + VARIANTS[variant]
            if phase == "formal":
                command.append(f"--completion_output={completion}")
            metrics = run(command, raw / f"{tag}.stdout.log", raw / f"{tag}.stderr.log",
                          command_log, {"phase": phase, "dataset": dataset,
                                        "variant": variant, "repetition": repetition})
            if phase == "warmup":
                continue
            measurements.append({"dataset": dataset, "variant": variant,
                                 "repetition": repetition, **metrics})
            with completion.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1024:
                raise RuntimeError(f"{dataset}: incomplete completion output")
            for row in rows:
                query_id = int(row["query_id"]); meta = audit[query_id]
                expected = int(meta["reference_rounds"])
                if meta["tail_label"] == "long" and int(row["service_rounds"]) != expected:
                    raise RuntimeError(f"{dataset}: long-query service-round mismatch")
                latencies.append({"dataset": dataset, "variant": variant,
                                  "repetition": repetition, "query_id": query_id,
                                  "tail_label": meta["tail_label"], "batch": meta["batch"],
                                  "reference_rounds": expected,
                                  **{key: row[key] for key in
                                     ("waiting_ms", "service_ms", "submit_to_completion_ms",
                                      "waiting_rounds", "service_rounds",
                                      "submit_to_completion_rounds")}})
    write_csv(case / "raw_measurements.csv", measurements,
              ("dataset", "variant", "repetition"))
    write_csv(case / "query_latency.csv", latencies,
              ("dataset", "variant", "repetition", "query_id", "tail_label"))
    (case / "case_manifest.json").write_text(json.dumps({**manifest, "device": device,
        "fingerprints_match": True, "warmups": warmups, "formal_repetitions": repetitions}, indent=2) + "\n")
    return measurements, latencies


def summarize(measurements, latencies, output):
    summaries = []
    for dataset in sorted({row["dataset"] for row in measurements}):
        item = {"dataset": dataset}
        for variant in VARIANTS:
            selected = [row for row in measurements if row["dataset"] == dataset and row["variant"] == variant]
            for metric in ("workload_ms", "kernel_gpu_ms", "feature_ms", "selector_ms",
                           "recycle_ms", "group_refills", "active_slot_ratio",
                           "refill_admitted_groups", "refill_deferred_incompatible_groups"):
                item[f"{variant}_{metric}_median"] = statistics.median(float(row[metric]) for row in selected)
        item["e2e_speedup"] = item["A_no_refill_workload_ms_median"] / item["B_lsss_bridge_workload_ms_median"]
        for label in ("all", "short", "long"):
            for metric in ("submit_to_completion_ms", "waiting_ms", "service_ms"):
                distributions = {}
                for variant in VARIANTS:
                    selected = [row for row in latencies if row["dataset"] == dataset and
                                row["variant"] == variant and
                                (label == "all" or row["tail_label"] == label)]
                    ids = sorted({int(row["query_id"]) for row in selected})
                    values = [statistics.median(float(row[metric]) for row in selected
                                                if int(row["query_id"]) == query_id)
                              for query_id in ids]
                    distributions[variant] = values
                    item[f"{variant}_{label}_{metric}_mean"] = statistics.fmean(values)
                    for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99)):
                        item[f"{variant}_{label}_{metric}_{name}"] = quantile(values, fraction)
                baseline, candidate = distributions.values()
                item[f"{label}_{metric}_mean_speedup"] = statistics.fmean(baseline) / statistics.fmean(candidate)
                item[f"{label}_{metric}_p99_speedup"] = quantile(baseline, .99) / quantile(candidate, .99)
        summaries.append(item)
    write_csv(output / "summary.csv", summaries, ("dataset", "e2e_speedup"))
    (output / "metrics.json").write_text(json.dumps({"all_fingerprints_match": True,
        "formal_measurements": len(measurements), "query_latency_rows": len(latencies),
        "cases": summaries}, indent=2) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--devices", default="1,2,3,4,5")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    datasets = sorted(path.name for path in args.prepared.iterdir()
                      if path.is_dir() and (path / "manifest.json").exists())
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) < len(datasets):
        raise SystemExit("one GPU per dataset is required")
    config = {"started_at": datetime.now().astimezone().isoformat(), "datasets": datasets,
              "N": 1024, "Q": 128, "G": 32, "variants": VARIANTS,
              "warmups": args.warmups, "formal_repetitions": args.repetitions,
              "binary": str(args.cli.resolve()), "binary_sha256": sha256(args.cli),
              "devices": dict(zip(datasets, devices))}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_measurements, all_latencies = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(datasets)) as executor:
        futures = {executor.submit(run_dataset, dataset, devices[index], args.prepared,
                                   args.output, args.cli.resolve(), args.warmups, args.repetitions): dataset
                   for index, dataset in enumerate(datasets)}
        for future in concurrent.futures.as_completed(futures):
            dataset = futures[future]
            measurements, latencies = future.result()
            all_measurements.extend(measurements); all_latencies.extend(latencies)
            print(f"completed {dataset}", flush=True)
    write_csv(args.output / "raw_measurements.csv", all_measurements,
              ("dataset", "variant", "repetition"))
    write_csv(args.output / "query_latency.csv", all_latencies,
              ("dataset", "variant", "repetition", "query_id", "tail_label"))
    for row in summarize(all_measurements, all_latencies, args.output):
        print(row["dataset"], f"e2e={row['e2e_speedup']:.4f}",
              f"latency_mean={row['all_submit_to_completion_ms_mean_speedup']:.4f}",
              f"latency_p99={row['all_submit_to_completion_ms_p99_speedup']:.4f}", flush=True)


if __name__ == "__main__":
    main()
