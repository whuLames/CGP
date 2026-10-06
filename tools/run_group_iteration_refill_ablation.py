#!/usr/bin/env python3
"""SSSP strong-tail ablation for global versus per-group Iteration mapping."""
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
DATA_ROOT = PROJECT / "experiments/20260929_strong_tail_workloads"
DATASETS = {
    "soc-orkut": {"directed": False},
    "soc-twitter": {"directed": True},
}
VARIANTS = {
    "A_no_refill": ["--same_algorithm_groups"],
    "B_eager_global": ["--group_refill", "--eager_sssp_refill"],
    "D_eager_group": ["--group_refill", "--eager_sssp_refill",
                      "--group_iteration_mapping"],
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
    fields = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_workload(dataset, q, output):
    source = DATA_ROOT / dataset / "workloads_predicted/sssp_strong_tail.csv"
    target = output / "workloads" / f"{dataset}_sssp_strong_tail_q{q}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    text, replacements = re.subn(r"(?m)^# capacity=\d+$", f"# capacity={q}", text, count=1)
    if replacements != 1:
        raise RuntimeError(f"missing capacity header in {source}")
    target.write_text(text)
    return target


def common_command(cli, dataset, q, device, workload):
    root = DATA_ROOT / dataset
    command = [
        str(cli), f"--graph={root/'derived_graph'}", f"--queries={workload}",
        "--n=1024", f"--q={q}", "--layout=grouped", "--group_width=32",
        "--algorithm=sssp", "--selector=threshold", "--pull_threshold=0.20",
        "--frontier=unordered", "--frontier_build=direct", "--frontier_mask64=true",
        "--push_mapping=iteration", "--legacy_int_weights", "--planner=length",
        "--predictor=import_key", "--profile_kernel", "--log_level=warn",
        f"--device={device}",
    ]
    if q == 128:
        command.append("--memory_fraction=0.95")
    if DATASETS[dataset]["directed"]:
        command.append("--directed")
    return command


def run_one(command, stdout_path, stderr_path, command_log, metadata):
    started = datetime.now().astimezone().isoformat()
    run = subprocess.run(command, text=True, capture_output=True)
    stdout_path.write_text(run.stdout)
    stderr_path.write_text(run.stderr)
    with command_log.open("a") as handle:
        handle.write(json.dumps({**metadata, "started": started,
                                 "returncode": run.returncode,
                                 "command": command}) + "\n")
    if run.returncode:
        raise RuntimeError(f"command failed ({run.returncode}): {' '.join(command)}\n{run.stderr[-2000:]}")
    return parse_metrics(run.stdout)


def fingerprints(path):
    with path.open(newline="") as handle:
        return {int(row["query_id"]): (row["source"], row["vertices"], row["sum64"], row["xor64"])
                for row in csv.DictReader(handle)}


def read_labels(dataset):
    path = DATA_ROOT / dataset / "workloads_predicted/workload_audit.csv"
    with path.open(newline="") as handle:
        return {int(row["id"]): row for row in csv.DictReader(handle)
                if row["workload"] == "sssp"}


def run_case(dataset, q, device, output, cli, warmups, repetitions):
    case = output / dataset / f"q{q}"
    raw = case / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    command_log = case / "commands.jsonl"
    workload = prepare_workload(dataset, q, output)
    base = common_command(cli, dataset, q, device, workload)
    labels = read_labels(dataset)

    reference = None
    for variant, flags in VARIANTS.items():
        fingerprint_path = raw / f"validate_{variant}_fingerprints.csv"
        command = base + flags + [f"--result_fingerprints={fingerprint_path}"]
        if variant == "A_no_refill":
            command.append(f"--plan_output={raw/'planned_queries.csv'}")
        run_one(command, raw / f"validate_{variant}.stdout.log",
                raw / f"validate_{variant}.stderr.log", command_log,
                {"phase": "validate", "dataset": dataset, "q": q,
                 "variant": variant})
        current = fingerprints(fingerprint_path)
        if reference is None:
            reference = current
        elif current != reference:
            raise RuntimeError(f"fingerprint mismatch: {dataset}/Q{q}/{variant}")

    measurements, latencies = [], []
    for repetition in range(-warmups, repetitions):
        order = list(VARIANTS)
        if repetition % 2:
            order.reverse()
        for variant in order:
            phase = "warmup" if repetition < 0 else "formal"
            rep = repetition if repetition >= 0 else repetition + warmups
            tag = f"{phase}_{variant}_r{rep}"
            completion = raw / f"{tag}_completion.csv"
            rounds = raw / f"{tag}_rounds.csv"
            command = base + VARIANTS[variant]
            if phase == "formal":
                command += [f"--completion_output={completion}", f"--round_metrics={rounds}"]
            metrics = run_one(command, raw / f"{tag}.stdout.log",
                              raw / f"{tag}.stderr.log", command_log,
                              {"phase": phase, "dataset": dataset, "q": q,
                               "variant": variant, "repetition": repetition})
            if phase == "warmup":
                continue
            measurements.append({"dataset": dataset, "q": q, "variant": variant,
                                 "repetition": repetition, **metrics})
            with completion.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    query_id = int(row["query_id"])
                    label = labels[query_id]
                    if int(row["service_rounds"]) != int(label["reference_rounds"]):
                        raise RuntimeError(f"service-round mismatch: {dataset}/Q{q}/q{query_id}")
                    latencies.append({
                        "dataset": dataset, "q": q, "variant": variant,
                        "repetition": repetition, "query_id": query_id,
                        "tail_label": label["label"],
                        "reference_rounds": int(label["reference_rounds"]),
                        **{key: row[key] for key in (
                            "activation_round", "completion_round", "waiting_rounds",
                            "service_rounds", "submit_to_completion_rounds",
                            "activation_ms", "completion_ms", "waiting_ms",
                            "service_ms", "submit_to_completion_ms")},
                    })
    write_csv(case / "raw_measurements.csv", measurements,
              ("dataset", "q", "variant", "repetition"))
    write_csv(case / "query_latency.csv", latencies,
              ("dataset", "q", "variant", "repetition", "query_id", "tail_label"))
    return measurements, latencies


def summarize(measurements, latencies, output):
    rows = []
    for dataset, q in sorted({(row["dataset"], row["q"]) for row in measurements}):
        summary = {"dataset": dataset, "q": q}
        for variant in VARIANTS:
            selected = [row for row in measurements if row["dataset"] == dataset and
                        row["q"] == q and row["variant"] == variant]
            for metric in ("workload_ms", "kernel_gpu_ms", "feature_ms", "selector_ms",
                           "recycle_ms", "group_refills", "active_slot_ratio",
                           "group_mapping_rounds", "group_mapping_divergent_rounds",
                           "group_mapping_launches"):
                summary[f"{variant}_{metric}_median"] = statistics.median(
                    float(row[metric]) for row in selected)
        for candidate in ("B_eager_global", "D_eager_group"):
            summary[f"{candidate}_e2e_speedup_vs_A"] = (
                summary["A_no_refill_workload_ms_median"] /
                summary[f"{candidate}_workload_ms_median"])
        summary["D_eager_group_e2e_speedup_vs_B"] = (
            summary["B_eager_global_workload_ms_median"] /
            summary["D_eager_group_workload_ms_median"])

        for label in ("all", "short", "long"):
            subset = [row for row in latencies if row["dataset"] == dataset and row["q"] == q and
                      (label == "all" or row["tail_label"] == label)]
            for metric in ("submit_to_completion_ms", "service_ms"):
                medians = {}
                for variant in VARIANTS:
                    per_query = []
                    ids = sorted({int(row["query_id"]) for row in subset if row["variant"] == variant})
                    for query_id in ids:
                        values = [float(row[metric]) for row in subset
                                  if row["variant"] == variant and int(row["query_id"]) == query_id]
                        per_query.append(statistics.median(values))
                    medians[variant] = statistics.fmean(per_query)
                    summary[f"{variant}_{label}_{metric}_mean_of_query_medians"] = medians[variant]
                for candidate in ("B_eager_global", "D_eager_group"):
                    summary[f"{candidate}_{label}_{metric}_speedup_vs_A"] = (
                        medians["A_no_refill"] / medians[candidate])
                summary[f"D_eager_group_{label}_{metric}_speedup_vs_B"] = (
                    medians["B_eager_global"] / medians["D_eager_group"])
        rows.append(summary)
    write_csv(output / "summary.csv", rows, ("dataset", "q"))
    metrics = {
        "all_fingerprints_match": True,
        "formal_measurements": len(measurements),
        "query_latency_rows": len(latencies),
        "cases": rows,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--devices", default="0,1,2,3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = [int(value) for value in args.devices.split(",")]
    cases = [(dataset, q) for dataset in DATASETS for q in (64, 128)]
    if len(devices) < len(cases):
        raise SystemExit("four devices are required for the four concurrent cases")
    config = {
        "started_at": datetime.now().astimezone().isoformat(),
        "datasets": list(DATASETS), "workload": "sssp_strong_tail",
        "Q": [64, 128], "G": 32, "N": 1024,
        "variants": VARIANTS, "warmups": args.warmups,
        "formal_repetitions": args.repetitions,
        "devices": {f"{dataset}_q{q}": devices[index]
                    for index, (dataset, q) in enumerate(cases)},
        "binary": str(args.cli.resolve()), "binary_sha256": sha256(args.cli),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_measurements, all_latencies = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_case, dataset, q, devices[index], args.output,
                            args.cli.resolve(), args.warmups, args.repetitions): (dataset, q)
            for index, (dataset, q) in enumerate(cases)
        }
        for future in concurrent.futures.as_completed(futures):
            dataset, q = futures[future]
            measurements, latencies = future.result()
            all_measurements.extend(measurements)
            all_latencies.extend(latencies)
            print(f"completed {dataset} Q={q}", flush=True)
    write_csv(args.output / "raw_measurements.csv", all_measurements,
              ("dataset", "q", "variant", "repetition"))
    write_csv(args.output / "query_latency.csv", all_latencies,
              ("dataset", "q", "variant", "repetition", "query_id", "tail_label"))
    rows = summarize(all_measurements, all_latencies, args.output)
    for row in rows:
        print(row["dataset"], f"Q={row['q']}",
              f"global/A={row['B_eager_global_e2e_speedup_vs_A']:.4f}",
              f"group/A={row['D_eager_group_e2e_speedup_vs_A']:.4f}",
              f"group/global={row['D_eager_group_e2e_speedup_vs_B']:.4f}")


if __name__ == "__main__":
    main()
