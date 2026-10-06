#!/usr/bin/env python3
"""Build and run controlled SSSP head-of-line refill traces."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "experiments/20260929_strong_tail_workloads"
DATASETS = {"soc-orkut": False, "soc-twitter": True}
TRACES = {"tail128": 128, "tail256_hol": 256}
VARIANTS = {
    "A_no_refill": ["--same_algorithm_groups"],
    "B_refill": ["--group_refill", "--eager_sssp_refill"],
    "C_interference_g2": ["--group_refill", "--interference_aware_refill",
                            "--refill_max_active_groups=2"],
    "D_interference_g3": ["--group_refill", "--interference_aware_refill",
                            "--refill_max_active_groups=3"],
    "E_interference_bridge": ["--group_refill", "--interference_bridge_refill"],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_queries(path):
    rows = []
    identity = None
    with path.open() as handle:
        for line in handle:
            if line.startswith("# graph_identity="):
                identity = line.strip().split("=", 1)[1]
            if not line.strip() or line.startswith("#"):
                continue
            values = next(csv.reader([line]))
            rows.append(dict(zip(("id", "source", "score", "offset", "feature_key",
                                  "algorithm", "reference_rounds"), values)))
    if not identity:
        raise RuntimeError(f"missing graph identity: {path}")
    return identity, rows


def read_labels(dataset):
    path = DATA_ROOT / dataset / "workloads_predicted/workload_audit.csv"
    with path.open(newline="") as handle:
        return {int(row["id"]): row for row in csv.DictReader(handle)
                if row["workload"] == "sssp"}


def build_trace(dataset, trace, long_count, output):
    source = DATA_ROOT / dataset / "workloads_predicted/sssp_strong_tail.csv"
    identity, rows = read_queries(source)
    labels = read_labels(dataset)
    long_rows = [row for row in rows if labels[int(row["id"])]["label"] == "long"]
    short_rows = [row for row in rows if labels[int(row["id"])]["label"] == "short"]
    if len(long_rows) != 64 or len(short_rows) != 960 or long_count not in (128, 256):
        raise RuntimeError("unexpected frozen strong-tail composition")
    short_count = 1024 - long_count
    rng = random.Random(45 + long_count)
    selected_short = rng.sample(short_rows, short_count)
    selected_short.sort(key=lambda row: (int(row["feature_key"]), int(row["id"])))

    expanded_long = []
    for index in range(long_count):
        row = dict(long_rows[index % len(long_rows)])
        row["id"] = str(20_000_000 + (0 if dataset == "soc-orkut" else 1_000_000) + index)
        row["base_id"] = long_rows[index % len(long_rows)]["id"]
        expanded_long.append(row)
    expanded_long.sort(key=lambda row: (int(row["reference_rounds"]), int(row["base_id"]), int(row["id"])))

    long_groups = [expanded_long[i:i + 32] for i in range(0, long_count, 32)]
    short_groups = [selected_short[i:i + 32] for i in range(0, short_count, 32)]
    ordered_groups = []
    long_at = set(range(8)) if long_count == 256 else {0, 2, 4, 6}
    long_cursor = short_cursor = 0
    for batch in range(8):
        groups = []
        if batch in long_at:
            groups.append(("long", long_groups[long_cursor]))
            long_cursor += 1
        while len(groups) < 4:
            groups.append(("short", short_groups[short_cursor]))
            short_cursor += 1
        ordered_groups.append(groups)
    if long_cursor != len(long_groups) or short_cursor != len(short_groups):
        raise RuntimeError("trace construction did not consume all groups")

    workload = output / "workloads" / dataset / f"sssp_{trace}.csv"
    audit = output / "workloads" / dataset / f"sssp_{trace}_audit.csv"
    workload.parent.mkdir(parents=True, exist_ok=True)
    with workload.open("w", newline="") as handle:
        handle.write(f"# graph_identity={identity}\n# capacity=128\n")
        handle.write(f"# controlled {trace}: FIFO groups, G=32, Q=128, long_queries={long_count}\n")
        handle.write("# id,source,score,offset,feature_key,algorithm,reference_rounds\n")
        writer = csv.writer(handle)
        for groups in ordered_groups:
            for _, group in groups:
                for row in group:
                    writer.writerow([row[key] for key in ("id", "source", "score", "offset",
                                                          "feature_key", "algorithm", "reference_rounds")])
    audit_rows = []
    order = 0
    for batch, groups in enumerate(ordered_groups):
        for group_in_batch, (kind, group) in enumerate(groups):
            for row in group:
                query_id = int(row["id"])
                base_id = int(row.get("base_id", row["id"]))
                audit_rows.append({"order": order, "query_id": query_id, "base_id": base_id,
                                   "source": int(row["source"]), "tail_label": kind,
                                   "batch": batch, "group_in_batch": group_in_batch,
                                   "reference_rounds": int(row["reference_rounds"]),
                                   "feature_key": int(row["feature_key"])})
                order += 1
    write_csv(audit, audit_rows, ("order", "query_id", "base_id", "source", "tail_label",
                                  "batch", "group_in_batch", "reference_rounds", "feature_key"))
    manifest = {
        "dataset": dataset, "trace": trace, "graph_identity": identity,
        "N": 1024, "Q": 128, "G": 32, "long_queries": long_count,
        "short_queries": short_count, "long_groups": len(long_groups),
        "short_groups": len(short_groups), "long_batches": sorted(long_at),
        "source_workload": str(source.resolve()), "source_sha256": sha256(source),
        "workload_sha256": sha256(workload),
    }
    (workload.parent / f"sssp_{trace}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return workload, {row["query_id"]: row for row in audit_rows}


def write_csv(path, rows, preferred=()):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_metrics(text):
    result = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def common_command(cli, dataset, device, workload):
    root = DATA_ROOT / dataset
    command = [str(cli), f"--graph={root/'derived_graph'}", f"--queries={workload}",
               "--n=1024", "--q=128", "--layout=grouped", "--group_width=32",
               "--algorithm=sssp", "--selector=threshold", "--pull_threshold=0.20",
               "--frontier=unordered", "--frontier_build=direct", "--frontier_mask64=true",
               "--push_mapping=iteration", "--legacy_int_weights", "--planner=fifo",
               "--predictor=import_key",
               "--profile_kernel", "--log_level=warn", "--memory_fraction=0.95",
               f"--device={device}"]
    if DATASETS[dataset]:
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
        raise RuntimeError(f"command failed ({run.returncode}): {run.stderr[-2000:]}")
    return parse_metrics(run.stdout)


def read_fingerprints(path):
    with path.open(newline="") as handle:
        return {int(row["query_id"]): (row["source"], row["vertices"], row["sum64"], row["xor64"])
                for row in csv.DictReader(handle)}


def run_case(dataset, trace, device, output, cli, warmups, repetitions):
    workload, audit = build_trace(dataset, trace, TRACES[trace], output)
    case = output / dataset / trace
    raw = case / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    command_log = case / "commands.jsonl"
    base = common_command(cli, dataset, device, workload)
    reference = None
    for variant, flags in VARIANTS.items():
        path = raw / f"validate_{variant}_fingerprints.csv"
        run_one(base + flags + [f"--result_fingerprints={path}"],
                raw / f"validate_{variant}.stdout.log", raw / f"validate_{variant}.stderr.log",
                command_log, {"phase": "validate", "dataset": dataset,
                              "trace": trace, "variant": variant})
        current = read_fingerprints(path)
        if reference is None:
            reference = current
        elif current != reference:
            raise RuntimeError(f"fingerprint mismatch: {dataset}/{trace}")

    measurements, latency = [], []
    for repetition in range(-warmups, repetitions):
        order = list(VARIANTS)
        if repetition % 2:
            order.reverse()
        for variant in order:
            phase = "warmup" if repetition < 0 else "formal"
            rep = repetition if repetition >= 0 else repetition + warmups
            tag = f"{phase}_{variant}_r{rep}"
            completion = raw / f"{tag}_completion.csv"
            command = base + VARIANTS[variant]
            if phase == "formal":
                command.append(f"--completion_output={completion}")
            metrics = run_one(command, raw / f"{tag}.stdout.log", raw / f"{tag}.stderr.log",
                              command_log, {"phase": phase, "dataset": dataset,
                                            "trace": trace, "variant": variant,
                                            "repetition": repetition})
            if phase == "warmup":
                continue
            measurements.append({"dataset": dataset, "trace": trace, "variant": variant,
                                 "repetition": repetition, **metrics})
            with completion.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    query_id = int(row["query_id"])
                    meta = audit[query_id]
                    if int(row["service_rounds"]) != meta["reference_rounds"]:
                        raise RuntimeError(f"service-round mismatch: {dataset}/{trace}/q{query_id}")
                    latency.append({"dataset": dataset, "trace": trace, "variant": variant,
                                    "repetition": repetition, "query_id": query_id,
                                    "tail_label": meta["tail_label"], "batch": meta["batch"],
                                    "group_in_batch": meta["group_in_batch"],
                                    **{key: row[key] for key in ("waiting_ms", "service_ms",
                                                                 "submit_to_completion_ms",
                                                                 "waiting_rounds", "service_rounds",
                                                                 "submit_to_completion_rounds")}})
    write_csv(case / "raw_measurements.csv", measurements,
              ("dataset", "trace", "variant", "repetition"))
    write_csv(case / "query_latency.csv", latency,
              ("dataset", "trace", "variant", "repetition", "query_id", "tail_label"))
    return measurements, latency


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] * (high - position) + values[high] * (position - low)


def summarize(measurements, latency, output):
    rows = []
    for dataset, trace in sorted({(row["dataset"], row["trace"]) for row in measurements}):
        result = {"dataset": dataset, "trace": trace, "long_queries": TRACES[trace]}
        for variant in VARIANTS:
            selected = [row for row in measurements if row["dataset"] == dataset and
                        row["trace"] == trace and row["variant"] == variant]
            for metric in ("workload_ms", "kernel_gpu_ms", "feature_ms", "recycle_ms",
                           "group_refills", "active_slot_ratio", "refill_admitted_groups",
                           "refill_deferred_groups", "refill_deferred_pull_groups",
                           "refill_deferred_capacity_groups",
                           "refill_deferred_incompatible_groups"):
                result[f"{variant}_{metric}_median"] = statistics.median(float(row[metric]) for row in selected)
        for candidate in list(VARIANTS)[1:]:
            result[f"{candidate}_e2e_speedup"] = (result["A_no_refill_workload_ms_median"] /
                                                    result[f"{candidate}_workload_ms_median"])
        for label in ("all", "short", "long"):
            for metric in ("submit_to_completion_ms", "waiting_ms", "service_ms"):
                distributions = {}
                for variant in VARIANTS:
                    selected = [row for row in latency if row["dataset"] == dataset and
                                row["trace"] == trace and row["variant"] == variant and
                                (label == "all" or row["tail_label"] == label)]
                    ids = sorted({int(row["query_id"]) for row in selected})
                    values = [statistics.median(float(row[metric]) for row in selected
                                                if int(row["query_id"]) == query_id)
                              for query_id in ids]
                    distributions[variant] = values
                    result[f"{variant}_{label}_{metric}_mean"] = statistics.fmean(values)
                    for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99)):
                        result[f"{variant}_{label}_{metric}_{name}"] = quantile(values, fraction)
                for candidate in list(VARIANTS)[1:]:
                    result[f"{candidate}_{label}_{metric}_mean_speedup"] = (
                        statistics.fmean(distributions["A_no_refill"]) /
                        statistics.fmean(distributions[candidate]))
        rows.append(result)
    write_csv(output / "summary.csv", rows, ("dataset", "trace", "long_queries"))
    (output / "metrics.json").write_text(json.dumps({"all_fingerprints_match": True,
        "formal_measurements": len(measurements), "query_latency_rows": len(latency),
        "cases": rows}, indent=2) + "\n")
    return rows


def main():
    global VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        help="Subset to run; A_no_refill must be included for comparisons")
    args = parser.parse_args()
    if args.variants:
        if "A_no_refill" not in args.variants:
            raise SystemExit("--variants must include A_no_refill")
        VARIANTS = {name: VARIANTS[name] for name in args.variants}
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [(dataset, trace) for dataset in DATASETS for trace in TRACES]
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) < len(cases):
        raise SystemExit("four devices are required")
    config = {"started_at": datetime.now().astimezone().isoformat(), "N": 1024,
              "Q": 128, "G": 32, "datasets": list(DATASETS), "traces": TRACES,
              "variants": VARIANTS, "warmups": args.warmups,
              "formal_repetitions": args.repetitions, "binary": str(args.cli.resolve()),
              "binary_sha256": sha256(args.cli),
              "devices": {f"{dataset}_{trace}": devices[index]
                          for index, (dataset, trace) in enumerate(cases)}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_measurements, all_latency = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_case, dataset, trace, devices[index], args.output,
                                   args.cli.resolve(), args.warmups, args.repetitions): (dataset, trace)
                   for index, (dataset, trace) in enumerate(cases)}
        for future in concurrent.futures.as_completed(futures):
            dataset, trace = futures[future]
            measurements, latency = future.result()
            all_measurements.extend(measurements)
            all_latency.extend(latency)
            print(f"completed {dataset} {trace}", flush=True)
    write_csv(args.output / "raw_measurements.csv", all_measurements,
              ("dataset", "trace", "variant", "repetition"))
    write_csv(args.output / "query_latency.csv", all_latency,
              ("dataset", "trace", "variant", "repetition", "query_id", "tail_label"))
    for row in summarize(all_measurements, all_latency, args.output):
        for candidate in list(VARIANTS)[1:]:
            print(row["dataset"], row["trace"], candidate,
                  f"e2e={row[f'{candidate}_e2e_speedup']:.4f}",
                  f"all_latency={row[f'{candidate}_all_submit_to_completion_ms_mean_speedup']:.4f}",
                  f"short_latency={row[f'{candidate}_short_submit_to_completion_ms_mean_speedup']:.4f}",
                  f"long_latency={row[f'{candidate}_long_submit_to_completion_ms_mean_speedup']:.4f}")


if __name__ == "__main__":
    main()
