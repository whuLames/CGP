#!/usr/bin/env python3
"""Run the fixed GraphWeft Hybrid/direct/mask64 campaign.

Five dataset workers run concurrently on GPUs 0--4.  Within a dataset every
workload is serial, with the six configurations rotated for each repetition.
Warmups, validations, formal commands, logs, and completion records are all
retained under one campaign directory.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOADS = PROJECT / "experiments/20260927_fixed_n1024_m64"
DEFAULT_BASELINES = PROJECT / "experiments"
GRAPHS = {
    "cit-Patents": Path("/home/zyl/data/csr_data/cit-Patents"),
    "soc-LiveJournal1": Path("/home/zyl/data/csr_data/soc-LiveJournal1"),
    "indochina": Path("/home/zyl/data/csr_data/indochina"),
    "soc-orkut": Path("/home/zyl/data/csr_data/soc-orkut"),
    "soc-twitter": Path("/home/zyl/data/csr_data/soc-twitter"),
}
DIRECTED_IDENTITIES = {
    "cit-Patents": "ca71a63d2bc4aa14",
    "soc-LiveJournal1": "61876d811dd317b5",
    "indochina": "ab252d0d7a2c312e",
    # The frozen sources/keys were selected from the same symmetric edge set
    # through the undirected loader.  This campaign is explicitly directed,
    # whose graph identity differs only because directedness is hash-bound.
    "soc-orkut": "c476f4df35515f1e",
    "soc-twitter": "54434ea949424dbb",
}
WORKLOAD_FILES = {"bfs": "bfs_frozen.csv", "sssp": "sssp_frozen.csv",
                  "mixed_tail": "mixed_tail.csv"}
CONFIGS = (
    ("P", ("--selector=push", "--frontier_mask64=true", "--planner=fifo", "--same_algorithm_groups")),
    ("H0", ("--selector=threshold", "--frontier_mask64=false", "--planner=fifo", "--same_algorithm_groups")),
    ("A", ("--selector=threshold", "--frontier_mask64=true", "--planner=fifo", "--same_algorithm_groups")),
    ("B", ("--selector=threshold", "--frontier_mask64=true", "--planner=length", "--predictor=import_key", "--same_algorithm_groups")),
    ("C", ("--selector=threshold", "--frontier_mask64=true", "--planner=fifo", "--group_refill")),
    ("D", ("--selector=threshold", "--frontier_mask64=true", "--planner=length", "--predictor=import_key", "--group_refill")),
)
CONFIG_MAP = dict(CONFIGS)
ENGINE_METRICS = (
    "workload_ms", "throughput_qps", "kernel_ms", "kernel_gpu_ms", "frontier_ms",
    "copy_ms", "feature_ms", "selector_ms", "initialization_ms", "recycle_ms",
    "planning_ms", "prediction_ms", "transfer_ms", "execution_ms", "task_wall_ms",
    "round_ms", "rounds", "push_rounds", "pull_rounds", "group_refills",
    "completed_slot_rounds", "active_slot_ratio", "final_drain_rounds",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(command):
    return subprocess.run(["git", *command], cwd=PROJECT, text=True,
                          capture_output=True, check=True).stdout.strip()


def parse_metrics(text):
    parsed = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def distribution(rows, column, prefix, selected_ids=None):
    values = [float(row[column]) for row in rows
              if selected_ids is None or int(row["query_id"]) in selected_ids]
    if not values:
        return {}
    return {f"{prefix}_mean": statistics.fmean(values),
            f"{prefix}_p50": percentile(values, .50),
            f"{prefix}_p95": percentile(values, .95),
            f"{prefix}_p99": percentile(values, .99),
            f"{prefix}_max": max(values)}


def completion_metrics(path, labels):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"activation_ms", "completion_ms", "waiting_ms", "service_ms",
                "submit_to_completion_ms"}
    if len(rows) != 1024 or not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"invalid completion metadata: {path}")
    result = {}
    result.update(distribution(rows, "submit_to_completion_ms", "latency_ms"))
    result.update(distribution(rows, "waiting_ms", "waiting_ms"))
    result.update(distribution(rows, "service_ms", "service_ms"))
    for label, ids in labels.items():
        result.update(distribution(rows, "submit_to_completion_ms", f"{label}_latency_ms", ids))
    groups = {}
    for row in rows:
        key = (int(row["group"]), int(row["activation_round"]))
        groups.setdefault(key, []).append(float(row["completion_ms"]))
    waits = [max(values) - value for values in groups.values() for value in values]
    result["group_member_wait_ms_total"] = sum(waits)
    result["group_member_wait_ms_mean"] = statistics.fmean(waits) if waits else 0.0
    return result


def read_labels(path):
    if not path.exists():
        return {}
    labels = {"short": set(), "long": set()}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("label") in labels:
                labels[row["label"]].add(int(row["id"]))
    return labels


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    preferred = [key for key in ("dataset", "workload", "repetition", "variant") if key in fields]
    fields = preferred + [key for key in fields if key not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows, configs=CONFIGS):
    output = []
    for workload in WORKLOAD_FILES:
        for variant, _ in configs:
            selected = [row for row in rows if row["workload"] == workload and row["variant"] == variant]
            item = {"workload": workload, "variant": variant, "samples": len(selected)}
            numeric = sorted(set().union(*(row.keys() for row in selected)) -
                             {"dataset", "workload", "repetition", "variant"}) if selected else []
            for metric in numeric:
                try:
                    values = [float(row[metric]) for row in selected if row.get(metric, "") != ""]
                except (TypeError, ValueError):
                    continue
                if not values:
                    continue
                median = statistics.median(values)
                item[f"{metric}_median"] = median
                item[f"{metric}_min"] = min(values)
                item[f"{metric}_max"] = max(values)
                item[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
                item[f"{metric}_range_pct"] = ((max(values) - min(values)) / median * 100.0
                                                if median else 0.0)
            output.append(item)
    return output


class DatasetRunner:
    def __init__(self, args, dataset, device):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.root = args.output / dataset
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.commands = self.root / "commands.jsonl"
        self.rows = []
        self.configs = tuple((name, CONFIG_MAP[name]) for name in args.variants)

    def command(self, workload, variant, completion=None, hashes=None, small=False):
        graph = PROJECT / "tests/data/profile_graph.el" if small else GRAPHS[self.dataset]
        queries = PROJECT / "tests/data/mixed_queries.csv" if small else (
            self.args.input_root / self.dataset / WORKLOAD_FILES[workload])
        q = 4 if small else 64
        group_width = 2 if small else self.args.group_width
        algorithm = "bfs" if workload == "mixed_tail" else workload
        command = [str(self.args.cli), f"--graph={graph}", f"--queries={queries}",
                   f"--n={12 if small else 1024}", f"--q={q}", "--layout=grouped",
                   f"--group_width={group_width}", f"--algorithm={algorithm}",
                   "--frontier=unordered", "--frontier_build=direct", "--push_mapping=shared",
                   "--pull_threshold=0.20", "--directed", "--legacy_int_weights",
                   "--profile_kernel", "--log_level=warn", f"--device={self.device}",
                   *CONFIG_MAP[variant]]
        if completion:
            command.append(f"--completion_output={completion}")
        if hashes:
            command.append(f"--result_fingerprints={hashes}")
        return command

    def execute(self, command, stem, kind, workload, variant, repetition=None):
        start = time.time()
        run = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
        wall = (time.time() - start) * 1000.0
        stdout_path = self.raw / f"{stem}.stdout.log"
        stderr_path = self.raw / f"{stem}.stderr.log"
        stdout_path.write_text(run.stdout)
        stderr_path.write_text(run.stderr)
        record = {"time": datetime.now().astimezone().isoformat(), "dataset": self.dataset,
                  "device": self.device, "kind": kind, "workload": workload,
                  "variant": variant, "repetition": repetition, "returncode": run.returncode,
                  "process_wall_ms": wall, "command": command,
                  "stdout": str(stdout_path.relative_to(self.args.output)),
                  "stderr": str(stderr_path.relative_to(self.args.output))}
        with self.commands.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if run.returncode:
            raise RuntimeError(f"{self.dataset}/{stem} failed; see {stderr_path}")
        return parse_metrics(run.stdout), run.stdout

    @staticmethod
    def hash_values(path):
        with path.open(newline="") as handle:
            return {int(row["query_id"]): (row["source"], row["vertices"], row["sum64"], row["xor64"])
                    for row in csv.DictReader(handle)}

    def validate_small(self):
        hashes = {}
        for variant, _ in self.configs:
            path = self.raw / f"validate_small_{variant}_hashes.csv"
            self.execute(self.command("mixed_tail", variant, hashes=path, small=True),
                         f"validate_small_{variant}", "small_validation", "mixed_tail", variant)
            hashes[variant] = self.hash_values(path)
        reference = hashes[self.configs[0][0]]
        if any(value != reference for value in hashes.values()):
            raise RuntimeError(f"{self.dataset}: small-graph hash mismatch")

    def validate_formal(self, workload):
        hashes, metrics = {}, {}
        for variant, _ in self.configs:
            path = self.raw / f"validate_{workload}_{variant}_hashes.csv"
            parsed, _ = self.execute(self.command(workload, variant, hashes=path),
                                     f"validate_{workload}_{variant}", "formal_validation",
                                     workload, variant)
            hashes[variant] = self.hash_values(path)
            metrics[variant] = parsed
        reference_name = self.configs[0][0]
        if any(value != hashes[reference_name] for value in hashes.values()):
            raise RuntimeError(f"{self.dataset}/{workload}: result hash mismatch")
        if "H0" in metrics and "A" in metrics:
            for key in ("rounds", "push_rounds", "pull_rounds"):
                if metrics["H0"].get(key) != metrics["A"].get(key):
                    raise RuntimeError(f"{self.dataset}/{workload}: A/H0 {key} mismatch")

    def run(self):
        if not self.args.skip_validation:
            self.validate_small()
        for workload in WORKLOAD_FILES:
            labels = read_labels(self.args.workload_root / self.dataset / "workloads" /
                                 "mixed_tail_audit.csv") if workload == "mixed_tail" else {}
            if not self.args.skip_validation:
                self.validate_formal(workload)
            for variant, _ in self.configs:
                self.execute(self.command(workload, variant), f"warmup_{workload}_{variant}",
                             "warmup", workload, variant)
            for repetition in range(self.args.repetitions):
                rotated = self.configs[repetition % len(self.configs):] + self.configs[:repetition % len(self.configs)]
                for variant, _ in rotated:
                    stem = f"formal_{workload}_r{repetition}_{variant}"
                    completion = self.raw / f"{stem}_completion.csv"
                    metrics, _ = self.execute(self.command(workload, variant, completion=completion),
                                              stem, "formal", workload, variant, repetition)
                    row = {"dataset": self.dataset, "workload": workload,
                           "repetition": repetition, "variant": variant,
                           **{key: metrics[key] for key in ENGINE_METRICS if key in metrics},
                           **completion_metrics(completion, labels)}
                    total_rounds = float(row.get("push_rounds", 0)) + float(row.get("pull_rounds", 0))
                    row["push_round_ratio"] = float(row.get("push_rounds", 0)) / total_rounds if total_rounds else 0
                    row["pull_round_ratio"] = float(row.get("pull_rounds", 0)) / total_rounds if total_rounds else 0
                    self.rows.append(row)
                    write_csv(self.root / "raw_measurements.csv", self.rows)
        write_csv(self.root / "summary.csv", summarize_rows(self.rows, self.configs))
        return {"dataset": self.dataset, "formal_samples": len(self.rows), "status": "complete"}


def run_preflight(args):
    logs = args.output / "preflight"
    logs.mkdir(parents=True, exist_ok=True)
    commands = (["cmake", "--build", str(PROJECT / "build"), "-j4"],
                [str(PROJECT / "build/graphweft_frontier_validate")],
                [str(PROJECT / "build/graphweft_validate")],
                [str(PROJECT / "build/graphweft_partition_validate")],
                [str(PROJECT / "build/graphweft_pull_partition_validate")])
    for index, command in enumerate(commands):
        run = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
        (logs / f"{index}.stdout.log").write_text(run.stdout)
        (logs / f"{index}.stderr.log").write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(f"preflight failed: {' '.join(command)}")


def prepare_inputs(args):
    """Copy frozen rows verbatim while binding them to directed graph loading."""
    manifest = []
    args.input_root = args.output / "inputs"
    for dataset in args.datasets:
        destination = args.input_root / dataset
        destination.mkdir(parents=True, exist_ok=True)
        for filename in WORKLOAD_FILES.values():
            source = args.workload_root / dataset / "workloads" / filename
            target = destination / filename
            lines = source.read_text().splitlines()
            if not lines or not lines[0].startswith("# graph_identity="):
                raise RuntimeError(f"missing frozen graph identity: {source}")
            source_identity = lines[0].split("=", 1)[1]
            lines[0] = "# graph_identity=" + DIRECTED_IDENTITIES[dataset]
            target.write_text("\n".join(lines) + "\n")
            # Excluding the identity header, this assertion proves source
            # order, feature keys, algorithm tags, and reference lengths are
            # byte-for-byte unchanged.
            if source.read_text().splitlines()[1:] != target.read_text().splitlines()[1:]:
                raise RuntimeError(f"normalized workload rows changed: {source}")
            manifest.append({"dataset": dataset, "file": filename,
                             "source": str(source), "source_identity": source_identity,
                             "directed_identity": DIRECTED_IDENTITIES[dataset],
                             "source_sha256": sha256(source), "campaign_sha256": sha256(target),
                             "rows_unchanged": True})
    (args.input_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOADS)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--group-width", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=tuple(CONFIG_MAP), default=list(CONFIG_MAP))
    parser.add_argument("--datasets", nargs="+", choices=tuple(GRAPHS), default=list(GRAPHS))
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve(); args.cli = args.cli.resolve()
    args.workload_root = args.workload_root.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    if len(args.devices) < len(args.datasets):
        parser.error("one device is required per dataset")
    if args.group_width <= 0 or 64 % args.group_width:
        parser.error("group width must be a positive divisor of M=64")
    missing = [str(path) for dataset in args.datasets for path in
               (GRAPHS[dataset], *(args.workload_root / dataset / "workloads" / name
                                   for name in WORKLOAD_FILES.values())) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing campaign inputs:\n" + "\n".join(missing))
    diff = git(["diff", "--binary"])
    config = {"name": "GraphWeft Hybrid + bitmask64", "started_at": datetime.now().astimezone().isoformat(),
              "project": str(PROJECT), "git_commit": git(["rev-parse", "HEAD"]),
              "git_status": git(["status", "--short"]),
              "working_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
              "binary_sha256": sha256(args.cli), "datasets": args.datasets,
              "devices": dict(zip(args.datasets, args.devices)), "N": 1024, "M": 64, "G": args.group_width,
              "pull_threshold": .20, "warmups": 1, "formal_repetitions": args.repetitions,
              "variants": {name: list(CONFIG_MAP[name]) for name in args.variants},
              "workload_root": str(args.workload_root), "baseline_root": str(args.baseline_root),
              "environment": {"platform": platform.platform(), "python": platform.python_version()}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "working_tree.diff").write_text(diff)
    (args.output / "stdout.log").touch(); (args.output / "stderr.log").touch()
    prepare_inputs(args)
    if not args.skip_preflight:
        run_preflight(args)
        config["binary_sha256"] = sha256(args.cli)
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.datasets)) as pool:
        futures = {pool.submit(DatasetRunner(args, dataset, device).run): dataset
                   for dataset, device in zip(args.datasets, args.devices)}
        for future in concurrent.futures.as_completed(futures):
            dataset = futures[future]
            try:
                result = future.result()
                print(f"complete {dataset}: {result['formal_samples']} formal samples", flush=True)
                results.append(result)
            except Exception as error:
                print(f"failed {dataset}: {error}", file=sys.stderr, flush=True)
                results.append({"dataset": dataset, "status": "failed", "error": str(error)})
    metrics = {"status": "success" if all(row["status"] == "complete" for row in results) else "failed",
               "datasets": results, "expected_test_cases": len(args.datasets) * 3 * len(args.variants),
               "expected_formal_samples": len(args.datasets) * 3 * len(args.variants) * args.repetitions,
               "completed_formal_samples": sum(row.get("formal_samples", 0) for row in results)}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if metrics["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
