#!/usr/bin/env python3
"""Seven-graph online-cohort ablation: mapping, length grouping, and LSSS."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = {
    "cit-Patents": (Path("/home/zyl/data/csr_data/cit-Patents"), True),
    "soc-LiveJournal1": (Path("/home/zyl/data/csr_data/soc-LiveJournal1"), True),
    "indochina": (Path("/home/zyl/data/csr_data/indochina"), True),
    "soc-orkut": (Path("/home/zyl/data/csr_data/soc-orkut"), True),
    "soc-twitter": (Path("/home/zyl/data/csr_data/soc-twitter"), True),
    "roadNet-CA": (Path("/home/zyl/data/csr_data/roadNet-CA"), False),
    "roadNet-TX": (Path("/home/zyl/data/csr_data/roadNet-TX"), False),
}
VARIANTS = {
    "A_shared_fifo": ("raw", ["--push_mapping=shared", "--same_algorithm_groups"]),
    "B_iteration_fifo": ("raw", ["--push_mapping=iteration", "--same_algorithm_groups"]),
    "C_iteration_length": ("windowed", ["--push_mapping=iteration", "--same_algorithm_groups"]),
    "D_iteration_lsss": ("windowed", ["--push_mapping=iteration", "--group_refill",
                                               "--interference_bridge_refill"]),
}
METRICS = ("workload_ms", "throughput_qps", "kernel_gpu_ms", "feature_ms", "frontier_ms",
           "copy_ms", "selector_ms", "round_ms", "rounds", "push_rounds", "pull_rounds",
           "group_refills", "refill_admitted_groups", "refill_deferred_incompatible_groups",
           "active_slot_ratio")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): value.update(block)
    return value.hexdigest()


def parse_metrics(text):
    output = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try: output[key] = float(value)
        except ValueError: output[key] = value
    return output


def read_workload(path):
    comments, rows = [], []
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"): comments.append(line.rstrip())
            elif line.strip(): rows.append(next(csv.reader([line])))
    if len(rows) != 1024: raise RuntimeError(f"{path}: expected 1024 rows")
    return comments, rows


def write_workload(path, comments, rows, note):
    with path.open("w", newline="") as handle:
        for line in comments:
            if not line.startswith("# capacity="): handle.write(line + "\n")
        handle.write("# capacity=128\n" + f"# {note}\n")
        csv.writer(handle).writerows(rows)


def prepare_inputs(source_root, output):
    target = output / "inputs"; target.mkdir(parents=True)
    manifest = []
    for dataset in DATASETS:
        destination = target / dataset; destination.mkdir()
        for workload in ("bfs", "sssp", "sswp"):
            comments, rows = read_workload(source_root / dataset / f"{workload}.csv")
            raw = destination / f"{workload}_raw.csv"
            windowed = destination / f"{workload}_windowed.csv"
            write_workload(raw, comments, rows, "natural seed-42 arrival order")
            ordered = []
            for begin in range(0, len(rows), 128):
                cohort = rows[begin:begin + 128]
                cohort.sort(key=lambda row: (int(row[4]), int(row[0])))
                ordered.extend(cohort)
            write_workload(windowed, comments, ordered,
                           "online proxy: stable length ordering only within each Q=128 admission cohort")
            manifest.append({"dataset": dataset, "workload": workload,
                             "raw_sha256": digest(raw), "windowed_sha256": digest(windowed),
                             "cohort_size": 128, "group_width": 32})
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    preferred = [key for key in ("dataset", "workload", "variant", "repetition", "query_id") if key in fields]
    fields = preferred + [key for key in fields if key not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def fingerprints(path):
    with path.open(newline="") as handle:
        return {int(row["query_id"]): (row["source"], row["vertices"], row["sum64"], row["xor64"])
                for row in csv.DictReader(handle)}


class Worker:
    def __init__(self, args, dataset, device):
        self.args, self.dataset, self.device = args, dataset, device
        self.graph, self.directed = DATASETS[dataset]
        self.root = args.output / dataset; self.raw = self.root / "raw"; self.raw.mkdir(parents=True)
        self.log = self.root / "commands.jsonl"; self.measurements, self.latencies = [], []

    def command(self, workload, variant, completion=None, result_fingerprints=None):
        input_kind, flags = VARIANTS[variant]
        command = [str(self.args.cli), f"--graph={self.graph}",
                   f"--queries={self.args.output/'inputs'/self.dataset/(workload+'_'+input_kind+'.csv')}",
                   "--n=1024", "--q=128", "--layout=grouped", "--group_width=32",
                   f"--algorithm={workload}", "--selector=threshold", "--pull_threshold=0.20",
                   "--frontier=unordered", "--frontier_build=direct", "--frontier_mask64=true",
                   "--legacy_int_weights", "--planner=fifo", "--predictor=import_key",
                   "--profile_kernel", "--memory_fraction=0.95", "--log_level=warn",
                   f"--device={self.device}", *flags]
        if self.directed: command.append("--directed")
        if completion: command.append(f"--completion_output={completion}")
        if result_fingerprints: command.append(f"--result_fingerprints={result_fingerprints}")
        return command

    def execute(self, command, stem, metadata):
        started = time.time(); result = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
        stdout, stderr = self.raw / f"{stem}.stdout.log", self.raw / f"{stem}.stderr.log"
        stdout.write_text(result.stdout); stderr.write_text(result.stderr)
        with self.log.open("a") as handle:
            handle.write(json.dumps({**metadata, "started": datetime.now().astimezone().isoformat(),
                "returncode": result.returncode, "wall_ms": (time.time()-started)*1000,
                "command": command}) + "\n")
        if result.returncode: raise RuntimeError(f"{self.dataset}/{stem}: {result.stderr[-3000:]}")
        return parse_metrics(result.stdout)

    def run(self):
        for workload in self.args.workloads:
            reference = None
            for variant in VARIANTS:
                path = self.raw / f"validate_{workload}_{variant}.csv"
                self.execute(self.command(workload, variant, result_fingerprints=path),
                             f"validate_{workload}_{variant}",
                             {"phase": "validate", "workload": workload, "variant": variant})
                current = fingerprints(path)
                if len(current) != 1024: raise RuntimeError("incomplete fingerprints")
                if reference is None: reference = current
                elif current != reference: raise RuntimeError(f"fingerprint mismatch {workload}/{variant}")
            for variant in VARIANTS:
                self.execute(self.command(workload, variant), f"warmup_{workload}_{variant}",
                             {"phase": "warmup", "workload": workload, "variant": variant})
                for repetition in range(self.args.repetitions):
                    stem = f"formal_{workload}_{variant}_r{repetition}"
                    completion = self.raw / f"{stem}_completion.csv"
                    values = self.execute(self.command(workload, variant, completion=completion), stem,
                        {"phase": "formal", "workload": workload,
                         "variant": variant, "repetition": repetition})
                    self.measurements.append({"dataset": self.dataset, "workload": workload,
                        "variant": variant, "repetition": repetition,
                        **{key: values[key] for key in METRICS if key in values}})
                    with completion.open(newline="") as handle:
                        for row in csv.DictReader(handle):
                            self.latencies.append({"dataset": self.dataset, "workload": workload,
                                "variant": variant, "repetition": repetition,
                                **{key: row[key] for key in ("query_id", "waiting_ms", "service_ms",
                                    "submit_to_completion_ms", "waiting_rounds", "service_rounds")}})
                    write_csv(self.root / "raw_measurements.csv", self.measurements)
                    write_csv(self.root / "query_latency.csv", self.latencies)
        return self.measurements, self.latencies


def geomean(values): return math.exp(statistics.fmean(math.log(value) for value in values))


def summarize(measurements):
    rows = []
    for dataset, workload in sorted({(row["dataset"], row["workload"]) for row in measurements}):
        item = {"dataset": dataset, "workload": workload}; medians = {}
        for variant in VARIANTS:
            selected = [row for row in measurements if row["dataset"] == dataset and
                        row["workload"] == workload and row["variant"] == variant]
            medians[variant] = statistics.median(float(row["workload_ms"]) for row in selected)
            item[f"{variant}_workload_ms_median"] = medians[variant]
            for metric in ("rounds", "active_slot_ratio", "group_refills", "refill_admitted_groups"):
                item[f"{variant}_{metric}_median"] = statistics.median(float(row[metric]) for row in selected)
        item["mapping_speedup_A_over_B"] = medians["A_shared_fifo"] / medians["B_iteration_fifo"]
        item["length_speedup_B_over_C"] = medians["B_iteration_fifo"] / medians["C_iteration_length"]
        item["refill_speedup_C_over_D"] = medians["C_iteration_length"] / medians["D_iteration_lsss"]
        item["full_speedup_A_over_D"] = medians["A_shared_fifo"] / medians["D_iteration_lsss"]
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--workloads", nargs="+", choices=("bfs", "sssp", "sswp"), default=["bfs", "sssp", "sswp"])
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve(); args.cli = args.cli.resolve(); args.source_inputs = args.source_inputs.resolve()
    prepare_inputs(args.source_inputs, args.output)
    devices = [int(value) for value in args.devices.split(",")]
    config = {"started_at": datetime.now().astimezone().isoformat(), "N": 1024, "Q": 128,
              "G": 32, "datasets": list(DATASETS), "workloads": args.workloads,
              "variants": VARIANTS, "formal_repetitions": args.repetitions,
              "devices": dict(zip(DATASETS, devices)), "binary": str(args.cli),
              "binary_sha256": digest(args.cli), "arrival_model": "seed42 order; reorder only within Q=128 cohort"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    measurements, latencies = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(Worker(args, dataset, devices[index]).run): dataset
                   for index, dataset in enumerate(DATASETS)}
        for future in concurrent.futures.as_completed(futures):
            dataset = futures[future]; current_m, current_l = future.result()
            measurements.extend(current_m); latencies.extend(current_l); print(f"completed {dataset}", flush=True)
    write_csv(args.output / "raw_measurements.csv", measurements)
    write_csv(args.output / "query_latency.csv", latencies)
    summary = summarize(measurements); write_csv(args.output / "summary.csv", summary)
    metrics = {"status": "success", "fingerprints_match": True,
               "formal_measurements": len(measurements), "query_latency_rows": len(latencies)}
    for key in ("mapping_speedup_A_over_B", "length_speedup_B_over_C",
                "refill_speedup_C_over_D", "full_speedup_A_over_D"):
        metrics[key + "_geomean"] = geomean([row[key] for row in summary])
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
