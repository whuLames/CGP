#!/usr/bin/env python3
"""Seven-graph formal Shared/Static/Iteration matrix for All-Push and Hybrid."""
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
STATIC = {"cit-Patents": (16, 1), "soc-LiveJournal1": (16, 1),
          "indochina": (2, 2), "soc-orkut": (16, 2), "soc-twitter": (4, 3)}
METRICS = ("workload_ms", "throughput_qps", "kernel_gpu_ms", "feature_ms",
           "frontier_ms", "copy_ms", "selector_ms", "round_ms", "rounds",
           "push_rounds", "pull_rounds", "active_slot_ratio")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def parse(text):
    output = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try: output[key] = float(value)
        except ValueError: output[key] = value
    return output


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    preferred = [key for key in ("dataset", "workload", "mode", "mapping", "repetition") if key in fields]
    fields = preferred + [key for key in fields if key not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def read_fingerprints(path):
    with path.open(newline="") as handle:
        return {int(row["query_id"]): (row["source"], row["vertices"], row["sum64"], row["xor64"])
                for row in csv.DictReader(handle)}


class Worker:
    def __init__(self, args, dataset, device):
        self.args, self.dataset, self.device = args, dataset, device
        self.graph, self.directed = DATASETS[dataset]
        self.root = args.output / dataset; self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True); self.log = self.root / "commands.jsonl"
        self.rows = []

    def mappings(self):
        return ("shared", "static", "iteration") if self.dataset in STATIC else ("shared", "iteration")

    def command(self, workload, mode, mapping, fingerprints=None, rounds=None):
        command = [str(self.args.cli), f"--graph={self.graph}",
                   f"--queries={self.args.inputs/self.dataset/(workload+'.csv')}",
                   "--n=1024", "--q=64", "--layout=grouped", "--group_width=32",
                   f"--algorithm={workload}", f"--selector={'push' if mode == 'all_push' else 'threshold'}",
                   "--pull_threshold=0.20", "--frontier=unordered", "--frontier_build=direct",
                   "--frontier_mask64=true", "--same_algorithm_groups", "--profile_kernel",
                   "--legacy_int_weights", "--log_level=warn", f"--device={self.device}",
                   f"--push_mapping={mapping}"]
        if self.directed: command.append("--directed")
        if mapping == "static":
            lanes, grain = STATIC[self.dataset]
            command += [f"--push_query_lanes={lanes}", f"--push_grain={grain}"]
        if fingerprints: command.append(f"--result_fingerprints={fingerprints}")
        if rounds: command.append(f"--round_metrics={rounds}")
        return command

    def execute(self, command, stem, metadata):
        started = time.time(); result = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
        stdout, stderr = self.raw / f"{stem}.stdout.log", self.raw / f"{stem}.stderr.log"
        stdout.write_text(result.stdout); stderr.write_text(result.stderr)
        with self.log.open("a") as handle:
            handle.write(json.dumps({**metadata, "started": datetime.now().astimezone().isoformat(),
                "returncode": result.returncode, "wall_ms": (time.time()-started)*1000,
                "command": command}) + "\n")
        if result.returncode:
            raise RuntimeError(f"{self.dataset}/{stem}: {result.stderr[-3000:]}")
        return parse(result.stdout)

    def run(self):
        for workload in self.args.workloads:
            for mode in self.args.modes:
                reference = None
                for mapping in self.mappings():
                    path = self.raw / f"validate_{mode}_{workload}_{mapping}.csv"
                    self.execute(self.command(workload, mode, mapping, fingerprints=path),
                                 f"validate_{mode}_{workload}_{mapping}",
                                 {"phase": "validate", "workload": workload, "mode": mode, "mapping": mapping})
                    current = read_fingerprints(path)
                    if len(current) != 1024: raise RuntimeError("incomplete fingerprints")
                    if reference is None: reference = current
                    elif current != reference: raise RuntimeError(f"fingerprint mismatch {workload}/{mode}/{mapping}")
                for mapping in self.mappings():
                    self.execute(self.command(workload, mode, mapping),
                                 f"warmup_{mode}_{workload}_{mapping}",
                                 {"phase": "warmup", "workload": workload, "mode": mode, "mapping": mapping})
                    for repetition in range(self.args.repetitions):
                        stem = f"formal_{mode}_{workload}_{mapping}_r{repetition}"
                        rounds = self.raw / f"{stem}_rounds.csv" if mapping == "iteration" and repetition == 0 else None
                        values = self.execute(self.command(workload, mode, mapping, rounds=rounds), stem,
                            {"phase": "formal", "workload": workload, "mode": mode,
                             "mapping": mapping, "repetition": repetition})
                        self.rows.append({"dataset": self.dataset, "workload": workload,
                            "mode": mode, "mapping": mapping, "repetition": repetition,
                            **{key: values[key] for key in METRICS if key in values}})
                        write_csv(self.root / "raw_measurements.csv", self.rows)
        return self.rows


def summarize(rows):
    output = []
    groups = sorted({(row["dataset"], row["workload"], row["mode"]) for row in rows})
    for dataset, workload, mode in groups:
        selected = [row for row in rows if (row["dataset"], row["workload"], row["mode"]) ==
                    (dataset, workload, mode)]
        medians = {mapping: statistics.median(float(row["workload_ms"]) for row in selected
                   if row["mapping"] == mapping) for mapping in {row["mapping"] for row in selected}}
        for mapping, value in medians.items():
            output.append({"dataset": dataset, "workload": workload, "mode": mode,
                           "mapping": mapping, "samples": sum(row["mapping"] == mapping for row in selected),
                           "workload_ms_median": value,
                           "speedup_vs_shared": medians["shared"] / value,
                           "speedup_vs_static": medians.get("static", float("nan")) / value})
    return output


def geomean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--workloads", nargs="+", choices=("bfs", "sssp", "sswp"), default=["bfs", "sssp", "sswp"])
    parser.add_argument("--modes", nargs="+", choices=("all_push", "hybrid"), default=["all_push", "hybrid"])
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    args.inputs = args.inputs.resolve(); args.output = args.output.resolve(); args.cli = args.cli.resolve()
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) < len(DATASETS): raise SystemExit("seven devices required")
    config = {"started_at": datetime.now().astimezone().isoformat(), "N": 1024, "Q": 64, "G": 32,
              "datasets": list(DATASETS), "workloads": args.workloads, "modes": args.modes,
              "formal_repetitions": args.repetitions, "devices": dict(zip(DATASETS, devices)),
              "binary": str(args.cli), "binary_sha256": digest(args.cli), "inputs": str(args.inputs),
              "road_static_policy": "not reported; no seed-45 frozen road configuration"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(Worker(args, dataset, devices[index]).run): dataset
                   for index, dataset in enumerate(DATASETS)}
        for future in concurrent.futures.as_completed(futures):
            dataset = futures[future]; rows.extend(future.result()); print(f"completed {dataset}", flush=True)
    write_csv(args.output / "raw_measurements.csv", rows)
    summary = summarize(rows); write_csv(args.output / "summary.csv", summary)
    iteration = [row for row in summary if row["mapping"] == "iteration"]
    metrics = {"status": "success", "fingerprints_match": True,
               "formal_measurements": len(rows),
               "iteration_speedup_vs_shared_geomean": geomean([row["speedup_vs_shared"] for row in iteration]),
               "iteration_speedup_vs_static_five_graph_geomean": geomean([
                   row["speedup_vs_static"] for row in iteration if row["dataset"] in STATIC])}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
