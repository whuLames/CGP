#!/usr/bin/env python3
"""Formal G=32 Shared/Static/Adaptive Hybrid campaign."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
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
WORKLOAD_ROOT = PROJECT / "experiments/20260927_fixed_n1024_m64"
GRAPHS = {
    "cit-Patents": Path("/home/zyl/data/csr_data/cit-Patents"),
    "soc-LiveJournal1": Path("/home/zyl/data/csr_data/soc-LiveJournal1"),
    "indochina": Path("/home/zyl/data/csr_data/indochina"),
    "soc-orkut": Path("/home/zyl/data/csr_data/soc-orkut"),
    "soc-twitter": Path("/home/zyl/data/csr_data/soc-twitter"),
}
IDENTITIES = {
    "cit-Patents": "ca71a63d2bc4aa14",
    "soc-LiveJournal1": "61876d811dd317b5",
    "indochina": "ab252d0d7a2c312e",
    "soc-orkut": "c476f4df35515f1e",
    "soc-twitter": "54434ea949424dbb",
}
STATIC = {
    "cit-Patents": (16, 1, "q16_w2"),
    "soc-LiveJournal1": (16, 1, "q16_w2"),
    "indochina": (2, 2, "q2_w4"),
    "soc-orkut": (16, 2, "q16_w4"),
    "soc-twitter": (4, 3, "q4_b2"),
}
WORKLOADS = ("bfs", "sssp", "sswp")
MAPPINGS = ("shared", "static", "adaptive")
METRICS = ("workload_ms", "throughput_qps", "kernel_ms", "kernel_gpu_ms",
           "feature_ms", "adaptive_preparation_ms", "frontier_ms", "copy_ms",
           "selector_ms", "round_ms", "rounds", "push_rounds", "pull_rounds")


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


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    preferred = [x for x in ("dataset", "workload", "mapping", "repetition") if x in fields]
    fields = preferred + [x for x in fields if x not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fingerprints(path):
    with path.open(newline="") as handle:
        return {int(r["query_id"]): (r["source"], r["completion_local_round"],
                r["vertices"], r["sum64"], r["xor64"]) for r in csv.DictReader(handle)}


def round_aggregates(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {"round_metric_rows": len(rows)}
    for name in ("adaptive_w1_vertices", "adaptive_w2_vertices", "adaptive_w4_vertices",
                 "adaptive_b2_vertices", "adaptive_b4_vertices"):
        result[name + "_sum"] = sum(int(r[name]) for r in rows)
    total = sum(result[x + "_sum"] for x in ("adaptive_w1_vertices", "adaptive_w2_vertices",
                                               "adaptive_w4_vertices", "adaptive_b2_vertices",
                                               "adaptive_b4_vertices"))
    if total:
        for name in ("adaptive_w1_vertices", "adaptive_w2_vertices", "adaptive_w4_vertices",
                     "adaptive_b2_vertices", "adaptive_b4_vertices"):
            result[name + "_fraction"] = result[name + "_sum"] / total
    return result


def prepare_inputs(output):
    manifest = []
    for dataset in GRAPHS:
        target_dir = output / "inputs" / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        for workload in WORKLOADS:
            source_name = "bfs_frozen.csv" if workload == "bfs" else "sssp_frozen.csv"
            source = WORKLOAD_ROOT / dataset / "workloads" / source_name
            lines = source.read_text().splitlines()
            lines[0] = "# graph_identity=" + IDENTITIES[dataset]
            if workload == "sswp":
                converted = []
                for line in lines:
                    if not line or line.startswith("#"):
                        converted.append(line)
                        continue
                    fields = line.split(",")
                    fields[5] = "2"
                    converted.append(",".join(fields))
                lines = converted
            target = target_dir / f"{workload}.csv"
            target.write_text("\n".join(lines) + "\n")
            data_rows = sum(bool(line and not line.startswith("#")) for line in lines)
            if data_rows != 1024:
                raise RuntimeError(f"{target}: expected 1024 queries, got {data_rows}")
            manifest.append({"dataset": dataset, "workload": workload, "source": str(source),
                             "target": str(target), "rows": data_rows,
                             "source_sha256": sha256(source), "target_sha256": sha256(target)})
    (output / "inputs" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class DatasetRunner:
    def __init__(self, args, dataset, device):
        self.args, self.dataset, self.device = args, dataset, device
        self.root = args.output / dataset
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.commands = self.root / "commands.jsonl"

    def command(self, workload, mapping, rounds=None, hashes=None):
        command = [str(self.args.cli), f"--graph={GRAPHS[self.dataset]}", "--directed",
                   "--legacy_int_weights", f"--queries={self.args.output / 'inputs' / self.dataset / (workload + '.csv')}",
                   "--n=1024", "--q=64", "--layout=grouped", "--group_width=32",
                   f"--algorithm={workload}", "--selector=threshold", "--pull_threshold=0.20",
                   "--frontier=unordered", "--frontier_build=direct", "--frontier_mask64=true",
                   "--same_algorithm_groups", "--profile_kernel", "--log_level=warn",
                   f"--device={self.device}", f"--push_mapping={mapping}"]
        if mapping == "static":
            lanes, grain, _ = STATIC[self.dataset]
            command += [f"--push_query_lanes={lanes}", f"--push_grain={grain}"]
        if rounds:
            command.append(f"--round_metrics={rounds}")
        if hashes:
            command.append(f"--result_fingerprints={hashes}")
        return command

    def execute(self, command, stem, kind, workload, mapping, repetition=None):
        started = time.time()
        run = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
        stdout_path, stderr_path = self.raw / f"{stem}.stdout.log", self.raw / f"{stem}.stderr.log"
        stdout_path.write_text(run.stdout); stderr_path.write_text(run.stderr)
        record = {"timestamp": datetime.now().astimezone().isoformat(), "dataset": self.dataset,
                  "device": self.device, "kind": kind, "workload": workload, "mapping": mapping,
                  "repetition": repetition, "returncode": run.returncode,
                  "process_wall_ms": (time.time() - started) * 1000, "command": command,
                  "stdout": str(stdout_path.relative_to(self.args.output)),
                  "stderr": str(stderr_path.relative_to(self.args.output))}
        with self.commands.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if run.returncode:
            raise RuntimeError(f"{self.dataset}/{stem} failed; see {stderr_path}")
        return parse_metrics(run.stdout)

    def validate(self, workload):
        values = {}
        for mapping in MAPPINGS:
            path = self.raw / f"validate_{workload}_{mapping}_fingerprints.csv"
            self.execute(self.command(workload, mapping, hashes=path),
                         f"validate_{workload}_{mapping}", "validation", workload, mapping)
            values[mapping] = fingerprints(path)
        if values["shared"] != values["static"] or values["shared"] != values["adaptive"]:
            raise RuntimeError(f"{self.dataset}/{workload}: fingerprint mismatch")

    def run(self):
        for workload in WORKLOADS:
            self.validate(workload)
            for mapping in MAPPINGS:
                self.execute(self.command(workload, mapping), f"warmup_{workload}_{mapping}",
                             "warmup", workload, mapping)
            for repetition in range(self.args.repetitions):
                order = MAPPINGS[repetition % 3:] + MAPPINGS[:repetition % 3]
                for mapping in order:
                    stem = f"formal_{workload}_{mapping}_r{repetition}"
                    rounds = self.raw / f"{stem}_rounds.csv"
                    parsed = self.execute(self.command(workload, mapping, rounds=rounds), stem,
                                          "formal", workload, mapping, repetition)
                    row = {"dataset": self.dataset, "workload": workload, "mapping": mapping,
                           "repetition": repetition, "static_source": STATIC[self.dataset][2],
                           **{key: parsed[key] for key in METRICS if key in parsed},
                           **round_aggregates(rounds)}
                    self.rows.append(row)
                    write_csv(self.root / "raw_measurements.csv", self.rows)
        return self.rows


def summaries(rows):
    output = []
    for dataset in GRAPHS:
        for workload in WORKLOADS:
            medians = {}
            for mapping in MAPPINGS:
                selected = [r for r in rows if r["dataset"] == dataset and
                            r["workload"] == workload and r["mapping"] == mapping]
                item = {"dataset": dataset, "workload": workload, "mapping": mapping,
                        "samples": len(selected), "static_source": STATIC[dataset][2]}
                for metric in METRICS:
                    values = [float(r[metric]) for r in selected if metric in r]
                    if values:
                        item[metric + "_median"] = statistics.median(values)
                        item[metric + "_min"] = min(values)
                        item[metric + "_max"] = max(values)
                for bucket in ("w1", "w2", "w4", "b2", "b4"):
                    key = f"adaptive_{bucket}_vertices_fraction"
                    values = [float(r[key]) for r in selected if key in r]
                    if values: item[key + "_median"] = statistics.median(values)
                medians[mapping] = item.get("workload_ms_median", 0)
                output.append(item)
            for item in output[-3:]:
                current = medians[item["mapping"]]
                if current:
                    item["speedup_vs_shared"] = medians["shared"] / current
                    item["speedup_vs_static"] = medians["static"] / current
    return output


def write_readme(output, status, summary):
    adaptive = [r for r in summary if r["mapping"] == "adaptive"]
    lines = ["# G=32 Vertex-Adaptive Push performance campaign", "", "## Purpose", "",
             "Compare Shared, independently frozen Static, and Adaptive Push under fixed Hybrid 0.20.", "",
             "## Scenario", "", "- Five real graphs; BFS, SSSP, and SSWP; N=1024, M=64, grouped G=32.",
             "- Direct unordered frontier, bitmask64 enabled, no refill.",
             "- One warmup and five formal repetitions per test case; mappings rotate by repetition.",
             "- One untimed full-result fingerprint validation per graph/workload/mapping.", "",
             "## Environment", "", f"- Platform: {platform.platform()}",
             "- Hardware: five Tesla V100-SXM2-32GB GPUs (devices 0-4).", "- Runtime: CUDA build in `build/graphweft_cli`.", "",
             "## Results", "", f"Campaign status: **{status}**.", "",
             "| Dataset | Workload | Adaptive speedup vs Shared | Adaptive speedup vs Static |", "|---|---|---:|---:|"]
    for row in adaptive:
        lines.append(f"| {row['dataset']} | {row['workload']} | {row.get('speedup_vs_shared', 0):.3f}x | {row.get('speedup_vs_static', 0):.3f}x |")
    lines += ["", "Raw commands, stdout/stderr, per-round metrics, fingerprints, and measurements are retained below this directory.", "",
              "## Observations", "", "See `summary.csv` and `metrics.json`.", "", "## Conclusion", "",
              "No hard speedup threshold was applied; gains and regressions are reported symmetrically.", "", "## Issues", "",
              "None." if status == "success" else "See `stderr.log` and per-case logs.", "", "## Next Steps", "",
              "Profile representative wins and regressions after reviewing the complete bucket distributions.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--devices", nargs=5, type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    args.output = args.output.resolve(); args.cli = args.cli.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "artifacts").mkdir(exist_ok=True)
    (args.output / "stdout.log").touch(); (args.output / "stderr.log").touch()
    prepare_inputs(args.output)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
                                capture_output=True, check=True).stdout.strip()
    config = {"name": "G=32 Vertex-Adaptive Push", "started_at": datetime.now().astimezone().isoformat(),
              "git_commit": git_commit, "binary_sha256": sha256(args.cli), "N": 1024, "M": 64,
              "group_width": 32, "selector": "Hybrid 0.20", "frontier": "direct unordered mask64",
              "refill": False, "warmups": 1, "formal_repetitions": args.repetitions,
              "graphs": {k: str(v) for k, v in GRAPHS.items()}, "devices": dict(zip(GRAPHS, args.devices)),
              "workloads": WORKLOADS, "mappings": MAPPINGS,
              "static_mappings": {k: v[2] for k, v in STATIC.items()},
              "environment": {"platform": platform.platform(), "python": platform.python_version()}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_rows, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(DatasetRunner(args, dataset, device).run): dataset
                for dataset, device in zip(GRAPHS, args.devices)}
        for future in concurrent.futures.as_completed(jobs):
            dataset = jobs[future]
            try:
                rows = future.result(); all_rows.extend(rows)
                print(f"complete {dataset}: {len(rows)} formal samples", flush=True)
            except Exception as error:
                failures.append({"dataset": dataset, "error": str(error)})
                print(f"failed {dataset}: {error}", file=sys.stderr, flush=True)
    write_csv(args.output / "result.csv", all_rows)
    summary = summaries(all_rows); write_csv(args.output / "summary.csv", summary)
    expected = len(GRAPHS) * len(WORKLOADS) * len(MAPPINGS) * args.repetitions
    status = "success" if not failures and len(all_rows) == expected else "failed"
    speedups = [r["speedup_vs_shared"] for r in summary if r["mapping"] == "adaptive"]
    metrics = {"status": status, "expected_test_cases": 45, "expected_formal_samples": expected,
               "completed_formal_samples": len(all_rows), "failures": failures,
               "adaptive_geomean_speedup_vs_shared": math.exp(statistics.fmean(math.log(x) for x in speedups)) if speedups else None}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_readme(args.output, status, summary)
    if status != "success": raise SystemExit(1)


if __name__ == "__main__":
    main()
