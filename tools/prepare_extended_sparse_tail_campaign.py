#!/usr/bin/env python3
"""Prepare N=1024 SSSP traces with one sparse long G=32 group per Q=128 batch."""
import argparse
import csv
import json
import random
import subprocess
from array import array
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "experiments/20260927_fixed_n1024_m64"
DATASETS = {
    "cit-Patents": {"graph": Path("/home/zyl/data/csr_data/cit-Patents"), "directed": True,
                    "min": 256, "max": 1024},
    "soc-LiveJournal1": {"graph": Path("/home/zyl/data/csr_data/soc-LiveJournal1"), "directed": True,
                         "min": 256, "max": 1024},
    "indochina": {"graph": Path("/home/zyl/data/csr_data/indochina"), "directed": True,
                   "min": 256, "max": 1024},
    "roadNet-CA": {"graph": Path("/home/zyl/data/csr_data/roadNet-CA"), "directed": False,
                   "min": 2048, "max": 8192},
    "roadNet-TX": {"graph": Path("/home/zyl/data/csr_data/roadNet-TX"), "directed": False,
                   "min": 2048, "max": 8192},
}


def read_query_sources(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                values = next(csv.reader([line]))
                rows.append((int(values[0]), int(values[1])))
    return rows


def completion_rounds(path):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {int(row["query_id"]): int(row["service_rounds"])
                for row in csv.DictReader(handle)}


def tx_sources(graph, count, seed):
    offsets = array("i")
    with (graph / "csr_vlist.bin").open("rb") as handle:
        offsets.fromfile(handle, (graph / "csr_vlist.bin").stat().st_size // 4)
    eligible = [vertex for vertex in range(len(offsets) - 1)
                if offsets[vertex + 1] > offsets[vertex]]
    return [(30_000_000 + index, source) for index, source in
            enumerate(random.Random(seed).sample(eligible, count))]


def prepare_one(name, spec, output, cli):
    root = output / name
    root.mkdir(parents=True, exist_ok=True)
    derived = root / "derived_graph"
    command = ["python3", str(PROJECT / "tools/prepare_strong_tail_workloads.py"),
               "augment", "--graph", str(spec["graph"]), "--output", str(derived),
               "--paths", "256", "--min-edges", str(spec["min"]),
               "--max-edges", str(spec["max"]), "--alpha", "1.5", "--seed", "45"]
    if spec["directed"]:
        command.append("--directed")
    if not derived.exists():
        subprocess.run(command, cwd=PROJECT, check=True)
    manifest = json.loads((derived / "augmentation_manifest.json").read_text())

    candidate_path = BASE / name / "workloads/sssp_candidates.csv"
    if candidate_path.exists():
        candidates = read_query_sources(candidate_path)
    elif name == "roadNet-TX":
        candidates = tx_sources(spec["graph"], 4096, 43)
    else:
        raise RuntimeError(f"no candidates for {name}")
    rounds = completion_rounds(BASE / name / "sssp_candidate_completion.csv")
    if rounds:
        candidates.sort(key=lambda row: (rounds[row[0]], row[1]))
        pool = candidates[:len(candidates) // 2]
    else:
        pool = candidates
    short = random.Random(47).sample(pool, 768)
    long = list(zip(manifest["path_heads"][:256], manifest["path_edges"][:256]))
    long.sort(key=lambda row: (row[1], row[0]))
    short.sort(key=lambda row: (rounds.get(row[0], 0), row[1]))

    workload = root / "sssp_lsss_tail256.csv"
    audit = root / "workload_audit.csv"
    audit_rows = []
    with workload.open("w", newline="") as handle:
        # The exact derived identity is captured from every runtime command;
        # FIFO import intentionally does not require a frozen-plan identity.
        handle.write("# capacity=128\n")
        handle.write("# LSSS controlled sparse tail: one long G=32 group per Q=128 batch\n")
        handle.write("# id,source,score,offset,feature_key,algorithm,reference_rounds\n")
        writer = csv.writer(handle)
        order = 0
        for batch in range(8):
            groups = [("long", long[batch * 32:(batch + 1) * 32])]
            for group in range(3):
                begin = (batch * 3 + group) * 32
                groups.append(("short", short[begin:begin + 32]))
            for group_index, (label, values) in enumerate(groups):
                for item in values:
                    if label == "long":
                        source, path_edges = item
                        query_id = 40_000_000 + order
                        reference = path_edges + 1
                        feature_key = 0
                    else:
                        candidate_id, source = item
                        query_id = 40_000_000 + order
                        reference = rounds.get(candidate_id, 0)
                        feature_key = 1
                    writer.writerow((query_id, source, 0, 0, feature_key, 1, reference))
                    audit_rows.append({"order": order, "query_id": query_id, "source": source,
                                       "tail_label": label, "batch": batch,
                                       "group_in_batch": group_index,
                                       "reference_rounds": reference,
                                       "feature_key": feature_key})
                    order += 1
    with audit.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader(); writer.writerows(audit_rows)
    report = {"dataset": name, "original_graph": str(spec["graph"]),
              "derived_graph": str(derived), "graph_identity": "captured_at_runtime",
              "directed": spec["directed"], "N": 1024, "Q": 128, "G": 32,
              "long_queries": 256, "short_queries": 768,
              "long_path_edges_min": min(value for _, value in long),
              "long_path_edges_median": sorted(value for _, value in long)[127:129],
              "long_path_edges_max": max(value for _, value in long),
              "construction": "256 disjoint Pareto path components; one frontier vertex per query/round",
              "feature_key_policy": "synthetic long=0, original-graph short=1",
              "candidate_rounds_available": bool(rounds)}
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.datasets:
        print(f"preparing {name}", flush=True)
        reports.append(prepare_one(name, DATASETS[name], args.output, args.cli.resolve()))
        print(f"completed {name}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps({"datasets": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
