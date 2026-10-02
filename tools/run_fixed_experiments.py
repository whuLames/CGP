#!/usr/bin/env python3
"""Run the fixed N=1024/M=64 scheduling matrix and retain every raw log."""
import argparse
import csv
import json
import re
import statistics
import subprocess
from pathlib import Path


def variants(include_hybrid, include_oracle, include_g, only):
    base = [
        ("A_fifo_batch", ["--same_algorithm_groups", "--planner=fifo"]),
        ("B_length_batch", ["--same_algorithm_groups", "--planner=length", "--predictor=import_key"]),
        ("C_fifo_refill", ["--group_refill", "--planner=fifo"]),
        ("D_length_refill", ["--group_refill", "--planner=length", "--predictor=import_key"]),
    ]
    result = [(name + "_push", flags + ["--selector=push"]) for name, flags in base]
    if only != "all" and only != "base":
        result = []
    if include_hybrid:
        result += [(name + "_hybrid", flags + ["--selector=threshold"]) for name, flags in (base[0], base[3])]
    if include_oracle:
        result += [("oracle_refill_push", ["--group_refill", "--oracle_order", "--selector=push"])]
    if include_g:
        for groups in (1, 4, 8, 16, 32, 64):
            result.append((f"G{groups}_length_refill_push",
                           ["--group_refill", "--planner=length", "--predictor=import_key",
                            "--selector=push", f"--group_width={64//groups}"]))
    return result


def fields(text):
    found = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", text):
        try: found[key] = float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError: found[key] = value
    return found


def group_wait(path):
    groups = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault((int(row["group"]), int(row["activation_round"])), []).append(int(row["completion_round"]))
    return sum(max(values) * len(values) - sum(values) for values in groups.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli", type=Path, default=Path("build/graphweft_cli"))
    p.add_argument("--graph", type=Path, required=True); p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--directed", action="store_true")
    p.add_argument("--legacy-int-weights", action="store_true"); p.add_argument("--device", type=int, default=0)
    p.add_argument("--hybrid", action="store_true"); p.add_argument("--oracle", action="store_true")
    p.add_argument("--g-sensitivity", action="store_true"); p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--only", choices=("base", "hybrid", "oracle", "g", "all"), default="base",
                   help="Run one suite without repeating the base A-D measurements")
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    # The legacy flags retain their additive behavior.  --only provides the
    # non-overlapping stages used by the fixed experiment campaign.
    hybrid = args.hybrid or args.only in ("hybrid", "all")
    oracle = args.oracle or args.only in ("oracle", "all")
    g_sensitivity = args.g_sensitivity or args.only in ("g", "all")
    configs = variants(hybrid, oracle, g_sensitivity, args.only)
    if args.only == "hybrid": configs = [x for x in configs if x[0].endswith("_hybrid")]
    elif args.only == "oracle": configs = [x for x in configs if x[0].startswith("oracle_")]
    elif args.only == "g": configs = [x for x in configs if x[0].startswith("G")]
    common = [str(args.cli), f"--graph={args.graph}", f"--queries={args.queries}", "--n=1024", "--q=64",
              "--layout=grouped", "--group_width=8", "--algorithm=bfs", "--log_level=warn"]
    if args.directed: common.append("--directed")
    if args.legacy_int_weights: common.append("--legacy_int_weights")
    common.append(f"--device={args.device}")
    (args.output / "config.json").write_text(json.dumps({"N": 1024, "M": 64, "warmups": 1,
        "measured_repetitions": args.repetitions, "variants": [x[0] for x in configs]}, indent=2) + "\n")
    rows = []
    # One warmup per configuration.  Measured order rotates each repetition.
    for name, flags in configs:
        subprocess.run(common + flags, check=True, stdout=subprocess.DEVNULL)
    for repetition in range(args.repetitions):
        order = configs[repetition % len(configs):] + configs[:repetition % len(configs)]
        for name, flags in order:
            stem = f"rep{repetition}_{name}"
            completion_path = args.output / (stem + "_completion.csv")
            command = common + flags + [f"--completion_output={completion_path}",
                                        f"--schedule_events={args.output/(stem+'_events.csv')}"]
            run = subprocess.run(command, check=True, text=True, capture_output=True)
            (args.output / (stem + ".log")).write_text(run.stdout + run.stderr)
            record={"repetition": repetition, "variant": name, **fields(run.stdout)}
            record["completed_slot_rounds"] = group_wait(completion_path)
            rows.append(record)
    keys = ["repetition", "variant"] + sorted(set().union(*(r.keys() for r in rows)) - {"repetition", "variant"})
    with (args.output / "timings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    metrics = ("workload_ms", "throughput_qps", "kernel_ms", "copy_ms", "frontier_ms",
               "planning_ms", "recycle_ms", "completed_slot_rounds", "active_slot_ratio",
               "final_drain_rounds")
    summary = []
    for name, _ in configs:
        selected = [r for r in rows if r["variant"] == name]
        item = {"variant": name, "samples": len(selected)}
        for metric in metrics:
            values = sorted(float(r[metric]) for r in selected if metric in r)
            if not values: continue
            item[metric + "_median"] = statistics.median(values)
            item[metric + "_min"] = values[0]; item[metric + "_max"] = values[-1]
        summary.append(item)
    summary_keys = ["variant", "samples"] + sorted(set().union(*(r.keys() for r in summary)) - {"variant", "samples"})
    with (args.output / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_keys); w.writeheader(); w.writerows(summary)


if __name__ == "__main__":
    main()
