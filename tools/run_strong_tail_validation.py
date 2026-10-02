#!/usr/bin/env python3
"""Run each frozen strong-tail workload once and retain completion/fingerprint evidence."""
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--directed", action="store_true")
    p.add_argument("--cli", type=Path, default=Path("build/graphweft_cli"))
    args = p.parse_args()
    root = args.dataset_root.resolve(); output = root / "validation"; output.mkdir(exist_ok=True)
    cli = args.cli.resolve()
    for workload, algorithm in (("bfs", "bfs"), ("sssp", "sssp"), ("mixed", "bfs")):
        query = root / "workloads" / f"{workload}_strong_tail.csv"
        completion = output / f"{workload}_completion.csv"
        fingerprints = output / f"{workload}_fingerprints.csv"
        command = [str(cli), f"--graph={root/'derived_graph'}", f"--queries={query}",
                   "--n=1024", "--q=64", "--layout=grouped", "--group_width=32",
                   f"--algorithm={algorithm}", "--frontier=unordered", "--frontier_build=direct",
                   "--frontier_mask64=true", "--push_mapping=shared", "--pull_threshold=0.20",
                   "--group_refill", "--legacy_int_weights", "--log_level=warn",
                   f"--completion_output={completion}", f"--result_fingerprints={fingerprints}"]
        if args.directed: command.append("--directed")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device}
        started = datetime.now().astimezone().isoformat()
        run = subprocess.run(command, text=True, capture_output=True, env=env)
        (output / f"{workload}.stdout.log").write_text(run.stdout)
        (output / f"{workload}.stderr.log").write_text(run.stderr)
        with (output / "commands.jsonl").open("a") as f:
            f.write(json.dumps({"started": started, "workload": workload,
                                "returncode": run.returncode, "command": command}) + "\n")
        if run.returncode:
            raise SystemExit(f"{workload} failed: {run.stderr}")
        print(f"complete {root.name}/{workload}", flush=True)


if __name__ == "__main__": main()
