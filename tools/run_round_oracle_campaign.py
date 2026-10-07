#!/usr/bin/env python3
"""Run resumable matched-state oracle and production-baseline experiments.

The manifest owns graph/workload placement; this runner never downloads or
converts data.  Invoke one process per GPU with disjoint --job entries.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
GIB = 1 << 30


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def free_bytes(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def pause_for_space(root: Path, threshold: int, context):
    state = {"status": "paused_insufficient_space", "utc": datetime.now(timezone.utc).isoformat(),
             "free_bytes": free_bytes(root), "threshold_bytes": threshold, **context}
    atomic_json(root / "PAUSED_INSUFFICIENT_SPACE.json", state)
    print("\n*** GRAPHWEFT CAMPAIGN PAUSED: /data space is below the safety threshold. ***", flush=True)
    print(json.dumps(state, indent=2), flush=True)


def telemetry(device: int):
    fields = "uuid,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,memory.free"
    command = ["nvidia-smi", f"--id={device}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True)
    return {"command": command, "returncode": result.returncode, "value": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def complete(run_dir: Path, outputs):
    state = run_dir / "run.json"
    if not state.exists():
        return False
    try:
        record = json.loads(state.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "success" and all(path.exists() and path.stat().st_size for path in outputs)


def monitored_run(command, run_dir: Path, storage_root: Path, device: int, outputs, reserve_space: int, low_space: int):
    if complete(run_dir, outputs):
        print("skip complete", run_dir, flush=True)
        return True
    if free_bytes(storage_root) < reserve_space:
        pause_for_space(storage_root, reserve_space, {"next_run": str(run_dir)})
        return False
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "cwd": str(PROJECT), "device": device,
              "telemetry_before": telemetry(device)}
    atomic_json(run_dir / "run.json", record)
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(device)
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stdout, stderr=stderr, text=True)
        while process.poll() is None:
            if free_bytes(storage_root) < low_space:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                record.update({"status": "paused_insufficient_space", "returncode": process.returncode,
                               "ended_utc": datetime.now(timezone.utc).isoformat()})
                atomic_json(run_dir / "run.json", record)
                pause_for_space(storage_root, low_space, {"interrupted_run": str(run_dir)})
                return False
            time.sleep(1)
    record.update({"status": "success" if process.returncode == 0 else "failed",
                   "returncode": process.returncode,
                   "ended_utc": datetime.now(timezone.utc).isoformat(),
                   "telemetry_after": telemetry(device),
                   "outputs": {str(path): sha256(path) for path in outputs if path.exists()}})
    atomic_json(run_dir / "run.json", record)
    if process.returncode:
        raise RuntimeError(f"failed run {run_dir}; see stderr.log")
    return True


def parse_job(spec):
    fields = spec.split(":")
    if len(fields) != 5:
        raise ValueError("--job must be DATASET:WORKLOAD:M:REFILL:GPU")
    dataset, workload, capacity, refill, device = fields
    return dataset, workload, int(capacity), refill, int(device)


def common_command(args, dataset, workload, capacity, refill):
    graph = Path(dataset["graph"])
    query = Path(dataset["workloads"][workload])
    command = [str(args.cli), f"--graph={graph}", f"--queries={query}", "--n=1024",
               f"--q={capacity}", "--layout=grouped", "--group_width=32", "--algorithm=sssp",
               "--selector=threshold", f"--pull_threshold={args.pull_threshold}", "--pull_kernel=auto",
               "--frontier=unordered", "--frontier_build=fused", "--frontier_mask64=true",
               "--same_algorithm_groups", "--device=0", "--log_level=warn"]
    if dataset.get("directed", False): command.append("--directed")
    if dataset.get("legacy_int_weights", True): command.append("--legacy_int_weights")
    if refill != "none": command += ["--group_refill", "--eager_sssp_refill"]
    if refill not in {"none", "eager_global", "eager_group"}:
        raise ValueError(f"unknown refill policy: {refill}")
    return command


def run_case(args, manifest, job):
    dataset_name, workload, capacity, refill, device = job
    if dataset_name not in manifest["datasets"]:
        raise ValueError(f"unknown dataset: {dataset_name}")
    dataset = manifest["datasets"][dataset_name]
    if workload not in dataset.get("workloads", {}):
        raise ValueError(f"unknown workload {dataset_name}/{workload}")
    case = args.output / dataset_name / workload / f"m{capacity}" / refill
    case.mkdir(parents=True, exist_ok=True)
    atomic_json(case / "config.json", {"dataset": dataset_name, "dataset_config": dataset,
        "workload": workload, "N": 1024, "M": capacity, "Q": capacity, "G": 32,
        "refill": refill, "warmups": 1, "formal_repetitions": 2,
        "pull_threshold": args.pull_threshold, "device": device,
        "binary": str(args.cli), "binary_sha256": sha256(args.cli)})
    common = common_command(args, dataset, workload, capacity, refill)

    variants = ("oracle", "adaptive", "iteration")
    for variant in variants:
        if refill == "eager_group" and variant == "adaptive":
            continue  # per-group Iteration mapping has no AdaptivePush equivalent
        variant_root = case / variant
        for repetition in (-1, 0, 1):
            label = "warmup" if repetition < 0 else f"rep{repetition}"
            run_dir = variant_root / label
            command = list(common)
            if variant == "oracle":
                command += ["--push_mapping=iteration", "--round_oracle_profile=core",
                            f"--round_oracle_order={'reverse' if repetition == 1 else 'forward'}"]
                if refill == "eager_group": command.append("--group_iteration_mapping")
                oracle = run_dir / "oracle.csv" if repetition >= 0 else Path("/dev/null")
                command.append(f"--round_oracle_output={oracle}")
                outputs = [] if repetition < 0 else [oracle]
            else:
                command.append(f"--push_mapping={variant}")
                if refill == "eager_group": command.append("--group_iteration_mapping")
                if repetition >= 0:
                    rounds, completions, fingerprints = (run_dir / "rounds.csv", run_dir / "completions.csv",
                                                          run_dir / "fingerprints.csv")
                    command += [f"--round_metrics={rounds}", f"--completion_output={completions}",
                                f"--result_fingerprints={fingerprints}"]
                    outputs = [rounds, completions, fingerprints]
                else: outputs = []
            if not monitored_run(command, run_dir, args.output, device, outputs,
                                 int(args.reserve_gib*GIB),int(args.pause_gib*GIB)):
                return False
    atomic_json(case / "COMPLETE.json", {"status": "success", "completed_utc": datetime.now(timezone.utc).isoformat()})
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=PROJECT / "build/graphweft_cli")
    parser.add_argument("--pull-threshold", type=float, default=.2)
    parser.add_argument("--reserve-gib",type=float,default=30)
    parser.add_argument("--pause-gib",type=float,default=25)
    parser.add_argument("--job", action="append", required=True,
                        help="DATASET:WORKLOAD:M:REFILL:GPU; repeated jobs run sequentially")
    args = parser.parse_args()
    if args.reserve_gib<=args.pause_gib or args.pause_gib<=0:
        parser.error("--reserve-gib must be greater than --pause-gib, both positive")
    args.manifest=args.manifest.resolve();args.output=args.output.resolve();args.cli=args.cli.resolve()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    paused=args.output/"PAUSED_INSUFFICIENT_SPACE.json"
    if paused.exists() and free_bytes(args.output)>=int(args.reserve_gib*GIB):paused.unlink()
    campaign_path=args.output/"campaign.json";previous={}
    if campaign_path.exists():
        try:previous=json.loads(campaign_path.read_text())
        except json.JSONDecodeError:previous={}
    requested=list(previous.get("jobs",[]))
    for job in args.job:
        if job not in requested:requested.append(job)
    atomic_json(campaign_path, {"schema": 1, "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest), "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "started_utc": previous.get("started_utc",datetime.now(timezone.utc).isoformat()), "jobs": requested})
    for specification in args.job:
        if not run_case(args, manifest, parse_job(specification)):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
