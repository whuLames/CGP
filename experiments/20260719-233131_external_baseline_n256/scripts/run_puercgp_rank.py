#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
PUER_ROOT = Path("/home/zyl/Projects/ocgp/puercgp")
DATA_ROOT = Path("/home/zyl/data/csr_data")
DATASETS = ("cit-Patents", "soc-orkut", "soc-twitter", "soc-sinaweibo")
ALGORITHMS = ("pagerank", "ppr")


def command_for(dataset: str, algorithm: str, output: Path) -> list[str]:
    graph = DATA_ROOT / dataset
    sources = EXP_ROOT / "query_sets" / dataset / "sources.csv"
    return [
        str(PUER_ROOT / "build/bench_rank_baseline"),
        str(graph),
        str(output),
        f"--algorithm={algorithm}",
        f"--sources={sources}",
        "--total-queries=256",
        "--batch-size=32",
        "--iterations=10",
        "--warmups=2",
        "--repeats=5",
    ]


def run_case(dataset: str, algorithm: str, gpu: int, resume: bool) -> int:
    case = f"puercgp-{dataset}-{algorithm}-fixed10"
    output = EXP_ROOT / "artifacts" / "puercgp_rank" / case
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if resume and status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("exit_code") == 0:
            print(f"SKIP {case}", flush=True)
            return 0

    command = command_for(dataset, algorithm, output)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.monotonic()
    process = subprocess.run(
        command,
        cwd=PUER_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wall_seconds = time.monotonic() - start
    (output / "stdout.log").write_text(process.stdout)
    (output / "stderr.log").write_text(process.stderr)
    status = {
        "case": case,
        "dataset": dataset,
        "algorithm": algorithm,
        "variants": ["push", "hybrid", "pull"],
        "gpu": gpu,
        "command": command,
        "cwd": str(PUER_ROOT),
        "exit_code": process.returncode,
        "wall_seconds": wall_seconds,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    with (EXP_ROOT / "stdout.log").open("a") as stream:
        stream.write(
            f"PUER_RANK {case} exit={process.returncode} "
            f"wall_s={wall_seconds:.3f}\n"
        )
    if process.stderr:
        with (EXP_ROOT / "stderr.log").open("a") as stream:
            stream.write(f"\n[{case}]\n{process.stderr}")
    print(
        f"DONE {case} exit={process.returncode} wall_s={wall_seconds:.3f}",
        flush=True,
    )
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
    failures = 0
    for dataset in datasets:
        for algorithm in algorithms:
            failures += run_case(dataset, algorithm, args.gpu, args.resume) != 0
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
