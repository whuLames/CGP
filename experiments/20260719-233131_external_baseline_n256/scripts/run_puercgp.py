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
ALGORITHMS = ("bfs", "sssp", "sswp")
VARIANTS = ("push_online", "hybrid_online", "pull")
INDEX_NAMES = {
    "cit-Patents": "phase_index_sample_l64.bin",
    "soc-orkut": "phase_index_sample_l64.bin",
    "soc-twitter": "phase_index_sample_l32.bin",
    "soc-sinaweibo": "phase_index_sample_l16.bin",
}


def command_for(dataset: str, algorithm: str, variant: str, output: Path) -> list[str]:
    if variant == "push_online":
        mode, measure = "push", "online-only"
    elif variant == "hybrid_online":
        mode, measure = "hybrid", "online-only"
    else:
        mode, measure = "pull", "baseline-only"
    graph = DATA_ROOT / dataset
    index = graph / INDEX_NAMES[dataset]
    sources = EXP_ROOT / "query_sets" / dataset / "sources.csv"
    return [
        str(PUER_ROOT / "build/bench_online_runner"),
        str(graph),
        str(index),
        str(output),
        f"--algorithm={algorithm}",
        f"--mode={mode}",
        "--online-policy=full",
        f"--measure={measure}",
        "--total-queries=256",
        "--batch-size=64",
        "--repeats=5",
        "--warmups=2",
        f"--sources={sources}",
        "--batch-swaps=0",
        "--max-offset=16",
    ]


def run_case(dataset: str, algorithm: str, variant: str, gpu: int, resume: bool) -> int:
    case = f"puercgp-{dataset}-{algorithm}-{variant}"
    output = EXP_ROOT / "artifacts" / "puercgp" / case
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if resume and status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("exit_code") == 0:
            print(f"SKIP {case}", flush=True)
            return 0

    command = command_for(dataset, algorithm, variant, output)
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
        "variant": variant,
        "gpu": gpu,
        "command": command,
        "cwd": str(PUER_ROOT),
        "exit_code": process.returncode,
        "wall_seconds": wall_seconds,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    with (EXP_ROOT / "stdout.log").open("a") as stream:
        stream.write(
            f"PUER {case} exit={process.returncode} wall_s={wall_seconds:.3f}\n"
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
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
    variants = (args.variant,) if args.variant else VARIANTS
    failed = 0
    for dataset in datasets:
        for algorithm in algorithms:
            for variant in variants:
                failed += run_case(
                    dataset, algorithm, variant, args.gpu, args.resume
                ) != 0
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
