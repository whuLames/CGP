#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


EXP_ROOT = Path("/home/zyl/Projects/ocgp/experiment/experiments/20260609_forkgraph_glign_ibfs_bfs")
FORKGRAPH = Path("/home/zyl/Projects/ocgp/baselines/ForkGraph")
GLIGN = Path("/home/zyl/Projects/ocgp/baselines/Glign")
IBFS = Path("/home/zyl/Projects/ocgp/baselines/iBFS")
GGR_DIR = Path("/home/zyl/data/ggr_data/singlegpu")

DATASETS = ["cit-Patents", "soc-sinaweibo", "soc-twitter", "soc-orkut"]
CONCURRENCIES = [2, 4, 8, 16, 32, 64]
BASELINES = ["forkgraph", "glign", "ibfs"]
REPEATS = [1, 2, 3]

CSV_FIELDS = [
    "dataset", "baseline", "concurrency", "repeat", "query_file", "sources",
    "algo_time_ms", "wall_time_sec", "exit_code", "command", "log_path",
]


def load_sources(path: Path):
    return [int(x) for x in path.read_text().split()]


def command_for(baseline: str, dataset: str, concurrency: int, query_file: Path):
    graph = GGR_DIR / f"{dataset}.gr"
    if baseline == "forkgraph":
        part_dir = FORKGRAPH / "inputs_gr_partitioned"
        return {
            "cwd": FORKGRAPH,
            "cmd": [
                "./build/bfs", "-gr", "-p", "8",
                str(query_file),
                str(part_dir / f"{dataset}.inter.gr"),
                str(part_dir / f"{dataset}.intra.gr"),
                str(part_dir / f"{dataset}.part"),
            ],
        }
    if baseline == "glign":
        return {
            "cwd": GLIGN / "apps",
            "cmd": [
                "./BFS_Batch",
                "-option", "glign",
                "-mode", "3",
                "-delay",
                "-gr",
                "-batch", str(concurrency),
                "-max_combination", str(concurrency),
                "-qf", str(query_file),
                str(graph),
            ],
        }
    if baseline == "ibfs":
        return {
            "cwd": IBFS,
            "cmd": [
                "./gpu-ibfs",
                str(graph),
                str(concurrency),
                "8",
                str(concurrency),
            ],
        }
    raise ValueError(baseline)


def parse_algo_time_ms(baseline: str, text: str):
    if baseline == "forkgraph":
        m = re.search(r"exuection time us:\s*([0-9.]+)\s*us", text)
        if m:
            return float(m.group(1)) / 1000.0
        m = re.search(r"exuection time:\s*([0-9.]+)\s*ms", text)
        if m:
            return float(m.group(1))
    if baseline == "glign":
        matches = re.findall(r"evaluation time:\s*([0-9.eE+-]+)", text)
        if matches:
            return float(matches[-1]) * 1000.0
        matches = re.findall(r"batching evaluation time[: ]+([0-9.eE+-]+)", text)
        if matches:
            return float(matches[-1]) * 1000.0
    if baseline == "ibfs":
        matches = re.findall(r"Traversal-iter-\d+:\s*([0-9.eE+-]+)\s*second", text)
        if matches:
            return sum(float(x) for x in matches) * 1000.0
        matches = re.findall(r"Traversal-time:\s*([0-9.eE+-]+)\s*second", text)
        if matches:
            return sum(float(x) for x in matches) * 1000.0
        m = re.search(r"Total time:\s*([0-9.eE+-]+)", text)
        if m:
            return float(m.group(1)) * 1000.0
    return ""


def append_row(csv_path: Path, row: dict):
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_keys(csv_path: Path):
    keys = set()
    if not csv_path.exists():
        return keys
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            keys.add((row["dataset"], row["baseline"], int(row["concurrency"]), int(row["repeat"])))
    return keys


def run_one(dataset: str, baseline: str, concurrency: int, repeat: int, timeout_sec: int):
    query_file = EXP_ROOT / "query_sets" / dataset / f"q{concurrency}.sources"
    sources = load_sources(query_file)
    raw_dir = EXP_ROOT / "raw_logs" / dataset / baseline
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"q{concurrency}_r{repeat}.log"
    spec = command_for(baseline, dataset, concurrency, query_file)
    cmd = spec["cmd"]
    cwd = spec["cwd"]

    if baseline == "ibfs":
        shutil.copyfile(query_file, IBFS / "input.dat")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "96")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout_sec,
        )
        exit_code = proc.returncode
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        output = (exc.stdout or "") + f"\nTIMEOUT after {timeout_sec}s\n"
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
    wall = time.monotonic() - start
    log_text = (
        f"DATASET={dataset}\nBASELINE={baseline}\nCONCURRENCY={concurrency}\n"
        f"REPEAT={repeat}\nCWD={cwd}\nCOMMAND={' '.join(cmd)}\n"
        f"WALL_SEC={wall:.6f}\nEXIT_CODE={exit_code}\n\n{output}"
    )
    log_path.write_text(log_text)
    row = {
        "dataset": dataset,
        "baseline": baseline,
        "concurrency": concurrency,
        "repeat": repeat,
        "query_file": str(query_file),
        "sources": " ".join(str(s) for s in sources),
        "algo_time_ms": parse_algo_time_ms(baseline, output),
        "wall_time_sec": f"{wall:.6f}",
        "exit_code": exit_code,
        "command": " ".join(cmd),
        "log_path": str(log_path),
    }
    append_row(EXP_ROOT / "results.csv", row)
    return row


def summarize():
    import statistics
    rows = []
    csv_path = EXP_ROOT / "results.csv"
    if not csv_path.exists():
        return
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    groups = {}
    for row in rows:
        key = (row["dataset"], row["baseline"], row["concurrency"])
        groups.setdefault(key, []).append(row)
    out_path = EXP_ROOT / "summary_by_config.csv"
    fields = [
        "dataset", "baseline", "concurrency", "runs", "ok_runs",
        "median_algo_time_ms", "min_algo_time_ms", "max_algo_time_ms",
        "median_wall_time_sec", "min_wall_time_sec", "max_wall_time_sec",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, vals in sorted(groups.items()):
            ok = [v for v in vals if v["exit_code"] == "0" and v["algo_time_ms"]]
            algo = [float(v["algo_time_ms"]) for v in ok]
            wall = [float(v["wall_time_sec"]) for v in ok]
            writer.writerow({
                "dataset": key[0],
                "baseline": key[1],
                "concurrency": key[2],
                "runs": len(vals),
                "ok_runs": len(ok),
                "median_algo_time_ms": statistics.median(algo) if algo else "",
                "min_algo_time_ms": min(algo) if algo else "",
                "max_algo_time_ms": max(algo) if algo else "",
                "median_wall_time_sec": statistics.median(wall) if wall else "",
                "min_wall_time_sec": min(wall) if wall else "",
                "max_wall_time_sec": max(wall) if wall else "",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "one", "summarize"], required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--baselines", nargs="+", choices=BASELINES)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--concurrency", type=int, choices=CONCURRENCIES)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode == "summarize":
        summarize()
        return
    if args.mode == "one":
        if not (args.dataset and args.baseline and args.concurrency):
            raise SystemExit("--mode one requires --dataset --baseline --concurrency")
        rows = [(args.dataset, args.baseline, args.concurrency, args.repeat)]
    elif args.mode == "smoke":
        rows = [("cit-Patents", baseline, 2, 1) for baseline in BASELINES]
    else:
        datasets = args.datasets or DATASETS
        baselines = args.baselines or BASELINES
        rows = [
            (dataset, baseline, concurrency, repeat)
            for dataset in datasets
            for concurrency in CONCURRENCIES
            for baseline in baselines
            for repeat in REPEATS
        ]

    done = completed_keys(EXP_ROOT / "results.csv") if args.resume else set()
    for dataset, baseline, concurrency, repeat in rows:
        if (dataset, baseline, concurrency, repeat) in done:
            print(f"SKIP {dataset} {baseline} q{concurrency} r{repeat}", flush=True)
            continue
        print(f"RUN {dataset} {baseline} q{concurrency} r{repeat}", flush=True)
        row = run_one(dataset, baseline, concurrency, repeat, args.timeout_sec)
        print(
            f"  exit={row['exit_code']} algo_ms={row['algo_time_ms']} wall={row['wall_time_sec']}",
            flush=True,
        )
    summarize()


if __name__ == "__main__":
    main()
