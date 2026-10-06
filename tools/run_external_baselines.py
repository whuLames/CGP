#!/usr/bin/env python3
"""Run frozen GraphWeft workloads on Gunrock-S/C and Groute-S/C.

Mixed-tail is partitioned into its BFS and SSSP members because both native
baseline runners accept exactly one algorithm per invocation.  The reported
mixed runtime is the sum of those two resident-graph runtimes; it is not
heterogeneous co-residency.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKLOAD_ROOT = PROJECT / "experiments/20260927_fixed_n1024_m64"
GUNROCK_ROOT = Path("/home/zyl/Projects/ocgp/baselines/gunrockV2.2")
GUNROCK_BIN = GUNROCK_ROOT / "build_six_cuda2/bin"
GROUTE_ROOT = Path("/home/zyl/Projects/ocgp/baselines/groute")
GROUTE_BIN = GROUTE_ROOT / "build-loader/groute-batch"
GRAPHS = {
    "cit-Patents": Path("/home/zyl/data/ggr_data/singlegpu/cit-Patents.gr"),
    "soc-LiveJournal1": Path("/home/zyl/data/ggr_data/graphweft-exact/soc-LiveJournal1.gr"),
    "indochina": Path("/home/zyl/data/ggr_data/graphweft-exact/indochina.gr"),
    "soc-orkut": Path("/home/zyl/data/ggr_data/singlegpu/soc-orkut.gr"),
    "soc-twitter": Path("/home/zyl/data/ggr_data/singlegpu/soc-twitter.gr"),
    "roadNet-CA": Path("/home/zyl/data/ggr_data/roadNet-CA.gr"),
    "roadNet-TX": Path("/home/zyl/data/ggr_data/roadNet-TX.gr"),
}


def load_workload(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if not row:
                continue
            try:
                rows.append({"id": int(row[0]), "source": int(row[1]),
                             "algorithm": int(row[5]) if len(row) > 5 else -1})
            except ValueError:
                continue
    return rows


def write_sources(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row['source']}\n" for row in rows))


def run_command(command, cwd, env, stdout_path, stderr_path, timeout):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                capture_output=True, timeout=timeout)
        code = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nTIMEOUT\n"
    wall_ms = (time.perf_counter() - started) * 1000.0
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    (stdout_path.parent / (stdout_path.stem + ".command.json")).write_text(
        json.dumps({"command": command, "cwd": str(cwd),
                    "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                    "wall_ms": wall_ms, "exit_code": code}, indent=2) + "\n")
    return code, wall_ms


def gunrock_runtime(path):
    document = json.loads(path.read_text())
    return sum(float(batch["wall_ms"]) for batch in document["batches"]), float(document["total_wall_ms"])


def groute_runtime(path):
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise RuntimeError(f"empty Groute output: {path}")
    row = rows[-1]
    return float(row["runtime_ms"]), int(row["actual_q"]), int(row["iterations"]), int(row["peak_used_bytes"])


def probe_capacity(system, dataset, algorithm, graph, sources, output, env, timeout):
    marker = output / "capacity" / system / dataset / f"{algorithm}.json"
    if marker.exists():
        return int(json.loads(marker.read_text())["actual_m"])
    attempts = []
    probe_sources = output / "inputs" / dataset / f"{algorithm}-capacity-full.txt"
    write_sources(probe_sources, sources)
    for candidate in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        run_dir = output / "capacity" / system / dataset / algorithm / f"m{candidate}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if system == "gunrock":
            result_path = run_dir / "result.json"
            command = [str(GUNROCK_BIN / f"{algorithm}_concurrent"), "-m", str(graph),
                       "--query_file", str(probe_sources), "--num_queries", str(len(sources)),
                       "--num_streams", str(candidate), "-d", str(run_dir),
                       "-f", result_path.name]
            code, wall = run_command(command, GUNROCK_ROOT, env, run_dir / "stdout.log",
                                     run_dir / "stderr.log", timeout)
            success = code == 0 and result_path.exists()
        else:
            result_path = run_dir / "runs.csv"
            command = [str(GROUTE_BIN), f"--graph={graph}", f"--graph-name={dataset}",
                       f"--sources={probe_sources}", f"--algorithm={algorithm}",
                       "--mode=concurrent", "--gpu=0", f"--n={len(sources)}",
                       f"--q={candidate}", "--no-auto-q", "--warmups=0", "--repeats=1",
                       f"--output={result_path}"]
            code, wall = run_command(command, GROUTE_ROOT, env, run_dir / "stdout.log",
                                     run_dir / "stderr.log", timeout)
            success = code == 0 and result_path.exists()
        attempts.append({"m": candidate, "exit_code": code, "wall_ms": wall,
                         "success": success})
        if success:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"system": system, "dataset": dataset,
                "algorithm": algorithm, "actual_m": candidate, "attempts": attempts}, indent=2) + "\n")
            return candidate
    raise RuntimeError(f"no supported M for {system} {dataset} {algorithm}")


def run_gunrock(dataset, algorithm, graph, sources_path, n, m, run_dir, env,
                repetitions, timeout):
    rows = []
    for repetition in range(-1, repetitions):
        kind = "warmup" if repetition < 0 else f"rep{repetition}"
        one = run_dir / kind
        one.mkdir(parents=True, exist_ok=True)
        result_path = one / "result.json"
        command_record = one / "stdout.command.json"
        if result_path.exists() and command_record.exists():
            process_ms = float(json.loads(command_record.read_text())["wall_ms"])
        else:
            command = [str(GUNROCK_BIN / f"{algorithm}_concurrent"), "-m", str(graph),
                       "--query_file", str(sources_path), "--num_queries", str(n),
                       "--num_streams", str(m), "-d", str(one), "-f", result_path.name]
            code, process_ms = run_command(command, GUNROCK_ROOT, env, one / "stdout.log",
                                           one / "stderr.log", timeout)
            if code or not result_path.exists():
                raise RuntimeError(f"Gunrock failed: {dataset} {algorithm} M={m} {kind}")
        runtime_ms, resident_ms = gunrock_runtime(result_path)
        rows.append({"run_kind": "warmup" if repetition < 0 else "formal",
                     "repeat": max(0, repetition), "runtime_ms": runtime_ms,
                     "resident_wall_ms": resident_ms, "process_wall_ms": process_ms})
    return rows


def run_groute(dataset, algorithm, graph, sources_path, n, m, run_dir, env,
               repetitions, timeout):
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "runs.csv"
    command = [str(GROUTE_BIN), f"--graph={graph}", f"--graph-name={dataset}",
               f"--sources={sources_path}", f"--algorithm={algorithm}",
               f"--mode={'sequential' if m == 1 else 'concurrent'}", "--gpu=0",
               f"--n={n}", f"--q={m}", "--no-auto-q", "--warmups=1",
               f"--repeats={repetitions}", f"--output={result_path}"]
    code, process_ms = run_command(command, GROUTE_ROOT, env, run_dir / "stdout.log",
                                   run_dir / "stderr.log", timeout)
    if code or not result_path.exists():
        raise RuntimeError(f"Groute failed: {dataset} {algorithm} M={m}")
    rows = []
    for row in csv.DictReader(result_path.open(newline="")):
        rows.append({"run_kind": row["run_kind"], "repeat": int(row["repeat"]),
                     "runtime_ms": float(row["runtime_ms"]),
                     "resident_wall_ms": float(row["runtime_ms"]),
                     "process_wall_ms": process_ms,
                     "iterations": int(row["iterations"]),
                     "peak_gpu_bytes": int(row["peak_used_bytes"])})
    return rows


def append_rows(path, rows):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "experiments/20260928_external_baselines")
    parser.add_argument("--workload-root", type=Path, default=WORKLOAD_ROOT,
                        help="Root containing <dataset>/<algorithm>.csv or <dataset>/workloads/*.csv")
    parser.add_argument("--datasets", nargs="+", choices=tuple(GRAPHS), default=list(GRAPHS))
    parser.add_argument("--systems", nargs="+", choices=("gunrock", "groute"), default=["gunrock", "groute"])
    parser.add_argument("--workloads", nargs="+", choices=("bfs", "sssp", "sswp", "mixed_tail"),
                        default=["bfs", "sssp", "mixed_tail"])
    parser.add_argument("--modes", nargs="+", choices=("sequential", "concurrent"),
                        default=["sequential", "concurrent"])
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    # Native runners execute with their own repository as cwd.  Normalize a
    # caller-supplied relative output directory before passing query/result
    # paths across that cwd boundary.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    raw_path = args.output / "raw.csv"
    completed = set()
    if raw_path.exists():
        for row in csv.DictReader(raw_path.open(newline="")):
            completed.add((row["system"], row["dataset"], row["workload"], row["mode"],
                           row["algorithm"], row["run_kind"], row["repeat"]))
    for dataset in args.datasets:
        graph = GRAPHS[dataset]
        ordinary = {}
        required_algorithms = {workload for workload in args.workloads if workload != "mixed_tail"}
        if "mixed_tail" in args.workloads:
            required_algorithms.update(("bfs", "sssp"))
        for algorithm in sorted(required_algorithms):
            direct = args.workload_root / dataset / f"{algorithm}.csv"
            workload_path = direct if direct.exists() else args.workload_root / dataset / "workloads" / f"{algorithm}.csv"
            rows = load_workload(workload_path)
            ordinary[algorithm] = rows
            write_sources(args.output / "inputs" / dataset / f"{algorithm}.txt", rows)
        mixed_parts = {}
        if "mixed_tail" in args.workloads:
            mixed_path = args.workload_root / dataset / "mixed_tail.csv"
            if not mixed_path.exists():
                mixed_path = args.workload_root / dataset / "workloads/mixed_tail.csv"
            mixed = load_workload(mixed_path)
            mixed_parts = {"bfs": [r for r in mixed if r["algorithm"] == 0],
                           "sssp": [r for r in mixed if r["algorithm"] == 1]}
            for algorithm, rows in mixed_parts.items():
                write_sources(args.output / "inputs" / dataset / f"mixed_tail-{algorithm}.txt", rows)
        for system in args.systems:
            capacities = {}
            if "concurrent" in args.modes:
                for algorithm in sorted(required_algorithms):
                    capacities[algorithm] = probe_capacity(system, dataset, algorithm, graph,
                        ordinary[algorithm], args.output, env, args.timeout)
            for workload in args.workloads:
                algorithms = ("bfs", "sssp") if workload == "mixed_tail" else (workload,)
                for mode in args.modes:
                    for algorithm in algorithms:
                        rows = mixed_parts[algorithm] if workload == "mixed_tail" else ordinary[algorithm]
                        n = len(rows)
                        m = 1 if mode == "sequential" else min(capacities[algorithm], n)
                        source_name = f"mixed_tail-{algorithm}.txt" if workload == "mixed_tail" else f"{algorithm}.txt"
                        sources_path = args.output / "inputs" / dataset / source_name
                        expected = {("warmup", "0")} | {("formal", str(i)) for i in range(args.repetitions)}
                        if all((system, dataset, workload, mode, algorithm, k, r) in completed for k, r in expected):
                            continue
                        run_dir = args.output / "runs" / system / dataset / workload / mode / algorithm
                        runner = run_gunrock if system == "gunrock" else run_groute
                        measured = runner(dataset, algorithm, graph, sources_path, n, m, run_dir,
                                          env, args.repetitions, args.timeout)
                        output_rows = []
                        for row in measured:
                            output_rows.append({"system": system, "dataset": dataset,
                                "workload": workload, "mixed_execution":
                                "algorithm_partitioned" if workload == "mixed_tail" else "homogeneous",
                                "mode": mode, "algorithm": algorithm, "n": n, "m": m,
                                "run_kind": row["run_kind"], "repeat": row["repeat"],
                                "runtime_ms": row["runtime_ms"],
                                "resident_wall_ms": row["resident_wall_ms"],
                                "process_wall_ms": row["process_wall_ms"],
                                "iterations": row.get("iterations", ""),
                                "peak_gpu_bytes": row.get("peak_gpu_bytes", "")})
                        append_rows(raw_path, output_rows)
                        completed.update((r["system"], r["dataset"], r["workload"], r["mode"],
                                          r["algorithm"], r["run_kind"], str(r["repeat"])) for r in output_rows)
                        print(f"DONE {system} {dataset} {workload}/{algorithm} {mode} N={n} M={m}", flush=True)
    # Produce component and workload summaries. Mixed rows pair repetitions by
    # summing the BFS and SSSP native runtimes.
    rows = list(csv.DictReader(raw_path.open(newline="")))
    grouped = {}
    for row in rows:
        if row["run_kind"] != "formal":
            continue
        key = (row["system"], row["dataset"], row["workload"], row["mode"], row["repeat"])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    by_config = {}
    for key, parts in grouped.items():
        system, dataset, workload, mode, repetition = key
        expected_parts = 2 if workload == "mixed_tail" else 1
        if len(parts) != expected_parts:
            continue
        runtime = sum(float(p["runtime_ms"]) for p in parts)
        config = (system, dataset, workload, mode)
        by_config.setdefault(config, []).append(runtime)
    for (system, dataset, workload, mode), values in sorted(by_config.items()):
        aggregate.append({"system": system, "dataset": dataset, "workload": workload,
            "mode": mode, "samples": len(values), "runtime_ms_median": statistics.median(values),
            "runtime_ms_min": min(values), "runtime_ms_max": max(values),
            "throughput_qps": 1024000.0 / statistics.median(values)})
    summary_path = args.output / "summary.csv"
    if aggregate:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)


if __name__ == "__main__":
    main()
