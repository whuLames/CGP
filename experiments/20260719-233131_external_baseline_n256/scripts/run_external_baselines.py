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


EXP_ROOT = Path(__file__).resolve().parents[1]
GR_ROOT = Path("/home/zyl/data/ggr_data/singlegpu")
IBFS_ROOT = Path("/home/zyl/Projects/ocgp/baselines/iBFS")
GLIGN_ROOT = Path("/home/zyl/Projects/ocgp/baselines/Glign")
FORKGRAPH_ROOT = Path("/home/zyl/Projects/ocgp/baselines/ForkGraph")
FORKGRAPH_DATA_ROOT = FORKGRAPH_ROOT / "inputs_gr_llc_range"
GUNROCK_ROOT = Path("/home/zyl/Projects/ocgp/baselines/gunrockV2.2")

DATASETS = ("cit-Patents", "soc-orkut", "soc-twitter", "soc-sinaweibo")
SYSTEMS = ("ibfs", "glign", "forkgraph", "gunrock")
DEFAULT_WARMUPS = 2
DEFAULT_REPEATS = 5


class CaseFailure(RuntimeError):
    pass


def run_process(command, cwd, env, stdout_path, stderr_path, timeout):
    start = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        wall_seconds = time.monotonic() - start
        stdout_path.write_text(error.stdout or "")
        stderr_path.write_text(error.stderr or "")
        raise CaseFailure(f"timeout after {wall_seconds:.3f}s") from error
    wall_seconds = time.monotonic() - start
    stdout_path.write_text(process.stdout)
    stderr_path.write_text(process.stderr)
    if process.returncode != 0:
        raise CaseFailure(f"exit code {process.returncode}")
    return process.stdout, wall_seconds


def source_path(dataset):
    return EXP_ROOT / "query_sets" / dataset / "sources.txt"


def graph_path(dataset):
    return GR_ROOT / f"{dataset}.gr"


def parse_single(pattern, output, label):
    matches = re.findall(pattern, output, flags=re.MULTILINE)
    if len(matches) != 1:
        raise CaseFailure(f"expected one {label}, found {len(matches)}")
    return float(matches[0])


def command_ibfs(dataset, case_dir):
    shutil.copyfile(source_path(dataset), case_dir / "input.dat")
    return [
        str(IBFS_ROOT / "gpu-ibfs"),
        str(graph_path(dataset)),
        "64",
        "8",
        "256",
    ]


def command_glign(dataset, algorithm):
    executable = {
        "bfs": "BFS_Batch",
        "sssp": "SSSP_Batch",
        "sswp": "SSWP_Batch",
    }[algorithm]
    return [
        str(GLIGN_ROOT / "apps" / executable),
        "-option",
        "glign",
        "-mode",
        "3",
        "-delay",
        "-gr",
        "-batch",
        "64",
        "-max_combination",
        "256",
        "-qf",
        str(source_path(dataset)),
        str(graph_path(dataset)),
    ]


def command_forkgraph(dataset, algorithm, query_file):
    summary = json.loads(
        (FORKGRAPH_DATA_ROOT / f"{dataset}.summary.json").read_text()
    )
    return [
        str(FORKGRAPH_ROOT / "build" / algorithm),
        "-p",
        str(summary["partitions"]),
        "-gr",
        str(query_file),
        str(FORKGRAPH_DATA_ROOT / f"{dataset}.intra.gr"),
        str(FORKGRAPH_DATA_ROOT / f"{dataset}.inter.gr"),
        str(FORKGRAPH_DATA_ROOT / f"{dataset}.part"),
    ]


def command_gunrock_bfs(dataset, variant, json_dir, json_file):
    streams = {"streams1": 1, "streams64": 64}[variant]
    return [
        str(GUNROCK_ROOT / "build_cuda" / "bin" / "bfs_concurrent"),
        "-m",
        str(graph_path(dataset)),
        "--query_file",
        str(source_path(dataset)),
        "--num_streams",
        str(streams),
        "--json_dir",
        str(json_dir),
        "--json_file",
        json_file,
    ]


def command_gunrock_sequential(dataset, algorithm, json_dir, json_file):
    command = [
        str(GUNROCK_ROOT / "build_cuda" / "bin" / algorithm),
        "-m",
        str(graph_path(dataset)),
        "--export_metrics",
        "--json_dir",
        str(json_dir),
        "--json_file",
        json_file,
    ]
    if algorithm == "sssp":
        sources = source_path(dataset).read_text().split()
        command.extend(("--src", ",".join(sources)))
    else:
        command.extend(("--num_runs", "256", "--max_iterations", "10"))
    return command


def parse_ibfs(output):
    values = re.findall(
        r"^Traversal-iter-\d+:\s*([0-9.eE+-]+)\s+second\(s\)$",
        output,
        flags=re.MULTILINE,
    )
    if len(values) != 4:
        raise CaseFailure(f"expected four iBFS groups, found {len(values)}")
    return sum(float(value) for value in values) * 1000.0, 64


def parse_glign(output):
    seconds = parse_single(
        r"^Glign evaluation time:\s*([0-9.eE+-]+)$",
        output,
        "Glign evaluation time",
    )
    return seconds * 1000.0, 64


def parse_forkgraph(output, algorithm):
    if algorithm in ("bfs", "sssp"):
        value = parse_single(
            r"^exuection time:\s*([0-9.eE+-]+)\s+ms$",
            output,
            "ForkGraph execution time",
        )
        return value
    seconds = parse_single(
        r"^computation time:\s*([0-9.eE+-]+)$",
        output,
        "ForkGraph PPR computation time",
    )
    return seconds * 1000.0


def run_regular_case(
    system, dataset, algorithm, variant, case_dir, env, warmups, repeats
):
    rows = []
    for run in range(-warmups, repeats):
        kind = "warmup" if run < 0 else "repeat"
        index = run + warmups if run < 0 else run
        stdout_path = case_dir / f"{kind}-{index}.stdout.log"
        stderr_path = case_dir / f"{kind}-{index}.stderr.log"
        if system == "ibfs":
            command = command_ibfs(dataset, case_dir)
            cwd = case_dir
            timeout = 3600
        elif system == "glign":
            command = command_glign(dataset, algorithm)
            cwd = GLIGN_ROOT / "apps"
            timeout = 7200
        else:
            json_file = f"{kind}-{index}.json"
            if algorithm == "bfs":
                command = command_gunrock_bfs(
                    dataset, variant, case_dir, json_file
                )
            else:
                command = command_gunrock_sequential(
                    dataset, algorithm, case_dir, json_file
                )
            cwd = GUNROCK_ROOT
            timeout = 7200
        stdout, wall_seconds = run_process(
            command, cwd, env, stdout_path, stderr_path, timeout
        )
        if system == "ibfs":
            compute_ms, effective_q = parse_ibfs(stdout)
        elif system == "glign":
            compute_ms, effective_q = parse_glign(stdout)
        else:
            payload = json.loads((case_dir / json_file).read_text())
            if algorithm == "bfs":
                compute_ms = float(payload["total_wall_ms"])
                effective_q = int(payload["effective_batch_size"])
            else:
                process_times = payload["process_times"]
                if len(process_times) != 256:
                    raise CaseFailure(
                        f"expected 256 Gunrock runs, found {len(process_times)}"
                    )
                compute_ms = sum(float(value) for value in process_times)
                effective_q = 1
        print(
            f"PROGRESS {system}-{dataset}-{algorithm}-{variant} "
            f"{kind}={index} compute_ms={compute_ms:.3f} "
            f"wall_s={wall_seconds:.3f}",
            flush=True,
        )
        if run >= 0:
            rows.append(
                {
                    "repeat": run,
                    "compute_ms": compute_ms,
                    "process_wall_ms": wall_seconds * 1000.0,
                    "chunks": 1,
                    "effective_q": effective_q,
                }
            )
    return rows


def run_forkgraph_case(
    dataset, algorithm, case_dir, env, warmups, repeats, resume
):
    batch_size = 32 if algorithm == "ppr" else 64
    batch_count = 256 // batch_size
    rows = []
    for run in range(-warmups, repeats):
        kind = "warmup" if run < 0 else "repeat"
        index = run + warmups if run < 0 else run
        compute_ms = 0.0
        wall_seconds = 0.0
        for chunk in range(batch_count):
            query_file = (
                EXP_ROOT
                / "query_sets"
                / dataset
                / f"q{batch_size}_batch{chunk}.sources"
            )
            command = command_forkgraph(dataset, algorithm, query_file)
            stdout_path = case_dir / f"{kind}-{index}-chunk-{chunk}.stdout.log"
            stderr_path = case_dir / f"{kind}-{index}-chunk-{chunk}.stderr.log"
            metadata_path = case_dir / f"{kind}-{index}-chunk-{chunk}.meta.json"
            if resume and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                chunk_compute_ms = float(metadata["compute_ms"])
                chunk_wall = float(metadata["wall_seconds"])
                action = "resumed"
            else:
                stdout, chunk_wall = run_process(
                    command,
                    FORKGRAPH_ROOT,
                    env,
                    stdout_path,
                    stderr_path,
                    3600,
                )
                chunk_compute_ms = parse_forkgraph(stdout, algorithm)
                metadata_path.write_text(
                    json.dumps(
                        {
                            "compute_ms": chunk_compute_ms,
                            "wall_seconds": chunk_wall,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                action = "completed"
            compute_ms += chunk_compute_ms
            wall_seconds += chunk_wall
            print(
                f"PROGRESS forkgraph-{dataset}-{algorithm} {kind}={index} "
                f"chunk={chunk + 1}/{batch_count} "
                f"compute_ms={chunk_compute_ms:.3f} wall_s={chunk_wall:.3f} "
                f"action={action}",
                flush=True,
            )
        print(
            f"PROGRESS forkgraph-{dataset}-{algorithm} {kind}={index} "
            f"total_compute_ms={compute_ms:.3f} total_wall_s={wall_seconds:.3f}",
            flush=True,
        )
        if run >= 0:
            rows.append(
                {
                    "repeat": run,
                    "compute_ms": compute_ms,
                    "process_wall_ms": wall_seconds * 1000.0,
                    "chunks": batch_count,
                    "effective_q": batch_size,
                }
            )
    return rows


def validate_case(system, algorithm, variant):
    if system == "ibfs" and algorithm == "bfs" and variant == "native":
        return
    if system == "glign" and algorithm in ("bfs", "sssp", "sswp") \
            and variant == "native":
        return
    if system == "forkgraph" and algorithm in ("bfs", "sssp") \
            and variant in ("native", "llc48"):
        return
    if system == "forkgraph" and algorithm == "ppr" \
            and variant == "residual_native":
        return
    if system == "gunrock" and algorithm == "bfs" \
            and variant in ("streams1", "streams64"):
        return
    if system == "gunrock" and algorithm in ("sssp", "pr") \
            and variant == "sequential":
        return
    raise ValueError("unsupported system/algorithm/variant combination")


def write_rows(path, system, dataset, algorithm, variant, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "system",
                "dataset",
                "algorithm",
                "variant",
                "repeat",
                "compute_ms",
                "process_wall_ms",
                "chunks",
                "effective_q",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "system": system,
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "variant": variant,
                    **row,
                }
            )


def run_case(
    system, dataset, algorithm, variant, gpu, resume, warmups, repeats
):
    validate_case(system, algorithm, variant)
    case = f"{system}-{dataset}-{algorithm}-{variant}"
    case_dir = EXP_ROOT / "artifacts" / "external" / case
    case_dir.mkdir(parents=True, exist_ok=True)
    status_path = case_dir / "status.json"
    if resume and status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("state") == "complete":
            print(f"SKIP {case}", flush=True)
            return 0

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "96"
    env["OMP_DYNAMIC"] = "FALSE"
    if system == "forkgraph" and variant == "llc48":
        env["OMP_NUM_THREADS"] = "48"
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "spread"
    elif system == "gunrock":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif system == "ibfs":
        # This artifact has physical device 4 compiled into main.cpp.
        env.pop("CUDA_VISIBLE_DEVICES", None)

    start = time.monotonic()
    status = {
        "case": case,
        "system": system,
        "dataset": dataset,
        "algorithm": algorithm,
        "variant": variant,
        "gpu": 4 if system == "ibfs" else (gpu if system == "gunrock" else None),
        "warmups": warmups,
        "repeats": repeats,
        "measurement_policy": (
            "single_cold_run"
            if warmups == 0 and repeats == 1
            else "warmup_then_measured_repeats"
        ),
        "state": "running",
    }
    if system == "forkgraph" and variant == "llc48":
        partition_summary = json.loads(
            (FORKGRAPH_DATA_ROOT / f"{dataset}.summary.json").read_text()
        )
        status["openmp"] = {
            key: env[key]
            for key in (
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_PLACES",
                "OMP_PROC_BIND",
            )
        }
        status["partitioning"] = partition_summary
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        if system == "forkgraph":
            rows = run_forkgraph_case(
                dataset,
                algorithm,
                case_dir,
                env,
                warmups,
                repeats,
                resume,
            )
        else:
            rows = run_regular_case(
                system,
                dataset,
                algorithm,
                variant,
                case_dir,
                env,
                warmups,
                repeats,
            )
        write_rows(
            case_dir / "runs.csv",
            system,
            dataset,
            algorithm,
            variant,
            rows,
        )
        status["state"] = "complete"
        status["exit_code"] = 0
    except Exception as error:
        status["state"] = "failed"
        status["exit_code"] = 1
        status["error"] = str(error)
    status["wall_seconds"] = time.monotonic() - start
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    with (EXP_ROOT / "stdout.log").open("a") as stream:
        stream.write(
            f"EXTERNAL {case} state={status['state']} "
            f"wall_s={status['wall_seconds']:.3f}\n"
        )
    print(
        f"DONE {case} state={status['state']} "
        f"wall_s={status['wall_seconds']:.3f}",
        flush=True,
    )
    return status["exit_code"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")
    raise SystemExit(
        run_case(
            args.system,
            args.dataset,
            args.algorithm,
            args.variant,
            args.gpu,
            args.resume,
            args.warmups,
            args.repeats,
        )
    )


if __name__ == "__main__":
    main()
