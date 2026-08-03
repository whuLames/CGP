#!/usr/bin/env python3

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parent
GUNROCK_ROOT = Path("/home/zyl/Projects/ocgp/baselines/gunrockV2.2")
BIN_ROOT = GUNROCK_ROOT / "build_cuda" / "bin"
GRAPH_ROOT = Path("/home/zyl/data/ggr_data/singlegpu")
QUERY_ROOT = Path(
    "/home/zyl/Projects/ocgp/experiments/"
    "20260719-233131_external_baseline_n256/query_sets"
)
ALGORITHMS = ("bfs", "sssp", "pagerank")
Q_CANDIDATES = (64, 32, 16)
WARMUPS = 2
REPEATS = 5
TOTAL_QUERIES = 256
GPU_BUSY_LIMIT_MIB = 64


class RunFailure(RuntimeError):
    def __init__(self, message, oom=False):
        super().__init__(message)
        self.oom = oom


class GpuBusyError(RuntimeError):
    pass


def gpu_memory_mib(gpu):
    process = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    fields = [int(value.strip()) for value in process.stdout.strip().split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"unexpected nvidia-smi output: {process.stdout!r}")
    return fields[0], fields[1]


def require_idle_gpu(gpu):
    used_mib, free_mib = gpu_memory_mib(gpu)
    print(
        f"PREFLIGHT gpu={gpu} used_mib={used_mib} free_mib={free_mib}",
        flush=True,
    )
    if used_mib > GPU_BUSY_LIMIT_MIB:
        raise GpuBusyError(
            f"gpu {gpu} is not idle: {used_mib} MiB already allocated"
        )


@contextlib.contextmanager
def reserve_gpu(gpu):
    lock_path = Path("/tmp") / f"puercgp-gunrock-gpu-{gpu}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GpuBusyError(
                f"gpu {gpu} is reserved by another benchmark worker"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def source_metadata(dataset):
    path = QUERY_ROOT / dataset / "sources.txt"
    content = path.read_bytes()
    sources = [int(value) for value in content.split()]
    if len(sources) != TOTAL_QUERIES:
        raise RuntimeError(f"{dataset}: expected 256 sources, found {len(sources)}")
    return path, hashlib.sha256(content).hexdigest()


def command_for(dataset, algorithm, q, output_dir, json_name):
    graph = GRAPH_ROOT / f"{dataset}.gr"
    sources = QUERY_ROOT / dataset / "sources.txt"
    common = [
        "-m",
        str(graph),
        "--num_streams",
        str(q),
        "-d",
        str(output_dir),
        "-f",
        json_name,
    ]
    if algorithm == "bfs":
        return [
            str(BIN_ROOT / "bfs_concurrent"),
            *common,
            "--query_file",
            str(sources),
        ]
    if algorithm == "sssp":
        return [
            str(BIN_ROOT / "sssp_concurrent"),
            *common,
            "--query_file",
            str(sources),
            "--num_queries",
            str(TOTAL_QUERIES),
        ]
    return [
        str(BIN_ROOT / "pr_concurrent"),
        *common,
        "--num_queries",
        str(TOTAL_QUERIES),
        "--max_iterations",
        "10",
    ]


def run_once(dataset, algorithm, q, gpu, output_dir, kind, index):
    with reserve_gpu(gpu):
        return run_once_locked(
            dataset, algorithm, q, gpu, output_dir, kind, index
        )


def run_once_locked(dataset, algorithm, q, gpu, output_dir, kind, index):
    require_idle_gpu(gpu)
    json_name = f"{kind}-{index}.json"
    command = command_for(dataset, algorithm, q, output_dir, json_name)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.monotonic()
    process = subprocess.run(
        command,
        cwd=GUNROCK_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,
    )
    process_wall_ms = (time.monotonic() - start) * 1000.0
    (output_dir / f"{kind}-{index}.stdout.log").write_text(process.stdout)
    (output_dir / f"{kind}-{index}.stderr.log").write_text(process.stderr)
    combined = (process.stdout + "\n" + process.stderr).lower()
    oom = any(
        token in combined
        for token in (
            "out of memory",
            "cudaerrormemoryallocation",
            "bad_alloc",
            "memory allocation",
        )
    )
    if oom:
        used_mib, _ = gpu_memory_mib(gpu)
        if used_mib > GPU_BUSY_LIMIT_MIB:
            raise GpuBusyError(
                f"gpu {gpu} remained externally occupied after failure: "
                f"{used_mib} MiB allocated"
            )
    json_path = output_dir / json_name
    if process.returncode != 0 or not json_path.exists():
        raise RunFailure(
            f"exit={process.returncode}, json={json_path.exists()}", oom=oom
        )
    payload = json.loads(json_path.read_text())
    effective_q = int(payload["effective_batch_size"])
    submitted = int(payload["submitted_queries"])
    if effective_q != q or submitted != TOTAL_QUERIES:
        raise RunFailure(
            f"unexpected workload: effective_q={effective_q}, submitted={submitted}"
        )
    return float(payload["total_wall_ms"]), process_wall_ms


def write_runs(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("kind", "index", "q", "total_wall_ms", "process_wall_ms"),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_algorithm(dataset, algorithm, gpu, q_candidates, artifact_root):
    case_root = artifact_root / dataset / algorithm
    case_root.mkdir(parents=True, exist_ok=True)
    source_path, source_sha256 = source_metadata(dataset)
    summary_path = case_root / "summary.json"
    attempts = []
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("status", "").startswith("oom_at_q"):
            attempts = previous.get("attempts", [])

    for q in q_candidates:
        output_dir = case_root / f"q{q}"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        try:
            for kind, count in (("warmup", WARMUPS), ("repeat", REPEATS)):
                for index in range(count):
                    print(
                        f"START dataset={dataset} algorithm={algorithm} "
                        f"q={q} {kind}={index} gpu={gpu}",
                        flush=True,
                    )
                    total_ms, process_ms = run_once(
                        dataset, algorithm, q, gpu, output_dir, kind, index
                    )
                    rows.append(
                        {
                            "kind": kind,
                            "index": index,
                            "q": q,
                            "total_wall_ms": f"{total_ms:.8f}",
                            "process_wall_ms": f"{process_ms:.3f}",
                        }
                    )
                    write_runs(output_dir / "runs.csv", rows)
                    print(
                        f"DONE dataset={dataset} algorithm={algorithm} "
                        f"q={q} {kind}={index} total_ms={total_ms:.3f}",
                        flush=True,
                    )
        except subprocess.TimeoutExpired as error:
            attempts.append({"q": q, "status": "timeout", "error": str(error)})
            raise
        except GpuBusyError as error:
            attempts.append(
                {"q": q, "status": "environment_busy", "error": str(error)}
            )
            (output_dir / "failure.json").write_text(
                json.dumps(attempts[-1], indent=2) + "\n"
            )
            raise
        except RunFailure as error:
            if error.oom and rows:
                attempts.append(
                    {
                        "q": q,
                        "status": "environment_unstable",
                        "error": (
                            f"{error}; the same Q already completed "
                            f"{len(rows)} run(s)"
                        ),
                    }
                )
                (output_dir / "failure.json").write_text(
                    json.dumps(attempts[-1], indent=2) + "\n"
                )
                raise GpuBusyError(
                    f"{dataset} {algorithm} Q={q} became OOM after successful runs"
                ) from error
            attempts.append(
                {"q": q, "status": "oom" if error.oom else "failed", "error": str(error)}
            )
            (output_dir / "failure.json").write_text(
                json.dumps(attempts[-1], indent=2) + "\n"
            )
            print(
                f"FALLBACK dataset={dataset} algorithm={algorithm} q={q} "
                f"oom={error.oom} error={error}",
                flush=True,
            )
            if error.oom:
                time.sleep(2)
                continue
            raise

        repeat_values = [
            float(row["total_wall_ms"]) for row in rows if row["kind"] == "repeat"
        ]
        summary = {
            "dataset": dataset,
            "algorithm": algorithm,
            "status": "complete",
            "gpu": gpu,
            "total_queries": TOTAL_QUERIES,
            "effective_q": q,
            "warmups": WARMUPS,
            "repeats": REPEATS,
            "median_ms": statistics.median(repeat_values),
            "min_ms": min(repeat_values),
            "max_ms": max(repeat_values),
            "source_file": str(source_path) if algorithm != "pagerank" else None,
            "source_sha256": source_sha256 if algorithm != "pagerank" else None,
            "pagerank_iterations": 10 if algorithm == "pagerank" else None,
            "attempts": attempts + [{"q": q, "status": "complete"}],
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"COMPLETE dataset={dataset} algorithm={algorithm} q={q} "
            f"median_ms={summary['median_ms']:.3f}",
            flush=True,
        )
        return summary

    summary = {
        "dataset": dataset,
        "algorithm": algorithm,
        "status": f"oom_at_q{q_candidates[-1]}",
        "gpu": gpu,
        "total_queries": TOTAL_QUERIES,
        "source_file": str(source_path) if algorithm != "pagerank" else None,
        "source_sha256": source_sha256 if algorithm != "pagerank" else None,
        "pagerank_iterations": 10 if algorithm == "pagerank" else None,
        "attempts": attempts,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"FAILED dataset={dataset} algorithm={algorithm} all_q_oom", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--q-candidates", nargs="+", type=int, default=Q_CANDIDATES)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=EXP_ROOT / "artifacts",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    summaries = []
    for algorithm in args.algorithms:
        summaries.append(
            run_algorithm(
                args.dataset,
                algorithm,
                args.gpu,
                args.q_candidates,
                artifact_root,
            )
        )
    worker_output = (
        EXP_ROOT / f"worker-{args.dataset}.json"
        if artifact_root == (EXP_ROOT / "artifacts").resolve()
        else artifact_root / f"worker-{args.dataset}.json"
    )
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    worker_output.write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
