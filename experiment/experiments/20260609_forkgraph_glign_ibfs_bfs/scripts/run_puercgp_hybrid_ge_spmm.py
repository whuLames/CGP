#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
import subprocess
import time
from pathlib import Path


EXP_ROOT = Path("/home/zyl/Projects/ocgp/experiment/experiments/20260609_forkgraph_glign_ibfs_bfs")
PUERCGP_VALIDATE_BFS = Path("/home/zyl/Projects/ocgp/puercgp/build/validate_bfs")
PUERCGP_CWD = Path("/home/zyl/Projects/ocgp/puercgp/build")
CSR_ROOT = Path("/home/zyl/data/csr_data")

DATASETS = ["cit-Patents", "soc-sinaweibo", "soc-twitter", "soc-orkut"]
CONCURRENCIES = [2, 4, 8, 16, 32, 64]
REPEATS = [1, 2, 3]
BASELINE = "puercgp_hybrid_shared_node_warp_ge_spmm"
TRAVERSAL_MODE = "hybrid"
PUSH_STRATEGY = "shared_node_warp"
PULL_STRATEGY = "ge_spmm"

RAW_FIELDS = [
    "dataset",
    "baseline",
    "concurrency",
    "repeat",
    "query_file",
    "sources",
    "algo_time_ms",
    "puercgp_wall_ms",
    "wall_time_sec",
    "exit_code",
    "distance_mismatches",
    "traversal_mode",
    "push_strategy",
    "pull_strategy",
    "command",
    "log_path",
]

SUMMARY_FIELDS = [
    "dataset",
    "baseline",
    "concurrency",
    "runs",
    "ok_runs",
    "median_algo_time_ms",
    "min_algo_time_ms",
    "max_algo_time_ms",
    "median_wall_time_sec",
    "min_wall_time_sec",
    "max_wall_time_sec",
    "median_puercgp_wall_ms",
    "min_puercgp_wall_ms",
    "max_puercgp_wall_ms",
    "distance_mismatches",
    "traversal_mode",
    "push_strategy",
    "pull_strategy",
]


def load_sources(query_file: Path) -> list[int]:
    return [int(x) for x in query_file.read_text().split()]


def parse_key(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_float(text: str, key: str) -> str:
    value = parse_key(text, key)
    if not value:
        return ""
    try:
        return str(float(value))
    except ValueError:
        return ""


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    meta = {}
    for key in [
        "DATASET",
        "CONCURRENCY",
        "REPEAT",
        "QUERY_FILE",
        "WALL_SEC",
        "EXIT_CODE",
        "COMMAND",
    ]:
        meta[key] = parse_key(text, key)

    query_file = Path(meta["QUERY_FILE"])
    source_list = " ".join(str(s) for s in load_sources(query_file))
    comma_sources = ",".join(source_list.split())
    output_sources = parse_key(text, "sources")

    row = {
        "dataset": meta["DATASET"],
        "baseline": BASELINE,
        "concurrency": meta["CONCURRENCY"],
        "repeat": meta["REPEAT"],
        "query_file": str(query_file),
        "sources": source_list,
        "algo_time_ms": parse_float(text, "gpu_ms_median"),
        "puercgp_wall_ms": parse_float(text, "wall_ms_median"),
        "wall_time_sec": meta["WALL_SEC"],
        "exit_code": meta["EXIT_CODE"],
        "distance_mismatches": parse_key(text, "distance_mismatches"),
        "traversal_mode": parse_key(text, "traversal_mode"),
        "push_strategy": parse_key(text, "push_strategy"),
        "pull_strategy": parse_key(text, "pull_strategy"),
        "command": meta["COMMAND"],
        "log_path": str(log_path),
    }

    errors = []
    if output_sources != comma_sources:
        errors.append(f"sources mismatch: output={output_sources!r} query_file={comma_sources!r}")
    if row["exit_code"] != "0":
        errors.append(f"exit_code={row['exit_code']}")
    if row["distance_mismatches"] != "0":
        errors.append(f"distance_mismatches={row['distance_mismatches']}")
    if row["traversal_mode"] != TRAVERSAL_MODE:
        errors.append(f"traversal_mode={row['traversal_mode']}")
    if row["push_strategy"] != PUSH_STRATEGY:
        errors.append(f"push_strategy={row['push_strategy']}")
    if row["pull_strategy"] != PULL_STRATEGY:
        errors.append(f"pull_strategy={row['pull_strategy']}")
    if not row["algo_time_ms"]:
        errors.append("missing gpu_ms_median")
    if not row["puercgp_wall_ms"]:
        errors.append("missing wall_ms_median")
    if errors:
        raise RuntimeError(f"{log_path}: " + "; ".join(errors))

    return row


def log_path_for(dataset: str, concurrency: int, repeat: int) -> Path:
    return EXP_ROOT / "puercgp_hybrid_ge_spmm_raw_logs" / dataset / f"q{concurrency}_r{repeat}.log"


def command_for(dataset: str, concurrency: int) -> tuple[Path, list[str]]:
    query_file = EXP_ROOT / "query_sets" / dataset / f"q{concurrency}.sources"
    sources = ",".join(str(s) for s in load_sources(query_file))
    cmd = [
        str(PUERCGP_VALIDATE_BFS),
        str(CSR_ROOT / dataset),
        sources,
        "1",
        TRAVERSAL_MODE,
        PUSH_STRATEGY,
        PULL_STRATEGY,
    ]
    return query_file, cmd


def run_one(dataset: str, concurrency: int, repeat: int, timeout_sec: int) -> dict:
    query_file, cmd = command_for(dataset, concurrency)
    log_path = log_path_for(dataset, concurrency, repeat)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=PUERCGP_CWD,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        exit_code = proc.returncode
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\nTIMEOUT after {timeout_sec}s\n"

    wall = time.monotonic() - start
    sources_text = " ".join(str(s) for s in load_sources(query_file))
    log_path.write_text(
        f"DATASET={dataset}\n"
        f"BASELINE={BASELINE}\n"
        f"CONCURRENCY={concurrency}\n"
        f"REPEAT={repeat}\n"
        f"QUERY_FILE={query_file}\n"
        f"SOURCES_FILE_CONTENT={sources_text}\n"
        f"CWD={PUERCGP_CWD}\n"
        f"COMMAND={' '.join(cmd)}\n"
        f"WALL_SEC={wall:.6f}\n"
        f"EXIT_CODE={exit_code}\n\n"
        f"{output}"
    )
    return parse_log(log_path)


def expected_runs(datasets: list[str]) -> list[tuple[str, int, int]]:
    return [
        (dataset, concurrency, repeat)
        for dataset in datasets
        for concurrency in CONCURRENCIES
        for repeat in REPEATS
    ]


def write_puercgp_results(datasets: list[str]) -> list[dict]:
    rows = [parse_log(log_path_for(d, c, r)) for d, c, r in expected_runs(datasets)]
    rows.sort(key=lambda row: (row["dataset"], int(row["concurrency"]), int(row["repeat"])))
    with (EXP_ROOT / "puercgp_hybrid_ge_spmm_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["dataset"], row["baseline"], row["concurrency"])
        groups.setdefault(key, []).append(row)

    summary = []
    for key, vals in sorted(groups.items()):
        ok = [
            v
            for v in vals
            if v["exit_code"] == "0"
            and v["algo_time_ms"]
            and (v.get("distance_mismatches", "") in ("", "0"))
        ]
        algo = [float(v["algo_time_ms"]) for v in ok]
        wall = [float(v["wall_time_sec"]) for v in ok if v["wall_time_sec"]]
        puercgp_wall = [float(v["puercgp_wall_ms"]) for v in ok if v.get("puercgp_wall_ms")]
        mismatch_values = sorted({v.get("distance_mismatches", "") for v in vals if v.get("distance_mismatches", "")})
        summary.append(
            {
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
                "median_puercgp_wall_ms": statistics.median(puercgp_wall) if puercgp_wall else "",
                "min_puercgp_wall_ms": min(puercgp_wall) if puercgp_wall else "",
                "max_puercgp_wall_ms": max(puercgp_wall) if puercgp_wall else "",
                "distance_mismatches": ";".join(mismatch_values),
                "traversal_mode": vals[0].get("traversal_mode", ""),
                "push_strategy": vals[0].get("push_strategy", ""),
                "pull_strategy": vals[0].get("pull_strategy", ""),
            }
        )
    return summary


def read_existing_results() -> list[dict]:
    with (EXP_ROOT / "results.csv").open() as f:
        return list(csv.DictReader(f))


def read_existing_summary() -> list[dict]:
    with (EXP_ROOT / "summary_by_config.csv").open() as f:
        return list(csv.DictReader(f))


def write_combined(puercgp_rows: list[dict]) -> list[dict]:
    combined_rows = []
    for row in read_existing_results():
        combined = {field: "" for field in RAW_FIELDS}
        combined.update({field: row.get(field, "") for field in RAW_FIELDS})
        combined_rows.append(combined)
    combined_rows.extend(puercgp_rows)
    with (EXP_ROOT / "combined_results_with_puercgp_hybrid_ge_spmm.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(combined_rows)

    existing_summary = []
    for row in read_existing_summary():
        combined = {field: "" for field in SUMMARY_FIELDS}
        combined.update({field: row.get(field, "") for field in SUMMARY_FIELDS})
        existing_summary.append(combined)
    combined_summary = existing_summary + summarize_rows(puercgp_rows)
    combined_summary.sort(key=lambda row: (row["dataset"], row["baseline"], int(row["concurrency"])))
    with (EXP_ROOT / "combined_summary_with_puercgp_hybrid_ge_spmm.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(combined_summary)
    return combined_summary


def fmt_ms(value: str) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


def write_markdown(summary: list[dict]) -> None:
    by_dataset = {}
    for row in summary:
        by_dataset.setdefault(row["dataset"], {})[(row["baseline"], int(row["concurrency"]))] = row

    baselines = ["forkgraph", "glign", "ibfs", BASELINE]
    labels = {
        "forkgraph": "ForkGraph ms",
        "glign": "Glign ms",
        "ibfs": "iBFS ms",
        BASELINE: "puercgp hybrid GE-SpMM ms",
    }
    lines = [
        "# BFS Combined Summary with puercgp Hybrid GE-SpMM",
        "",
        "Median algorithm time in milliseconds. puercgp uses `gpu_ms_median` from `validate_bfs`; `ok_runs/runs` is noted when not all three runs succeeded.",
        "",
        "puercgp configuration: `traversal_mode=hybrid`, `push_strategy=shared_node_warp`, `pull_strategy=ge_spmm`.",
        "",
    ]
    for dataset in sorted(by_dataset):
        lines.extend([f"## {dataset}", ""])
        lines.append("| concurrency | ForkGraph ms | Glign ms | iBFS ms | puercgp hybrid GE-SpMM ms | notes |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for concurrency in CONCURRENCIES:
            notes = []
            cells = []
            for baseline in baselines:
                row = by_dataset[dataset].get((baseline, concurrency))
                if row is None:
                    cells.append("")
                    notes.append(f"{baseline} missing")
                    continue
                cells.append(fmt_ms(row["median_algo_time_ms"]))
                if row["ok_runs"] != row["runs"]:
                    notes.append(f"{baseline} {row['ok_runs']}/{row['runs']} ok")
                if baseline == BASELINE and row.get("distance_mismatches") not in ("", "0"):
                    notes.append(f"puercgp mismatches {row.get('distance_mismatches')}")
            lines.append(
                f"| {concurrency} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {'; '.join(notes)} |"
            )
        lines.append("")
    (EXP_ROOT / "COMBINED_SUMMARY_WITH_PUERCGP_HYBRID_GE_SPMM.md").write_text("\n".join(lines))


def run_full(args: argparse.Namespace) -> None:
    datasets = args.datasets or DATASETS
    for dataset, concurrency, repeat in expected_runs(datasets):
        log_path = log_path_for(dataset, concurrency, repeat)
        if args.resume and log_path.exists():
            try:
                row = parse_log(log_path)
                print(
                    f"SKIP {dataset} q{concurrency} r{repeat} "
                    f"gpu_ms={row['algo_time_ms']} wall_ms={row['puercgp_wall_ms']}",
                    flush=True,
                )
                continue
            except Exception as exc:
                print(f"RERUN invalid existing log {log_path}: {exc}", flush=True)

        print(f"RUN {dataset} q{concurrency} r{repeat}", flush=True)
        row = run_one(dataset, concurrency, repeat, args.timeout_sec)
        print(
            f"  exit={row['exit_code']} gpu_ms={row['algo_time_ms']} "
            f"puercgp_wall_ms={row['puercgp_wall_ms']} wall_sec={row['wall_time_sec']}",
            flush=True,
        )

    puercgp_rows = write_puercgp_results(DATASETS)
    summary = write_combined(puercgp_rows)
    write_markdown(summary)
    print(f"wrote puercgp rows={len(puercgp_rows)} combined_summary_rows={len(summary)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "summarize"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode == "full":
        run_full(args)
    else:
        puercgp_rows = write_puercgp_results(DATASETS)
        summary = write_combined(puercgp_rows)
        write_markdown(summary)
        print(f"wrote puercgp rows={len(puercgp_rows)} combined_summary_rows={len(summary)}", flush=True)


if __name__ == "__main__":
    main()
