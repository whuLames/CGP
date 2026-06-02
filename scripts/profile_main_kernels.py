#!/usr/bin/env python3
"""Collect per-iteration main-kernel NCU metrics for Gunrock, CGP, and GE-SpMM."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path("/home/zyl/Projects/ocgp")
OUT_DIR = ROOT / "profile_results" / "main_kernel_bottlenecks"
PLAIN_DIR = OUT_DIR / "plain"
NCU_RAW_DIR = OUT_DIR / "ncu_raw"
NCU_REPORT_DIR = OUT_DIR / "ncu_reports"
MAIN_CSV = OUT_DIR / "main_kernel_metrics.csv"
NCU = Path(
    "/home/zyl/.conda/pkgs/nsight-compute-2025.2.1.3-0/"
    "nsight-compute-2025.2.1/ncu"
)

GUNROCK_BFS = ROOT / "baselines/gunrockV2.2/build_cuda/bin/bfs"
GUNROCK_PR = ROOT / "baselines/gunrockV2.2/build_cuda/bin/pr"
CGP_BFS = ROOT / "cgp/build/bin/cgp_bfs"
GE_SPMM = Path("/home/zyl/Projects/TCRGraph/src/kernels/ge_spmm_bench")

GGR_GRAPHS = {
    "cit-Patents": Path("/home/zyl/data/ggr_data/singlegpu/cit-Patents.gr"),
    "soc-orkut": Path("/home/zyl/data/ggr_data/singlegpu/soc-orkut.gr"),
    "soc-sinaweibo": Path("/home/zyl/data/ggr_data/singlegpu/soc-sinaweibo.gr"),
    "soc-twitter": Path("/home/zyl/data/ggr_data/singlegpu/soc-twitter.gr"),
}
CSR_GRAPHS = {
    "cit-Patents": Path("/home/zyl/data/csr_data/cit-Patents"),
    "soc-orkut": Path("/home/zyl/data/csr_data/soc-orkut"),
    "soc-sinaweibo": Path("/home/zyl/data/csr_data/soc-sinaweibo"),
    "soc-twitter": Path("/home/zyl/data/csr_data/soc-twitter"),
}
BFS_SOURCES = {
    "cit-Patents": 0,
    "soc-orkut": 1506298,
    "soc-sinaweibo": 53297474,
    "soc-twitter": 11702603,
}
CGP_Q = {
    "cit-Patents": [2, 4, 8, 16],
    "soc-orkut": [2, 4, 8, 16],
    "soc-sinaweibo": [2, 4, 8],
    "soc-twitter": [2, 4, 8],
}
M_VALUES = [2, 4, 8, 16]

METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_issued.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "lts__t_sectors.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "smsp__sass_branch_targets_threads_divergent.sum",
    "smsp__sass_branch_targets_threads_uniform.sum",
    "smsp__inst_executed_op_generic_atom.sum",
    "smsp__inst_executed_op_global_red.sum",
    "smsp__sass_inst_executed_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "lts__t_requests_op_atom.sum",
    "lts__t_requests_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "lts__t_sectors_op_red.sum",
    "gpu__time_duration.sum",
]
DERIVED = [
    "eligible_ratio",
    "issue_utilization",
    "warp_active_lane_efficiency",
    "predicated_lane_efficiency",
    "dram_bytes_per_edge",
    "l2_bytes_per_edge",
    "atomic_instructions_per_edge",
]
BASE_COLUMNS = [
    "scenario",
    "implementation",
    "algorithm",
    "dataset",
    "q",
    "M",
    "source_or_query_file",
    "iteration",
    "level_mode",
    "main_kernel_name",
    "main_kernel_range",
    "iteration_elapsed_ms",
    "main_kernel_elapsed_ms",
    "processed_edges",
    "active_vertices",
    "input_frontier",
    "output_frontier",
    "actual_edge_count",
    "virtual_edge_count",
]


@dataclass
class Experiment:
    scenario: str
    implementation: str
    algorithm: str
    dataset: str
    q: str
    m: str
    source_or_query_file: str
    plain_csv: Path
    ncu_csv: Path
    ncu_report: Path
    plain_cmd: List[str]
    ncu_cmd: List[str]


def query_file(dataset: str, q: int) -> Path:
    if dataset == "cit-Patents":
        return ROOT / f"cgp/build/cgp_bfs_compare_queries/cit-Patents_q{q}.txt"
    return ROOT / f"baselines/gunrockV2.2/bfs_concurrent_{dataset}_q{q}.txt"


def safe_name(*parts: object) -> str:
    text = "_".join(str(p) for p in parts if str(p) != "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def make_experiments(smoke: bool) -> List[Experiment]:
    experiments: List[Experiment] = []
    datasets = ["cit-Patents"] if smoke else list(GGR_GRAPHS)

    for dataset in datasets:
        graph = GGR_GRAPHS[dataset]
        source = str(BFS_SOURCES[dataset])
        name = safe_name("serial_gunrock", "bfs", dataset)
        plain = PLAIN_DIR / f"{name}.csv"
        experiments.append(
            Experiment(
                "serial_gunrock",
                "gunrockV2.2",
                "bfs",
                dataset,
                "",
                "",
                source,
                plain,
                NCU_RAW_DIR / f"{name}.csv",
                NCU_REPORT_DIR / f"{name}.ncu-rep",
                [str(GUNROCK_BFS), "-m", str(graph), "-n", "1", "-s", source,
                 "--iter_profile", str(plain)],
                [str(GUNROCK_BFS), "-m", str(graph), "-n", "1", "-s", source,
                 "--iter_profile", str(plain)],
            )
        )

        pr_iters = "2" if smoke else "20"
        name = safe_name("serial_gunrock", "pr", dataset)
        plain = PLAIN_DIR / f"{name}.csv"
        experiments.append(
            Experiment(
                "serial_gunrock",
                "gunrockV2.2",
                "pr",
                dataset,
                "",
                "",
                "",
                plain,
                NCU_RAW_DIR / f"{name}.csv",
                NCU_REPORT_DIR / f"{name}.ncu-rep",
                [str(GUNROCK_PR), "-m", str(graph), "-n", "1",
                 "--max_iterations", pr_iters, "--iter_profile", str(plain)],
                [str(GUNROCK_PR), "-m", str(graph), "-n", "1",
                 "--max_iterations", pr_iters, "--iter_profile", str(plain)],
            )
        )

    for dataset in datasets:
        qs = [2] if smoke else CGP_Q[dataset]
        for q in qs:
            qfile = query_file(dataset, q)
            name = safe_name("cgp_shared_node_push", dataset, f"q{q}")
            plain = PLAIN_DIR / f"{name}.csv"
            cmd = [
                str(CGP_BFS), "-m", str(GGR_GRAPHS[dataset]),
                "--query_file", str(qfile),
                "--traversal_mode", "push",
                "--push_strategy", "shared_node",
                "--pull_strategy", "bitmap",
                "--profile_levels",
                "--iter_profile", str(plain),
            ]
            experiments.append(
                Experiment(
                    "cgp_shared_node_push", "cgp", "bfs", dataset, str(q), "",
                    str(qfile), plain, NCU_RAW_DIR / f"{name}.csv",
                    NCU_REPORT_DIR / f"{name}.ncu-rep", cmd, cmd,
                )
            )

    for dataset in datasets:
        ms = [2] if smoke else M_VALUES
        for m_value in ms:
            iters = "1" if smoke else "5"
            name = safe_name("ge_spmm", dataset, f"M{m_value}")
            plain = PLAIN_DIR / f"{name}.csv"
            cmd = [
                str(GE_SPMM), str(CSR_GRAPHS[dataset]), "--M", str(m_value),
                "--iters", iters, "--warmup", "1", "--csv-only",
                f"--iter_profile={plain}",
            ]
            experiments.append(
                Experiment(
                    "ge_spmm", "TCRGraph", "ge_spmm", dataset, "", str(m_value),
                    str(CSR_GRAPHS[dataset]), plain, NCU_RAW_DIR / f"{name}.csv",
                    NCU_REPORT_DIR / f"{name}.ncu-rep", cmd, cmd,
                )
            )

    return experiments


def run_cmd(cmd: List[str], stdout_path: Optional[Path] = None) -> None:
    env = os.environ.copy()
    env["TMPDIR"] = "/home/zyl/tmp"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if stdout_path is None:
        subprocess.run(cmd, check=True, env=env)
    else:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w") as out:
            subprocess.run(cmd, check=True, env=env, stdout=out,
                           stderr=subprocess.STDOUT)


def run_plain(experiments: Iterable[Experiment],
              resume: bool = False,
              continue_on_error: bool = False) -> List[Experiment]:
    failed: List[Experiment] = []
    for exp in experiments:
        exp.plain_csv.parent.mkdir(parents=True, exist_ok=True)
        if resume and exp.plain_csv.exists() and exp.plain_csv.stat().st_size > 0:
            print(f"[plain skip] {exp.scenario} {exp.algorithm} {exp.dataset} q={exp.q} M={exp.m}")
            continue
        if exp.plain_csv.exists():
            exp.plain_csv.unlink()
        print(f"[plain] {exp.scenario} {exp.algorithm} {exp.dataset} q={exp.q} M={exp.m}")
        try:
            run_cmd(exp.plain_cmd, PLAIN_DIR / f"{exp.plain_csv.stem}.log")
        except subprocess.CalledProcessError:
            failed.append(exp)
            print(f"[plain failed] {exp.scenario} {exp.algorithm} {exp.dataset} q={exp.q} M={exp.m}")
            if not continue_on_error:
                raise
    return failed


def ncu_command(exp: Experiment) -> List[str]:
    ncu_iter_profile = NCU_RAW_DIR / f"{exp.plain_csv.stem}.iter_profile.csv"
    def rewrite_iter_profile_arg(arg: str) -> str:
        if arg == str(exp.plain_csv):
            return str(ncu_iter_profile)
        prefix = "--iter_profile="
        if arg.startswith(prefix) and arg[len(prefix):] == str(exp.plain_csv):
            return f"{prefix}{ncu_iter_profile}"
        return arg

    app_cmd = [rewrite_iter_profile_arg(arg) for arg in exp.ncu_cmd]
    return [
        str(NCU), "--nvtx", "--csv", "--page", "raw",
        "--print-nvtx-rename", "kernel",
        "--metrics", ",".join(METRICS),
        "--export", str(exp.ncu_report),
        "--force-overwrite",
        *app_cmd,
    ]


def run_ncu(experiments: Iterable[Experiment],
            resume: bool = False,
            continue_on_error: bool = False) -> List[Experiment]:
    failed: List[Experiment] = []
    for exp in experiments:
        exp.ncu_csv.parent.mkdir(parents=True, exist_ok=True)
        exp.ncu_report.parent.mkdir(parents=True, exist_ok=True)
        if resume and exp.ncu_csv.exists() and exp.ncu_csv.stat().st_size > 0:
            print(f"[ncu skip] {exp.scenario} {exp.algorithm} {exp.dataset} q={exp.q} M={exp.m}")
            continue
        if exp.ncu_csv.exists():
            exp.ncu_csv.unlink()
        if exp.ncu_report.exists():
            exp.ncu_report.unlink()
        print(f"[ncu] {exp.scenario} {exp.algorithm} {exp.dataset} q={exp.q} M={exp.m}")
        try:
            run_cmd(ncu_command(exp), exp.ncu_csv)
        except subprocess.CalledProcessError:
            failed.append(exp)
            print(f"[ncu failed] {exp.scenario} {exp.algorithm} {exp.dataset} q={exp.q} M={exp.m}")
            if not continue_on_error:
                raise
    return failed


def to_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().replace(",", "")
    if text == "" or text.lower() in {"nan", "n/a"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def get(row: Dict[str, str], *names: str) -> str:
    lower = {k.lower(): v for k, v in row.items() if k is not None}
    for name in names:
        if name in row:
            return row[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return ""


def read_plain(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ncu_range(row: Dict[str, str]) -> str:
    for key, value in row.items():
        lk = key.lower()
        if "nvtx" in lk and "range" in lk and value:
            return value.strip()
    return ""


def ncu_metric_name(row: Dict[str, str]) -> str:
    return get(row, "Metric Name", "Metric Name:", "Name", "Metric")


def ncu_metric_value(row: Dict[str, str]) -> str:
    return get(row, "Metric Value", "Metric Value:", "Value", "Avg")


def read_ncu_kernel_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows

    with path.open(newline="", errors="replace") as f:
        lines: List[str] = []
        for line in f:
            if line.startswith('"ID"') or lines:
                lines.append(line)
    if not lines:
        return rows

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        return rows

    for row in reader:
        if not get(row, "ID"):
            continue
        if not get(row, "Kernel Name"):
            continue
        rows.append({k: v for k, v in row.items() if k is not None})
    return rows


def read_ncu_metrics(path: Path) -> Dict[str, Dict[str, str]]:
    metrics_by_range: Dict[str, Dict[str, str]] = {}
    for row in read_ncu_kernel_rows(path):
        range_name = ncu_range(row)
        if not range_name:
            continue
        metrics_by_range.setdefault(range_name, {}).update(
            {metric: row.get(metric, "") for metric in METRICS}
        )
    return metrics_by_range


def select_gunrock(exp: Experiment, plain_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    by_iter: Dict[str, Dict[str, Dict[str, str]]] = {}
    suffix = "advance" if exp.algorithm == "bfs" else "edge_spread"
    for row in plain_rows:
        iteration = get(row, "iteration")
        range_name = get(row, "range_name")
        bucket = by_iter.setdefault(iteration, {})
        if range_name.endswith(f":{suffix}"):
            bucket["main"] = row
        elif range_name == f"gunrock:{exp.algorithm}:iter:{iteration}":
            bucket["iter"] = row
    for iteration, rows in sorted(by_iter.items(), key=lambda item: int(item[0])):
        if "iter" not in rows or "main" not in rows:
            continue
        it = rows["iter"]
        main = rows["main"]
        out.append(base_record(
            exp, iteration, "", suffix, get(main, "range_name"),
            to_float(get(it, "elapsed_ms")), to_float(get(main, "elapsed_ms")),
            to_float(get(main, "active_edges")), to_float(get(main, "active_vertices")),
            to_float(get(main, "input_frontier")), to_float(get(main, "output_frontier")),
            to_float(get(main, "active_edges")), to_float(get(main, "active_edges")),
        ))
    return out


def select_cgp(exp: Experiment, plain_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    by_iter: Dict[str, List[Dict[str, str]]] = {}
    for row in plain_rows:
        by_iter.setdefault(get(row, "iteration"), []).append(row)

    candidates = {"degree_scan", "shared_push", "compact"}
    for iteration, rows in sorted(by_iter.items(), key=lambda item: int(item[0])):
        iter_rows = [r for r in rows if get(r, "range_name").count(":") == 4]
        cand_rows = [
            r for r in rows
            if get(r, "range_name").split(":")[-1] in candidates
        ]
        if not iter_rows or not cand_rows:
            continue
        iter_row = iter_rows[0]
        shared_push_rows = [
            r for r in cand_rows
            if get(r, "range_name").endswith(":shared_push")
        ]
        main = shared_push_rows[0] if shared_push_rows else max(
            cand_rows, key=lambda r: to_float(get(r, "elapsed_ms")))
        main_name = get(main, "range_name").split(":")[-1]
        out.append(base_record(
            exp, iteration, get(iter_row, "level_mode"), main_name,
            get(main, "range_name"), to_float(get(iter_row, "elapsed_ms")),
            to_float(get(main, "elapsed_ms")), to_float(get(main, "processed_edges")),
            to_float(get(main, "active_vertices")), to_float(get(main, "input_frontier")),
            to_float(get(main, "output_frontier")), to_float(get(main, "actual_edge_count")),
            to_float(get(main, "virtual_edge_count")),
        ))
    return out


def select_ge(exp: Experiment, plain_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in plain_rows:
        elapsed = to_float(get(row, "elapsed_ms"))
        out.append(base_record(
            exp, get(row, "iteration"), "", "main", get(row, "range_name"),
            elapsed, elapsed, to_float(get(row, "processed_edges")),
            to_float(get(row, "active_vertices")), to_float(get(row, "input_frontier")),
            to_float(get(row, "output_frontier")), to_float(get(row, "processed_edges")),
            to_float(get(row, "processed_edges")),
        ))
    return out


def base_record(exp: Experiment, iteration: object, level_mode: str,
                main_name: str, main_range: str, iteration_ms: float,
                main_ms: float, processed_edges: float, active_vertices: float,
                input_frontier: float, output_frontier: float,
                actual_edges: float, virtual_edges: float) -> Dict[str, object]:
    def counter(value: float) -> object:
        if isinstance(value, float) and (math.isnan(value) or value < 0):
            return ""
        return value

    return {
        "scenario": exp.scenario,
        "implementation": exp.implementation,
        "algorithm": exp.algorithm,
        "dataset": exp.dataset,
        "q": exp.q,
        "M": exp.m,
        "source_or_query_file": exp.source_or_query_file,
        "iteration": iteration,
        "level_mode": level_mode,
        "main_kernel_name": main_name,
        "main_kernel_range": main_range,
        "iteration_elapsed_ms": iteration_ms,
        "main_kernel_elapsed_ms": main_ms,
        "processed_edges": counter(processed_edges),
        "active_vertices": counter(active_vertices),
        "input_frontier": counter(input_frontier),
        "output_frontier": counter(output_frontier),
        "actual_edge_count": counter(actual_edges),
        "virtual_edge_count": counter(virtual_edges),
    }


def add_metrics(records: List[Dict[str, object]], metrics_by_range: Dict[str, Dict[str, str]]) -> None:
    for record in records:
        metric_values = metrics_by_range.get(str(record["main_kernel_range"]), {})
        if not metric_values:
            for range_name, values in metrics_by_range.items():
                if str(record["main_kernel_range"]) in range_name:
                    metric_values = values
                    break
        for metric in METRICS:
            record[metric] = metric_values.get(metric, "")
        derive(record)


def kernel_matches(record: Dict[str, object], kernel_name: str) -> bool:
    scenario = str(record["scenario"])
    main_name = str(record["main_kernel_name"])
    if scenario == "ge_spmm":
        return "ge_spmm_" in kernel_name
    if scenario == "cgp_shared_node_push":
        if main_name == "shared_push":
            return "expand_shared_node" in kernel_name
        if main_name == "degree_scan":
            return "compute_shared_node_degrees" in kernel_name
        if main_name == "compact":
            return "to_shared_frontier" in kernel_name or "to_list_kernel" in kernel_name
        return False
    if scenario == "serial_gunrock" and str(record["algorithm"]) == "bfs":
        return "block_mapped_kernel" in kernel_name or "advance" in kernel_name
    if scenario == "serial_gunrock" and str(record["algorithm"]) == "pr":
        return "operators::execute" in kernel_name or "op_wrapper" in kernel_name
    return False


def add_metrics_from_kernel_order(records: List[Dict[str, object]],
                                  kernel_rows: List[Dict[str, str]]) -> None:
    cursor = 0
    for record in records:
        selected: Optional[Dict[str, str]] = None
        for index in range(cursor, len(kernel_rows)):
            row = kernel_rows[index]
            if kernel_matches(record, get(row, "Kernel Name")):
                selected = row
                cursor = index + 1
                break
        if selected is None:
            for row in kernel_rows:
                if kernel_matches(record, get(row, "Kernel Name")):
                    selected = row
                    break
        if selected is not None:
            for metric in METRICS:
                record[metric] = selected.get(metric, "")
        else:
            for metric in METRICS:
                record.setdefault(metric, "")
        derive(record)


def divide(a: float, b: float) -> str:
    if math.isnan(a) or math.isnan(b) or b == 0:
        return ""
    return f"{a / b:.10g}"


def derive(record: Dict[str, object]) -> None:
    active = to_float(record.get("smsp__warps_active.avg.per_cycle_active"))
    eligible = to_float(record.get("smsp__warps_eligible.avg.per_cycle_active"))
    ipc = to_float(record.get("smsp__inst_issued.avg.per_cycle_active"))
    lane = to_float(record.get("smsp__thread_inst_executed_per_inst_executed.ratio"))
    pred_lane = to_float(record.get("smsp__thread_inst_executed_pred_on_per_inst_executed.ratio"))
    edges = to_float(record.get("processed_edges"))
    if not math.isnan(edges) and edges <= 0:
        edges = math.nan
    dram_bytes = to_float(record.get("dram__bytes.sum"))
    l2_bytes = to_float(record.get("lts__t_bytes.sum"))
    atom = to_float(record.get("smsp__inst_executed_op_generic_atom.sum"))
    record["eligible_ratio"] = divide(eligible, active)
    record["issue_utilization"] = "" if math.isnan(ipc) else f"{ipc:.10g}"
    record["warp_active_lane_efficiency"] = divide(lane, 32.0)
    record["predicated_lane_efficiency"] = divide(pred_lane, 32.0)
    record["dram_bytes_per_edge"] = divide(dram_bytes, edges)
    record["l2_bytes_per_edge"] = divide(l2_bytes, edges)
    record["atomic_instructions_per_edge"] = divide(atom, edges)


def merge(experiments: Iterable[Experiment]) -> List[Dict[str, object]]:
    all_records: List[Dict[str, object]] = []
    for exp in experiments:
        plain_rows = read_plain(exp.plain_csv)
        if not plain_rows:
            continue
        if exp.scenario == "serial_gunrock":
            records = select_gunrock(exp, plain_rows)
        elif exp.scenario == "cgp_shared_node_push":
            records = select_cgp(exp, plain_rows)
        else:
            records = select_ge(exp, plain_rows)
        range_metrics = read_ncu_metrics(exp.ncu_csv)
        if range_metrics:
            add_metrics(records, range_metrics)
        else:
            add_metrics_from_kernel_order(records, read_ncu_kernel_rows(exp.ncu_csv))
        all_records.extend(records)
    return all_records


def write_main(records: List[Dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = BASE_COLUMNS + METRICS + DERIVED
    with MAIN_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_failures(failures: List[Experiment]) -> None:
    if not failures:
        return
    path = OUT_DIR / "failed_experiments.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "implementation", "algorithm", "dataset", "q", "M",
            "source_or_query_file", "plain_csv", "ncu_csv",
        ])
        writer.writeheader()
        for exp in failures:
            writer.writerow({
                "scenario": exp.scenario,
                "implementation": exp.implementation,
                "algorithm": exp.algorithm,
                "dataset": exp.dataset,
                "q": exp.q,
                "M": exp.m,
                "source_or_query_file": exp.source_or_query_file,
                "plain_csv": exp.plain_csv,
                "ncu_csv": exp.ncu_csv,
            })


def validate(records: List[Dict[str, object]], smoke: bool) -> None:
    if not records:
        raise RuntimeError("No merged records produced")
    if not smoke:
        ge_rows = [r for r in records if r["scenario"] == "ge_spmm"]
        if len(ge_rows) != 80:
            raise RuntimeError(f"Expected 80 GE-SpMM rows, got {len(ge_rows)}")
    missing_timing = [
        r for r in records
        if r["iteration_elapsed_ms"] == "" or r["main_kernel_elapsed_ms"] == ""
    ]
    if missing_timing:
        raise RuntimeError(f"{len(missing_timing)} records missing plain timings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run one small experiment per scenario")
    parser.add_argument("--skip-plain", action="store_true")
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing non-empty plain/NCU outputs")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue remaining experiments and write failed_experiments.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in (OUT_DIR, PLAIN_DIR, NCU_RAW_DIR, NCU_REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    experiments = make_experiments(args.smoke)
    failures: List[Experiment] = []
    if not args.merge_only and not args.skip_plain:
        failures.extend(run_plain(experiments, args.resume,
                                  args.continue_on_error))
    if not args.merge_only and not args.skip_ncu:
        failures.extend(run_ncu(experiments, args.resume,
                                args.continue_on_error))
    records = merge(experiments)
    if not args.continue_on_error:
        validate(records, args.smoke)
    write_main(records)
    write_failures(failures)
    print(f"Wrote {len(records)} rows to {MAIN_CSV}")
    if failures:
        print(f"Wrote {len(failures)} failures to {OUT_DIR / 'failed_experiments.csv'}")


if __name__ == "__main__":
    main()
