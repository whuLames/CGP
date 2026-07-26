#!/usr/bin/env python3
"""Analyze ncu CSVs from run_bfs_ipc.sh.

Output 5 levels of aggregation:
  L0 per_dispatch.csv:   one row per kernel dispatch
  L1 per_iteration.csv:  one row per BFS iteration (= dispatch_idx by default)
  L2 per_query.csv:      one row per (framework, ds, N, query, repeat) - time-weighted
  L3 per_config.csv:     one row per (framework, ds, N) - cross-query + median over repeats
  L4 hot_iteration.csv:  per-run, the iteration with max time

Plus summary.md for human reading.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PEAK_BW_GBS = 900.0

# Kernel name filters (substring match)
GUNROCK_KEEP = ["block_mapped_kernel"]
GUNROCK_EXCLUDE = ["static_kernel", "uninitialized", "__fill", "DeviceReduce", "fill"]
PUE_KEEP = ["expand_shared_node_warp_kernel"]
PUE_EXCLUDE = ["clear_shared_frontier_mask", "reset_counter", "init_shared_sources",
               "static_kernel", "uninitialized", "__fill"]

# Pattern to split gunrock -n 5 multi-BFS CSV into individual BFS runs.
# In gunrock, each BFS starts with init kernels. We use a "fill of distances"
# pattern (static_kernel that fills dist array) as BFS boundary marker.
# Simpler approach: just record all dispatch_idx as continuous, then in
# per_query we group by "5 BFS per file" using even splitting by total count.


def to_int(s: str) -> int:
    if not s:
        return 0
    s = s.replace('"', '').replace(',', '').strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def to_float(s: str) -> float:
    if not s:
        return 0.0
    s = s.replace('"', '').replace(',', '').strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def find_header_line(lines: List[str]) -> int:
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            return i
    raise ValueError("no CSV header")


def keep_kernel(name: str, framework: str) -> bool:
    keep_list = GUNROCK_KEEP if framework == "gunrock" else PUE_KEEP
    excl_list = GUNROCK_EXCLUDE if framework == "gunrock" else PUE_EXCLUDE
    if any(x in name for x in excl_list):
        return False
    return any(x in name for x in keep_list)


def parse_ncu_csv(path: Path, framework: str) -> List[Dict]:
    """Parse one ncu CSV. Return list of target kernel dispatches."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    try:
        header_idx = find_header_line(lines)
    except ValueError:
        return []

    reader = csv.DictReader(lines[header_idx:])
    out = []
    for row in reader:
        if not row.get("ID"):
            continue
        name = row.get("Kernel Name", "")
        if not keep_kernel(name, framework):
            continue
        out.append({
            "kernel_name": name[:120],
            "bytes_read": to_int(row.get("dram__bytes_read.sum", "0")),
            "bytes_write": to_int(row.get("dram__bytes_write.sum", "0")),
            "time_ns": to_int(row.get("gpu__time_duration.sum", "0")),
            "ipc": to_float(row.get("smsp__inst_issued.avg.per_cycle_active", "0")),
        })
    return out


def scan_gunrock(raw_dir: Path) -> List[Dict]:
    """Parse gunrock raw CSVs. File name: <ds>_N<N>_q<qidx>.csv (含 5 repeats via -n 5)."""
    pattern = re.compile(r"^(?P<ds>[^_]+(?:_[^_]+)*?)_N(?P<N>\d+)_q(?P<q>\d+)\.csv$")
    rows = []
    files = sorted(raw_dir.glob("*.csv"))
    for f in files:
        m = pattern.match(f.name)
        if not m:
            continue
        ds, N, q = m.group("ds"), int(m.group("N")), int(m.group("q"))
        dispatches = parse_ncu_csv(f, "gunrock")
        if not dispatches:
            continue
        # gunrock -n 5 ran 5 BFS in one process. We need to split dispatches into 5 groups.
        # Heuristic: total dispatches / 5 (assume each BFS has the same # of advance kernels).
        # If the count is not divisible by 5, fall back to 1 group (mark all as repeat 1).
        total = len(dispatches)
        if total >= 5 and total % 5 == 0:
            per_bfs = total // 5
            for rep in range(5):
                start = rep * per_bfs
                end = start + per_bfs
                for local_idx, d in enumerate(dispatches[start:end]):
                    rows.append({
                        "framework": "gunrock",
                        "dataset": ds, "N": N, "query_idx": q,
                        "repeat": rep + 1, "dispatch_idx": local_idx,
                        **d,
                    })
        else:
            # Fallback: treat all as repeat 1
            print(f"WARN: {f.name} has {total} dispatches (not divisible by 5), treating as repeat 1")
            for local_idx, d in enumerate(dispatches):
                rows.append({
                    "framework": "gunrock",
                    "dataset": ds, "N": N, "query_idx": q,
                    "repeat": 1, "dispatch_idx": local_idx,
                    **d,
                })
    return rows


def scan_puercgp(raw_dir: Path) -> List[Dict]:
    """Parse puercgp raw CSVs. File name: <ds>_N<N>_r<r>.csv."""
    pattern = re.compile(r"^(?P<ds>[^_]+(?:_[^_]+)*?)_N(?P<N>\d+)_r(?P<r>\d+)\.csv$")
    rows = []
    files = sorted(raw_dir.glob("*.csv"))
    for f in files:
        m = pattern.match(f.name)
        if not m:
            continue
        ds, N, r = m.group("ds"), int(m.group("N")), int(m.group("r"))
        dispatches = parse_ncu_csv(f, "puercgp")
        for local_idx, d in enumerate(dispatches):
            rows.append({
                "framework": "puercgp",
                "dataset": ds, "N": N,
                "query_idx": -1,  # shared kernel processes N queries
                "repeat": r, "dispatch_idx": local_idx,
                **d,
            })
    return rows


def time_weighted_ipc(records: List[Dict]) -> float:
    """Σ(ipc × time) / Σ(time)"""
    total_t = sum(r["time_ns"] for r in records)
    if total_t <= 0:
        return 0.0
    return sum(r["ipc"] * r["time_ns"] for r in records) / total_t


def aggregate_l2_per_query(dispatch_rows: List[Dict]) -> List[Dict]:
    """Group by (framework, ds, N, query, repeat) -> one row."""
    groups: Dict[Tuple, List[Dict]] = {}
    for r in dispatch_rows:
        key = (r["framework"], r["dataset"], r["N"], r["query_idx"], r["repeat"])
        groups.setdefault(key, []).append(r)

    out = []
    for (fw, ds, N, q, rep), recs in sorted(groups.items()):
        total_t = sum(r["time_ns"] for r in recs)
        total_br = sum(r["bytes_read"] for r in recs)
        total_bw = sum(r["bytes_write"] for r in recs)
        ipc = time_weighted_ipc(recs)
        dram_bw_gbs = (total_br + total_bw) / (total_t / 1e9) / 1e9 if total_t > 0 else 0
        out.append({
            "framework": fw, "dataset": ds, "N": N,
            "query_idx": q, "repeat": rep,
            "total_bytes_read": total_br,
            "total_bytes_write": total_bw,
            "total_bytes_total": total_br + total_bw,
            "total_time_ns": total_t,
            "total_dispatches": len(recs),
            "ipc_time_weighted": round(ipc, 4),
            "dram_bw_gbs": round(dram_bw_gbs, 2),
            "bp_peak": round(dram_bw_gbs / PEAK_BW_GBS, 4),
        })
    return out


def aggregate_l3_per_config(l2_rows: List[Dict]) -> List[Dict]:
    """Group by (framework, ds, N) -> median over repeats + cross-query weighted IPC."""
    groups: Dict[Tuple, List[Dict]] = {}
    for r in l2_rows:
        key = (r["framework"], r["dataset"], r["N"])
        groups.setdefault(key, []).append(r)

    out = []
    for (fw, ds, N), recs in sorted(groups.items()):
        n_queries = len(set(r["query_idx"] for r in recs))
        n_repeats = len(set(r["repeat"] for r in recs))
        ipcs = [r["ipc_time_weighted"] for r in recs]
        bytes_totals = [r["total_bytes_total"] for r in recs]
        dram_bws = [r["dram_bw_gbs"] for r in recs]
        bps = [r["bp_peak"] for r in recs]
        times = [r["total_time_ns"] for r in recs]

        # cross-query + cross-repeat time-weighted IPC
        total_t_all = sum(times)
        cross_ipc = sum(r["ipc_time_weighted"] * r["total_time_ns"] for r in recs) / total_t_all \
                    if total_t_all > 0 else 0

        out.append({
            "framework": fw, "dataset": ds, "N": N,
            "n_query_runs": len(recs),
            "n_distinct_queries": n_queries,
            "n_repeats": n_repeats,
            "median_ipc": round(statistics.median(ipcs), 4),
            "mean_ipc": round(statistics.mean(ipcs), 4),
            "stdev_ipc": round(statistics.pstdev(ipcs), 4) if len(ipcs) > 1 else 0.0,
            "cross_query_weighted_ipc": round(cross_ipc, 4),
            "median_dram_bytes_total": int(statistics.median(bytes_totals)),
            "median_dram_bw_gbs": round(statistics.median(dram_bws), 2),
            "median_bp_peak": round(statistics.median(bps), 4),
            "median_total_time_ms": round(statistics.median(times) / 1e6, 2),
        })
    return out


def aggregate_l4_hot(dispatch_rows: List[Dict]) -> List[Dict]:
    """For each (framework, ds, N, query, repeat), find the dispatch with max time."""
    groups: Dict[Tuple, List[Dict]] = {}
    for r in dispatch_rows:
        key = (r["framework"], r["dataset"], r["N"], r["query_idx"], r["repeat"])
        groups.setdefault(key, []).append(r)

    out = []
    for (fw, ds, N, q, rep), recs in sorted(groups.items()):
        if not recs:
            continue
        hot = max(recs, key=lambda x: x["time_ns"])
        total_t = sum(r["time_ns"] for r in recs)
        bytes_total = hot["bytes_read"] + hot["bytes_write"]
        bw_gbs = bytes_total / (hot["time_ns"] / 1e9) / 1e9 if hot["time_ns"] > 0 else 0
        out.append({
            "framework": fw, "dataset": ds, "N": N,
            "query_idx": q, "repeat": rep,
            "hot_iteration_idx": hot["dispatch_idx"],
            "hot_time_ns": hot["time_ns"],
            "hot_bytes_total": bytes_total,
            "hot_ipc": round(hot["ipc"], 4),
            "hot_bw_gbs": round(bw_gbs, 2),
            "hot_time_pct_of_query": round(hot["time_ns"] / total_t * 100, 2) if total_t > 0 else 0,
        })
    return out


def write_csv(rows: List[Dict], fieldnames: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_summary_md(l3_rows: List[Dict], l4_rows: List[Dict], path: Path) -> None:
    """Human-readable summary focusing on gunrock vs puercgp comparison."""
    lines = ["# BFS IPC/DRAM Comparison Summary\n"]

    # Group by (dataset, N)
    by_key: Dict[Tuple, Dict[str, Dict]] = {}
    for r in l3_rows:
        key = (r["dataset"], r["N"])
        by_key.setdefault(key, {})[r["framework"]] = r

    lines.append("## L3 per-config comparison (median over repeats)\n")
    lines.append("| Dataset | N | Framework | total DRAM (GB) | total time (ms) | IPC (median) | hot-iter IPC | BP_peak |")
    lines.append("|---|---|---|---|---|---|---|---|")

    # hot iteration lookup
    hot_lookup: Dict[Tuple, float] = {}
    for h in l4_rows:
        key = (h["framework"], h["dataset"], h["N"], h["query_idx"], h["repeat"])
        hot_lookup[key] = h["hot_ipc"]
    hot_by_config: Dict[Tuple, List[float]] = {}
    for h in l4_rows:
        key = (h["framework"], h["dataset"], h["N"])
        hot_by_config.setdefault(key, []).append(h["hot_ipc"])

    for (ds, N), frameworks in sorted(by_key.items()):
        for fw in ["gunrock", "puercgp"]:
            r = frameworks.get(fw)
            if not r:
                continue
            total_gb = r["median_dram_bytes_total"] / 1e9
            # gunrock serial: each query independent, so multiply by N for total
            if fw == "gunrock":
                total_gb_all_queries = total_gb * N
                time_all_queries_ms = r["median_total_time_ms"] * N
            else:
                total_gb_all_queries = total_gb
                time_all_queries_ms = r["median_total_time_ms"]
            hot_ipc_median = statistics.median(hot_by_config.get((fw, ds, N), [0]))
            lines.append(f"| {ds} | {N} | {fw} | {total_gb_all_queries:.2f} | "
                         f"{time_all_queries_ms:.2f} | {r['median_ipc']} | "
                         f"{hot_ipc_median:.4f} | {r['median_bp_peak']} |")

    lines.append("\n## Notes\n")
    lines.append("- gunrock total DRAM = (per-query median DRAM) × N (serial N queries)")
    lines.append("- puercgp total DRAM = per-config median (already N-query shared)")
    lines.append("- IPC is SMSP-level (V100 upper bound ≈ 1.0)")
    lines.append("- hot-iter IPC = median over repeats of the longest iteration's IPC")

    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gunrock-dir", type=Path, required=True)
    ap.add_argument("--puercgp-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning gunrock raw: {args.gunrock_dir}")
    g_rows = scan_gunrock(args.gunrock_dir)
    print(f"  {len(g_rows)} gunrock dispatches")

    print(f"Scanning puercgp raw: {args.puercgp_dir}")
    p_rows = scan_puercgp(args.puercgp_dir)
    print(f"  {len(p_rows)} puercgp dispatches")

    all_dispatch = g_rows + p_rows
    if not all_dispatch:
        print("ERROR: no dispatches parsed")
        return 1

    # L0
    l0_fields = ["framework", "dataset", "N", "query_idx", "repeat", "dispatch_idx",
                 "kernel_name", "bytes_read", "bytes_write", "bytes_total",
                 "time_ns", "ipc"]
    l0_out = []
    for r in all_dispatch:
        l0_out.append({
            **r,
            "bytes_total": r["bytes_read"] + r["bytes_write"],
            "ipc": round(r["ipc"], 4),
        })
    write_csv(l0_out, l0_fields, args.out_dir / "per_dispatch.csv")
    print(f"Wrote L0: {len(l0_out)} rows")

    # L1 = L0 essentially (iteration_idx = dispatch_idx, default mode)
    l1_fields = ["framework", "dataset", "N", "query_idx", "repeat", "iteration_idx",
                 "bytes_read", "bytes_write", "bytes_total", "time_ns", "ipc_time_weighted"]
    l1_out = [{
        "framework": r["framework"], "dataset": r["dataset"], "N": r["N"],
        "query_idx": r["query_idx"], "repeat": r["repeat"],
        "iteration_idx": r["dispatch_idx"],
        "bytes_read": r["bytes_read"], "bytes_write": r["bytes_write"],
        "bytes_total": r["bytes_read"] + r["bytes_write"],
        "time_ns": r["time_ns"], "ipc_time_weighted": round(r["ipc"], 4),
    } for r in all_dispatch]
    write_csv(l1_out, l1_fields, args.out_dir / "per_iteration.csv")
    print(f"Wrote L1: {len(l1_out)} rows")

    # L2
    l2_rows = aggregate_l2_per_query(all_dispatch)
    l2_fields = ["framework", "dataset", "N", "query_idx", "repeat",
                 "total_bytes_read", "total_bytes_write", "total_bytes_total",
                 "total_time_ns", "total_dispatches",
                 "ipc_time_weighted", "dram_bw_gbs", "bp_peak"]
    write_csv(l2_rows, l2_fields, args.out_dir / "per_query.csv")
    print(f"Wrote L2: {len(l2_rows)} rows")

    # L3
    l3_rows = aggregate_l3_per_config(l2_rows)
    l3_fields = ["framework", "dataset", "N", "n_query_runs", "n_distinct_queries", "n_repeats",
                 "median_ipc", "mean_ipc", "stdev_ipc", "cross_query_weighted_ipc",
                 "median_dram_bytes_total", "median_dram_bw_gbs", "median_bp_peak",
                 "median_total_time_ms"]
    write_csv(l3_rows, l3_fields, args.out_dir / "per_config.csv")
    print(f"Wrote L3: {len(l3_rows)} rows")

    # L4
    l4_rows = aggregate_l4_hot(all_dispatch)
    l4_fields = ["framework", "dataset", "N", "query_idx", "repeat",
                 "hot_iteration_idx", "hot_time_ns", "hot_bytes_total", "hot_ipc",
                 "hot_bw_gbs", "hot_time_pct_of_query"]
    write_csv(l4_rows, l4_fields, args.out_dir / "hot_iteration.csv")
    print(f"Wrote L4: {len(l4_rows)} rows")

    # Summary
    write_summary_md(l3_rows, l4_rows, args.out_dir / "summary.md")
    print(f"Wrote summary.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
