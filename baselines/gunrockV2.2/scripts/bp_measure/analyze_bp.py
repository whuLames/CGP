#!/usr/bin/env python3
"""Parse ncu raw CSVs from run_bp_measure.sh and compute Bandwidth Pressure (BP).

Two-stage analysis:
  Stage 1: scan all raw CSVs, extract per-kernel (dram_bytes, time_ns)
           filtered to BFS main-loop kernels only.
  Stage 2: find max achieved_BW across all runs, compute BP_relative.

Outputs:
  bp_profile.csv: long table, one row per (dataset, source, run)
  bp_summary.csv: wide table, one row per dataset
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PEAK_BW_GBS = 900.0  # V100 HBM2 peak

# BFS main-loop kernel name patterns
# - block_mapped_kernel / thread_mapped_kernel / merge_path_kernel: advance operator
# - DeviceReduce*: frontier size reduction (part of BFS convergence check)
BFS_MAIN_PATTERNS = [
    r"block_mapped_kernel",
    r"thread_mapped_kernel",
    r"merge_path_kernel",
    r"DeviceReduce",
]

# Init/reset/fill kernels to exclude
EXCLUDE_PATTERNS = [
    r"static_kernel",
    r"uninitialized_copy",
    r"__fill::functor",
    r"uninitialized",
]


def is_bfs_main(name: str) -> bool:
    """Check if kernel name matches BFS main-loop patterns."""
    if any(re.search(p, name) for p in EXCLUDE_PATTERNS):
        return False
    return any(re.search(p, name) for p in BFS_MAIN_PATTERNS)


def to_int(s: str) -> int:
    """Convert ncu CSV cell value (quoted, comma-grouped) to int."""
    if not s:
        return 0
    s = s.replace('"', '').replace(',', '').strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def find_header_line(lines: List[str]) -> int:
    """Find the line index of CSV header (starts with '"ID"')."""
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            return i
    raise ValueError("no CSV header found (line starting with '\"ID\"')")


def parse_ncu_csv(path: Path) -> Dict:
    """Parse one ncu raw CSV file.

    Returns dict with:
      - total_bytes_read, total_bytes_write, total_time_ns (all kernels)
      - bfs_main: {bytes_read, bytes_write, time_ns, kernel_count}
      - all_kernels: {bytes_read, bytes_write, time_ns, kernel_count}
      - parse_ok: bool
    """
    result = {
        "parse_ok": False,
        "bfs_main": {"bytes_read": 0, "bytes_write": 0, "time_ns": 0, "kernel_count": 0},
        "all_kernels": {"bytes_read": 0, "bytes_write": 0, "time_ns": 0, "kernel_count": 0},
    }
    try:
        text = path.read_text()
    except Exception as e:
        result["error"] = str(e)
        return result

    lines = text.splitlines()
    try:
        header_idx = find_header_line(lines)
    except ValueError as e:
        result["error"] = str(e)
        return result

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        if not row.get("ID"):
            continue
        name = row.get("Kernel Name", "")
        br = to_int(row.get("dram__bytes_read.sum", "0"))
        bw = to_int(row.get("dram__bytes_write.sum", "0"))
        t = to_int(row.get("gpu__time_duration.sum", "0"))

        # all kernels
        result["all_kernels"]["bytes_read"] += br
        result["all_kernels"]["bytes_write"] += bw
        result["all_kernels"]["time_ns"] += t
        result["all_kernels"]["kernel_count"] += 1

        # BFS main only
        if is_bfs_main(name):
            result["bfs_main"]["bytes_read"] += br
            result["bfs_main"]["bytes_write"] += bw
            result["bfs_main"]["time_ns"] += t
            result["bfs_main"]["kernel_count"] += 1

    result["parse_ok"] = True
    return result


def compute_bw_bp(bytes_total: int, time_ns: int, peak_bw_gbs: float = PEAK_BW_GBS) -> Tuple[float, float]:
    """Compute achieved BW (GB/s) and BP_peak."""
    if time_ns <= 0:
        return 0.0, 0.0
    bw_gbs = bytes_total / (time_ns / 1e9) / 1e9
    bp_peak = bw_gbs / peak_bw_gbs
    return bw_gbs, bp_peak


def scan_raw_dir(raw_dir: Path, peak_bw_gbs: float = PEAK_BW_GBS) -> List[Dict]:
    """Scan raw directory, parse each CSV.

    Returns list of per-run dicts with keys:
      dataset, source, run, raw_file,
      bytes_read, bytes_write, time_ns, kernel_count,
      achieved_bw_gbs, bp_peak
    (computed for bfs_main scope)
    """
    rows = []
    pattern = re.compile(r"^(.+)_s(\d+)_r(\d+)\.csv$")
    files = sorted(raw_dir.glob("*.csv"))
    for f in files:
        m = pattern.match(f.name)
        if not m:
            continue
        dataset, source, run = m.group(1), int(m.group(2)), int(m.group(3))
        parsed = parse_ncu_csv(f)
        if not parsed.get("parse_ok"):
            print(f"WARN: failed to parse {f}: {parsed.get('error')}")
            continue
        bfs = parsed["bfs_main"]
        bytes_total = bfs["bytes_read"] + bfs["bytes_write"]
        bw_gbs, bp_peak = compute_bw_bp(bytes_total, bfs["time_ns"], peak_bw_gbs)
        rows.append({
            "dataset": dataset,
            "source": source,
            "run": run,
            "raw_file": f.name,
            "bfs_kernel_count": bfs["kernel_count"],
            "bfs_bytes_read": bfs["bytes_read"],
            "bfs_bytes_write": bfs["bytes_write"],
            "bfs_bytes_total": bytes_total,
            "bfs_time_ms": bfs["time_ns"] / 1e6,
            "achieved_bw_gbs": round(bw_gbs, 2),
            "bp_peak": round(bp_peak, 4),
            # all-kernels variant for reference
            "all_bytes_total": parsed["all_kernels"]["bytes_read"] + parsed["all_kernels"]["bytes_write"],
            "all_time_ms": parsed["all_kernels"]["time_ns"] / 1e6,
        })
    return rows


def build_summary(profile_rows: List[Dict]) -> List[Dict]:
    """Aggregate per-dataset stats."""
    # find max achieved BW across all runs (BFS main only)
    max_bfs_bw = max((r["achieved_bw_gbs"] for r in profile_rows), default=1.0)
    if max_bfs_bw <= 0:
        max_bfs_bw = 1.0

    # group by dataset
    by_dataset: Dict[str, List[Dict]] = {}
    for r in profile_rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    summary = []
    for dataset, rows in sorted(by_dataset.items()):
        bp_peaks = [r["bp_peak"] for r in rows]
        bws = [r["achieved_bw_gbs"] for r in rows]
        bp_rels = [r["achieved_bw_gbs"] / max_bfs_bw for r in rows]

        summary.append({
            "dataset": dataset,
            "n_runs": len(rows),
            "max_bfs_bw_used_gbs": round(max_bfs_bw, 2),
            "median_bp_peak": round(statistics.median(bp_peaks), 4),
            "mean_bp_peak": round(statistics.mean(bp_peaks), 4),
            "stdev_bp_peak": round(statistics.pstdev(bp_peaks) if len(bp_peaks) > 1 else 0.0, 4),
            "median_bp_relative": round(statistics.median(bp_rels), 4),
            "mean_bp_relative": round(statistics.mean(bp_rels), 4),
            "median_1_over_bp_peak": round(1.0 / statistics.median(bp_peaks), 4) if statistics.median(bp_peaks) > 0 else 0.0,
            "median_achieved_bw_gbs": round(statistics.median(bws), 2),
            "median_bfs_time_ms": round(statistics.median([r["bfs_time_ms"] for r in rows]), 2),
            "median_bfs_bytes_gb": round(statistics.median([r["bfs_bytes_total"] for r in rows]) / 1e9, 3),
        })
    return summary


def write_csv(rows: List[Dict], fieldnames: List[str], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--profile-out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--peak-bw-gbs", type=float, default=PEAK_BW_GBS,
                    help=f"Peak HBM BW in GB/s (default: {PEAK_BW_GBS} for V100)")
    args = ap.parse_args()

    # 用 args.peak_bw_gbs 作为运行时峰值（默认就是 PEAK_BW_GBS）
    peak_bw = args.peak_bw_gbs

    print(f"Scanning raw dir: {args.raw_dir}")
    profile_rows = scan_raw_dir(args.raw_dir, peak_bw)
    if not profile_rows:
        print("ERROR: no profile rows extracted. Check raw dir / file naming.")
        return 1
    print(f"  Parsed {len(profile_rows)} runs")

    summary_rows = build_summary(profile_rows)

    profile_fields = [
        "dataset", "source", "run", "raw_file",
        "bfs_kernel_count",
        "bfs_bytes_read", "bfs_bytes_write", "bfs_bytes_total",
        "bfs_time_ms", "achieved_bw_gbs", "bp_peak",
        "all_bytes_total", "all_time_ms",
    ]
    summary_fields = [
        "dataset", "n_runs",
        "max_bfs_bw_used_gbs",
        "median_bp_peak", "mean_bp_peak", "stdev_bp_peak",
        "median_bp_relative", "mean_bp_relative",
        "median_1_over_bp_peak",
        "median_achieved_bw_gbs",
        "median_bfs_time_ms", "median_bfs_bytes_gb",
    ]

    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(profile_rows, profile_fields, args.profile_out)
    write_csv(summary_rows, summary_fields, args.summary_out)

    print(f"\nWrote: {args.profile_out} ({len(profile_rows)} rows)")
    print(f"Wrote: {args.summary_out} ({len(summary_rows)} rows)")

    print("\n=== Summary ===")
    for s in summary_rows:
        print(f"\n{s['dataset']}:")
        print(f"  runs                       = {s['n_runs']}")
        print(f"  max BFS BW (all datasets) = {s['max_bfs_bw_used_gbs']} GB/s")
        print(f"  median BP_peak            = {s['median_bp_peak']:.4f}")
        print(f"  median BP_relative        = {s['median_bp_relative']:.4f}")
        print(f"  median 1/BP_peak          = {s['median_1_over_bp_peak']:.4f}")
        print(f"  median achieved BW        = {s['median_achieved_bw_gbs']} GB/s")
        print(f"  median BFS time           = {s['median_bfs_time_ms']} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
