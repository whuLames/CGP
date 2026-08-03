#!/usr/bin/env python3
"""Compute exact CSR degree-distribution statistics without sorting vertices."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "cit-Patents": Path("/home/zyl/data/csr_data/cit-Patents/csr_vlist.bin"),
    "soc-orkut": Path("/home/zyl/data/csr_data/soc-orkut/csr_vlist.bin"),
    "soc-twitter": Path("/home/zyl/data/csr_data/soc-twitter/csr_vlist.bin"),
    "soc-sinaweibo": Path("/home/zyl/data/csr_data/soc-sinaweibo/csr_vlist.bin"),
    "roadNet-CA": Path("/home/zyl/data/csr_data/roadNet-CA/csr_vlist.bin"),
}


def quantile_from_histogram(counts: np.ndarray, quantile: float) -> int:
    target = max(1, math.ceil(quantile * int(counts.sum())))
    return int(np.searchsorted(np.cumsum(counts, dtype=np.int64), target))


def top_edge_share(counts: np.ndarray, fraction: float) -> float:
    vertex_count = int(counts.sum())
    remaining = max(1, math.ceil(vertex_count * fraction))
    selected_edges = 0
    for degree in range(len(counts) - 1, -1, -1):
        take = min(remaining, int(counts[degree]))
        selected_edges += take * degree
        remaining -= take
        if remaining == 0:
            break
    total_edges = int(np.dot(np.arange(len(counts), dtype=np.float64), counts))
    return selected_edges / total_edges if total_edges else 0.0


def gini_from_histogram(counts: np.ndarray) -> float:
    vertex_count = int(counts.sum())
    degrees = np.arange(len(counts), dtype=np.float64)
    total_degree = float(np.dot(degrees, counts))
    if total_degree == 0:
        return 0.0
    ranks_before = np.cumsum(counts, dtype=np.int64) - counts
    rank_sums = counts * (2.0 * ranks_before + counts + 1.0) / 2.0
    weighted_rank_sum = float(np.dot(degrees, rank_sums))
    return (2.0 * weighted_rank_sum / (vertex_count * total_degree) -
            (vertex_count + 1.0) / vertex_count)


def analyze(path: Path) -> dict[str, float | int]:
    offsets = np.memmap(path, dtype=np.int32, mode="r")
    degrees = np.diff(offsets)
    counts = np.bincount(degrees)
    degree_values = np.arange(len(counts), dtype=np.float64)
    vertex_count = len(degrees)
    edge_count = int(offsets[-1])
    mean = edge_count / vertex_count
    second_moment = (
        float(np.dot(degree_values * degree_values, counts)) / vertex_count
    )
    standard_deviation = math.sqrt(max(0.0, second_moment - mean * mean))
    result: dict[str, float | int] = {
        "vertices": vertex_count,
        "edges": edge_count,
        "mean_degree": mean,
        "std_degree": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
        "gini": gini_from_histogram(counts),
        "degree_p50": quantile_from_histogram(counts, 0.50),
        "degree_p90": quantile_from_histogram(counts, 0.90),
        "degree_p99": quantile_from_histogram(counts, 0.99),
        "degree_p999": quantile_from_histogram(counts, 0.999),
        "max_degree": len(counts) - 1,
        "degree_le_2_pct": 100.0 * int(counts[:3].sum()) / vertex_count,
        "degree_ge_64_pct": 100.0 * int(counts[64:].sum()) / vertex_count,
        "top_1pct_edge_share": top_edge_share(counts, 0.01),
        "top_0_1pct_edge_share": top_edge_share(counts, 0.001),
    }
    return result


def main() -> None:
    rows = []
    for dataset, path in DATASETS.items():
        row = {"dataset": dataset}
        row.update(analyze(path))
        rows.append(row)
    output = ROOT / "results/degree_distribution.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
