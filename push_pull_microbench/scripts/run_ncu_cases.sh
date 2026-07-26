#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <graph>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH="$1"
METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"

ncu --metrics "$METRICS" "$ROOT/build/push_pull_bench" --graph="$GRAPH" --mode=density --exec=all_push --update=spmm_sum --Q=32 --rho-v=0.001 --rho-q=0.5 --repeat=1 --warmup=0
ncu --metrics "$METRICS" "$ROOT/build/push_pull_bench" --graph="$GRAPH" --mode=density --exec=all_pull --update=spmm_sum --Q=32 --rho-v=0.2 --rho-q=0.5 --repeat=1 --warmup=0
ncu --metrics "$METRICS" "$ROOT/build/push_pull_bench" --graph="$GRAPH" --mode=mixed --exec=mixed_concurrent --update=spmm_sum --Q-push=16 --Q-pull=16 --rho-push=0.005 --rho-pull=0.2 --repeat=1 --warmup=0
