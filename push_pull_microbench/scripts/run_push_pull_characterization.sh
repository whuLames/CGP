#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <graph> <out_dir> [per_query_frontier_size]" >&2
  echo "env: Q=32 RHO_Q=0.5 REPEAT=7 WARMUP=2 SEEDS='1 2 3 4 5'" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH="$1"
OUT_DIR="$2"
PER_Q="${3:-0}"

Q="${Q:-32}"
RHO_Q="${RHO_Q:-0.5}"
REPEAT="${REPEAT:-7}"
WARMUP="${WARMUP:-2}"
SEEDS="${SEEDS:-1 2 3 4 5}"

mkdir -p "$OUT_DIR"

PUSH_FRONTIER_OUT="$OUT_DIR/push_frontier_size.csv"
PUSH_OVERLAP_OUT="$OUT_DIR/push_overlap.csv"
PULL_Q_OUT="$OUT_DIR/pull_q.csv"
rm -f "$PUSH_FRONTIER_OUT" "$PUSH_OVERLAP_OUT" "$PULL_Q_OUT"

# Experiment 1:
# Push cost as the union active vertex set grows. Q and rho_q are fixed, so the
# main independent variable is unique_vertices, reported in the CSV.
for seed in $SEEDS; do
  for rho_v in 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2; do
    "$ROOT/build/push_pull_bench" --graph="$GRAPH" --csv="$PUSH_FRONTIER_OUT" \
      --mode=density --exec=all_push --update=spmm_sum \
      --Q="$Q" --rho-v="$rho_v" --rho-q="$RHO_Q" \
      --repeat="$REPEAT" --warmup="$WARMUP" --seed="$seed"
  done
done

# Experiment 2:
# Push cost as query frontiers overlap. Each query keeps the same frontier size,
# while overlap controls how many frontier vertices are shared across queries.
OVERLAP_ARGS=(--rho-v=0.001)
if [[ "$PER_Q" != "0" ]]; then
  OVERLAP_ARGS=(--frontier-size-per-query="$PER_Q")
fi

for seed in $SEEDS; do
  for overlap in 0.0 0.1 0.25 0.5 0.75 0.9 1.0; do
    "$ROOT/build/push_pull_bench" --graph="$GRAPH" --csv="$PUSH_OVERLAP_OUT" \
      --mode=overlap --exec=all_push --update=spmm_sum \
      --Q="$Q" "${OVERLAP_ARGS[@]}" --overlap="$overlap" \
      --repeat="$REPEAT" --warmup="$WARMUP" --seed="$seed"
  done
done

# Experiment 3:
# Pull GE-SpMM scaling with Q. Frontier density is intentionally fixed; the pull
# kernel scans transpose CSR for every query lane, so Q is the relevant variable.
for seed in $SEEDS; do
  for pull_q in 1 2 4 8 16 32 64; do
    "$ROOT/build/push_pull_bench" --graph="$GRAPH" --csv="$PULL_Q_OUT" \
      --mode=density --exec=all_pull --update=spmm_sum \
      --Q="$pull_q" --rho-v=0.001 --rho-q=0.5 \
      --repeat="$REPEAT" --warmup="$WARMUP" --seed="$seed"
  done
done

echo "wrote:"
echo "  $PUSH_FRONTIER_OUT"
echo "  $PUSH_OVERLAP_OUT"
echo "  $PULL_Q_OUT"
