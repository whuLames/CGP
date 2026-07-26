#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <out_root> <graph1.gr> [graph2.gr ...]" >&2
  echo "env: DEVICE=0 Q=64 REPEAT=3 WARMUP=2 SEEDS='1 2 3 4 5'" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$1"
shift

DEVICE="${DEVICE:-0}"
Q="${Q:-64}"
REPEAT="${REPEAT:-3}"
WARMUP="${WARMUP:-2}"
SEEDS="${SEEDS:-1 2 3 4 5}"

RHO_V_LIST="${RHO_V_LIST:-0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2}"
RHO_Q_LIST="${RHO_Q_LIST:-0.1 0.2 0.3 0.4 0.5 0.6 0.7}"
PULL_Q_LIST="${PULL_Q_LIST:-1 2 4 8 16 32 64}"

mkdir -p "$OUT_ROOT"

for graph in "$@"; do
  name="$(basename "$graph")"
  name="${name%.gr}"
  out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  push_out="$out_dir/push_density.csv"
  pull_out="$out_dir/pull_q.csv"
  log_out="$out_dir/run.log"
  rm -f "$push_out" "$pull_out" "$log_out"

  echo "[$(date '+%F %T')] graph=$graph" | tee -a "$log_out"
  echo "[$(date '+%F %T')] push density sweep" | tee -a "$log_out"
  for seed in $SEEDS; do
    for rho_v in $RHO_V_LIST; do
      for rho_q in $RHO_Q_LIST; do
        echo "[$(date '+%F %T')] push seed=$seed rho_v=$rho_v rho_q=$rho_q" | tee -a "$log_out"
        CUDA_VISIBLE_DEVICES="$DEVICE" "$ROOT/build/push_pull_bench" \
          --graph="$graph" --csv="$push_out" \
          --mode=density --exec=all_push --update=spmm_sum \
          --Q="$Q" --rho-v="$rho_v" --rho-q="$rho_q" \
          --repeat="$REPEAT" --warmup="$WARMUP" --seed="$seed"
      done
    done
  done

  echo "[$(date '+%F %T')] pull Q sweep" | tee -a "$log_out"
  for seed in $SEEDS; do
    for pull_q in $PULL_Q_LIST; do
      echo "[$(date '+%F %T')] pull seed=$seed Q=$pull_q" | tee -a "$log_out"
      CUDA_VISIBLE_DEVICES="$DEVICE" "$ROOT/build/push_pull_bench" \
        --graph="$graph" --csv="$pull_out" \
        --mode=density --exec=all_pull --update=spmm_sum \
        --Q="$pull_q" --rho-v=0.001 --rho-q=0.5 \
        --repeat="$REPEAT" --warmup="$WARMUP" --seed="$seed"
    done
  done

  echo "[$(date '+%F %T')] done graph=$graph" | tee -a "$log_out"
done
