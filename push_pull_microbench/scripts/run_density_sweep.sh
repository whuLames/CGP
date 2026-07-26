#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <graph> <out.csv>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH="$1"
OUT="$2"
rm -f "$OUT"

for Q in 1 4 8 16 32 64; do
  for rho_v in 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.4; do
    for exec in all_push all_pull; do
      "$ROOT/build/push_pull_bench" --graph="$GRAPH" --csv="$OUT" \
        --mode=density --exec="$exec" --update=spmm_sum \
        --Q="$Q" --rho-v="$rho_v" --rho-q=0.5 --repeat=7 --warmup=2
    done
  done
done
