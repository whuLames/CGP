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

for rho_push in 0.001 0.005 0.01; do
  for rho_pull in 0.05 0.1 0.2 0.4; do
    for exec in all_push all_pull mixed_serial; do
      "$ROOT/build/push_pull_bench" --graph="$GRAPH" --csv="$OUT" \
        --mode=mixed --exec="$exec" --update=spmm_sum \
        --Q-push=16 --Q-pull=16 --rho-push="$rho_push" --rho-pull="$rho_pull" \
        --repeat=7 --warmup=2
    done
  done
done
