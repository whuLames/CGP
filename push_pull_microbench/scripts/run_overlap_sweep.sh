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

for overlap in 0.0 0.1 0.25 0.5 0.75 0.9 1.0; do
  for exec in all_push all_pull; do
    "$ROOT/build/push_pull_bench" --graph="$GRAPH" --csv="$OUT" \
      --mode=overlap --exec="$exec" --update=spmm_sum \
      --Q=32 --rho-v=0.01 --overlap="$overlap" --repeat=7 --warmup=2
  done
done
