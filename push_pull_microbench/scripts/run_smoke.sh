#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/build/push_pull_bench" --self-test
"$ROOT/build/push_pull_bench" --graph=toy --mode=density --exec=all_push --update=spmm_sum --Q=4 --rho-v=0.5 --rho-q=0.5 --repeat=1 --warmup=0
"$ROOT/build/push_pull_bench" --graph=toy --mode=density --exec=all_pull --update=spmm_sum --Q=4 --rho-v=0.5 --rho-q=0.5 --repeat=1 --warmup=0
