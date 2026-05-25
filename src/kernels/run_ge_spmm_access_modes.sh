#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  /home/zyl/data/csr_data/cit-Patents
  /home/zyl/data/csr_data/soc-LiveJournal1
)

MS=(4 16 32 64 128)
MODES=(full load_only graph_only)

TILE_ROW="${TILE_ROW:-8}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-30}"
GPU="${GPU:-0}"
BIN="${BIN:-./ge_spmm_bench}"
VERIFY="${VERIFY:-0}"

echo "CSV_HEADER: dataset,kernel,mode,M,tile_row,iters,avg_ms,per_comp_ms,max_rel_err"

for dataset in "${DATASETS[@]}"; do
  for m in "${MS[@]}"; do
    for mode in "${MODES[@]}"; do
      extra_args=()
      if [[ "${VERIFY}" != "1" ]]; then
        extra_args+=(--skip-verify)
      fi
      "${BIN}" "${dataset}" \
        --M="${m}" \
        --tile-row="${TILE_ROW}" \
        --warmup="${WARMUP}" \
        --iters="${ITERS}" \
        --gpu="${GPU}" \
        --mode="${mode}" \
        --csv-only \
        "${extra_args[@]}"
    done
  done
done
