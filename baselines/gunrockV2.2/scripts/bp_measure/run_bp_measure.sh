#!/usr/bin/env bash
# 跑 BP 测量实验：4 数据集 × 3 source × 5 重复 = 60 次 ncu + bfs 运行
# 用法: bash run_bp_measure.sh [--out-dir DIR] [--runs N] [--datasets a,b,c]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BFS_BIN="${REPO_DIR}/build_cuda/bin/bfs"
NCU_BIN="/home/zyl/.conda/envs/torch2.8/bin/ncu"
DATA_DIR="/home/zyl/data/ggr_data/singlegpu"
OUT_DIR="${REPO_DIR}/bp_profiles/raw"
RUNS=5
DATASETS="cit-Patents,soc-orkut,soc-twitter,soc-sinaweibo"
DEVICE=2
SLEEP_BETWEEN=2

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)    OUT_DIR="$2"; shift 2 ;;
    --runs)       RUNS="$2"; shift 2 ;;
    --datasets)   DATASETS="$2"; shift 2 ;;
    --device)     DEVICE="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--out-dir DIR] [--runs N] [--datasets a,b,c] [--device N]"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}"

# 每个 dataset 的 source 列表（前 3 个，来自 bfs_concurrent_multidataset_sources.csv）
declare -A SOURCES
SOURCES[cit-Patents]="0 1 2"
SOURCES[soc-orkut]="1506298 1900229 69341"
SOURCES[soc-twitter]="11702603 18140996 12437392"
SOURCES[soc-sinaweibo]="53297474 23176989 23515621"

# ncu metric 列表（注意：lts__t_request_hit_rate 在 V100 上不支持，已去掉）
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"

IFS=',' read -ra DATASET_LIST <<< "${DATASETS}"

echo "=========================================="
echo "BP measure experiment"
echo "  Output dir: ${OUT_DIR}"
echo "  Datasets:   ${DATASETS}"
echo "  Runs/repeat: ${RUNS}"
echo "  Device:     ${DEVICE}"
echo "=========================================="

total_runs=0
failed_runs=0

for dataset in "${DATASET_LIST[@]}"; do
  graph="${DATA_DIR}/${dataset}.gr"
  if [[ ! -f "${graph}" ]]; then
    echo "ERROR: missing graph file: ${graph}" >&2
    continue
  fi
  sources="${SOURCES[${dataset}]:-}"
  if [[ -z "${sources}" ]]; then
    echo "ERROR: no sources configured for ${dataset}" >&2
    continue
  fi

  echo ""
  echo "===== Dataset: ${dataset} ====="
  for src in ${sources}; do
    for ((run=1; run<=RUNS; run++)); do
      out_csv="${OUT_DIR}/${dataset}_s${src}_r${run}.csv"
      total_runs=$((total_runs+1))

      echo "[${total_runs}] ${dataset} src=${src} run=${run}"
      CUDA_VISIBLE_DEVICES=${DEVICE} \
        "${NCU_BIN}" \
          --metrics "${METRICS}" \
          --csv --page raw \
          "${BFS_BIN}" \
          -m "${graph}" \
          -s "${src}" \
          -n 1 \
          > "${out_csv}" 2>&1 || {
        echo "  FAILED (see ${out_csv})" >&2
        failed_runs=$((failed_runs+1))
        continue
      }
      # 校验 CSV 有内容
      line_count=$(wc -l < "${out_csv}")
      if [[ ${line_count} -lt 7 ]]; then
        echo "  WARNING: only ${line_count} lines in output (expected 7+)" >&2
      fi
      sleep ${SLEEP_BETWEEN}
    done
  done
done

echo ""
echo "=========================================="
echo "Done. Total: ${total_runs}, Failed: ${failed_runs}"
echo "Output: ${OUT_DIR}"
echo "=========================================="
