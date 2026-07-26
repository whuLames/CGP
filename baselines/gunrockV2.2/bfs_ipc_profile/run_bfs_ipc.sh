#!/usr/bin/env bash
# 跑 puercgp vs gunrock BFS IPC/DRAM 对比实验
# 用法: bash run_bfs_ipc.sh [--gunrock-only|--puercgp-only] [--datasets a,b,c] [--repeats N]
#
# 矩阵: 3 数据集 × 5 N 值 × 2 框架
# - gunrock: 每 query 1 个 ncu launch，含 5 repeats via -n 5
# - puercgp: 每 N 5 个 ncu launches（外层 repeats）

set -uo pipefail  # 不用 -e，让单次失败不退出整个脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GUNROCK_BIN="${REPO_DIR}/build_cuda/bin/bfs"
PUE_BINARY="/home/zyl/Projects/ocgp/puercgp/build/validate_bfs"
NCU_BIN="/home/zyl/.conda/envs/torch2.8/bin/ncu"
GUNROCK_DATA="/home/zyl/data/ggr_data/singlegpu"
PUE_DATA="/home/zyl/data/csr_data"
SOURCES_CSV="${REPO_DIR}/bfs_concurrent_multidataset_sources.csv"

OUT_BASE="${SCRIPT_DIR}/raw"
GUNROCK_OUT="${OUT_BASE}/gunrock"
PUE_OUT="${OUT_BASE}/puercgp"

DATASETS="soc-orkut,soc-twitter,soc-sinaweibo"
N_VALUES="2,4,8,16,32"
REPEATS_GUNROCK=5    # 通过 -n 5 合并到单 launch
REPEATS_PUE=5        # 外层 repeats
DEVICE=2
SLEEP_BETWEEN=1

RUN_GUNROCK=1
RUN_PUERCGP=1

METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_issued.avg.per_cycle_active"

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gunrock-only) RUN_PUERCGP=0; shift ;;
    --puercgp-only) RUN_GUNROCK=0; shift ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --n-values) N_VALUES="$2"; shift 2 ;;
    --repeats)  REPEATS_GUNROCK="$2"; REPEATS_PUE="$2"; shift 2 ;;
    --device)   DEVICE="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--gunrock-only|--puercgp-only] [--datasets a,b,c] [--n-values 2,4,8] [--repeats N]"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${GUNROCK_OUT}" "${PUE_OUT}"

IFS=',' read -ra DS_ARRAY <<< "${DATASETS}"
IFS=',' read -ra N_ARRAY  <<< "${N_VALUES}"

# 为每个 dataset 读 sources（前 32 个）
declare -A SOURCES_BY_DS
for ds in "${DS_ARRAY[@]}"; do
  # sources CSV 格式: dataset,query_id,source,iteration_length
  sources=$(awk -F',' -v d="$ds" '$1==d {print $3}' "${SOURCES_CSV}" | head -32 | tr '\n' ' ')
  SOURCES_BY_DS["$ds"]="${sources}"
  echo "[setup] ${ds}: ${#sources} sources"
done

echo ""
echo "=========================================="
echo "BFS IPC/DRAM comparison"
echo "  Datasets:    ${DATASETS}"
echo "  N values:    ${N_VALUES}"
echo "  Gunrock rep: ${REPEATS_GUNROCK} (via -n inside single ncu launch)"
echo "  Puercgp rep: ${REPEATS_PUE}"
echo "  Device:      ${DEVICE}"
echo "=========================================="

total=0
failed=0

# ============== gunrock ==============
if [[ ${RUN_GUNROCK} -eq 1 ]]; then
  echo ""
  echo "========== GUNROCK =========="
  for ds in "${DS_ARRAY[@]}"; do
    sources_str="${SOURCES_BY_DS[$ds]}"
    read -ra src_arr <<< "${sources_str}"
    graph="${GUNROCK_DATA}/${ds}.gr"

    for N in "${N_ARRAY[@]}"; do
      for ((qi=0; qi<N; qi++)); do
        src="${src_arr[$qi]}"
        out="${GUNROCK_OUT}/${ds}_N${N}_q${qi}.csv"
        total=$((total+1))

        # Skip if already exists (resumable)
        if [[ -f "${out}" && $(wc -l < "${out}") -gt 10 ]]; then
          echo "[${total}] gunrock ${ds} N=${N} q=${qi} src=${src} -- SKIP (exists)"
          continue
        fi

        echo "[${total}] gunrock ${ds} N=${N} q=${qi} src=${src}"
        CUDA_VISIBLE_DEVICES=${DEVICE} \
          "${NCU_BIN}" \
            --metrics "${METRICS}" \
            --csv --page raw \
            "${GUNROCK_BIN}" \
              -m "${graph}" \
              -s "${src}" \
              -n "${REPEATS_GUNROCK}" \
            > "${out}" 2>&1 || {
          echo "  FAILED (see ${out})"
          failed=$((failed+1))
        }
        sleep ${SLEEP_BETWEEN}
      done
    done
  done
fi

# ============== puercgp ==============
if [[ ${RUN_PUERCGP} -eq 1 ]]; then
  echo ""
  echo "========== PUERCGP =========="
  for ds in "${DS_ARRAY[@]}"; do
    sources_str="${SOURCES_BY_DS[$ds]}"
    read -ra src_arr <<< "${sources_str}"
    csr_dir="${PUE_DATA}/${ds}"

    for N in "${N_ARRAY[@]}"; do
      # 取前 N 个 source，逗号连接
      src_csv=""
      for ((qi=0; qi<N; qi++)); do
        [[ $qi -gt 0 ]] && src_csv+=","
        src_csv+="${src_arr[$qi]}"
      done

      for ((r=1; r<=REPEATS_PUE; r++)); do
        out="${PUE_OUT}/${ds}_N${N}_r${r}.csv"
        total=$((total+1))

        # Skip if already exists (resumable)
        if [[ -f "${out}" && $(wc -l < "${out}") -gt 10 ]]; then
          echo "[${total}] puercgp ${ds} N=${N} r=${r} -- SKIP (exists)"
          continue
        fi

        echo "[${total}] puercgp ${ds} N=${N} r=${r}"
        CUDA_VISIBLE_DEVICES=${DEVICE} \
          "${NCU_BIN}" \
            --metrics "${METRICS}" \
            --csv --page raw \
            "${PUE_BINARY}" \
              "${csr_dir}" \
              "${src_csv}" \
              1 \
              push shared_node_warp bitmap \
            > "${out}" 2>&1 || {
          echo "  FAILED (see ${out})"
          failed=$((failed+1))
        }
        sleep ${SLEEP_BETWEEN}
      done
    done
  done
fi

echo ""
echo "=========================================="
echo "Done. Total: ${total}, Failed: ${failed}"
echo "Gunrock output: ${GUNROCK_OUT}"
echo "Puercgp output: ${PUE_OUT}"
echo "=========================================="
