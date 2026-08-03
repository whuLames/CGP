#!/usr/bin/env bash
# Run the five-dataset pure pull GE-SpMM group-slots comparison.

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
data_root=${DATA_ROOT:-/home/zyl/data/csr_data}
gpu_id=${GPU_ID:-0}
warmup=${WARMUP:-5}
iterations=${ITERS:-30}
tile_row=${TILE_ROW:-8}
seed=${SEED:-42}
nvcc_path=${NVCC:-/home/zyl/.conda/envs/torch2.8/bin/nvcc}
timestamp=$(date +%Y%m%d-%H%M%S)
experiment_dir=${EXP_DIR:-"${repo_root}/experiments/${timestamp}_ge_spmm_pull_group_slots"}
binary="${script_dir}/ge_spmm_group_slots_bench"
summary="${script_dir}/summarize_ge_spmm_group_slots.py"
datasets=(cit-Patents soc-orkut soc-twitter soc-sinaweibo roadNet-CA)
git_commit=$(git -C "${repo_root}" rev-parse HEAD)

python3 "${summary}" "${experiment_dir}" --initialize \
  "--gpu=${gpu_id}" "--warmup=${warmup}" "--iterations=${iterations}" \
  "--tile-row=${tile_row}" "--seed=${seed}" "--data-root=${data_root}" \
  "--git-commit=${git_commit}" "--nvcc=${nvcc_path}"
printf 'dataset\tstatus\texit_code\n' \
  > "${experiment_dir}/artifacts/status.tsv"

{
  echo "experiment_dir=${experiment_dir}"
  echo "gpu=${gpu_id} warmup=${warmup} iters=${iterations} tile_row=${tile_row} seed=${seed}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
  "${nvcc_path}" --version
} >> "${experiment_dir}/stdout.log" 2>> "${experiment_dir}/stderr.log"

if [[ ! -x "${binary}" ]]; then
  make -C "${script_dir}" ge_spmm_group_slots_bench NVCC="${nvcc_path}" \
    >> "${experiment_dir}/stdout.log" 2>> "${experiment_dir}/stderr.log"
  build_status=$?
  if (( build_status != 0 )); then
    echo "benchmark build failed with exit code ${build_status}" \
      >> "${experiment_dir}/stderr.log"
    exit "${build_status}"
  fi
fi

overall_status=0
for dataset in "${datasets[@]}"; do
  graph_dir="${data_root}/${dataset}"
  csv_path="${experiment_dir}/artifacts/${dataset}.csv"
  command=(
    "${binary}" "${graph_dir}"
    "--case=both"
    "--gpu=${gpu_id}"
    "--warmup=${warmup}"
    "--iters=${iterations}"
    "--tile-row=${tile_row}"
    "--seed=${seed}"
    "--csv=${csv_path}"
  )
  printf 'RUN'
  printf ' %q' "${command[@]}"
  printf '\n'
  {
    printf 'RUN'
    printf ' %q' "${command[@]}"
    printf '\n'
  } >> "${experiment_dir}/stdout.log"
  "${command[@]}" >> "${experiment_dir}/stdout.log" \
    2>> "${experiment_dir}/stderr.log"
  exit_code=$?
  if (( exit_code == 0 )); then
    status=success
  else
    status=failed
    overall_status=1
  fi
  printf '%s\t%s\t%s\n' "${dataset}" "${status}" "${exit_code}" \
    >> "${experiment_dir}/artifacts/status.tsv"
done

python3 "${summary}" "${experiment_dir}" \
  >> "${experiment_dir}/stdout.log" 2>> "${experiment_dir}/stderr.log"
summary_status=$?
if (( summary_status != 0 )); then
  overall_status=1
fi

echo "results=${experiment_dir}/result.csv"
exit "${overall_status}"
