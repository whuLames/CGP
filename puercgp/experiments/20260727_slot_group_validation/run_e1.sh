#!/bin/bash
set -u
BIN=./build/bench_concurrent_push_groups
OUT=experiments/20260727_slot_group_validation/runs
declare -A GRAPHS=(
  [cit-Patents]=/home/zyl/data/csr_data/cit-Patents
  [soc-orkut]=/home/zyl/data/csr_data/soc-orkut
  [soc-twitter]=/home/zyl/data/csr_data/soc-twitter
  [roadNet-CA]=/home/zyl/data/csr_data/roadNet-CA
)
run() {  # name graph args...
  local name=$1 graph=$2; shift 2
  echo "=== $name ==="
  $BIN "$graph" --n=64 --seed=42 --repeats=5 --warmup=1 "$@" \
    > $OUT/e1_${name}.log 2>&1 || echo "FAILED $name"
}
for G in cit-Patents soc-orkut soc-twitter roadNet-CA; do
  P=${GRAPHS[$G]}
  run ${G}_A0 $P --groups=1
  run ${G}_A1 $P --groups=2 --sms=40,40
  run ${G}_A2 $P --groups=2 --sms=16,16
  run ${G}_A3 $P --groups=4 --sms=20,20,20,20
  run ${G}_A4 $P --groups=2 --sequential
  run ${G}_alone40 $P --groups=2 --sms=40,40 --alone=0
  # 正确性：A3 4 组切分指纹 vs A0（单独跑，不计时）
  $BIN $P --n=64 --seed=42 --repeats=1 --warmup=0 --groups=1 --verify \
    > $OUT/e1_${G}_A0_fp.log 2>&1
  $BIN $P --n=64 --seed=42 --repeats=1 --warmup=0 --groups=4 --sms=20,20,20,20 --verify \
    > $OUT/e1_${G}_A3_fp.log 2>&1
  if diff <(grep ^fp $OUT/e1_${G}_A0_fp.log) <(grep ^fp $OUT/e1_${G}_A3_fp.log) >/dev/null; then
    echo "$G fingerprints MATCH"
  else
    echo "$G fingerprints MISMATCH"
  fi
done
echo E1_DONE
