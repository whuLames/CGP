#!/bin/bash
set -u
BIN=./build/bench_replenish
GRAPH=/home/zyl/data/csr_data/roadNet-CA
SRC=experiments/20260712_roadnet_replenish/sources
OUT=experiments/20260727_slot_group_validation/runs
for N in 400 800; do
  for CHUNK in 1 4 8 16 32; do
    RUN=replenish
    [ "$CHUNK" = 1 ] && RUN=both
    echo "=== N=$N chunk=$CHUNK run=$RUN ==="
    $BIN $GRAPH --sssp=0 --wcc=0 \
      --sources-file=$SRC/sources_n${N}.csv \
      --batch-size=32 --replenish-chunk=$CHUNK \
      --mode=push --push=warp --no-discard \
      --warmup=1 --repeats=3 --run=$RUN --max-iterations=20000 \
      > $OUT/e0_n${N}_chunk${CHUNK}.log 2>&1 || echo "FAILED N=$N chunk=$CHUNK"
  done
done
echo E0_DONE
