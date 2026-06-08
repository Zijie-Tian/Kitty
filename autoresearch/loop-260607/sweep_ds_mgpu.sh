#!/usr/bin/env bash
# Multi-GPU dataset sweep: one worker per GPU (round-robin), calls driver_ds.sh.
# Writes res_<DATASET>/<TAG>.tsv (shared with sweep_ds.sh; use distinct tags).
# Usage: GPUS=2,3,4,5 DATASET=qasper MAXS=-1 sweep_ds_mgpu.sh [CONFIG_FILE]
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-2,3,4,5}"; MAXS="${MAXS:--1}"; DATASET="${DATASET:-qasper}"
CFG="${1:-$LOOP_DIR/configs_qasper_boosts.txt}"
IFS=',' read -r -a GPUARR <<< "$GPUS"; NG=${#GPUARR[@]}
mkdir -p "$LOOP_DIR/res_${DATASET}" "$LOOP_DIR/logs"
rm -f "$LOOP_DIR/DONE_${DATASET}_mgpu"
mapfile -t CFGS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$CFG")
echo "[ds_mgpu] $(date '+%F %T') GPUS=$GPUS DATASET=$DATASET workers=$NG configs=${#CFGS[@]}"
worker(){
  local gpu="$1" idx
  for ((idx=$2; idx<${#CFGS[@]}; idx+=NG)); do
    read -r tag kb vb pb pr cs _ <<< "${CFGS[$idx]}"
    [ -z "${tag:-}" ] && continue
    echo "[ds_mgpu gpu$gpu] -> $DATASET $tag"
    DATASET="$DATASET" GPU="$gpu" bash "$LOOP_DIR/driver_ds.sh" "$tag" "$kb" "$vb" "$pb" "$pr" "$cs" "$MAXS"
  done
}
for ((w=0; w<NG; w++)); do worker "${GPUARR[$w]}" "$w" & sleep 1; done
wait
echo "DS_MGPU_${DATASET}_DONE $(date '+%F %T')" > "$LOOP_DIR/DONE_${DATASET}_mgpu"
echo "[ds_mgpu $DATASET] done $(date '+%F %T')"
