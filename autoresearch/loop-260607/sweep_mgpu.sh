#!/usr/bin/env bash
# Multi-GPU low-bit-K trec sweep: one persistent worker per GPU, round-robin
# config assignment (configs have ~equal cost, so striping balances well).
# Each config -> driver.sh -> res/<TAG>.tsv. On finish: merge -> results_mgpu.tsv
# + DONE_MGPU marker. Safe under nohup/setsid; poll res/*.tsv for progress.
#
# Usage: GPUS=2,3,4,5 MAXS=-1 sweep_mgpu.sh [CONFIG_FILE]
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-2,3,4,5}"; MAXS="${MAXS:--1}"
CFG_FILE="${1:-$LOOP_DIR/configs_expanded.txt}"
IFS=',' read -r -a GPUARR <<< "$GPUS"
NG=${#GPUARR[@]}
mkdir -p "$LOOP_DIR/res" "$LOOP_DIR/logs" "$LOOP_DIR/out"
rm -f "$LOOP_DIR/DONE_MGPU"
mapfile -t CFGS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$CFG_FILE")
echo "[mgpu] $(date '+%F %T') GPUS=$GPUS workers=$NG configs=${#CFGS[@]} cfg=$CFG_FILE"

worker(){   # $1=physical gpu id, $2=start index
  local gpu="$1" idx
  for ((idx=$2; idx<${#CFGS[@]}; idx+=NG)); do
    read -r tag kb vb pb pr cs _ <<< "${CFGS[$idx]}"
    [ -z "${tag:-}" ] && continue
    echo "[mgpu gpu$gpu] -> $tag"
    GPU="$gpu" bash "$LOOP_DIR/driver.sh" "$tag" "$kb" "$vb" "$pb" "$pr" "$cs" "$MAXS"
  done
}
for ((w=0; w<NG; w++)); do worker "${GPUARR[$w]}" "$w" & sleep 1; done
wait

{
  echo -e "tag\tkbits\tvbits\tpbit\tpratio\tchansel\teff_kbits\ttrec\tsecs"
  cat "$LOOP_DIR"/res/*.tsv 2>/dev/null | sort -t$'\t' -k7,7n -k8,8nr
} > "$LOOP_DIR/results_mgpu.tsv"
echo "ALL_DONE $(date '+%F %T')" > "$LOOP_DIR/DONE_MGPU"
echo "[mgpu] done $(date '+%F %T')"
