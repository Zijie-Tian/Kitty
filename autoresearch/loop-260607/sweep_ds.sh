#!/usr/bin/env bash
# Job-pool sweep of driver_ds.sh on one GPU for a chosen dataset.
# Usage: PAR=4 DATASET=qasper GPU=0 MAXS=-1 sweep_ds.sh [CONFIG_FILE]
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAR="${PAR:-4}"; MAXS="${MAXS:--1}"; DATASET="${DATASET:-qasper}"; GPU="${GPU:-0}"
export DATASET GPU
CFG="${1:-$LOOP_DIR/configs_survivors.txt}"
mkdir -p "$LOOP_DIR/res_${DATASET}" "$LOOP_DIR/logs"
rm -f "$LOOP_DIR/DONE_${DATASET}"
echo "[sweep_ds] $(date '+%F %T') DATASET=$DATASET GPU=$GPU PAR=$PAR cfg=$CFG"
sem(){ while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 3; done; }
while read -r tag kb vb pb pr cs _; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue ;; esac
  sem
  ( bash "$LOOP_DIR/driver_ds.sh" "$tag" "$kb" "$vb" "$pb" "$pr" "$cs" "$MAXS" ) &
  sleep 2
done < "$CFG"
wait
{
  echo -e "tag\tkbits\tvbits\tpbit\tpratio\tchansel\teff_kbits\t${DATASET}\tsecs"
  cat "$LOOP_DIR"/res_${DATASET}/*.tsv 2>/dev/null | sort -k7,7n
} > "$LOOP_DIR/results_${DATASET}.tsv"
echo "ALL_DONE $(date '+%F %T')" > "$LOOP_DIR/DONE_${DATASET}"
echo "[sweep_ds $DATASET] done $(date '+%F %T')"; cat "$LOOP_DIR/results_${DATASET}.tsv"
