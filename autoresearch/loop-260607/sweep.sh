#!/usr/bin/env bash
# Orchestrate the low-bit-K trec sweep with a bounded parallel job pool.
# Each config -> driver.sh -> res/<TAG>.tsv. When all finish, merge -> results.tsv
# and drop a DONE marker. Safe to run under nohup; poll res/*.tsv for progress.
#
# Usage: PAR=4 MAXS=-1 sweep.sh [CONFIG_FILE]
#   CONFIG_FILE lines: TAG KBITS VBITS PBIT PRATIO CHANSEL  ('#' comments ok)
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAR="${PAR:-4}"; MAXS="${MAXS:--1}"
CFG_FILE="${1:-$LOOP_DIR/configs.txt}"
mkdir -p "$LOOP_DIR/res" "$LOOP_DIR/logs" "$LOOP_DIR/out"
rm -f "$LOOP_DIR/DONE"

echo "[sweep] start $(date '+%F %T')  PAR=$PAR  MAXS=$MAXS  cfg=$CFG_FILE"
sem(){ while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 3; done; }
while read -r tag kb vb pb pr cs _; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue ;; esac
  sem
  ( bash "$LOOP_DIR/driver.sh" "$tag" "$kb" "$vb" "$pb" "$pr" "$cs" "$MAXS" ) &
  sleep 2
done < "$CFG_FILE"
wait

{
  echo -e "tag\tkbits\tvbits\tpbit\tpratio\tchansel\teff_kbits\ttrec\tsecs"
  cat "$LOOP_DIR"/res/*.tsv 2>/dev/null | sort -t$'\t' -k7,7n -k8,8nr
} > "$LOOP_DIR/results.tsv"
echo "ALL_DONE $(date '+%F %T')" > "$LOOP_DIR/DONE"
echo "[sweep] done $(date '+%F %T')"
echo "===== results.tsv ====="
cat "$LOOP_DIR/results.tsv"
