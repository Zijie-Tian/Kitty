#!/usr/bin/env bash
# Precision-at-fixed-count sweep (local 3090s, 3/card): configs_precision.txt, trec THEN qasper.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kitty
GPUS="${GPUS:-2,2,2,3,3,3,4,4,4,5,5,5}"   # 3 tasks per 3090, local only
rm -f "$HERE/DONE_precision"
for ds in trec qasper; do
  echo "[precision] === $ds === $(date '+%F %T')"
  GPUS="$GPUS" DATASET="$ds" MAXS=-1 bash "$HERE/sweep_ds_mgpu.sh" "$HERE/configs_precision.txt"
done
echo "PRECISION_DONE $(date '+%F %T')" > "$HERE/DONE_precision"
echo "[precision] done $(date '+%F %T')"
