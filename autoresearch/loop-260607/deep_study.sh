#!/usr/bin/env bash
# Deep study driver (local 3090-server): run configs_deep.txt for trec THEN qasper,
# 3 workers per card. Writes res_trec/ and res_qasper/ + DONE_deep when finished.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kitty
GPUS="${GPUS:-2,2,2,3,3,3,4,4,4,5,5,5}"   # 3 tasks per 3090
rm -f "$HERE/DONE_deep"
for ds in trec qasper; do
  echo "[deep] === $ds === $(date '+%F %T')"
  GPUS="$GPUS" DATASET="$ds" MAXS=-1 bash "$HERE/sweep_ds_mgpu.sh" "$HERE/configs_deep.txt"
done
echo "DEEP_DONE $(date '+%F %T')" > "$HERE/DONE_deep"
echo "[deep] all done $(date '+%F %T')"
