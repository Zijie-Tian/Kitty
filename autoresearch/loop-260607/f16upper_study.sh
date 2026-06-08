#!/usr/bin/env bash
# f16 upper-bound sweep (local 3090s, 3/card): configs_f16upper.txt for trec THEN qasper.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kitty
GPUS="${GPUS:-2,2,2,3,3,3,4,4,4,5,5,5}"   # 3 tasks per 3090
rm -f "$HERE/DONE_f16upper"
for ds in trec qasper; do
  echo "[f16upper] === $ds === $(date '+%F %T')"
  GPUS="$GPUS" DATASET="$ds" MAXS=-1 bash "$HERE/sweep_ds_mgpu.sh" "$HERE/configs_f16upper.txt"
done
echo "F16UPPER_DONE $(date '+%F %T')" > "$HERE/DONE_f16upper"
echo "[f16upper] done $(date '+%F %T')"
