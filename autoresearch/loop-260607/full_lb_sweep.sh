#!/usr/bin/env bash
# Full 21-subtask LongBench for each config in CFG, parallel across a worker pool
# (list a GPU N times in GPUS for N workers on it). Writes res_<ds>/<tag>.tsv.
# Usage: GPUS=0,0,0,0,0,0 full_lb_sweep.sh [CFG]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-0,0,0,0,0,0}"; CFG="${1:-$HERE/configs_top.txt}"
IFS=',' read -r -a GPUARR <<< "$GPUS"; NG=${#GPUARR[@]}
DATASETS=(2wikimqa dureader gov_report hotpotqa lcc lsht multi_news multifieldqa_en \
          multifieldqa_zh musique narrativeqa passage_count passage_retrieval_en \
          passage_retrieval_zh qasper qmsum repobench-p samsum trec triviaqa vcsum)
mapfile -t CFGS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$CFG")
JOBS=()
for c in "${CFGS[@]}"; do
  read -r tag kb vb pb pr cs _ <<< "$c"
  for ds in "${DATASETS[@]}"; do JOBS+=("$tag|$kb|$vb|$pb|$pr|$cs|$ds"); done
done
rm -f "$HERE/DONE_fulllb"
echo "[fulllb] $(date '+%F %T') configs=${#CFGS[@]} datasets=${#DATASETS[@]} jobs=${#JOBS[@]} workers=$NG"
worker(){
  local gpu="$1" idx
  for ((idx=$2; idx<${#JOBS[@]}; idx+=NG)); do
    IFS='|' read -r tag kb vb pb pr cs ds <<< "${JOBS[$idx]}"
    echo "[fulllb gpu$gpu] $ds $tag"
    DATASET="$ds" GPU="$gpu" bash "$HERE/driver_ds.sh" "$tag" "$kb" "$vb" "$pb" "$pr" "$cs" "${MAXS:--1}"
  done
}
for ((w=0; w<NG; w++)); do worker "${GPUARR[$w]}" "$w" & sleep 1; done
wait
echo "FULLLB_DONE $(date '+%F %T')" > "$HERE/DONE_fulllb"
echo "[fulllb] done $(date '+%F %T')"
