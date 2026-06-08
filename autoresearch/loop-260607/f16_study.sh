#!/usr/bin/env bash
# Focused study: K-cache 1-bit base + f16 MIXED channels (promote_bit=16), V fixed 4-bit.
# Sweeps the f16 channel fraction on ONE GPU and measures one dataset (default qasper,
# the binding/sensitive metric). Self-contained: activates the kitty env itself.
# Usage: GPU=2 DATASET=qasper bash f16_study.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kitty
GPU="${GPU:-2}"; DATASET="${DATASET:-qasper}"
M="${LLAMA32_MODEL_PATH:-/home/zijie/models/Llama-3.2-1B-Instruct}"
D="${LONGBENCH_DATA_ROOT:-/home/zijie/data/LongBench}"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/src" \
       TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUTD="$HERE/f16study"; mkdir -p "$OUTD/out"
RES="$OUTD/results_${DATASET}.tsv"; : > "$RES"
rm -f "$OUTD/DONE_${DATASET}"
# f16-channel fractions of head_dim=64: 1,2,3,4,6,8,12,16,24,32 channels.
RATIOS="0.015625 0.03125 0.046875 0.0625 0.09375 0.125 0.1875 0.25 0.375 0.5"
echo "[f16study] $(date '+%F %T') GPU=$GPU dataset=$DATASET"
for pr in $RATIOS; do
  tag="f16_pr${pr}"
  o="$OUTD/out/${DATASET}_${tag}"; rm -rf "$o"
  eff=$(python -c "print(round($pr*16+(1-$pr)*1,4))")
  ch=$(python -c "print(int(64*$pr+1e-6))")
  t0=$(date +%s)
  if python -m kitty_sim.cli.eval_longbench "$M" --model-family llama3 --model-tag llama32-1b-instruct \
       --variant custom --kbits 1 --vbits 4 --promote_bit 16 --promote_ratio "$pr" --channel_selection 1 \
       --sink_length 32 --buffer_length 128 --group_size 128 \
       --dataset "$DATASET" --data-root "$D" --max-model-len 32768 --local-files-only \
       --output-dir "$o" --flat-output-dir --overwrite > "$OUTD/${DATASET}_${tag}.log" 2>&1 \
     && python -m kitty_sim.cli.score_longbench --model "$o" >> "$OUTD/${DATASET}_${tag}.log" 2>&1; then
    sc=$(python -c "import json;print(json.load(open('$o/result.json'))['$DATASET'])")
  else sc=ERROR; fi
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$ch" "$eff" "$sc" "$((t1-t0))" >> "$RES"
  echo "[f16study] $tag ch=$ch/64 eff_kbits=$eff ${DATASET}=$sc ${t1}-${t0}=$((t1-t0))s"
done
echo "F16_${DATASET}_DONE $(date '+%F %T')" > "$OUTD/DONE_${DATASET}"
echo "[f16study] done $(date '+%F %T')"; cat "$RES"
