#!/usr/bin/env bash
# Generalized single-dataset LongBench eval for one low-bit-K config.
# DATASET via env (default trec). Writes res_<DATASET>/<TAG>.tsv + logs/<DATASET>_<TAG>.log.
# Native per-dataset generation length is used (no --max-gen override).
# Usage: DATASET=qasper GPU=0 driver_ds.sh TAG KBITS VBITS PBIT PRATIO CHANSEL [MAXSAMPLES]
#   KBITS=fp16 -> dense fp16 baseline.
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LOOP_DIR/../.." && pwd)"
TAG="$1"; KBITS="$2"; VBITS="$3"; PBIT="$4"; PRATIO="$5"; CHANSEL="$6"; MAXS="${7:--1}"
DATASET="${DATASET:-trec}"
MODEL="${LLAMA32_MODEL_PATH:-/home/zijie/models/Llama-3.2-1B-Instruct}"
DATAROOT="${LONGBENCH_DATA_ROOT:-/home/zijie/data/LongBench}"
GPU="${GPU:-0}"
RESDIR="$LOOP_DIR/res_${DATASET}"; OUT="$LOOP_DIR/out_${DATASET}/$TAG"; LOG="$LOOP_DIR/logs/${DATASET}_${TAG}.log"
mkdir -p "$OUT" "$RESDIR" "$LOOP_DIR/logs"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_ROOT/src" \
       TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ "$KBITS" = "fp16" ]; then
  VARGS=(--variant fp16); EFFB=16.0
else
  VARGS=(--variant custom --kbits "$KBITS" --vbits "$VBITS" --promote_bit "$PBIT" \
         --promote_ratio "$PRATIO" --channel_selection "$CHANSEL" \
         --sink_length 32 --buffer_length 128 --group_size 128)
  EFFB=$(python -c "print(round($PRATIO*$PBIT+(1-$PRATIO)*$KBITS,4))")
fi
T0=$(date +%s)
python -m kitty_sim.cli.eval_longbench "$MODEL" --model-family "${MODEL_FAMILY:-llama3}" --model-tag "${MODEL_TAG:-llama32-1b-instruct}" \
  "${VARGS[@]}" --dataset "$DATASET" --data-root "$DATAROOT" \
  --max-model-len 32768 --max-samples "$MAXS" --local-files-only \
  --output-dir "$OUT" --flat-output-dir --overwrite >"$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo -e "${TAG}\t${KBITS}\t${VBITS}\t${PBIT}\t${PRATIO}\t${CHANSEL}\t${EFFB}\tERROR\t0" >"$RESDIR/$TAG.tsv"
  echo "[FAIL] $DATASET $TAG rc=$rc (see $LOG)"; exit "$rc"
fi
python -m kitty_sim.cli.score_longbench --model "$OUT" >>"$LOG" 2>&1
T1=$(date +%s)
SC=$(python -c "import json;print(json.load(open('$OUT/result.json'))['$DATASET'])")
echo -e "${TAG}\t${KBITS}\t${VBITS}\t${PBIT}\t${PRATIO}\t${CHANSEL}\t${EFFB}\t${SC}\t$((T1-T0))" >"$RESDIR/$TAG.tsv"
echo "[done] $DATASET $TAG eff_kbits=$EFFB ${DATASET}=$SC $((T1-T0))s"
