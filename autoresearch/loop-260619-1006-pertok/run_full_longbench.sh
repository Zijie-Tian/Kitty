#!/bin/bash
# Full 21-dataset LongBench (32k) for the per-token autoresearch validation.
set -e
cd /home/zijie/Code/Kitty
PLAIN=/home/zijie/models/Llama-3.2-1B-Instruct
SMOOTH=/home/zijie/models/Llama-3.2-1B-Instruct-smooth
echo "[$(date +%H:%M)] fp16 ceiling ..."
LLAMA32_MODEL_PATH=$PLAIN MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 --variant fp16
echo "[$(date +%H:%M)] per-token nf2 baseline ..."
QLUT_BIN_CODEBOOKS=nf2 PERTOKEN_OUTLIER_K=0 LLAMA32_MODEL_PATH=$PLAIN \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_pertoken
echo "[$(date +%H:%M)] CHAMPION smooth+outlier8@4bit ..."
QLUT_BIN_CODEBOOKS=nf2 PERTOKEN_OUTLIER_K=8 PERTOKEN_OUTLIER_BITS=4 \
  LLAMA32_MODEL_PATH=$SMOOTH LLAMA32_MODEL_SLUG=llama32-1b-instruct-smooth \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_pertoken
echo "[$(date +%H:%M)] ALL DONE. results:"
for d in llama32-1b-instruct_fp16 llama32-1b-instruct_qlutattn-pertoken llama32-1b-instruct-smooth_qlutattn-pertoken; do
  echo "== $d =="; cat longbench_out/$d/pred/result.json 2>/dev/null
done
