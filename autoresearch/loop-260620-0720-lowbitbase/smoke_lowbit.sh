#!/bin/bash
cd /home/zijie/Code/Kitty
SMOOTH=/home/zijie/models/Llama-3.2-1B-Instruct-smooth
run() { # gpu codebook obits slug
  QLUT_BIN_CODEBOOKS=$2 PERTOKEN_OUTLIER_K=8 PERTOKEN_OUTLIER_BITS=$3   LLAMA32_MODEL_PATH=$SMOOTH LLAMA32_MODEL_SLUG=$4   MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa   bash scripts/run_exp.sh llama32 --gpu $1 --variant qlutattn_pertoken --max-samples 25
}
run 1 tern 4 smooth-ternbase  &
run 2 sign 4 smooth-signbase  &
run 3 nf2  3 smooth-out3bit   &
wait
echo "ALL_SMOKE_DONE"
for d in smooth-ternbase smooth-signbase smooth-out3bit; do
  echo "== $d =="; cat longbench_out/smoke/${d}_qlutattn-pertoken/pred/result.json 2>/dev/null
done
