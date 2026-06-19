#!/bin/bash
# Full 21-dataset LongBench (32k), fanned across 4 GPUs: GPU0 x6 + GPU1/2/3 x2 = 12 workers.
# baseline (per-token nf2) then champion (smooth+outlier8@4bit). fp16 already complete (27.59).
set -e
cd /home/zijie/Code/Kitty
PLAIN=/home/zijie/models/Llama-3.2-1B-Instruct
SMOOTH=/home/zijie/models/Llama-3.2-1B-Instruct-smooth
GPUS=0,0,0,0,0,0,1,1,2,2,3,3
echo "[$(date +%H:%M)] === baseline per-token nf2 (12 workers) ==="
QLUT_BIN_CODEBOOKS=nf2 PERTOKEN_OUTLIER_K=0 LLAMA32_MODEL_PATH=$PLAIN \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpus $GPUS --variant qlutattn_pertoken
echo "[$(date +%H:%M)] === CHAMPION smooth+outlier8@4bit (12 workers) ==="
QLUT_BIN_CODEBOOKS=nf2 PERTOKEN_OUTLIER_K=8 PERTOKEN_OUTLIER_BITS=4 \
  LLAMA32_MODEL_PATH=$SMOOTH LLAMA32_MODEL_SLUG=llama32-1b-instruct-smooth \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpus $GPUS --variant qlutattn_pertoken
echo "[$(date +%H:%M)] === ALL DONE — full-21 means ==="
python - <<'PY'
import json,os
for tag,d in [("fp16","llama32-1b-instruct_fp16"),
              ("per-token nf2 baseline","llama32-1b-instruct_qlutattn-pertoken"),
              ("CHAMPION smooth+outlier8@4b","llama32-1b-instruct-smooth_qlutattn-pertoken")]:
    f=f"longbench_out/{d}/pred/result.json"
    if os.path.exists(f):
        r=json.load(open(f)); v=[x for x in r.values() if isinstance(x,(int,float))]
        print(f"  {tag:30s} n={len(v):2d}  mean={sum(v)/len(v):.2f}")
    else: print(f"  {tag:30s} (no result)")
PY
