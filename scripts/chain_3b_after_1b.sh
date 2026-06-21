#!/usr/bin/env bash
# Auto-chain: wait until the 1B qlutattn-k168v4-pt full run finishes (its
# pred/result.json is written only after all 21 datasets are scored), then launch
# the 3B full run on the same variant. 3B at 32k context needs 1 worker/GPU
# (2/card OOMs on a 24GB 3090), so 6 GPUs -> 6 workers. LLAMA32_MODEL_SLUG is set
# so the 3B output dir does not collide with the 1B one.
set -u
REPO=/home/zijie/Code/Kitty
RESULT="$REPO/longbench_out/llama32-1b-instruct_qlutattn-k168v4-pt/pred/result.json"
LOG="$REPO/longbench_out/k168v4pt_3b_full_6w.log"

echo "[chain] $(date '+%F %T') waiting for 1B result.json: $RESULT"
WAITED=0; MAX=43200   # 12h safety cap so it never waits forever
until [ -f "$RESULT" ]; do
  sleep 180; WAITED=$((WAITED+180))
  if [ "$WAITED" -ge "$MAX" ]; then
    echo "[chain] $(date '+%F %T') TIMEOUT after ${MAX}s waiting for 1B; aborting 3B"; exit 1
  fi
done

echo "[chain] $(date '+%F %T') 1B finished; settling 30s for GPU release ..."
sleep 30
echo "[chain] $(date '+%F %T') launching 3B qlutattn-k168v4-pt full (6 GPU x 1 worker)"
cd "$REPO"
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-3B-Instruct \
LLAMA32_MODEL_SLUG=llama32-3b-instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash "$REPO/scripts/run_exp.sh" llama32 --gpus 0,1,2,3,4,5 --variant qlutattn_k168v4_pt > "$LOG" 2>&1
rc=$?
echo "[chain] $(date '+%F %T') 3B run exited rc=$rc -> longbench_out/llama32-3b-instruct_qlutattn-k168v4-pt/"
exit $rc
