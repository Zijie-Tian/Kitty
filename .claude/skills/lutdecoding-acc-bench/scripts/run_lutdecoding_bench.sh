#!/usr/bin/env bash
# lutdecoding-acc-bench driver -- run the STANDARD KV-cache accuracy benchmark for
# ONE model on LongBench, via the repo's sole LongBench entry point
# scripts/run_exp.sh. One run = these methods, in this fixed order:
#
#   1. fp16        F16 FULL          dense fp16 KV (the quality ceiling)
#   2. shadowkv    ShadowKV          pure-torch sparse-KV accuracy proxy
#   3. kitty       Kitty             paper-style k2/b4/v2/pr0.125
#   4. kivi_star   KIVI*-2           KIVI uniform K2V2 + sink=32
#   5. kivi        KIVI-2            KIVI uniform K2V2 (no sink)
#   6. qlutattn    QLUTATTN          snf-pt: per-token sign/nf2 offline sigma^2-mix,
#                                    sign-frac=0.5 -> ~1.875 bit K, V per-token 4-bit
#   7. qlutattn_fast QLUTATTN-fast   per-token PURE sign (qlutattn_k125v4_pt): ~1.25 bit
#                                    K, V per-token 4-bit. NO calibration, fastest decode.
#
# (QUEST was dropped from this codebase in commit 8adb49b and is intentionally
#  NOT part of this benchmark. If it is restored, add it as a 7th method here.)
#
# All but fp16/kivi run on the pure-torch sim fake-quant path: an ACCURACY PROXY,
# no real KV-memory/speed savings. QLUTATTN REQUIRES a one-time offline calibration
# (done automatically below); every other method runs config-only.
#
# Usage (cd into the Kitty repo, conda activate kitty, then):
#   TARGET=llama32 \
#   MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
#   MODEL_SLUG=llama32-1b-instruct \
#   MAX_GEN=256 GPUS=0,0,0,1,1,1,2,2,2 \
#   CALIB_DATA=/home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
#   bash run_lutdecoding_bench.sh [smoke|full] [method ...]
#
# Positional:
#   $1   mode: smoke (2 samples/dataset, subset, smoke/ layout) | full (all 21, full layout). default full
#   $2.. optional method subset (names from column 1 above); empty = all 6
#
# Env:
#   TARGET         run_exp.sh target -> decides model family + per-target env prefix:
#                  llama32|llama|qwen|glm|deepseek  (default llama32)
#   MODEL_PATH     local model dir (required)
#   MODEL_SLUG     output <model> dir segment (required; keeps non-default models from colliding)
#   MAX_GEN        per-target generation cap (llama32=256, qwen=2048, ...) (default 256)
#   GPUS           run_exp.sh --gpus layout, list a card N times for N workers (default 0)
#   CALIB_DATA     wikitext parquet for the QLUTATTN offline calibration (required if qlutattn runs)
#   MAX_MODEL_LEN  context cap (default 32768; keep 32768 for real full runs)
#   SIGN_FRAC      QLUTATTN sign fraction -> K bit = f*1.25+(1-f)*2.5 (default 0.5 -> 1.875)
#   REPO           Kitty repo root (default: current dir; must contain scripts/run_exp.sh)
#
# IMPORTANT: this launches GPU work. Per the project rule, confirm the GPU
# id(s)/scale with the user BEFORE running -- this script does not ask.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"; [ $# -ge 1 ] && shift || true
SEL_METHODS=("$@")            # optional subset; empty = all

REPO="${REPO:-$PWD}"
TARGET="${TARGET:-llama32}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_SLUG="${MODEL_SLUG:-}"
MAX_GEN="${MAX_GEN:-256}"
GPUS="${GPUS:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
CALIB_DATA="${CALIB_DATA:-}"
SIGN_FRAC="${SIGN_FRAC:-0.5}"
MODEL_ID="${MODEL_ID:-}"           # model-id/display label; defaults to MODEL_PATH below

# ---- validate -------------------------------------------------------------- #
[ -f "$REPO/scripts/run_exp.sh" ] || { echo "[err] REPO=$REPO has no scripts/run_exp.sh -- cd to the Kitty repo or pass REPO=" >&2; exit 1; }
{ [ -n "$MODEL_PATH" ] && [ -e "$MODEL_PATH" ]; } || { echo "[err] MODEL_PATH unset or missing: '$MODEL_PATH'" >&2; exit 1; }
[ -n "$MODEL_SLUG" ] || { echo "[err] MODEL_SLUG required (e.g. llama32-1b-instruct)" >&2; exit 1; }
case "$MODE" in smoke|full) ;; *) echo "[err] mode must be smoke|full (got '$MODE')" >&2; exit 1;; esac

case "$TARGET" in
  llama32|llama3.2) PFX=LLAMA32 ;;
  llama|llama3)     PFX=LLAMA ;;
  qwen|qwen3)       PFX=QWEN ;;
  glm|glm4)         PFX=GLM ;;
  deepseek)         PFX=DEEPSEEK ;;
  *) echo "[err] unknown TARGET '$TARGET' (llama32|llama|qwen|glm|deepseek)" >&2; exit 1;;
esac

cd "$REPO"
# id/display label: default to the real model path so run_exp.sh's built-in
# per-target default id (e.g. Qwen3-8B for `qwen`, Llama-3.2-1B for `llama32`)
# is NOT shown in logs when running a different model. Loading always uses PATH.
MODEL_ID="${MODEL_ID:-$MODEL_PATH}"
export "${PFX}_MODEL_ID=$MODEL_ID"
export "${PFX}_MODEL_PATH=$MODEL_PATH"
export "${PFX}_MODEL_SLUG=$MODEL_SLUG"
export "${PFX}_MAX_GEN=$MAX_GEN"
export MAX_MODEL_LEN

SMOKE_ARGS=(); LAYOUT="full"
if [ "$MODE" = smoke ]; then
  SMOKE_ARGS=(--max-samples 2)
  export DATASETS_CSV="${DATASETS_CSV:-multifieldqa_en,hotpotqa}"   # retrieval -> exercises the K path
  LAYOUT="smoke"
fi

echo "[bench] target=$TARGET prefix=$PFX model=$MODEL_SLUG mode=$MODE gpus=$GPUS ctx=$MAX_MODEL_LEN max_gen=$MAX_GEN"
echo "[bench] model_path=$MODEL_PATH"

# is a method selected? (empty subset = all)
want() { [ ${#SEL_METHODS[@]} -eq 0 ] && return 0; for m in "${SEL_METHODS[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }

# ---- QLUTATTN offline calibration (sign/nf2 sigma^2-mix; sign-frac -> bit) -- #
# Each post-RoPE K channel is fixed OFFLINE to sign (low sigma^2, 1.25b) or nf2
# (high sigma^2, 2.5b) by its wikitext residual variance; sign-frac sets HOW MANY
# go to sign -> the effective K bit. The per-channel mean is still self-calibrated
# at prefill at runtime (free for attention). The mask is model-intrinsic, so it
# is cached next to the model and reused.
MASK=""
if want qlutattn; then
  { [ -n "$CALIB_DATA" ] && [ -f "$CALIB_DATA" ]; } || { echo "[err] qlutattn needs CALIB_DATA=<wikitext parquet>; got '$CALIB_DATA'" >&2; exit 1; }
  fpct=$(awk "BEGIN{printf \"%02d\", $SIGN_FRAC*100}")
  MASK="${MODEL_PATH%/}.lutbench_snf_f${fpct}.pt"
  if [ -f "$MASK" ]; then
    echo "[calib] reuse existing mask: $MASK"
  else
    first_gpu="${GPUS%%,*}"
    echo "[calib] calibrating sign/nf2 (sign-frac=$SIGN_FRAC) on GPU$first_gpu -> $MASK"
    CUDA_VISIBLE_DEVICES="$first_gpu" PYTHONPATH=src python scripts/calibrate_k168v4_pt.py \
      --model "$MODEL_PATH" --calib-data "$CALIB_DATA" \
      --codebooks sign,nf2 --sign-frac "$SIGN_FRAC" --output "$MASK"
  fi
fi

# ---- methods: name | run_exp variant | output method-slug | extra inline env -#
ALL_METHODS=(
  "fp16|fp16|fp16|"
  "shadowkv|shadowkv|shadowkv|"
  "kitty|kitty|kitty-k2b4v2-pr0p125|"
  "kivi_star|kivi_star|kivi-star-k2v2|KBITS=2 VBITS=2"
  "kivi|kivi|kivi-k2v2|KBITS=2 VBITS=2"
  "qlutattn|qlutattn_k188v4_pt|qlutattn-k188v4-pt|QLUT_CB_MASK=$MASK"
  "qlutattn_fast|qlutattn_k125v4_pt|qlutattn-k125v4-pt|"
)

[ "$MODE" = full ] && BASE_DIR="$REPO/longbench_out" || BASE_DIR="$REPO/longbench_out/smoke"
for spec in "${ALL_METHODS[@]}"; do
  IFS='|' read -r name variant slug extra <<<"$spec"
  want "$name" || continue
  pred_dir="$BASE_DIR/${MODEL_SLUG}_${slug}/pred"
  # Per-method completeness SKIP: only (re)run a method that is missing/incomplete.
  # run_exp.sh already resumes at the DATASET level; this method-level check avoids
  # even loading the model + rescanning for a method whose datasets are all done.
  # Smoke is never skipped (run_exp.sh wipes the smoke dir each launch).
  # FORCE_RERUN=1 bypasses the skip and reruns everything.
  if [ "$MODE" = full ] && [ "${FORCE_RERUN:-0}" != 1 ]; then
    echo "[check] $name: $pred_dir"
    if PYTHONPATH=src python "$SCRIPT_DIR/check_method_complete.py" \
         --pred-dir "$pred_dir" ${DATASETS_CSV:+--datasets "$DATASETS_CSV"}; then
      echo "[skip] $name already complete -- skipping (FORCE_RERUN=1 to override)"
      continue
    fi
    echo "[run]  $name incomplete -> running (run_exp.sh fills in only the missing datasets)"
  fi
  echo "=============================================================="
  echo "[bench] >>> $name  (variant=$variant)  $(date '+%F %T')"
  echo "=============================================================="
  # shellcheck disable=SC2086  -- $extra must word-split into KEY=VAL pairs
  env $extra bash scripts/run_exp.sh "$TARGET" --gpus "$GPUS" --variant "$variant" "${SMOKE_ARGS[@]}"
done

echo "[bench] all selected methods done -- aggregating"
python "$SCRIPT_DIR/collect_lutdecoding_results.py" \
  --base "$REPO/longbench_out" --layout "$LAYOUT" \
  --model-slug "$MODEL_SLUG" ${MASK:+--mask "$MASK"} || true
echo "[bench] DONE ($(date '+%F %T'))"
