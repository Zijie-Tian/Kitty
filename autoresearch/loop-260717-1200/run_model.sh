#!/usr/bin/env bash
# Rollout driver: run the FINAL research config on one model, full 21 datasets.
# usage: run_model.sh <target: llama32|qwen> <model_path> <model_slug> <mask_path> <gpus_csv>
set -euo pipefail
TARGET="$1"; MODEL_PATH="$2"; SLUG="$3"; MASK="$4"; GPUS="$5"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
common=(QLUT_RESEARCH=1 QLUT_CB_MASK="${MASK}" MAX_MODEL_LEN=32768)
case "${TARGET}" in
  llama32)
    common+=(LLAMA32_MODEL_PATH="${MODEL_PATH}" LLAMA32_MODEL_SLUG="${SLUG}" LLAMA32_MAX_GEN=256)
    ;;
  qwen)
    common+=(QWEN_MODEL_PATH="${MODEL_PATH}" QWEN_MODEL_SLUG="${SLUG}" QWEN_MAX_GEN=256)
    ;;
  *)
    echo "ERROR: target must be llama32 or qwen" >&2; exit 2
    ;;
esac
echo "[run_model] target=${TARGET} model=${MODEL_PATH} slug=${SLUG} gpus=${GPUS}"
exec env "${common[@]}" bash scripts/run_exp.sh "${TARGET}" --gpus "${GPUS}" --variant qlutattn
