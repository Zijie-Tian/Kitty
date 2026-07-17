#!/usr/bin/env bash
# autoresearch driver: run one qlutattn K-mask config on a GPU pair.
# usage: run_config.sh <slug> <mask_path> <gpus_csv> [datasets_csv] [research01]
# research01 defaults to 1; pass 0 to run the pure canonical validation path.
set -euo pipefail
SLUG="$1"; MASK="$2"; GPUS="$3"
DS="${4:-qasper,multifieldqa_en}"   # pass "all" for the full 21-dataset list
RESEARCH="${5:-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
env_prefix=(
  QLUT_CB_MASK="${MASK}"
  LLAMA32_MODEL_PATH="${HOME}/models/Llama-3.2-1B-Instruct"
  LLAMA32_MODEL_SLUG="llama32-1b-rm-${SLUG}"
  MAX_MODEL_LEN=32768
  LLAMA32_MAX_GEN=256
)
if [[ "${DS}" != "all" ]]; then
  env_prefix+=(DATASETS_CSV="${DS}")
fi
if [[ "${RESEARCH}" == "1" ]]; then
  env_prefix+=(QLUT_RESEARCH=1)
fi
echo "[run_config] slug=${SLUG} mask=${MASK} gpus=${GPUS} datasets=${DS} research=${RESEARCH}"
exec env "${env_prefix[@]}" bash scripts/run_exp.sh llama32 --gpus "${GPUS}" --variant qlutattn
