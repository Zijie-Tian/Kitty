#!/usr/bin/env bash
# Per-GPU collection lane: runs (model,task) pairs sequentially on one GPU.
#
# usage: run_collection.sh <gpu_id> <lane_file>
#
# Env (optional; .env is loaded via accuracy_simulation/env.sh):
#   RUN_TAG                 default: sensitivity_overlap_ext
#   OUT_ROOT                default: <repo>/outputs/sensitivity_overlap/$RUN_TAG
#   KITTY_LLAMA32_1B_PATH / KITTY_LLAMA32_3B_PATH
#   LONGBENCH_DATA_ROOT / DATA_ROOT   LongBench root containing data/*.jsonl
#   KITTY_PYTHON_BIN / PYTHON_BIN
#
# NOTE: launching lanes on GPUs other than 1 is an explicit override of the
# repo GPU1-only evaluation rule. Smoke defaults stay on GPU1.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=accuracy_simulation/env.sh
source "${REPO_ROOT}/accuracy_simulation/env.sh"

GPU=$1
LANE=$2
RUN_TAG="${RUN_TAG:-sensitivity_overlap_ext}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/sensitivity_overlap/${RUN_TAG}}"
DATA_ROOT="${DATA_ROOT:-${LONGBENCH_DATA_ROOT:-${HOME}/data/LongBench}}"
PYTHON_BIN="${PYTHON_BIN:-${KITTY_PYTHON_BIN:-python}}"
LOGD="${OUT_ROOT}/logs"
mkdir -p "$LOGD"

if [[ ! -d "${DATA_ROOT}/data" ]]; then
  echo "ERROR: LongBench data dir not found: ${DATA_ROOT}/data" >&2
  echo "Hint: set LONGBENCH_DATA_ROOT or DATA_ROOT to the LongBench root." >&2
  exit 2
fi

while read -r MTAG TASK; do
  [ -z "${MTAG:-}" ] && continue
  case "$MTAG" in
    llama32-1b)
      MODEL="${KITTY_LLAMA32_1B_PATH:-${LLAMA32_MODEL_PATH:-}}"
      ;;
    llama32-3b)
      MODEL="${KITTY_LLAMA32_3B_PATH:-}"
      ;;
    *)
      echo "[lane$GPU] unknown model tag $MTAG"
      continue
      ;;
  esac
  if [[ -z "${MODEL}" || ! -d "${MODEL}" ]]; then
    echo "[lane$GPU] FAIL $MTAG/$TASK: model path missing for tag $MTAG" >&2
    echo "  set KITTY_LLAMA32_1B_PATH / KITTY_LLAMA32_3B_PATH in .env" >&2
    continue
  fi
  OUT="${OUT_ROOT}/${MTAG}/${TASK}"
  echo "[lane$GPU] start $MTAG/$TASK $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="${REPO_ROOT}/src" PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/collect_task_stats.py" \
    --model "$MODEL" --model-tag "$MTAG" --task "$TASK" \
    --data-root "${DATA_ROOT}" \
    --num-prompts 128 --row-skip 32 --max-len 2048 --skip-first 32 --group-size 128 \
    --out "$OUT" > "$LOGD/${MTAG}_${TASK}.log" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    (cd "$OUT" && sha256sum stats.pt > SHA256SUMS.txt)
    echo "[lane$GPU] done  $MTAG/$TASK rc=0 $(date +%T)"
  else
    echo "[lane$GPU] FAIL  $MTAG/$TASK rc=$RC $(date +%T)"
  fi
done < "$LANE"
echo "[lane$GPU] ALL FINISHED $(date +%T)"
