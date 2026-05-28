#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=accuracy_simulation/env.sh
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

# Hard task constraint: this scheduler is GPU1-only.
GPU_ID="${GPU_ID:-1}"
if [[ "${GPU_IDS_CSV:-1}" != "1" || "${GPU_ID}" != "1" ]]; then
  echo -e "${RED}[error] GPU1-only LongBench runner: set GPU_IDS_CSV=1 and GPU_ID=1 only.${NC}" >&2
  exit 2
fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_TAG="${MODEL_TAG:-}"
MODEL_FAMILY="${MODEL_FAMILY:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  case "${MODEL,,}" in
    *qwen3*8b*)
      MODEL_PATH="${KITTY_QWEN3_8B_PATH:-}"
      ;;
    *llama-3.1-8b*|*llama3.1-8b*|*llama31*8b*)
      MODEL_PATH="${KITTY_LLAMA31_8B_PATH:-}"
      ;;
  esac
fi
DATA_ROOT="${DATA_ROOT:-${LONGBENCH_DATA_ROOT:-data/LongBench}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ "${OUTPUT_DIR}" == "longbench_out/pred" || "${OUTPUT_DIR}" == "${REPO_ROOT}/longbench_out/pred" ]]; then
  echo -e "${RED}[error] longbench_out/pred is retired; use longbench_out/<model>-<method>/pred or omit OUTPUT_DIR for normalized defaults.${NC}" >&2
  exit 2
fi
LOG_DIR="${LOG_DIR:-logs/longbench_gpu1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_GEN="${MAX_GEN:-}"
PROMPT_TOKEN_RESERVE="${PROMPT_TOKEN_RESERVE:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"
STRICT_COMPLETE="${STRICT_COMPLETE:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="python"
fi

VARIANTS=("kitty")
if [[ -n "${VARIANTS_CSV:-}" ]]; then
  IFS=',' read -r -a VARIANTS <<< "${VARIANTS_CSV}"
fi

slugify() {
  local value="${1,,}"
  value="${value//\//-}"
  value="${value//_/-}"
  value="$(printf '%s' "${value}" | sed -E 's/[^a-z0-9.+-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  printf '%s\n' "${value}"
}

model_layout_slug() {
  local value="${MODEL,,}"
  local source="${MODEL_PATH:-${MODEL}}"
  case "${value}" in
    *llama-3.1-8b*|*llama3.1-8b*|*llama31*8b*)
      printf '%s\n' "llama31-8b-instruct"
      ;;
    *llama-3.2-1b*|*llama3.2-1b*|*llama32*1b*)
      printf '%s\n' "llama32-1b-instruct"
      ;;
    *qwen3*8b*)
      printf '%s\n' "qwen3-8b"
      ;;
    *glm-4-9b*|*glm4*9b*)
      printf '%s\n' "glm4-9b-chat-1m"
      ;;
    *deepseek*r1*distill*llama*8b*)
      printf '%s\n' "deepseek-r1-distill-llama-8b"
      ;;
    *)
      slugify "$(basename "${source}")"
      ;;
  esac
}

method_layout_slug() {
  case "${1,,}" in
    kitty_page16)
      printf '%s\n' "quest-kitty"
      ;;
    kitty)
      printf '%s\n' "kitty"
      ;;
    kitty_pro)
      printf '%s\n' "kitty-pro"
      ;;
    fp16)
      printf '%s\n' "fp16"
      ;;
    kivi_2)
      printf '%s\n' "kivi-2"
      ;;
    kivi_star_2)
      printf '%s\n' "kivi-star-2"
      ;;
    custom)
      printf '%s\n' "custom-kitty"
      ;;
    *)
      slugify "$1"
      ;;
  esac
}

is_smoke_run() {
  [[ "${MAX_SAMPLES}" =~ ^[0-9]+$ && "${MAX_SAMPLES}" -gt 0 ]]
}

default_output_dir_for_variant() {
  local variant="$1"
  local model_slug method_slug
  model_slug="$(model_layout_slug)"
  method_slug="$(method_layout_slug "${variant}")"
  if is_smoke_run; then
    printf 'longbench_out/smoke/%s-%s/pred\n' "${model_slug}" "${method_slug}"
  else
    printf 'longbench_out/%s-%s/pred\n' "${model_slug}" "${method_slug}"
  fi
}

DATASETS=(
  "narrativeqa"
  "qasper"
  "multifieldqa_en"
  "multifieldqa_zh"
  "hotpotqa"
  "2wikimqa"
  "musique"
  "dureader"
  "gov_report"
  "qmsum"
  "multi_news"
  "vcsum"
  "trec"
  "triviaqa"
  "samsum"
  "lsht"
  "passage_retrieval_en"
  "passage_count"
  "passage_retrieval_zh"
  "lcc"
  "repobench-p"
)
if [[ -n "${DATASETS_CSV:-}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
fi

mkdir -p "${LOG_DIR}"
REPORT_DIR="${LOG_DIR}/reports"
mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Kitty LongBench GPU1-only runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "GPU: ${GREEN}physical GPU1 only${NC}"
echo -e "MODEL: ${GREEN}${MODEL}${NC}"
echo -e "MODEL_PATH: ${GREEN}${MODEL_PATH:-<auto>}${NC}"
echo -e "MODEL_TAG: ${GREEN}${MODEL_TAG:-<auto>}${NC}"
echo -e "MODEL_FAMILY: ${GREEN}${MODEL_FAMILY:-<auto>}${NC}"
echo -e "PYTHON_BIN: ${GREEN}${PYTHON_BIN}${NC}"
echo -e "DATA_ROOT: ${GREEN}${DATA_ROOT}${NC}"
echo -e "OUTPUT_DIR: ${GREEN}${OUTPUT_DIR:-<normalized per variant>}${NC}"
echo -e "VARIANTS: ${GREEN}${VARIANTS[*]}${NC}"
echo -e "DATASETS: ${GREEN}${#DATASETS[@]}${NC}"
echo -e "MAX_SAMPLES: ${GREEN}${MAX_SAMPLES}${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ ! -d "${DATA_ROOT}/data" ]]; then
  echo -e "${RED}[error] LongBench data directory not found: ${DATA_ROOT}/data${NC}" >&2
  exit 3
fi

declare -A PRED_DIRS=()
TOTAL=$(( ${#VARIANTS[@]} * ${#DATASETS[@]} ))
COUNT=0

for variant in "${VARIANTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    COUNT=$((COUNT + 1))
    safe_variant="${variant//[^A-Za-z0-9_.-]/_}"
    safe_dataset="${dataset//[^A-Za-z0-9_.-]/_}"
    effective_output_dir="${OUTPUT_DIR:-$(default_output_dir_for_variant "${variant}")}"
    report_json="${REPORT_DIR}/report_${safe_variant}_${safe_dataset}.json"
    log_file="${LOG_DIR}/eval_${safe_variant}_${safe_dataset}.log"

    cmd=("${PYTHON_BIN}" -m kitty_sim.cli.eval_longbench "${MODEL}"
      --variant "${variant}"
      --dataset "${dataset}"
      --data-root "${DATA_ROOT}"
      --output-dir "${effective_output_dir}"
      --flat-output-dir
      --max-samples "${MAX_SAMPLES}"
      --prompt-token-reserve "${PROMPT_TOKEN_RESERVE}"
      --torch-dtype "${TORCH_DTYPE}"
      --require-gpu1
      --report-json "${report_json}"
    )
    [[ -n "${MODEL_PATH}" ]] && cmd+=(--model-path "${MODEL_PATH}")
    [[ -n "${MODEL_TAG}" ]] && cmd+=(--model-tag "${MODEL_TAG}")
    [[ -n "${MODEL_FAMILY}" ]] && cmd+=(--model-family "${MODEL_FAMILY}")
    [[ -n "${MAX_MODEL_LEN}" ]] && cmd+=(--max-model-len "${MAX_MODEL_LEN}")
    [[ -n "${MAX_GEN}" ]] && cmd+=(--max-gen "${MAX_GEN}")
    [[ "${LOCAL_FILES_ONLY}" == "1" ]] && cmd+=(--local-files-only)
    [[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
    [[ "${STRICT_COMPLETE}" != "1" ]] && cmd+=(--no-strict-complete)

    echo -e "${BLUE}[${COUNT}/${TOTAL}]${NC} ${variant} | ${dataset} -> GPU1"
    CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1 PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" \
      "${cmd[@]}" > "${log_file}" 2>&1

    pred_dir=$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
p = Path("${report_json}")
print(json.loads(p.read_text())["prediction_dir"] if p.exists() else "")
PY
)
    if [[ -n "${pred_dir}" ]]; then
      PRED_DIRS["${pred_dir}"]=1
      echo -e "${GREEN}  -> ok; pred_dir=${pred_dir}${NC}"
    else
      echo -e "${RED}  -> missing report; inspect ${log_file}${NC}" >&2
      exit 4
    fi
  done
done

echo -e "${YELLOW}[score] scoring generated LongBench outputs${NC}"
for pred_dir in "${!PRED_DIRS[@]}"; do
  model_dir="${pred_dir//\//_}"
  model_dir="${model_dir//[^A-Za-z0-9_.-]/_}"
  score_log="${LOG_DIR}/score_${model_dir}.log"
  echo -e "${BLUE}[score]${NC} ${model_dir}"
  PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" -m kitty_sim.cli.score_longbench --model "${pred_dir}" > "${score_log}" 2>&1
  test -f "${pred_dir}/result.json"
  echo -e "${GREEN}  -> result: ${pred_dir}/result.json${NC}"
done

echo -e "${GREEN}All GPU1-only LongBench tasks completed.${NC}"
