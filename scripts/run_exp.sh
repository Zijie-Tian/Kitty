#!/usr/bin/env bash
# Run the current 32k LongBench Kitty experiments with checkpoint-style continuation.
# It only runs subtasks that were partial or not started when this script was created.
#
# Default: run Llama, Qwen3, and GLM loops concurrently on GPU 0/1/2.
# DeepSeek is an explicit target for completing the existing Distill-Llama run.
# Usage:
#   bash scripts/run_exp.sh [all|llama|qwen|glm|deepseek|llama32] [--gpu GPU] [--max-samples N]
#   SERIAL=1 bash scripts/run_exp.sh all
#   bash scripts/run_exp.sh deepseek --gpu 0
#   bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 1
#
# Notes:
# - No nohup is used; keep the terminal/session alive.
# - Completed subtasks are skipped.
# - Partial subtasks in the list are deleted and rerun from scratch.
# - When a model finishes its listed subtasks, strict score_longbench regenerates result.json.
# - If any generated jsonl is still partial, scoring fails and writes result.partial.json.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${KITTY_PYTHON_BIN:-${HOME}/anaconda3/envs/kitty/bin/python}}"
DATA_ROOT="${DATA_ROOT:-${LONGBENCH_DATA_ROOT:-${HOME}/data/LongBench}}"
GPU_OVERRIDE="${GPU_OVERRIDE:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
RUN_VARIANT="${RUN_VARIANT:-kitty}"

KITTY_TAG_128="kitty_g128_b128_s32_sel1_k2_v2_pb4_pr0p125"
KITTY_TAG_PAGE16="kitty_page16_g16_b16_s32_sel1_k2_v2_pb4_pr0p125"
KITTY_TAG="${KITTY_TAG:-${KITTY_TAG_128}}"

LLAMA_GPU="${LLAMA_GPU:-0}"
LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${KITTY_LLAMA31_8B_PATH:-${HOME}/models/Llama-3.1-8B-Instruct}}"
LLAMA_MODEL_TAG="${LLAMA_MODEL_TAG:-llama31-8b-instruct-gpu0-full-32k}"
LLAMA_OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-longbench_out/llama31-8b-instruct/pred}"
LLAMA_PRED_DIR="${LLAMA_OUTPUT_DIR}/${LLAMA_MODEL_TAG}-${KITTY_TAG}"
LLAMA_REPORT_PREFIX="${LLAMA_REPORT_PREFIX:-longbench_out/llama31-8b-instruct/logs/report_full_32k}"
LLAMA_DATASETS=(
  narrativeqa qasper multifieldqa_en multifieldqa_zh
  hotpotqa 2wikimqa musique dureader
  gov_report qmsum multi_news vcsum
  trec triviaqa samsum lsht
  passage_retrieval_en passage_count passage_retrieval_zh
  lcc repobench-p
)

LLAMA32_GPU="${LLAMA32_GPU:-1}"
LLAMA32_MODEL_ID="${LLAMA32_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
LLAMA32_MODEL_PATH="${LLAMA32_MODEL_PATH:-${KITTY_LLAMA32_1B_PATH:-${HOME}/models/Llama-3.2-1B-Instruct}}"
LLAMA32_MODEL_TAG_PREFIX="${LLAMA32_MODEL_TAG_PREFIX:-llama32-1b-instruct-gpu1-page16}"
LLAMA32_OUTPUT_DIR="${LLAMA32_OUTPUT_DIR:-longbench_out/llama32-1b-instruct-page16-smoke/pred}"
LLAMA32_REPORT_PREFIX="${LLAMA32_REPORT_PREFIX:-longbench_out/llama32-1b-instruct-page16-smoke/logs/report_page16}"
LLAMA32_VARIANT="${LLAMA32_VARIANT:-kitty_page16}"
LLAMA32_DATASETS=(
  narrativeqa qasper multifieldqa_en multifieldqa_zh
  hotpotqa 2wikimqa musique dureader
  gov_report qmsum multi_news vcsum
  trec triviaqa samsum lsht
  passage_retrieval_en passage_count passage_retrieval_zh
  lcc repobench-p
)

QWEN_GPU="${QWEN_GPU:-1}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-8B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${KITTY_QWEN3_8B_PATH:-${HOME}/models/Qwen3-8B}}"
QWEN_MODEL_TAG="${QWEN_MODEL_TAG:-qwen3-8b-gpu1-full-32k-gen2048}"
QWEN_OUTPUT_DIR="${QWEN_OUTPUT_DIR:-longbench_out/qwen3-8b/pred}"
QWEN_PRED_DIR="${QWEN_OUTPUT_DIR}/${QWEN_MODEL_TAG}-${KITTY_TAG}"
QWEN_REPORT_PREFIX="${QWEN_REPORT_PREFIX:-longbench_out/qwen3-8b/logs/report_full_32k_gen2048}"
QWEN_DATASETS=(hotpotqa 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec triviaqa samsum lsht passage_retrieval_en passage_count passage_retrieval_zh lcc repobench-p)

GLM_GPU="${GLM_GPU:-2}"
GLM_MODEL_ID="${GLM_MODEL_ID:-THUDM/GLM-4-9B-Chat-1M}"
GLM_MODEL_PATH="${GLM_MODEL_PATH:-${KITTY_GLM4_9B_1M_PATH:-${HOME}/models/GLM-4-9B-Chat-1M}}"
GLM_MODEL_TAG="${GLM_MODEL_TAG:-glm4-9b-chat-1m-gpu2-full-32k}"
GLM_OUTPUT_DIR="${GLM_OUTPUT_DIR:-longbench_out/glm4-9b-chat-1m/pred}"
GLM_PRED_DIR="${GLM_OUTPUT_DIR}/${GLM_MODEL_TAG}-${KITTY_TAG}"
GLM_REPORT_PREFIX="${GLM_REPORT_PREFIX:-longbench_out/glm4-9b-chat-1m/logs/report_full_32k}"
GLM_DATASETS=(vcsum trec triviaqa samsum lsht passage_retrieval_en passage_count passage_retrieval_zh lcc repobench-p)

DEEPSEEK_GPU="${DEEPSEEK_GPU:-0}"
DEEPSEEK_MODEL_ID="${DEEPSEEK_MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-${HOME}/models/DeepSeek-R1-Distill-Llama-8B}"
DEEPSEEK_MODEL_TAG="${DEEPSEEK_MODEL_TAG:-deepseek-r1-distill-llama-8b}"
DEEPSEEK_OUTPUT_DIR="${DEEPSEEK_OUTPUT_DIR:-longbench_out/pred}"
DEEPSEEK_PRED_DIR="${DEEPSEEK_PRED_DIR:-${DEEPSEEK_OUTPUT_DIR}/${DEEPSEEK_MODEL_TAG}-${KITTY_TAG}}"
DEEPSEEK_REPORT_PREFIX="${DEEPSEEK_REPORT_PREFIX:-logs/longbench/reports/deepseek_r1_distill_llama_8b_kitty}"
DEEPSEEK_MAX_GEN="${DEEPSEEK_MAX_GEN:-1024}"
DEEPSEEK_DATASETS=(lsht passage_retrieval_en passage_count passage_retrieval_zh lcc repobench-p)

usage() {
  cat <<USAGE
Usage: bash scripts/run_exp.sh [all|llama|qwen|glm|deepseek|llama32] [--gpu GPU] [--max-samples N]

Default target is: all
Default all-mode runs three model loops concurrently on GPU 0/1/2.
Set SERIAL=1 to run all targets one by one.
DeepSeek is opt-in and is not included in the default all target.
Llama32 runs Llama-3.2-1B-Instruct on GPU1 by default with the kitty_page16 smoke variant.

Examples:
  bash scripts/run_exp.sh deepseek --gpu 0
  SERIAL=1 bash scripts/run_exp.sh all --gpu 0
  bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 1
  bash scripts/run_exp.sh llama32 2

Environment overrides:
  PYTHON_BIN=${PYTHON_BIN}
  DATA_ROOT=${DATA_ROOT}
  GPU_OVERRIDE=${GPU_OVERRIDE:-<unset>}
  MAX_SAMPLES=${MAX_SAMPLES}
  MAX_MODEL_LEN=${MAX_MODEL_LEN}
  RUN_VARIANT=${RUN_VARIANT}
  LLAMA_GPU=${LLAMA_GPU}
  LLAMA32_GPU=${LLAMA32_GPU}
  LLAMA32_MODEL_PATH=${LLAMA32_MODEL_PATH}
  QWEN_GPU=${QWEN_GPU}
  GLM_GPU=${GLM_GPU}
  DEEPSEEK_GPU=${DEEPSEEK_GPU}
  DEEPSEEK_MODEL_PATH=${DEEPSEEK_MODEL_PATH}
USAGE
}

parse_args() {
  TARGET=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      -h|--help|help)
        TARGET="help"
        shift
        ;;
      -g|--gpu)
        if [[ "$#" -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
          echo "ERROR: --gpu requires a GPU id, for example: --gpu 0" >&2
          return 2
        fi
        GPU_OVERRIDE="$2"
        shift 2
        ;;
      --gpu=*)
        GPU_OVERRIDE="${1#--gpu=}"
        if [[ -z "${GPU_OVERRIDE}" ]]; then
          echo "ERROR: --gpu requires a non-empty GPU id" >&2
          return 2
        fi
        shift
        ;;
      --max-samples)
        if [[ "$#" -lt 2 || -z "${2:-}" ]]; then
          echo "ERROR: --max-samples requires an integer, for example: --max-samples 1" >&2
          return 2
        fi
        MAX_SAMPLES="$2"
        shift 2
        ;;
      --max-samples=*)
        MAX_SAMPLES="${1#--max-samples=}"
        if [[ -z "${MAX_SAMPLES}" ]]; then
          echo "ERROR: --max-samples requires a non-empty integer" >&2
          return 2
        fi
        shift
        ;;
      --max-model-len)
        if [[ "$#" -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
          echo "ERROR: --max-model-len requires an integer, for example: --max-model-len 32768" >&2
          return 2
        fi
        MAX_MODEL_LEN="$2"
        shift 2
        ;;
      --max-model-len=*)
        MAX_MODEL_LEN="${1#--max-model-len=}"
        if [[ -z "${MAX_MODEL_LEN}" ]]; then
          echo "ERROR: --max-model-len requires a non-empty integer" >&2
          return 2
        fi
        shift
        ;;
      --variant)
        if [[ "$#" -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
          echo "ERROR: --variant requires a variant name, for example: --variant kitty_page16" >&2
          return 2
        fi
        RUN_VARIANT="$2"
        shift 2
        ;;
      --variant=*)
        RUN_VARIANT="${1#--variant=}"
        if [[ -z "${RUN_VARIANT}" ]]; then
          echo "ERROR: --variant requires a non-empty variant name" >&2
          return 2
        fi
        shift
        ;;
      -*)
        echo "ERROR: unknown option: $1" >&2
        usage >&2
        return 2
        ;;
      *)
        if [[ -n "${TARGET}" ]]; then
          if [[ "$1" =~ ^-?[0-9]+$ ]]; then
            MAX_SAMPLES="$1"
            shift
            continue
          fi
          echo "ERROR: only one target may be specified (got '${TARGET}' and '$1')" >&2
          return 2
        fi
        TARGET="$1"
        shift
        ;;
    esac
  done
  TARGET="${TARGET:-all}"
}

select_gpu() {
  local default_gpu="$1"
  printf '%s\n' "${GPU_OVERRIDE:-${default_gpu}}"
}

count_rows() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    printf '0\n'
    return 0
  fi
  awk 'NF { c += 1 } END { print c + 0 }' "${file}"
}

check_prereqs() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2
    return 2
  fi
  if [[ ! -d "${DATA_ROOT}/data" ]]; then
    echo "ERROR: LongBench data dir not found: ${DATA_ROOT}/data" >&2
    return 2
  fi
}

ensure_no_running_eval() {
  local model_tag="$1"
  local matches
  matches="$(pgrep -af "kitty_sim\.cli\.eval_longbench.*--model-tag[ =]${model_tag}" || true)"
  if [[ -n "${matches}" ]]; then
    cat >&2 <<MSG
ERROR: Found an existing LongBench eval process for model tag '${model_tag}'.
Stop it before running scripts/run_exp.sh, otherwise the same jsonl files may be written concurrently.

${matches}
MSG
    return 3
  fi
}

prepare_dataset() {
  local pred_dir="$1"
  local dataset="$2"
  local max_samples="${3:--1}"
  local data_file="${DATA_ROOT}/data/${dataset}.jsonl"
  local out_file="${pred_dir}/${dataset}.jsonl"
  local manifest_file="${pred_dir}/${dataset}.manifest.json"

  if [[ ! -f "${data_file}" ]]; then
    echo "ERROR: Missing LongBench dataset: ${data_file}" >&2
    return 2
  fi

  local expected current
  expected="$(count_rows "${data_file}")"
  if [[ "${max_samples}" -gt 0 && "${expected}" -gt "${max_samples}" ]]; then
    expected="${max_samples}"
  fi
  current="$(count_rows "${out_file}")"

  if [[ "${expected}" -le 0 ]]; then
    echo "ERROR: Dataset has no rows: ${data_file}" >&2
    return 2
  fi

  if [[ "${current}" -eq "${expected}" ]]; then
    echo "[skip] ${dataset}: already complete ${current}/${expected}"
    return 10
  fi

  if [[ "${current}" -gt 0 ]]; then
    echo "[rerun] ${dataset}: partial/corrupt ${current}/${expected}; deleting ${out_file}"
    rm -f "${out_file}" "${manifest_file}"
  else
    echo "[run] ${dataset}: missing 0/${expected}"
    rm -f "${manifest_file}"
  fi

  return 0
}

run_eval_dataset() {
  local gpu="$1"
  local model_id="$2"
  local model_path="$3"
  local model_tag="$4"
  local model_family="$5"
  local output_dir="$6"
  local report_json="$7"
  local dataset="$8"
  local max_gen="${9:-}"
  local transformers_verbosity="${10:-}"

  local -a env_cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "GPU_ID=${gpu}"
    "GPU_IDS_CSV=${gpu}"
    "TOKENIZERS_PARALLELISM=false"
    "HF_DATASETS_TRUST_REMOTE_CODE=1"
    "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
  )
  if [[ -n "${transformers_verbosity}" ]]; then
    env_cmd+=("TRANSFORMERS_VERBOSITY=${transformers_verbosity}")
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -m kitty_sim.cli.eval_longbench "${model_id}"
    --model-path "${model_path}"
    --model-tag "${model_tag}"
    --model-family "${model_family}"
    --variant "${RUN_VARIANT}"
    --dataset "${dataset}"
    --data-root "${DATA_ROOT}"
    --output-dir "${output_dir}"
    --max-samples "${MAX_SAMPLES}"
    --max-model-len "${MAX_MODEL_LEN}"
    --torch-dtype float16
    --local-files-only
    --report-json "${report_json}"
  )
  if [[ "${gpu}" == "1" ]]; then
    cmd+=(--require-gpu1)
  fi
  if [[ -n "${max_gen}" ]]; then
    cmd+=(--max-gen "${max_gen}")
  fi

  echo "[start] GPU${gpu} ${model_tag} dataset=${dataset} variant=${RUN_VARIANT} max_samples=${MAX_SAMPLES} max_model_len=${MAX_MODEL_LEN}"
  "${env_cmd[@]}" "${cmd[@]}"
  echo "[done]  GPU${gpu} ${model_tag} dataset=${dataset}"
}

score_pred_dir() {
  local pred_dir="$1"
  echo "[score] strict ${pred_dir} -> result.json"
  env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m kitty_sim.cli.score_longbench --model "${pred_dir}"
}

run_model_loop() {
  local label="$1"
  local gpu="$2"
  local model_id="$3"
  local model_path="$4"
  local model_tag="$5"
  local model_family="$6"
  local output_dir="$7"
  local pred_dir="$8"
  local report_prefix="$9"
  local max_gen="${10}"
  local transformers_verbosity="${11}"
  shift 11
  local -a datasets=("$@")

  echo "========== ${label}: GPU${gpu}, datasets=${datasets[*]} =========="
  ensure_no_running_eval "${model_tag}"

  local dataset rc
  for dataset in "${datasets[@]}"; do
    if prepare_dataset "${pred_dir}" "${dataset}" "${MAX_SAMPLES}"; then
      run_eval_dataset \
        "${gpu}" \
        "${model_id}" \
        "${model_path}" \
        "${model_tag}" \
        "${model_family}" \
        "${output_dir}" \
        "${report_prefix}_${dataset}.json" \
        "${dataset}" \
        "${max_gen}" \
        "${transformers_verbosity}"
    else
      rc=$?
      if [[ "${rc}" -eq 10 ]]; then
        continue
      fi
      return "${rc}"
    fi
  done

  score_pred_dir "${pred_dir}"
  echo "========== ${label}: complete =========="
}

run_llama() {
  run_model_loop \
    llama "$(select_gpu "${LLAMA_GPU}")" \
    "${LLAMA_MODEL_ID}" \
    "${LLAMA_MODEL_PATH}" \
    "${LLAMA_MODEL_TAG}" \
    llama3 \
    "${LLAMA_OUTPUT_DIR}" \
    "${LLAMA_PRED_DIR}" \
    "${LLAMA_REPORT_PREFIX}" \
    "" \
    "" \
    "${LLAMA_DATASETS[@]}"
}

run_llama32() {
  local sample_label="full"
  if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
    sample_label="smoke${MAX_SAMPLES}"
  fi
  local model_tag="${LLAMA32_MODEL_TAG_PREFIX}-${sample_label}"
  local variant_tag="${KITTY_TAG}"
  if [[ "${LLAMA32_VARIANT}" == "kitty_page16" && "${KITTY_TAG}" == "${KITTY_TAG_128}" ]]; then
    variant_tag="${KITTY_TAG_PAGE16}"
  fi
  local pred_dir="${LLAMA32_OUTPUT_DIR}/${model_tag}-${variant_tag}"

  RUN_VARIANT="${LLAMA32_VARIANT}" \
  run_model_loop \
    llama32 "$(select_gpu "${LLAMA32_GPU}")" \
    "${LLAMA32_MODEL_ID}" \
    "${LLAMA32_MODEL_PATH}" \
    "${model_tag}" \
    llama3 \
    "${LLAMA32_OUTPUT_DIR}" \
    "${pred_dir}" \
    "${LLAMA32_REPORT_PREFIX}_${sample_label}" \
    "" \
    "" \
    "${LLAMA32_DATASETS[@]}"
}

run_qwen() {
  run_model_loop \
    qwen3 "$(select_gpu "${QWEN_GPU}")" \
    "${QWEN_MODEL_ID}" \
    "${QWEN_MODEL_PATH}" \
    "${QWEN_MODEL_TAG}" \
    qwen \
    "${QWEN_OUTPUT_DIR}" \
    "${QWEN_PRED_DIR}" \
    "${QWEN_REPORT_PREFIX}" \
    2048 \
    "" \
    "${QWEN_DATASETS[@]}"
}

run_glm() {
  run_model_loop \
    glm4 "$(select_gpu "${GLM_GPU}")" \
    "${GLM_MODEL_ID}" \
    "${GLM_MODEL_PATH}" \
    "${GLM_MODEL_TAG}" \
    glm4 \
    "${GLM_OUTPUT_DIR}" \
    "${GLM_PRED_DIR}" \
    "${GLM_REPORT_PREFIX}" \
    "" \
    error \
    "${GLM_DATASETS[@]}"
}

run_deepseek() {
  run_model_loop \
    deepseek "$(select_gpu "${DEEPSEEK_GPU}")" \
    "${DEEPSEEK_MODEL_ID}" \
    "${DEEPSEEK_MODEL_PATH}" \
    "${DEEPSEEK_MODEL_TAG}" \
    llama3 \
    "${DEEPSEEK_OUTPUT_DIR}" \
    "${DEEPSEEK_PRED_DIR}" \
    "${DEEPSEEK_REPORT_PREFIX}" \
    "${DEEPSEEK_MAX_GEN}" \
    "" \
    "${DEEPSEEK_DATASETS[@]}"
}

run_all_parallel() {
  local -a pids=()
  cleanup() {
    if [[ "${#pids[@]}" -gt 0 ]]; then
      kill "${pids[@]}" 2>/dev/null || true
    fi
  }
  trap cleanup INT TERM

  (run_llama) 2>&1 | sed -u 's/^/[llama] /' & pids+=("$!")
  (run_qwen) 2>&1 | sed -u 's/^/[qwen3] /' & pids+=("$!")
  (run_glm) 2>&1 | sed -u 's/^/[glm4] /' & pids+=("$!")

  local status=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  trap - INT TERM
  return "${status}"
}

main() {
  parse_args "$@"
  local target="${TARGET}"
  if ! [[ "${MAX_SAMPLES}" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: max samples must be an integer, got: ${MAX_SAMPLES}" >&2
    return 2
  fi
  if ! [[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: max model length must be a positive integer, got: ${MAX_MODEL_LEN}" >&2
    return 2
  fi
  case "${target}" in
    help)
      usage
      return 0
      ;;
  esac

  if [[ "${target}" == "all" && -n "${GPU_OVERRIDE}" && "${SERIAL:-0}" != "1" ]]; then
    echo "ERROR: --gpu with target 'all' requires SERIAL=1, otherwise all model loops would share GPU${GPU_OVERRIDE} concurrently." >&2
    return 2
  fi

  check_prereqs

  case "${target}" in
    llama)
      run_llama
      ;;
    qwen|qwen3)
      run_qwen
      ;;
    llama32|llama3.2|llama-3.2)
      run_llama32
      ;;
    glm|glm4)
      run_glm
      ;;
    deepseek|deepseek-distill|deepseek-distill-llama-8b|deepseek-r1-distill-llama-8b)
      run_deepseek
      ;;
    all)
      if [[ "${SERIAL:-0}" == "1" ]]; then
        run_llama
        run_qwen
        run_glm
      else
        run_all_parallel
      fi
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
