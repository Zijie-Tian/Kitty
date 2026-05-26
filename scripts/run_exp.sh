#!/usr/bin/env bash
# Run the current 32k LongBench Kitty experiments with checkpoint-style continuation.
# It only runs subtasks that were partial or not started when this script was created.
#
# Default: run Llama, Qwen3, and GLM loops concurrently on GPU 0/1/2.
# Usage:
#   bash scripts/run_exp.sh [all|llama|qwen|glm]
#   SERIAL=1 bash scripts/run_exp.sh all
#
# Notes:
# - No nohup is used; keep the terminal/session alive.
# - Completed subtasks are skipped.
# - Partial subtasks in the list are deleted and rerun from scratch.
# - When a model finishes its listed subtasks, result.json is regenerated.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/tzj/anaconda3/envs/kitty/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/tzj/data/LongBench}"

KITTY_TAG="kitty_g128_b128_s32_sel1_k2_v2_pb4_pr0p125"

LLAMA_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
LLAMA_MODEL_PATH="/home/tzj/models/Llama-3.1-8B-Instruct"
LLAMA_MODEL_TAG="llama31-8b-instruct-gpu0-full-32k"
LLAMA_OUTPUT_DIR="longbench_out/llama31-8b-instruct/pred"
LLAMA_PRED_DIR="${LLAMA_OUTPUT_DIR}/${LLAMA_MODEL_TAG}-${KITTY_TAG}"
LLAMA_REPORT_PREFIX="longbench_out/llama31-8b-instruct/logs/report_full_32k"
LLAMA_DATASETS=(triviaqa samsum lsht passage_retrieval_en passage_count passage_retrieval_zh lcc repobench-p)

QWEN_MODEL_ID="Qwen/Qwen3-8B"
QWEN_MODEL_PATH="/home/tzj/models/Qwen3-8B"
QWEN_MODEL_TAG="qwen3-8b-gpu1-full-32k-gen2048"
QWEN_OUTPUT_DIR="longbench_out/qwen3-8b/pred"
QWEN_PRED_DIR="${QWEN_OUTPUT_DIR}/${QWEN_MODEL_TAG}-${KITTY_TAG}"
QWEN_REPORT_PREFIX="longbench_out/qwen3-8b/logs/report_full_32k_gen2048"
QWEN_DATASETS=(hotpotqa 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec triviaqa samsum lsht passage_retrieval_en passage_count passage_retrieval_zh lcc repobench-p)

GLM_MODEL_ID="THUDM/GLM-4-9B-Chat-1M"
GLM_MODEL_PATH="/home/tzj/models/GLM-4-9B-Chat-1M"
GLM_MODEL_TAG="glm4-9b-chat-1m-gpu2-full-32k"
GLM_OUTPUT_DIR="longbench_out/glm4-9b-chat-1m/pred"
GLM_PRED_DIR="${GLM_OUTPUT_DIR}/${GLM_MODEL_TAG}-${KITTY_TAG}"
GLM_REPORT_PREFIX="longbench_out/glm4-9b-chat-1m/logs/report_full_32k"
GLM_DATASETS=(vcsum trec triviaqa samsum lsht passage_retrieval_en passage_count passage_retrieval_zh lcc repobench-p)

usage() {
  cat <<USAGE
Usage: bash scripts/run_exp.sh [all|llama|qwen|glm]

Default target is: all
Default all-mode runs three model loops concurrently on GPU 0/1/2.
Set SERIAL=1 to run all targets one by one.

Environment overrides:
  PYTHON_BIN=${PYTHON_BIN}
  DATA_ROOT=${DATA_ROOT}
USAGE
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
  local data_file="${DATA_ROOT}/data/${dataset}.jsonl"
  local out_file="${pred_dir}/${dataset}.jsonl"
  local manifest_file="${pred_dir}/${dataset}.manifest.json"

  if [[ ! -f "${data_file}" ]]; then
    echo "ERROR: Missing LongBench dataset: ${data_file}" >&2
    return 2
  fi

  local expected current
  expected="$(count_rows "${data_file}")"
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
    --variant kitty
    --dataset "${dataset}"
    --data-root "${DATA_ROOT}"
    --output-dir "${output_dir}"
    --max-samples -1
    --max-model-len 32768
    --torch-dtype float16
    --local-files-only
    --report-json "${report_json}"
  )
  if [[ -n "${max_gen}" ]]; then
    cmd+=(--max-gen "${max_gen}")
  fi

  echo "[start] GPU${gpu} ${model_tag} dataset=${dataset}"
  "${env_cmd[@]}" "${cmd[@]}"
  echo "[done]  GPU${gpu} ${model_tag} dataset=${dataset}"
}

score_pred_dir() {
  local pred_dir="$1"
  echo "[score] ${pred_dir} -> result.json"
  env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${pred_dir}" <<'PY'
from pathlib import Path
import sys
from kitty_sim.longbench.scorer import score_directory

pred_dir = Path(sys.argv[1])
scores = score_directory(pred_dir, strict_complete=False, output_name="result.json")
print(scores)
PY
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
    if prepare_dataset "${pred_dir}" "${dataset}"; then
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
    llama 0 \
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

run_qwen() {
  run_model_loop \
    qwen3 1 \
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
    glm4 2 \
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
  local target="${1:-all}"
  case "${target}" in
    -h|--help|help)
      usage
      return 0
      ;;
  esac

  check_prereqs

  case "${target}" in
    llama)
      run_llama
      ;;
    qwen|qwen3)
      run_qwen
      ;;
    glm|glm4)
      run_glm
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
