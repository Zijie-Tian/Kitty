#!/usr/bin/env bash
# Sole entry point for LongBench experiments in this repo.
#
# Do NOT launch LongBench through any other script. run_exp.sh owns the
# deterministic output layout and the smoke/full separation:
#
#   full  (MAX_SAMPLES <= 0)  -> longbench_out/<model>_<method>/{pred,logs}
#   smoke (MAX_SAMPLES  > 0)  -> longbench_out/smoke/<model>_<method>/{pred,logs}
#
# Predictions (<dataset>.jsonl, <dataset>.manifest.json, result.json) live under
# pred/; per-dataset report json lives under logs/. <model> and <method> match
# the slugs used by src/kitty_sim/longbench/runner.py (model_layout_slug /
# method_layout_slug), separated by an underscore.
#
# Usage:
#   bash scripts/run_exp.sh [all|llama|qwen|glm|deepseek|llama32] [--gpu GPU | --gpus G0,G1,...] [--max-samples N]
#   SERIAL=1 bash scripts/run_exp.sh all
#   bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2
#   bash scripts/run_exp.sh llama32 --gpus 0,1,2 --max-samples 2   # fan datasets across GPUs (shell-level parallelism)
#
# Notes:
# - .env is sourced automatically (KITTY_PYTHON_BIN, KITTY_*_PATH, LONGBENCH_DATA_ROOT).
# - smoke: the target's previous smoke output dir is wiped on every launch
#          (clean slate -- smoke results are never resumed).
# - full:  completed datasets are kept; partial/missing datasets are (deleted and)
#          rerun, so an interrupted full run resumes and fills in the rest.
# - A more-complete existing output is never silently shrunk (set FORCE=1 to override).
# - RUN_MODE=auto|smoke|full forces the layout independently of MAX_SAMPLES.
# - DATASETS_CSV overrides the default full 21-dataset list.

set -Eeuo pipefail
trap 'echo "[run_exp] failed at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Load repo-local .env (KITTY_PYTHON_BIN, KITTY_*_PATH, LONGBENCH_DATA_ROOT, ...).
# env.sh preserves explicit environment/CLI values over .env values.
# shellcheck source=accuracy_simulation/env.sh
source "${REPO_ROOT}/accuracy_simulation/env.sh"

PYTHON_BIN="${PYTHON_BIN:-${KITTY_PYTHON_BIN:-${HOME}/anaconda3/envs/kitty/bin/python}}"
DATA_ROOT="${DATA_ROOT:-${LONGBENCH_DATA_ROOT:-${HOME}/data/LongBench}}"
GPU_OVERRIDE="${GPU_OVERRIDE:-}"
# Multi-GPU dataset fan-out (CSV), e.g. "0,1,2".
# Precedence: --gpus > --gpu > GPU_IDS_CSV env > per-target default.
GPUS_OVERRIDE="${GPUS_OVERRIDE:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
RUN_VARIANT="${RUN_VARIANT:-}"        # empty => per-target default; --variant/RUN_VARIANT overrides
RUN_MODE="${RUN_MODE:-auto}"          # auto|smoke|full -- forces layout independently of MAX_SAMPLES

# Canonical full LongBench dataset list (21). Override with DATASETS_CSV.
LONGBENCH_DATASETS=(
  narrativeqa qasper multifieldqa_en multifieldqa_zh
  hotpotqa 2wikimqa musique dureader
  gov_report qmsum multi_news vcsum
  trec triviaqa samsum lsht
  passage_retrieval_en passage_count passage_retrieval_zh
  lcc repobench-p
)

# Per-target config. model_slug values must match runner.py model_layout_slug.
LLAMA_GPU="${LLAMA_GPU:-0}"
LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${KITTY_LLAMA31_8B_PATH:-${HOME}/models/Llama-3.1-8B-Instruct}}"
LLAMA_MODEL_SLUG="${LLAMA_MODEL_SLUG:-llama31-8b-instruct}"
LLAMA_MAX_GEN="${LLAMA_MAX_GEN:-}"
LLAMA_DEFAULT_VARIANT="${LLAMA_DEFAULT_VARIANT:-kitty}"

LLAMA32_GPU="${LLAMA32_GPU:-1}"
LLAMA32_MODEL_ID="${LLAMA32_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
LLAMA32_MODEL_PATH="${LLAMA32_MODEL_PATH:-${KITTY_LLAMA32_1B_PATH:-${HOME}/models/Llama-3.2-1B-Instruct}}"
LLAMA32_MODEL_SLUG="${LLAMA32_MODEL_SLUG:-llama32-1b-instruct}"
LLAMA32_MAX_GEN="${LLAMA32_MAX_GEN:-256}"
LLAMA32_DEFAULT_VARIANT="${LLAMA32_DEFAULT_VARIANT:-quest_kitty_page16_sim}"

QWEN_GPU="${QWEN_GPU:-1}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-8B}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${KITTY_QWEN3_8B_PATH:-${HOME}/models/Qwen3-8B}}"
QWEN_MODEL_SLUG="${QWEN_MODEL_SLUG:-qwen3-8b}"
QWEN_MAX_GEN="${QWEN_MAX_GEN:-2048}"
QWEN_DEFAULT_VARIANT="${QWEN_DEFAULT_VARIANT:-kitty}"

GLM_GPU="${GLM_GPU:-2}"
GLM_MODEL_ID="${GLM_MODEL_ID:-THUDM/GLM-4-9B-Chat-1M}"
GLM_MODEL_PATH="${GLM_MODEL_PATH:-${KITTY_GLM4_9B_1M_PATH:-${HOME}/models/GLM-4-9B-Chat-1M}}"
GLM_MODEL_SLUG="${GLM_MODEL_SLUG:-glm4-9b-chat-1m}"
GLM_MAX_GEN="${GLM_MAX_GEN:-}"
GLM_DEFAULT_VARIANT="${GLM_DEFAULT_VARIANT:-kitty}"

DEEPSEEK_GPU="${DEEPSEEK_GPU:-0}"
DEEPSEEK_MODEL_ID="${DEEPSEEK_MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-${KITTY_DEEPSEEK_R1_DISTILL_LLAMA8B_PATH:-${HOME}/models/DeepSeek-R1-Distill-Llama-8B}}"
DEEPSEEK_MODEL_SLUG="${DEEPSEEK_MODEL_SLUG:-deepseek-r1-distill-llama-8b}"
DEEPSEEK_MAX_GEN="${DEEPSEEK_MAX_GEN:-1024}"
DEEPSEEK_DEFAULT_VARIANT="${DEEPSEEK_DEFAULT_VARIANT:-kitty}"

usage() {
  cat <<USAGE
Usage: bash scripts/run_exp.sh [all|llama|qwen|glm|deepseek|llama32] [--gpu GPU | --gpus G0,G1,...] [--max-samples N]

run_exp.sh is the ONLY supported entry point for LongBench in this repo.

Output layout (deterministic, smoke/full separated):
  full  -> longbench_out/<model>_<method>/{pred,logs}
  smoke -> longbench_out/smoke/<model>_<method>/{pred,logs}

Default target is: all (llama+qwen+glm concurrently on GPU 0/1/2; SERIAL=1 for serial).
DeepSeek is opt-in and not part of the default all target.
Llama32 runs Llama-3.2-1B-Instruct on GPU1 by default with the quest_kitty_page16_sim variant.

QUEST variants: quest_kitty_page16_sim (pure-torch QUEST+Kitty accuracy proxy, any arch)
and quest_kitty_page16_kernel (real Triton QUEST kernel, Llama/Qwen only).

Examples:
  bash scripts/run_exp.sh llama32 --gpu 0 --max-samples 2     # smoke (2 samples, sim QUEST)
  bash scripts/run_exp.sh llama32 --gpus 0,1,2 --max-samples 2  # fan datasets across GPUs 0,1,2
  bash scripts/run_exp.sh llama --gpu 1                       # full
  bash scripts/run_exp.sh qwen --variant quest_kitty_page16_sim   # full, pure-torch QUEST
  RUN_MODE=full bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2   # full layout, few samples
  DATASETS_CSV=trec,samsum bash scripts/run_exp.sh llama32 --gpu 1        # scope datasets

Environment overrides:
  PYTHON_BIN=${PYTHON_BIN}
  DATA_ROOT=${DATA_ROOT}
  GPU_OVERRIDE=${GPU_OVERRIDE:-<unset>}    GPUS_OVERRIDE=${GPUS_OVERRIDE:-<unset>}    GPU_IDS_CSV=${GPU_IDS_CSV:-<unset>}
  MAX_SAMPLES=${MAX_SAMPLES}    MAX_MODEL_LEN=${MAX_MODEL_LEN}    RUN_MODE=${RUN_MODE}
  RUN_VARIANT=${RUN_VARIANT:-<per-target default>}
  DATASETS_CSV=${DATASETS_CSV:-<full 21>}    FORCE=${FORCE:-0}
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
      --gpus)
        if [[ "$#" -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
          echo "ERROR: --gpus requires a comma-separated GPU list, for example: --gpus 0,1,2" >&2
          return 2
        fi
        GPUS_OVERRIDE="$2"
        shift 2
        ;;
      --gpus=*)
        GPUS_OVERRIDE="${1#--gpus=}"
        if [[ -z "${GPUS_OVERRIDE}" ]]; then
          echo "ERROR: --gpus requires a non-empty GPU list" >&2
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
          echo "ERROR: --variant requires a variant name, for example: --variant quest_kitty_page16_sim" >&2
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

# Resolve the GPU list (CSV) for a target. A single id ("1") keeps the original
# single-GPU behavior; a list ("0,1,2") fans datasets across GPUs.
# Precedence: --gpus > --gpu > GPU_IDS_CSV env > per-target default.
select_gpus() {
  local default_gpu="$1"
  if [[ -n "${GPUS_OVERRIDE:-}" ]]; then
    printf '%s\n' "${GPUS_OVERRIDE}"
  elif [[ -n "${GPU_OVERRIDE:-}" ]]; then
    printf '%s\n' "${GPU_OVERRIDE}"
  elif [[ -n "${GPU_IDS_CSV:-}" ]]; then
    printf '%s\n' "${GPU_IDS_CSV}"
  else
    printf '%s\n' "${default_gpu}"
  fi
}

# Map a variant name to its output method slug (mirrors runner.py method_layout_slug).
method_slug() {
  case "${1,,}" in
    quest_kitty_page16_kernel|quest_kitty_kernel) printf 'quest-kitty-kernel\n' ;;
    quest_kitty_page16_sim|quest_kitty_sim) printf 'quest-kitty-sim\n' ;;
    kitty) printf 'kitty\n' ;;
    kitty_pro) printf 'kitty-pro\n' ;;
    kitty_k1v2) printf 'kitty-k1v2\n' ;;
    kitty_k1v2_pr50) printf 'kitty-k1v2-pr50\n' ;;
    kitty_k1v2_pr75) printf 'kitty-k1v2-pr75\n' ;;
    kitty_k1v4) printf 'kitty-k1v4\n' ;;
    kitty_k1v4_pr50) printf 'kitty-k1v4-pr50\n' ;;
    kitty_k1v4_pr75) printf 'kitty-k1v4-pr75\n' ;;
    fp16) printf 'fp16\n' ;;
    kivi_2) printf 'kivi-2\n' ;;
    kivi_star_2) printf 'kivi-star-2\n' ;;
    custom) printf 'custom-kitty\n' ;;
    shadowkv) printf 'shadowkv\n' ;;
    *) printf '%s\n' "${1//_/-}" ;;
  esac
}

is_smoke() {
  case "${RUN_MODE:-auto}" in
    full) return 1 ;;
    smoke) return 0 ;;
    *) [[ "${MAX_SAMPLES}" =~ ^[0-9]+$ && "${MAX_SAMPLES}" -gt 0 ]] ;;
  esac
}

# Deterministic base dir for a (model_slug, variant): underscore-separated,
# smoke runs nested under longbench_out/smoke/.
resolve_base_dir() {
  local model_slug="$1" variant="$2" method
  method="$(method_slug "${variant}")"
  if is_smoke; then
    printf 'longbench_out/smoke/%s_%s\n' "${model_slug}" "${method}"
  else
    printf 'longbench_out/%s_%s\n' "${model_slug}" "${method}"
  fi
}

# Emit the datasets to run (DATASETS_CSV override, else the canonical full 21).
resolve_datasets() {
  if [[ -n "${DATASETS_CSV:-}" ]]; then
    local IFS=','
    local -a parsed
    read -r -a parsed <<< "${DATASETS_CSV}"
    printf '%s\n' "${parsed[@]}"
  else
    printf '%s\n' "${LONGBENCH_DATASETS[@]}"
  fi
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
    echo "Hint: set KITTY_PYTHON_BIN in .env (sourced automatically) or pass PYTHON_BIN=/path/to/python." >&2
    return 2
  fi
  if [[ ! -d "${DATA_ROOT}/data" ]]; then
    echo "ERROR: LongBench data dir not found: ${DATA_ROOT}/data" >&2
    echo "Hint: set LONGBENCH_DATA_ROOT in .env or pass DATA_ROOT=/path/to/LongBench." >&2
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

# smoke clean slate: remove this target's previous smoke output dir so every
# smoke launch starts fresh. Guarded to only ever touch longbench_out/smoke/*.
wipe_smoke_base() {
  local base="$1"
  case "${base}" in
    longbench_out/smoke/?*) ;;
    *)
      echo "ERROR: refusing to wipe non-smoke base dir: ${base}" >&2
      return 2
      ;;
  esac
  if [[ -d "${base}" ]]; then
    echo "[smoke] clean slate: removing previous smoke output ${base}"
    rm -rf "${base}"
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

  # Never silently shrink a more-complete existing output (e.g. full results when
  # asked for a smaller sample count). Require FORCE=1 to override.
  if [[ "${current}" -gt "${expected}" ]]; then
    if [[ "${FORCE:-0}" == "1" ]]; then
      echo "[force] ${dataset}: existing ${current} > target ${expected}; FORCE=1 deleting ${out_file}"
      rm -f "${out_file}" "${manifest_file}"
    else
      echo "ERROR: ${dataset}: existing output has ${current} rows (> target ${expected}) at ${out_file}." >&2
      echo "Refusing to shrink/delete it. Set FORCE=1 to override, or choose a different output mode." >&2
      return 2
    fi
  elif [[ "${current}" -gt 0 ]]; then
    echo "[rerun] ${dataset}: partial ${current}/${expected}; deleting ${out_file}"
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
  local variant="$6"
  local pred_dir="$7"
  local report_json="$8"
  local dataset="$9"
  local max_gen="${10:-}"
  local transformers_verbosity="${11:-}"

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
    --variant "${variant}"
    --dataset "${dataset}"
    --data-root "${DATA_ROOT}"
    --output-dir "${pred_dir}"
    --flat-output-dir
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
  # Real QUEST+Kitty kernel controls (only meaningful for the
  # quest_kitty_page16_kernel variant; ignored by other variants).
  if [[ -n "${QUEST_BUDGET:-}" ]]; then
    cmd+=(--quest-token-budget "${QUEST_BUDGET}")
  fi
  if [[ -n "${QUEST_SKIP_LAYERS:-}" ]]; then
    cmd+=(--quest-skip-layers "${QUEST_SKIP_LAYERS}")
  fi
  # ShadowKV sim controls (only meaningful for the shadowkv variant; ignored otherwise).
  if [[ -n "${SHADOWKV_BUDGET:-}" ]]; then
    cmd+=(--shadowkv-budget "${SHADOWKV_BUDGET}")
  fi
  if [[ -n "${SHADOWKV_RANK:-}" ]]; then
    cmd+=(--shadowkv-rank "${SHADOWKV_RANK}")
  fi
  if [[ -n "${SHADOWKV_CHUNK:-}" ]]; then
    cmd+=(--shadowkv-chunk-size "${SHADOWKV_CHUNK}")
  fi

  echo "[start] GPU${gpu} ${model_tag} dataset=${dataset} variant=${variant} max_samples=${MAX_SAMPLES} max_model_len=${MAX_MODEL_LEN}"
  "${env_cmd[@]}" "${cmd[@]}"
  echo "[done]  GPU${gpu} ${model_tag} dataset=${dataset}"
}

# Backgroundable worker: run one dataset on one GPU, prefix its output with the
# GPU id, and propagate the eval exit code past the sed pipe (PIPESTATUS[0]) so
# the dispatcher can detect per-dataset failures.
run_dataset_worker() {
  local gpu="$1"; shift
  local rc=0
  { run_eval_dataset "${gpu}" "$@" 2>&1 | sed -u "s/^/[gpu${gpu}] /"; } || rc="${PIPESTATUS[0]}"
  return "${rc}"
}

# Fan a list of datasets across a list of GPUs (shell-level task parallelism;
# the Python eval code is unchanged). One dataset per GPU at a time; a GPU that
# finishes immediately steals the next pending dataset (dynamic load balancing).
# Args: gpus_csv model_id model_path model_tag model_family variant pred_dir \
#       report_prefix max_gen verbosity -- dataset...
run_datasets_parallel() {
  local gpus_csv="$1"; shift
  local model_id="$1" model_path="$2" model_tag="$3" model_family="$4" variant="$5"
  local pred_dir="$6" report_prefix="$7" max_gen="$8" verbosity="$9"; shift 9
  local -a datasets=("$@")

  local -a gpus
  IFS=',' read -r -a gpus <<< "${gpus_csv}"
  local i
  for i in "${!gpus[@]}"; do gpus[$i]="${gpus[$i]//[[:space:]]/}"; done

  local -A slot_pid=() pid_gpu=() pid_ds=()
  local g
  for g in "${gpus[@]}"; do slot_pid["${g}"]=""; done

  local overall_rc=0
  local -a failures=()
  local ds assigned p rc

  for ds in "${datasets[@]}"; do
    assigned=""
    while [[ -z "${assigned}" ]]; do
      for g in "${gpus[@]}"; do
        p="${slot_pid[${g}]}"
        if [[ -z "${p}" ]]; then
          assigned="${g}"; break
        elif ! kill -0 "${p}" 2>/dev/null; then
          rc=0; wait "${p}" || rc=$?
          if [[ "${rc}" -ne 0 ]]; then
            overall_rc=1; failures+=("${pid_ds[${p}]}(gpu${g},rc=${rc})")
            echo "[parallel] FAILED dataset=${pid_ds[${p}]} on GPU${g} (rc=${rc})" >&2
          fi
          slot_pid["${g}"]=""; unset "pid_gpu[${p}]" "pid_ds[${p}]"
          assigned="${g}"; break
        fi
      done
      [[ -z "${assigned}" ]] && sleep 0.5
    done
    run_dataset_worker "${assigned}" \
      "${model_id}" "${model_path}" "${model_tag}" "${model_family}" \
      "${variant}" "${pred_dir}" "${report_prefix}_${ds}.json" \
      "${ds}" "${max_gen}" "${verbosity}" &
    p=$!
    slot_pid["${assigned}"]="${p}"; pid_gpu["${p}"]="${assigned}"; pid_ds["${p}"]="${ds}"
    echo "[parallel] dispatch dataset=${ds} -> GPU${assigned} (pid ${p})"
  done

  for g in "${gpus[@]}"; do
    p="${slot_pid[${g}]}"
    [[ -z "${p}" ]] && continue
    rc=0; wait "${p}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      overall_rc=1; failures+=("${pid_ds[${p}]}(gpu${g},rc=${rc})")
      echo "[parallel] FAILED dataset=${pid_ds[${p}]} on GPU${g} (rc=${rc})" >&2
    fi
    slot_pid["${g}"]=""
  done

  if [[ "${overall_rc}" -ne 0 ]]; then
    echo "[parallel] ${#failures[@]}/${#datasets[@]} dataset(s) failed: ${failures[*]}" >&2
  fi
  return "${overall_rc}"
}

score_pred_dir() {
  local pred_dir="$1"
  echo "[score] strict ${pred_dir} -> result.json"
  env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m kitty_sim.cli.score_longbench --model "${pred_dir}"
}

# run_model_loop label gpus_csv model_id model_path model_family model_slug default_variant max_gen verbosity
# gpus_csv may be a single id ("1") or a list ("0,1,2"); datasets fan out across the list.
run_model_loop() {
  local label="$1"
  local gpus_csv="$2"
  local model_id="$3"
  local model_path="$4"
  local model_family="$5"
  local model_slug="$6"
  local default_variant="$7"
  local max_gen="$8"
  local transformers_verbosity="$9"

  # --variant / RUN_VARIANT wins; otherwise this target's default.
  local variant="${RUN_VARIANT:-${default_variant}}"
  local base pred_dir report_prefix mode model_tag
  base="$(resolve_base_dir "${model_slug}" "${variant}")"
  pred_dir="${base}/pred"
  report_prefix="${base}/logs/report"
  if is_smoke; then mode="smoke${MAX_SAMPLES}"; else mode="full"; fi
  model_tag="${model_slug}_$(method_slug "${variant}")_${mode}"
  # smoke: start each launch from a clean slate; full: keep/resume existing output.
  if is_smoke; then
    wipe_smoke_base "${base}"
  fi
  mkdir -p "${pred_dir}" "${base}/logs"

  local -a datasets
  mapfile -t datasets < <(resolve_datasets)

  local -a gpu_arr
  IFS=',' read -r -a gpu_arr <<< "${gpus_csv}"
  echo "========== ${label}: GPU(s)=${gpus_csv} (${#gpu_arr[@]} parallel) variant=${variant} mode=${mode} datasets=${#datasets[@]} out=${pred_dir} =========="
  ensure_no_running_eval "${model_tag}"

  # Prepare serially in the main process (skip complete, rm stale/partial) so
  # workers never race on the same files; collect the datasets that must run.
  local dataset rc
  local -a to_run=()
  for dataset in "${datasets[@]}"; do
    if prepare_dataset "${pred_dir}" "${dataset}" "${MAX_SAMPLES}"; then
      to_run+=("${dataset}")
    else
      rc=$?
      if [[ "${rc}" -eq 10 ]]; then
        continue
      fi
      return "${rc}"
    fi
  done

  # Fan the pending datasets across the GPU list (one dataset per GPU at a time).
  if [[ "${#to_run[@]}" -gt 0 ]]; then
    run_datasets_parallel "${gpus_csv}" \
      "${model_id}" "${model_path}" "${model_tag}" "${model_family}" \
      "${variant}" "${pred_dir}" "${report_prefix}" "${max_gen}" "${transformers_verbosity}" \
      "${to_run[@]}"
  else
    echo "[parallel] nothing to run for ${label} (all datasets already complete)"
  fi

  score_pred_dir "${pred_dir}"
  echo "========== ${label}: complete -> ${pred_dir} =========="
}

run_llama() {
  run_model_loop \
    llama "$(select_gpus "${LLAMA_GPU}")" \
    "${LLAMA_MODEL_ID}" "${LLAMA_MODEL_PATH}" llama3 \
    "${LLAMA_MODEL_SLUG}" "${LLAMA_DEFAULT_VARIANT}" "${LLAMA_MAX_GEN}" ""
}

run_llama32() {
  run_model_loop \
    llama32 "$(select_gpus "${LLAMA32_GPU}")" \
    "${LLAMA32_MODEL_ID}" "${LLAMA32_MODEL_PATH}" llama3 \
    "${LLAMA32_MODEL_SLUG}" "${LLAMA32_DEFAULT_VARIANT}" "${LLAMA32_MAX_GEN}" ""
}

run_qwen() {
  run_model_loop \
    qwen3 "$(select_gpus "${QWEN_GPU}")" \
    "${QWEN_MODEL_ID}" "${QWEN_MODEL_PATH}" qwen \
    "${QWEN_MODEL_SLUG}" "${QWEN_DEFAULT_VARIANT}" "${QWEN_MAX_GEN}" ""
}

run_glm() {
  run_model_loop \
    glm4 "$(select_gpus "${GLM_GPU}")" \
    "${GLM_MODEL_ID}" "${GLM_MODEL_PATH}" glm4 \
    "${GLM_MODEL_SLUG}" "${GLM_DEFAULT_VARIANT}" "${GLM_MAX_GEN}" error
}

run_deepseek() {
  run_model_loop \
    deepseek "$(select_gpus "${DEEPSEEK_GPU}")" \
    "${DEEPSEEK_MODEL_ID}" "${DEEPSEEK_MODEL_PATH}" llama3 \
    "${DEEPSEEK_MODEL_SLUG}" "${DEEPSEEK_DEFAULT_VARIANT}" "${DEEPSEEK_MAX_GEN}" ""
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
  case "${RUN_MODE}" in
    auto|smoke|full) ;;
    *)
      echo "ERROR: RUN_MODE must be auto|smoke|full, got: ${RUN_MODE}" >&2
      return 2
      ;;
  esac
  case "${target}" in
    help)
      usage
      return 0
      ;;
  esac

  if [[ "${target}" == "all" && ( -n "${GPU_OVERRIDE}" || -n "${GPUS_OVERRIDE}" ) && "${SERIAL:-0}" != "1" ]]; then
    echo "ERROR: --gpu/--gpus with target 'all' requires SERIAL=1; otherwise the model loops (llama/qwen/glm) already run concurrently on their own GPUs." >&2
    echo "Hint: use a single target (e.g. 'llama32 --gpus 0,1,2') to fan one model's datasets across GPUs." >&2
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
