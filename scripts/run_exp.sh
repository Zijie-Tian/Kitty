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
#   bash scripts/run_exp.sh [all|llama|qwen|glm|deepseek|llama32] [--gpu GPU | --gpus G0,G1,...] [--max-samples N] [--variant NAME] [--v-tile-channels C]
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
# Empty string is an explicit unset sentinel (blocks .env re-injection of stale C).
V_TILE_CHANNELS="${V_TILE_CHANNELS-}"
V_TILE_CHANNELS_CLI=""
PRINT_METHOD_SLUG=0
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
LLAMA32_DEFAULT_VARIANT="${LLAMA32_DEFAULT_VARIANT:-fp16}"

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
Usage: bash scripts/run_exp.sh [all|llama|qwen|glm|deepseek|llama32] [--gpu GPU | --gpus G0,G1,...] [--max-samples N] [--variant NAME] [--v-tile-channels C]

run_exp.sh is the ONLY supported entry point for LongBench in this repo.

Output layout (deterministic, smoke/full separated):
  full  -> longbench_out/<model>_<method>/{pred,logs}
  smoke -> longbench_out/smoke/<model>_<method>/{pred,logs}

Default target is: all (llama+qwen+glm concurrently on GPU 0/1/2; SERIAL=1 for serial).
DeepSeek is opt-in and not part of the default all target.
Llama32 runs Llama-3.2-1B-Instruct on GPU1 by default with the fp16 variant.

Examples:
  bash scripts/run_exp.sh llama32 --gpu 0 --max-samples 2     # smoke (2 samples)
  bash scripts/run_exp.sh llama32 --gpus 0,1,2 --max-samples 2  # fan datasets across GPUs 0,1,2
  bash scripts/run_exp.sh llama --gpu 1                       # full
  bash scripts/run_exp.sh qwen --variant qlutattn_k1v4         # full, sigma^2-binned K quant
  RUN_MODE=full bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2   # full layout, few samples
  DATASETS_CSV=trec,samsum bash scripts/run_exp.sh llama32 --gpu 1        # scope datasets

Environment overrides:
  PYTHON_BIN=${PYTHON_BIN}
  DATA_ROOT=${DATA_ROOT}
  GPU_OVERRIDE=${GPU_OVERRIDE:-<unset>}    GPUS_OVERRIDE=${GPUS_OVERRIDE:-<unset>}    GPU_IDS_CSV=${GPU_IDS_CSV:-<unset>}
  MAX_SAMPLES=${MAX_SAMPLES}    MAX_MODEL_LEN=${MAX_MODEL_LEN}    RUN_MODE=${RUN_MODE}
  RUN_VARIANT=${RUN_VARIANT:-<per-target default>}
  DATASETS_CSV=${DATASETS_CSV:-<full 21>}    FORCE=${FORCE:-0}
  V_TILE_CHANNELS=${V_TILE_CHANNELS:-<unset>}    PERTOKEN_BLOCK=${PERTOKEN_BLOCK:-1}

V-cache 2-bit variants:
  qlutattn_k125v2_pt / qlutattn_k188v2_pt
      whole-head per-token asymmetric V2 (sign-K / SNF-K respectively)
  qlutattn_k125v2_pt_vtile16 / qlutattn_k188v2_pt_vtile16
      rescued tile16cC V2; requires --v-tile-channels C or V_TILE_CHANNELS=C
      and C must divide the model head_dim. Named V2 variants require VBITS=2.
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
          echo "ERROR: --variant requires a variant name, for example: --variant kitty" >&2
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
      --v-tile-channels)
        if [[ "$#" -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
          echo "ERROR: --v-tile-channels requires a positive integer" >&2
          return 2
        fi
        V_TILE_CHANNELS_CLI="$2"
        shift 2
        ;;
      --v-tile-channels=*)
        V_TILE_CHANNELS_CLI="${1#--v-tile-channels=}"
        if [[ -z "${V_TILE_CHANNELS_CLI}" ]]; then
          echo "ERROR: --v-tile-channels requires a non-empty integer" >&2
          return 2
        fi
        shift
        ;;
      --print-method-slug)
        PRINT_METHOD_SLUG=1
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

# Resolve the output method slug through the Python canonical resolver.  The
# shell deliberately owns no variant table: aliases, stale env validation,
# PERTOKEN_BLOCK support, V-tile C, rv version, and QUEST suffixes all come from
# the exact same build_variant()/method_layout_slug() path used by workers.
method_slug() {
  local variant="${1}"
  local resolved_c="${V_TILE_CHANNELS_CLI:-${V_TILE_CHANNELS:-}}"
  local -a pf_cmd=(
    env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
    "PERTOKEN_BLOCK=${PERTOKEN_BLOCK:-1}"
    "QLUT_CB_MASK=${QLUT_CB_MASK:-}"
    "QLUT_BIN_CODEBOOKS=${QLUT_BIN_CODEBOOKS:-}"
    "VBITS=${VBITS:-}"
    "V_TILE_CHANNELS=${resolved_c}"
    "${PYTHON_BIN}" -m kitty_sim.cli.preflight_longbench
    --variant "${variant}" --resolve-config-only --json
  )
  [[ -n "${KBITS:-}" ]] && pf_cmd+=(--kbits "${KBITS}")
  [[ -n "${VBITS:-}" ]] && pf_cmd+=(--vbits "${VBITS}")
  [[ -n "${PROMOTE_BIT:-}" ]] && pf_cmd+=(--promote_bit "${PROMOTE_BIT}")
  [[ -n "${PROMOTE_RATIO:-}" ]] && pf_cmd+=(--promote_ratio "${PROMOTE_RATIO}")
  [[ -n "${PROMOTE_RATIO_CONFIG:-}" ]] && pf_cmd+=(--promote-ratio-config "${PROMOTE_RATIO_CONFIG}")
  [[ -n "${resolved_c}" ]] && pf_cmd+=(--v-tile-channels "${resolved_c}")
  [[ -n "${SHADOWKV_BUDGET:-}" ]] && pf_cmd+=(--shadowkv-budget "${SHADOWKV_BUDGET}")
  [[ -n "${SHADOWKV_RANK:-}" ]] && pf_cmd+=(--shadowkv-rank "${SHADOWKV_RANK}")
  [[ -n "${SHADOWKV_CHUNK:-}" ]] && pf_cmd+=(--shadowkv-chunk-size "${SHADOWKV_CHUNK}")
  if [[ "${QUEST_KERNEL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${QUEST_TRITON:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${SIM_QUEST:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${QUEST_SIM:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    pf_cmd+=(--quest-kernel)
    pf_cmd+=(--quest-token-budget "${QUEST_TOKEN_BUDGET:-${QUEST_BUDGET:-2048}}")
    pf_cmd+=(--quest-skip-layers "${QUEST_SKIP_LAYERS:-0}")
  fi
  local pf_json
  pf_json="$("${pf_cmd[@]}")" || {
    echo "ERROR: Python preflight failed for variant=${variant}" >&2
    return 2
  }
  printf '%s\n' "$(printf '%s' "${pf_json}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["method_slug"])')"
}

# Resolve the exact per-dataset fingerprint with the same arguments the worker
# receives. This is CPU-only and never loads model weights.
longbench_preflight_json() {
  local model_id="$1" model_path="$2" model_family="$3" variant="$4" datasets_csv="$5" max_gen="$6"
  local -a cmd=(
    env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
    "PERTOKEN_BLOCK=${PERTOKEN_BLOCK:-1}"
    "QLUT_CB_MASK=${QLUT_CB_MASK:-}"
    "QLUT_BIN_CODEBOOKS=${QLUT_BIN_CODEBOOKS:-}"
    "VBITS=${VBITS:-}"
    "${PYTHON_BIN}" -m kitty_sim.cli.preflight_longbench "${model_id}"
    --model-path "${model_path}"
    --model-family "${model_family}"
    --variant "${variant}"
    --datasets-csv "${datasets_csv}"
    --data-root "${DATA_ROOT}"
    --max-samples "${MAX_SAMPLES}"
    --max-model-len "${MAX_MODEL_LEN}"
    --prompt-token-reserve "${PROMPT_TOKEN_RESERVE:-0}"
    --torch-dtype float16
    --local-files-only
    --json
  )
  [[ -n "${max_gen}" ]] && cmd+=(--max-gen "${max_gen}")
  [[ -n "${KBITS:-}" ]] && cmd+=(--kbits "${KBITS}")
  [[ -n "${VBITS:-}" ]] && cmd+=(--vbits "${VBITS}")
  [[ -n "${PROMOTE_BIT:-}" ]] && cmd+=(--promote_bit "${PROMOTE_BIT}")
  [[ -n "${PROMOTE_RATIO:-}" ]] && cmd+=(--promote_ratio "${PROMOTE_RATIO}")
  [[ -n "${PROMOTE_RATIO_CONFIG:-}" ]] && cmd+=(--promote-ratio-config "${PROMOTE_RATIO_CONFIG}")
  local resolved_c="${V_TILE_CHANNELS_CLI:-${V_TILE_CHANNELS:-}}"
  [[ -n "${resolved_c}" ]] && cmd+=(--v-tile-channels "${resolved_c}")
  [[ -n "${SHADOWKV_BUDGET:-}" ]] && cmd+=(--shadowkv-budget "${SHADOWKV_BUDGET}")
  [[ -n "${SHADOWKV_RANK:-}" ]] && cmd+=(--shadowkv-rank "${SHADOWKV_RANK}")
  [[ -n "${SHADOWKV_CHUNK:-}" ]] && cmd+=(--shadowkv-chunk-size "${SHADOWKV_CHUNK}")
  if [[ "${QUEST_KERNEL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${QUEST_TRITON:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${SIM_QUEST:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${QUEST_SIM:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    cmd+=(--quest-kernel)
    cmd+=(--quest-token-budget "${QUEST_TOKEN_BUDGET:-${QUEST_BUDGET:-2048}}")
    cmd+=(--quest-skip-layers "${QUEST_SKIP_LAYERS:-0}")
  fi
  "${cmd[@]}"
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
  local expected_hash="${4:-}"
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
    local manifest_summary="" actual_hash="" actual_status="" actual_written="" actual_expected=""
    if [[ -f "${manifest_file}" ]]; then
      manifest_summary="$("${PYTHON_BIN}" -c \
        'import json,sys; m=json.load(open(sys.argv[1], encoding="utf-8")); print("\t".join(str(m.get(k, "")) for k in ("run_config_hash", "status", "written_samples", "expected_samples")))' \
        "${manifest_file}" 2>/dev/null || true)"
      IFS=$'\t' read -r actual_hash actual_status actual_written actual_expected <<< "${manifest_summary}"
    fi
    if [[ -n "${expected_hash}" \
      && "${actual_hash}" == "${expected_hash}" \
      && "${actual_status}" == "ok" \
      && "${actual_written}" == "${expected}" \
      && "${actual_expected}" == "${expected}" ]]; then
      echo "[skip] ${dataset}: already complete ${current}/${expected}, run_config_hash matched"
      return 10
    fi
    if [[ "${FORCE:-0}" == "1" ]]; then
      echo "[force] ${dataset}: complete rows but stale/missing run_config_hash; rerunning"
      rm -f "${out_file}" "${manifest_file}"
      return 0
    fi
    echo "ERROR: ${dataset}: completed output has stale/missing run_config_hash." >&2
    echo "expected_hash=${expected_hash:-<missing>} manifest_hash=${actual_hash:-<missing>} status=${actual_status:-<missing>} written=${actual_written:-<missing>} manifest_expected=${actual_expected:-<missing>}" >&2
    echo "Set FORCE=1 to rerun, or migrate the manifest explicitly." >&2
    return 2
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
  local expected_run_hash="${10}"
  local max_gen="${11:-}"
  local transformers_verbosity="${12:-}"

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
    --expected-run-config-hash "${expected_run_hash}"
    --prompt-token-reserve "${PROMPT_TOKEN_RESERVE:-0}"
  )
  if [[ "${gpu}" == "1" ]]; then
    cmd+=(--require-gpu1)
  fi
  if [[ -n "${max_gen}" ]]; then
    cmd+=(--max-gen "${max_gen}")
  fi
  # K/V bit-width for the kivi / kivi_star variants (--kbits/--vbits, default 2/2).
  if [[ -n "${KBITS:-}" ]]; then
    cmd+=(--kbits "${KBITS}")
  fi
  if [[ -n "${VBITS:-}" ]]; then
    cmd+=(--vbits "${VBITS}")
  fi
  # kitty boost bit + boost fraction (affect the kitty variant; the slug encodes
  # them). PROMOTE_BIT = boost bit (default 4), PROMOTE_RATIO = scalar boost
  # fraction (default 0.125). PROMOTE_RATIO_CONFIG is the per-layer schedule
  # (kitty only); for a per-layer schedule also set <T>_MODEL_SLUG so different
  # schedules don't share an output dir.
  if [[ -n "${PROMOTE_BIT:-}" ]]; then
    cmd+=(--promote_bit "${PROMOTE_BIT}")
  fi
  if [[ -n "${PROMOTE_RATIO:-}" ]]; then
    cmd+=(--promote_ratio "${PROMOTE_RATIO}")
  fi
  if [[ -n "${PROMOTE_RATIO_CONFIG:-}" ]]; then
    cmd+=(--promote-ratio-config "${PROMOTE_RATIO_CONFIG}")
  fi
  # Rescued V tile channel block C (CLI > env). Empty string = explicit unset.
  local resolved_v_tile="${V_TILE_CHANNELS_CLI:-${V_TILE_CHANNELS:-}}"
  if [[ -n "${V_TILE_CHANNELS_CLI}" ]]; then
    cmd+=(--v-tile-channels "${V_TILE_CHANNELS_CLI}")
  elif [[ -n "${resolved_v_tile}" ]]; then
    cmd+=(--v-tile-channels "${resolved_v_tile}")
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
  # Triton QUEST overlay for Kitty/QLUTATTN fake-quant variants.
  if [[ "${QUEST_KERNEL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${QUEST_TRITON:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${SIM_QUEST:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ \
    || "${QUEST_SIM:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    cmd+=(--quest-kernel)
    cmd+=(--quest-token-budget "${QUEST_TOKEN_BUDGET:-${QUEST_BUDGET:-2048}}")
    cmd+=(--quest-skip-layers "${QUEST_SKIP_LAYERS:-0}")
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
  local -a jobs=("$@")

  local -a gpus
  IFS=',' read -r -a gpus <<< "${gpus_csv}"
  local i
  for i in "${!gpus[@]}"; do gpus[$i]="${gpus[$i]//[[:space:]]/}"; done

  # One scheduling slot per entry in the GPU list, keyed by slot INDEX (not GPU
  # id) so a GPU may appear multiple times to get multiple concurrent workers,
  # e.g. "0,0,0,1,1,1" runs 3 workers each on GPU0 and GPU1. slot_gpu[s] is the
  # physical GPU bound to slot s.
  local n_slots=${#gpus[@]}
  local -a slot_pid=() slot_gpu=()
  local -A pid_ds=()
  local s
  for ((s = 0; s < n_slots; s++)); do slot_pid[$s]=""; slot_gpu[$s]="${gpus[$s]}"; done

  local overall_rc=0
  local -a failures=()
  local job ds expected_hash assigned_slot assigned p rc

  for job in "${jobs[@]}"; do
    ds="${job%%|*}"
    expected_hash="${job#*|}"
    assigned_slot=-1
    while [[ "${assigned_slot}" -lt 0 ]]; do
      for ((s = 0; s < n_slots; s++)); do
        p="${slot_pid[$s]}"
        if [[ -z "${p}" ]]; then
          assigned_slot=$s; break
        elif ! kill -0 "${p}" 2>/dev/null; then
          rc=0; wait "${p}" || rc=$?
          if [[ "${rc}" -ne 0 ]]; then
            overall_rc=1; failures+=("${pid_ds[${p}]}(gpu${slot_gpu[$s]},rc=${rc})")
            echo "[parallel] FAILED dataset=${pid_ds[${p}]} on GPU${slot_gpu[$s]} (rc=${rc})" >&2
          fi
          slot_pid[$s]=""; unset "pid_ds[${p}]"
          assigned_slot=$s; break
        fi
      done
      [[ "${assigned_slot}" -lt 0 ]] && sleep 0.5
    done
    assigned="${slot_gpu[$assigned_slot]}"
    run_dataset_worker "${assigned}" \
      "${model_id}" "${model_path}" "${model_tag}" "${model_family}" \
      "${variant}" "${pred_dir}" "${report_prefix}_${ds}.json" \
      "${ds}" "${expected_hash}" "${max_gen}" "${verbosity}" &
    p=$!
    slot_pid[$assigned_slot]="${p}"; pid_ds["${p}"]="${ds}"
    echo "[parallel] dispatch dataset=${ds} -> GPU${assigned} (slot ${assigned_slot}, pid ${p})"
  done

  for ((s = 0; s < n_slots; s++)); do
    p="${slot_pid[$s]}"
    [[ -z "${p}" ]] && continue
    rc=0; wait "${p}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      overall_rc=1; failures+=("${pid_ds[${p}]}(gpu${slot_gpu[$s]},rc=${rc})")
      echo "[parallel] FAILED dataset=${pid_ds[${p}]} on GPU${slot_gpu[$s]} (rc=${rc})" >&2
    fi
    slot_pid[$s]=""
  done

  if [[ "${overall_rc}" -ne 0 ]]; then
    echo "[parallel] ${#failures[@]}/${#jobs[@]} dataset(s) failed: ${failures[*]}" >&2
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
  local -a datasets
  mapfile -t datasets < <(resolve_datasets)
  local datasets_csv
  datasets_csv="$(IFS=,; printf '%s' "${datasets[*]}")"

  # One config-only Python pass is the canonical source for the method slug and
  # every per-dataset run hash. It runs before any output is wiped or GPU worker
  # is launched.
  local preflight_json method
  preflight_json="$(longbench_preflight_json \
    "${model_id}" "${model_path}" "${model_family}" "${variant}" "${datasets_csv}" "${max_gen}")" || {
    echo "ERROR: LongBench preflight failed for ${label}" >&2
    return 2
  }
  method="$(printf '%s' "${preflight_json}" | "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(sys.stdin)["method_slug"])')"
  if [[ -z "${method}" ]]; then
    echo "ERROR: preflight returned no method_slug for ${label}" >&2
    return 2
  fi

  local base pred_dir report_prefix mode model_tag
  if is_smoke; then
    base="longbench_out/smoke/${model_slug}_${method}"
  else
    base="longbench_out/${model_slug}_${method}"
  fi
  pred_dir="${base}/pred"
  report_prefix="${base}/logs/report"
  if is_smoke; then mode="smoke${MAX_SAMPLES}"; else mode="full"; fi
  model_tag="${model_slug}_${method}_${mode}"
  # smoke: start each launch from a clean slate; full: keep/resume existing output.
  if is_smoke; then
    wipe_smoke_base "${base}"
  fi
  mkdir -p "${pred_dir}" "${base}/logs"

  local -a gpu_arr
  IFS=',' read -r -a gpu_arr <<< "${gpus_csv}"
  echo "========== ${label}: GPU(s)=${gpus_csv} (${#gpu_arr[@]} parallel) variant=${variant} mode=${mode} datasets=${#datasets[@]} out=${pred_dir} =========="
  ensure_no_running_eval "${model_tag}"

  # Prepare serially in the main process (skip complete, rm stale/partial) so
  # workers never race on the same files; collect the datasets that must run.
  local dataset rc expected_hash
  local -a to_run=()
  for dataset in "${datasets[@]}"; do
    expected_hash="$(printf '%s' "${preflight_json}" | "${PYTHON_BIN}" -c \
      'import json,sys; print(json.load(sys.stdin).get("datasets", {}).get(sys.argv[1], {}).get("run_config_hash") or "")' \
      "${dataset}")"
    if [[ -z "${expected_hash}" ]]; then
      echo "ERROR: preflight returned no run_config_hash for dataset=${dataset}" >&2
      return 2
    fi
    if prepare_dataset "${pred_dir}" "${dataset}" "${MAX_SAMPLES}" "${expected_hash}"; then
      to_run+=("${dataset}|${expected_hash}")
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

  if [[ "${PRINT_METHOD_SLUG}" == "1" ]]; then
    if [[ -z "${RUN_VARIANT}" ]]; then
      echo "ERROR: --print-method-slug requires --variant NAME" >&2
      return 2
    fi
    method_slug "${RUN_VARIANT}"
    return
  fi

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
