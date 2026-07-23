#!/usr/bin/env bash
# Sole cache-aware perplexity evaluation entry point for Kitty.
# Methods are CPU-preflighted first, then scheduled one model replica per GPU.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

capture_process_value() {
  local output_name="$1" input_name
  shift
  printf -v "${output_name}" '%s' ""
  for input_name in "$@"; do
    if [[ -v ${input_name} && -n "${!input_name}" ]]; then
      printf -v "${output_name}" '%s' "${!input_name}"
      return 0
    fi
  done
}

capture_process_value _PPL_PROCESS_PYTHON PYTHON_BIN KITTY_PYTHON_BIN
capture_process_value _PPL_PROCESS_MODEL MODEL LLAMA32_MODEL_ID
capture_process_value _PPL_PROCESS_MODEL_PATH MODEL_PATH LLAMA32_MODEL_PATH KITTY_LLAMA32_1B_PATH
capture_process_value _PPL_PROCESS_MODEL_TAG MODEL_TAG LLAMA32_MODEL_SLUG
capture_process_value _PPL_PROCESS_DATA PPL_DATA_PATH KITTY_WIKITEXT2_TEST_PATH
capture_process_value _PPL_PROCESS_DTYPE DTYPE TORCH_DTYPE
capture_process_value _PPL_PROCESS_GPU GPUS_OVERRIDE GPU_OVERRIDE GPU_IDS_CSV GPU_ID

# shellcheck source=accuracy_simulation/env.sh
source "${REPO_ROOT}/accuracy_simulation/env.sh"
cd "${REPO_ROOT}"

PYTHON_BIN="${_PPL_PROCESS_PYTHON:-${PYTHON_BIN:-${KITTY_PYTHON_BIN:-python}}}"
MODEL="${_PPL_PROCESS_MODEL:-${MODEL:-${LLAMA32_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}}}"
MODEL_PATH="${_PPL_PROCESS_MODEL_PATH:-${MODEL_PATH:-${LLAMA32_MODEL_PATH:-${KITTY_LLAMA32_1B_PATH:-${HOME}/models/Llama-3.2-1B-Instruct}}}}"
MODEL_TAG="${_PPL_PROCESS_MODEL_TAG:-${MODEL_TAG:-${LLAMA32_MODEL_SLUG:-}}}"
MODEL_FAMILY="${MODEL_FAMILY:-}"
DATA_PATH="${_PPL_PROCESS_DATA:-${PPL_DATA_PATH:-${KITTY_WIKITEXT2_TEST_PATH:-}}}"
VARIANTS="${VARIANTS_CSV:-fp16,kitty,shadowkv,qlutattn,kivi,kivi_star,llamacpp_q40,llamacpp_q40_star,custom}"
OUT_ROOT="${OUT_ROOT:-ppl_out}"
PREFILL_TOKENS="${PREFILL_TOKENS:-4096}"
SCORE_TOKENS="${SCORE_TOKENS:-256}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${_PPL_PROCESS_DTYPE:-${DTYPE:-${TORCH_DTYPE:-float16}}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FORCE="${FORCE:-${OVERWRITE:-0}}"

if [[ -n "${_PPL_PROCESS_GPU}" ]]; then
  GPU_CSV="${_PPL_PROCESS_GPU}"
elif [[ -n "${GPUS_OVERRIDE:-}" ]]; then
  GPU_CSV="${GPUS_OVERRIDE}"
elif [[ -n "${GPU_OVERRIDE:-}" ]]; then
  GPU_CSV="${GPU_OVERRIDE}"
elif [[ -n "${GPU_IDS_CSV:-}" ]]; then
  GPU_CSV="${GPU_IDS_CSV}"
else
  GPU_CSV="${GPU_ID:-1}"
fi

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_ppl.sh [OPTIONS]

The only supported Kitty perplexity entry point. It evaluates document-level
WikiText-2 with a 4096-token dense prefill followed by 256 one-token cache
handoffs. Only suffix targets are scored. Every method uses identical windows.

Model/data:
  --model ID                 Model id/alias (default Llama-3.2-1B-Instruct)
  --model-path PATH          Local checkpoint directory
  --model-tag TAG            Optional output model slug override
  --model-family FAMILY      Optional cache/model family override
  --data-path PATH           Local document-level WikiText-2 test parquet

Evaluation:
  --variants CSV             Default: fp16,kitty,shadowkv,qlutattn,kivi,
                             kivi_star,llamacpp_q40,llamacpp_q40_star,custom
                             Requires fp16 plus at least one comparison method.
  --prefill-tokens N         Dense prefill length (default 4096)
  --score-tokens N           One-token decode targets/window (default 256)
  --max-samples N            Window limit; N>0 uses smoke layout, <=0 full
  --max-model-len N          Hard context cap (default 32768)
  --dtype NAME               float16/fp16/bfloat16/bf16/float32/fp32
  --out-root PATH            Output root (default ppl_out)
  --local-files-only         Disable model/tokenizer network access (default)
  --no-local-files-only      Allow normal Hugging Face resolution
  --force                    Overwrite existing arm output

GPU scheduling:
  --gpu N                    Run methods serially on physical GPU N (default 1)
  --gpus CSV                 Run distinct methods concurrently, one per GPU;
                             later methods take the next free GPU

Output:
  full:  <out-root>/<model>_<method>/{pred,logs}
  smoke: <out-root>/smoke/<model>_<method>/{pred,logs}
  comparison: <layout-root>/<model>_ppl_comparison.{json,csv}

Environment (explicit CLI > process environment > repo .env):
  PYTHON_BIN, MODEL, MODEL_PATH, MODEL_TAG, MODEL_FAMILY
  PPL_DATA_PATH or KITTY_WIKITEXT2_TEST_PATH, VARIANTS_CSV, OUT_ROOT
  PREFILL_TOKENS, SCORE_TOKENS, MAX_SAMPLES, MAX_MODEL_LEN
  DTYPE/TORCH_DTYPE, LOCAL_FILES_ONLY, FORCE/OVERWRITE
  GPU_OVERRIDE, GPUS_OVERRIDE, GPU_ID, GPU_IDS_CSV
  QLUT_CB_MASK (required by qlutattn), plus canonical variant knobs:
  KBITS, VBITS, PROMOTE_BIT, PROMOTE_RATIO, PROMOTE_RATIO_CONFIG,
  SHADOWKV_BUDGET, SHADOWKV_RANK, SHADOWKV_CHUNK

Examples:
  bash scripts/run_ppl.sh --gpu 1 --variants fp16,qlutattn --max-samples 2

  bash scripts/run_ppl.sh --gpus 0,1,2,3,4,5 --max-samples 2
USAGE
}

die() {
  echo "[run-ppl] ERROR: $*" >&2
  exit 2
}

GPU_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) [[ $# -ge 2 ]] || die "--model needs a value"; MODEL="$2"; shift 2 ;;
    --model-path) [[ $# -ge 2 ]] || die "--model-path needs a value"; MODEL_PATH="$2"; shift 2 ;;
    --model-tag) [[ $# -ge 2 ]] || die "--model-tag needs a value"; MODEL_TAG="$2"; shift 2 ;;
    --model-family) [[ $# -ge 2 ]] || die "--model-family needs a value"; MODEL_FAMILY="$2"; shift 2 ;;
    --data-path) [[ $# -ge 2 ]] || die "--data-path needs a value"; DATA_PATH="$2"; shift 2 ;;
    --variants) [[ $# -ge 2 ]] || die "--variants needs a value"; VARIANTS="$2"; shift 2 ;;
    --prefill-tokens) [[ $# -ge 2 ]] || die "--prefill-tokens needs a value"; PREFILL_TOKENS="$2"; shift 2 ;;
    --score-tokens) [[ $# -ge 2 ]] || die "--score-tokens needs a value"; SCORE_TOKENS="$2"; shift 2 ;;
    --max-samples) [[ $# -ge 2 ]] || die "--max-samples needs a value"; MAX_SAMPLES="$2"; shift 2 ;;
    --max-model-len) [[ $# -ge 2 ]] || die "--max-model-len needs a value"; MAX_MODEL_LEN="$2"; shift 2 ;;
    --dtype) [[ $# -ge 2 ]] || die "--dtype needs a value"; DTYPE="$2"; shift 2 ;;
    --out-root) [[ $# -ge 2 ]] || die "--out-root needs a value"; OUT_ROOT="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --no-local-files-only) LOCAL_FILES_ONLY=0; shift ;;
    --force) FORCE=1; shift ;;
    --gpu)
      [[ $# -ge 2 ]] || die "--gpu needs a value"
      [[ -z "${GPU_FLAG}" ]] || die "--gpu and --gpus are mutually exclusive"
      GPU_FLAG="gpu"; GPU_CSV="$2"; shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus needs a value"
      [[ -z "${GPU_FLAG}" ]] || die "--gpu and --gpus are mutually exclusive"
      GPU_FLAG="gpus"; GPU_CSV="$2"; shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -z "${QLUT_CB_MASK:-}" ]]; then
  if [[ -n "${KITTY_LLAMA32_1B_QLUTATTN_MASK:-}" \
        && "${MODEL_PATH}" == "${KITTY_LLAMA32_1B_PATH:-}" ]]; then
    export QLUT_CB_MASK="${KITTY_LLAMA32_1B_QLUTATTN_MASK}"
  elif [[ -n "${KITTY_LLAMA32_3B_QLUTATTN_MASK:-}" \
          && "${MODEL_PATH}" == "${KITTY_LLAMA32_3B_PATH:-}" ]]; then
    export QLUT_CB_MASK="${KITTY_LLAMA32_3B_QLUTATTN_MASK}"
  elif [[ -n "${KITTY_MINICPM5_1B_QLUTATTN_MASK:-}" \
          && "${MODEL_PATH}" == "${KITTY_MINICPM5_1B_PATH:-}" ]]; then
    export QLUT_CB_MASK="${KITTY_MINICPM5_1B_QLUTATTN_MASK}"
  fi
fi

[[ -n "${MODEL}" ]] || die "model is empty"
command -v setsid >/dev/null 2>&1 || die "setsid is required for process-group-safe workers"
[[ -d "${MODEL_PATH}" ]] || die "local model directory does not exist: ${MODEL_PATH}"
[[ -n "${DATA_PATH}" && -f "${DATA_PATH}" ]] || die "local WikiText-2 parquet does not exist: ${DATA_PATH:-<unset>}"
[[ "${PREFILL_TOKENS}" =~ ^[0-9]+$ && "${PREFILL_TOKENS}" -gt 1 ]] || die "prefill tokens must be >1"
[[ "${SCORE_TOKENS}" =~ ^[0-9]+$ && "${SCORE_TOKENS}" -gt 0 ]] || die "score tokens must be >0"
[[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ && "${MAX_MODEL_LEN}" -gt 0 ]] || die "max model length must be positive"
[[ "${MAX_SAMPLES}" =~ ^-?[0-9]+$ ]] || die "max samples must be an integer"
case "${DTYPE}" in
  float16|fp16|bfloat16|bf16|float32|fp32) ;;
  *) die "unsupported dtype: ${DTYPE}" ;;
esac
[[ "${LOCAL_FILES_ONLY}" == "0" || "${LOCAL_FILES_ONLY}" == "1" ]] || die "LOCAL_FILES_ONLY must be 0 or 1"
[[ "${FORCE}" == "0" || "${FORCE}" == "1" ]] || die "FORCE/OVERWRITE must be 0 or 1"

IFS=',' read -r -a RAW_VARIANTS <<< "${VARIANTS}"
declare -a VARIANT_ARR=()
for variant in "${RAW_VARIANTS[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -n "${variant}" ]] || continue
  VARIANT_ARR+=("${variant}")
done
[[ "${#VARIANT_ARR[@]}" -gt 0 ]] || die "--variants resolved to an empty list"
[[ "${#VARIANT_ARR[@]}" -ge 2 ]] \
  || die "--variants must contain fp16 and at least one comparison method"

IFS=',' read -r -a RAW_GPUS <<< "${GPU_CSV}"
declare -a GPU_ARR=()
declare -A SEEN_GPUS=()
for gpu in "${RAW_GPUS[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id '${gpu}'"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate GPU id '${gpu}' is not allowed"
  SEEN_GPUS["${gpu}"]=1
  GPU_ARR+=("${gpu}")
done
[[ "${#GPU_ARR[@]}" -gt 0 ]] || die "GPU list is empty"
if [[ "${GPU_FLAG}" == "gpu" && "${#GPU_ARR[@]}" -ne 1 ]]; then
  die "--gpu accepts exactly one physical GPU"
fi

if (( MAX_SAMPLES > 0 )); then
  LAYOUT_ROOT="${OUT_ROOT%/}/smoke"
  SMOKE=1
else
  LAYOUT_ROOT="${OUT_ROOT%/}"
  SMOKE=0
fi
mkdir -p "${LAYOUT_ROOT}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/kitty-ppl.XXXXXX")"
declare -A ACTIVE_PIDS=()
cleanup() {
  local rc=$? pid
  trap - EXIT INT TERM
  for pid in "${!ACTIVE_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${!ACTIVE_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  rm -rf -- "${TMP_ROOT}"
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

append_variant_args() {
  local output_name="$1"
  local -n output_ref="${output_name}"
  [[ -n "${KBITS:-}" ]] && output_ref+=(--kbits "${KBITS}")
  [[ -n "${VBITS:-}" ]] && output_ref+=(--vbits "${VBITS}")
  [[ -n "${PROMOTE_BIT:-}" ]] && output_ref+=(--promote_bit "${PROMOTE_BIT}")
  [[ -n "${PROMOTE_RATIO:-}" ]] && output_ref+=(--promote_ratio "${PROMOTE_RATIO}")
  [[ -n "${PROMOTE_RATIO_CONFIG:-}" ]] && output_ref+=(--promote-ratio-config "${PROMOTE_RATIO_CONFIG}")
  [[ -n "${SHADOWKV_BUDGET:-}" ]] && output_ref+=(--shadowkv-budget "${SHADOWKV_BUDGET}")
  [[ -n "${SHADOWKV_RANK:-}" ]] && output_ref+=(--shadowkv-rank "${SHADOWKV_RANK}")
  [[ -n "${SHADOWKV_CHUNK:-}" ]] && output_ref+=(--shadowkv-chunk-size "${SHADOWKV_CHUNK}")
  if [[ "${QUEST_KERNEL:-0}" == "1" || "${QUEST_TRITON:-0}" == "1" || "${SIM_QUEST:-0}" == "1" || "${QUEST_SIM:-0}" == "1" ]]; then
    output_ref+=(--quest-kernel)
  fi
  [[ -n "${QUEST_TOKEN_BUDGET:-${QUEST_BUDGET:-}}" ]] && output_ref+=(--quest-token-budget "${QUEST_TOKEN_BUDGET:-${QUEST_BUDGET}}")
  [[ -n "${QUEST_SKIP_LAYERS:-}" ]] && output_ref+=(--quest-skip-layers "${QUEST_SKIP_LAYERS}")
}

build_common_args() {
  local variant="$1" output_name="$2"
  local -n output_ref="${output_name}"
  output_ref=(
    "${MODEL}"
    --model-path "${MODEL_PATH}"
    --variant "${variant}"
    --data-path "${DATA_PATH}"
    --prefill-tokens "${PREFILL_TOKENS}"
    --score-tokens "${SCORE_TOKENS}"
    --max-samples "${MAX_SAMPLES}"
    --max-model-len "${MAX_MODEL_LEN}"
    --torch-dtype "${DTYPE}"
  )
  [[ -n "${MODEL_TAG}" ]] && output_ref+=(--model-tag "${MODEL_TAG}")
  [[ -n "${MODEL_FAMILY}" ]] && output_ref+=(--model-family "${MODEL_FAMILY}")
  [[ "${LOCAL_FILES_ONLY}" == "1" ]] && output_ref+=(--local-files-only)
  append_variant_args "${output_name}"
}

declare -a PREFLIGHT_FILES=()
declare -a PREFLIGHT_LOGS=()
declare -a METHOD_SLUGS=()
declare -a MODEL_SLUGS=()
declare -a PREFLIGHT_HASHES=()
declare -a COMPARISON_HASHES=()
declare -a VALID_INDICES=()
declare -A SEEN_ARMS=()

preflight_variant() {
  local index="$1" variant="$2"
  local json_file="${TMP_ROOT}/preflight-${index}.json"
  local log_file="${TMP_ROOT}/preflight-${index}.log"
  local -a args=()
  build_common_args "${variant}" args
  local -a cmd=(
    env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -m kitty_sim.cli.preflight_ppl
    "${args[@]}" --json
  )
  echo "[run-ppl] CPU preflight: variant=${variant}"
  if ! "${cmd[@]}" >"${json_file}" 2>"${log_file}"; then
    echo "[run-ppl] preflight FAILED: variant=${variant}" >&2
    while IFS= read -r line; do
      printf '[preflight %s] %s\n' "${variant}" "${line}" >&2
    done < "${log_file}"
    return 1
  fi

  local metadata
  if ! metadata="$("${PYTHON_BIN}" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
values = [
    payload["method_slug"], payload["model_slug"], payload["preflight_hash"],
    payload["comparison_config_hash"], payload["canonical_variant"],
    payload["run_config"]["expected_samples"],
]
print("\t".join(str(value) for value in values))
' "${json_file}" 2>>"${log_file}")"; then
    echo "[run-ppl] invalid preflight JSON: variant=${variant}" >&2
    return 1
  fi

  local method_slug model_slug preflight_hash comparison_hash canonical_variant sample_count
  IFS=$'\t' read -r method_slug model_slug preflight_hash comparison_hash canonical_variant sample_count <<< "${metadata}"
  [[ "${method_slug}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "unsafe method slug '${method_slug}'"
  [[ "${model_slug}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "unsafe model slug '${model_slug}'"
  [[ -n "${preflight_hash}" && -n "${comparison_hash}" && -n "${canonical_variant}" ]] || die "preflight omitted hashes for '${variant}'"
  [[ "${sample_count}" =~ ^[0-9]+$ && "${sample_count}" -gt 0 ]] || die "preflight returned no PPL windows for '${variant}'"

  local arm_name="${model_slug}_${method_slug}"
  [[ -z "${SEEN_ARMS[${arm_name}]:-}" ]] || die "variants '${SEEN_ARMS[${arm_name}]}' and '${variant}' collide at '${arm_name}'"
  SEEN_ARMS["${arm_name}"]="${variant}"
  PREFLIGHT_FILES[index]="${json_file}"
  PREFLIGHT_LOGS[index]="${log_file}"
  METHOD_SLUGS[index]="${method_slug}"
  MODEL_SLUGS[index]="${model_slug}"
  PREFLIGHT_HASHES[index]="${preflight_hash}"
  COMPARISON_HASHES[index]="${comparison_hash}"
  VALID_INDICES+=("${index}")
  echo "[run-ppl] preflight ok: canonical=${canonical_variant} arm=${arm_name} windows=${sample_count}"
}

PREFLIGHT_RC=0
for index in "${!VARIANT_ARR[@]}"; do
  if ! preflight_variant "${index}" "${VARIANT_ARR[index]}"; then
    PREFLIGHT_RC=1
  fi
done
[[ "${PREFLIGHT_RC}" -eq 0 ]] || die "one or more CPU preflights failed; no GPU work started"

BASE_COMPARISON_HASH="${COMPARISON_HASHES[${VALID_INDICES[0]}]}"
FP16_COUNT=0
for index in "${VALID_INDICES[@]}"; do
  [[ "${COMPARISON_HASHES[index]}" == "${BASE_COMPARISON_HASH}" ]] || die "method preflights disagree on comparison target"
  [[ "${METHOD_SLUGS[index]}" == "fp16" ]] && FP16_COUNT=$((FP16_COUNT + 1))
done
[[ "${FP16_COUNT}" -eq 1 ]] || die "requested methods must contain exactly one fp16 baseline"

ARM_ACTIVE_PGID=""

terminate_arm() {
  local rc="${1:-130}"
  local pgid="${ARM_ACTIVE_PGID:-}"
  trap - INT TERM
  if [[ -n "${pgid}" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    wait "${pgid}" 2>/dev/null || true
  fi
  exit "${rc}"
}

run_logged_pipeline() {
  local log_path="$1" prefix="$2"
  shift 2
  setsid bash -o pipefail -c '
    log_path="$1"
    prefix="$2"
    shift 2
    "$@" 2>&1 | tee "${log_path}" | sed -u "s/^/[${prefix}] /"
  ' _ "${log_path}" "${prefix}" "$@" &
  ARM_ACTIVE_PGID=$!
  local status=0
  wait "${ARM_ACTIVE_PGID}" || status=$?
  ARM_ACTIVE_PGID=""
  return "${status}"
}

run_arm() {
  local index="$1" gpu="$2"
  trap - EXIT
  trap 'terminate_arm 130' INT TERM
  ARM_ACTIVE_PGID=""
  local variant="${VARIANT_ARR[index]}"
  local method_slug="${METHOD_SLUGS[index]}"
  local model_slug="${MODEL_SLUGS[index]}"
  local preflight_hash="${PREFLIGHT_HASHES[index]}"
  local arm_dir="${LAYOUT_ROOT}/${model_slug}_${method_slug}"
  local pred_dir="${arm_dir}/pred"
  local logs_dir="${arm_dir}/logs"
  local -a args=()
  build_common_args "${variant}" args

  [[ "${arm_dir}" == "${LAYOUT_ROOT}/"* && "${arm_dir}" != "${LAYOUT_ROOT}/" ]] || die "unsafe arm path: ${arm_dir}"
  if [[ "${SMOKE}" == "1" && -d "${arm_dir}" ]]; then
    echo "[run-ppl] smoke clean slate: ${arm_dir}"
    rm -rf -- "${arm_dir}"
  fi
  mkdir -p "${pred_dir}" "${logs_dir}"
  rm -f -- "${logs_dir}/worker.report.json" "${logs_dir}/run.log" "${logs_dir}/score.log"
  cp -- "${PREFLIGHT_FILES[index]}" "${logs_dir}/preflight.json"
  cp -- "${PREFLIGHT_LOGS[index]}" "${logs_dir}/preflight.log"

  local -a eval_cmd=(
    "${PYTHON_BIN}" -m kitty_sim.cli.eval_ppl
    "${args[@]}"
    --output-dir "${pred_dir}"
    --require-cuda-visible-devices "${gpu}"
    --report-json "${logs_dir}/worker.report.json"
    --expected-preflight-hash "${preflight_hash}"
  )
  [[ "${FORCE}" == "1" ]] && eval_cmd+=(--overwrite)
  local -a env_cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "GPU_ID=${gpu}"
    "GPU_IDS_CSV=${gpu}"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONUNBUFFERED=1"
    "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
  )

  echo "[run-ppl] launch GPU${gpu}: variant=${variant} arm=${model_slug}_${method_slug}"
  local status=0
  run_logged_pipeline \
    "${logs_dir}/run.log" "gpu${gpu} ${variant}" \
    "${env_cmd[@]}" "${eval_cmd[@]}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "[run-ppl] FAILED GPU${gpu}: variant=${variant} rc=${status}" >&2
    return "${status}"
  fi

  local -a score_cmd=(
    env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -m kitty_sim.cli.score_ppl "${pred_dir}"
  )
  status=0
  run_logged_pipeline \
    "${logs_dir}/score.log" "score ${variant}" \
    "${score_cmd[@]}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "[run-ppl] scoring FAILED: variant=${variant} rc=${status}" >&2
    return "${status}"
  fi
  trap - INT TERM
  echo "[run-ppl] done GPU${gpu}: variant=${variant}"
}

NEXT_INDEX=0
RUNNING=0
OVERALL_RC=0
declare -A PID_GPU=()
declare -A PID_INDEX=()

launch_next() {
  local gpu="$1" index pid
  (( NEXT_INDEX < ${#VALID_INDICES[@]} )) || return 1
  index="${VALID_INDICES[NEXT_INDEX]}"
  NEXT_INDEX=$((NEXT_INDEX + 1))
  run_arm "${index}" "${gpu}" &
  pid=$!
  ACTIVE_PIDS["${pid}"]=1
  PID_GPU["${pid}"]="${gpu}"
  PID_INDEX["${pid}"]="${index}"
  RUNNING=$((RUNNING + 1))
}

for gpu in "${GPU_ARR[@]}"; do
  launch_next "${gpu}" || break
done

while (( RUNNING > 0 )); do
  finished_pid=""
  set +e
  wait -n -p finished_pid
  status=$?
  set -e
  [[ -n "${finished_pid}" ]] || die "scheduler could not identify completed worker"
  gpu="${PID_GPU[${finished_pid}]}"
  index="${PID_INDEX[${finished_pid}]}"
  unset 'ACTIVE_PIDS['"${finished_pid}"']' 'PID_GPU['"${finished_pid}"']' 'PID_INDEX['"${finished_pid}"']'
  RUNNING=$((RUNNING - 1))
  if [[ "${status}" -ne 0 ]]; then
    echo "[run-ppl] method failed: variant=${VARIANT_ARR[index]} GPU${gpu} rc=${status}" >&2
    OVERALL_RC=1
  fi
  launch_next "${gpu}" || true
done

if [[ "${OVERALL_RC}" -ne 0 ]]; then
  echo "[run-ppl] one or more variants failed; comparison not written" >&2
  exit "${OVERALL_RC}"
fi

declare -a PRED_DIRS=()
MODEL_SLUG="${MODEL_SLUGS[${VALID_INDICES[0]}]}"
for index in "${VALID_INDICES[@]}"; do
  PRED_DIRS+=("${LAYOUT_ROOT}/${MODEL_SLUGS[index]}_${METHOD_SLUGS[index]}/pred")
done
COMPARISON_JSON="${LAYOUT_ROOT}/${MODEL_SLUG}_ppl_comparison.json"
COMPARISON_CSV="${LAYOUT_ROOT}/${MODEL_SLUG}_ppl_comparison.csv"
"${PYTHON_BIN}" -m kitty_sim.cli.score_ppl \
  "${PRED_DIRS[@]}" \
  --comparison-json "${COMPARISON_JSON}" \
  --comparison-csv "${COMPARISON_CSV}" \
  > "${LAYOUT_ROOT}/${MODEL_SLUG}_ppl_comparison.log"
echo "[run-ppl] all requested variants completed successfully"
echo "[run-ppl] comparison JSON: ${COMPARISON_JSON}"
echo "[run-ppl] comparison CSV:  ${COMPARISON_CSV}"
