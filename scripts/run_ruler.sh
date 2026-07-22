#!/usr/bin/env bash
# Sole RULER evaluation runner for Kitty.
#
# For each method, one Python worker per GPU owns a disjoint RULER task shard.
# Methods run sequentially; every worker loads one model on one physical GPU.
# Python preflight is the source of truth for canonical model/method slugs and
# run hashes; this shell intentionally contains no variant-to-slug table.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Preserve process-environment intent across aliases before .env is sourced.
# For example, an explicit GPU_ID=0 must outrank GPU_IDS_CSV=1 from .env.
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
capture_process_value _RULER_PROCESS_PYTHON PYTHON_BIN KITTY_PYTHON_BIN
capture_process_value _RULER_PROCESS_MODEL MODEL LLAMA32_MODEL_ID
capture_process_value _RULER_PROCESS_MODEL_PATH MODEL_PATH LLAMA32_MODEL_PATH KITTY_LLAMA32_1B_PATH
capture_process_value _RULER_PROCESS_MODEL_TAG MODEL_TAG LLAMA32_MODEL_SLUG
capture_process_value _RULER_PROCESS_DTYPE DTYPE TORCH_DTYPE
capture_process_value _RULER_PROCESS_GPU GPUS_OVERRIDE GPU_OVERRIDE GPU_IDS_CSV GPU_ID

# shellcheck source=accuracy_simulation/env.sh
source "${REPO_ROOT}/accuracy_simulation/env.sh"
cd "${REPO_ROOT}"

PYTHON_BIN="${_RULER_PROCESS_PYTHON:-${PYTHON_BIN:-${KITTY_PYTHON_BIN:-python}}}"
MODEL="${_RULER_PROCESS_MODEL:-${MODEL:-${LLAMA32_MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}}}"
MODEL_PATH="${_RULER_PROCESS_MODEL_PATH:-${MODEL_PATH:-${LLAMA32_MODEL_PATH:-${KITTY_LLAMA32_1B_PATH:-${HOME}/models/Llama-3.2-1B-Instruct}}}}"
MODEL_TAG="${_RULER_PROCESS_MODEL_TAG:-${MODEL_TAG:-${LLAMA32_MODEL_SLUG:-}}}"
MODEL_FAMILY="${MODEL_FAMILY:-}"
VARIANTS="${VARIANTS_CSV:-fp16,qlutattn}"
TASKS="${TASKS:-all}"
LENGTHS="${LENGTHS:-4096,8192,16384,32768}"
DATA_ROOT="${RULER_DATA_ROOT:-${HOME}/data/ruler/llama3}"
OUT_ROOT="${OUT_ROOT:-ruler_out}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${_RULER_PROCESS_DTYPE:-${DTYPE:-${TORCH_DTYPE:-float16}}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FORCE="${FORCE:-0}"

if [[ -n "${_RULER_PROCESS_GPU}" ]]; then
  GPU_CSV="${_RULER_PROCESS_GPU}"
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
Usage: bash scripts/run_ruler.sh [OPTIONS]

The only supported Kitty RULER evaluation entry point. Every method is
CPU-preflighted before any GPU worker starts. With --gpus, workers load one
model per GPU and automatically split the requested tasks between them.

Model:
  --model ID                 Model alias/Hugging Face id (default Llama-3.2-1B)
  --model-path PATH          Local checkpoint directory (required by preflight)
  --model-tag TAG            Optional path-safe output model slug override
  --model-family FAMILY      Optional prompt/cache family override (inferred otherwise)

Evaluation:
  --variants CSV             Methods (default: fp16,qlutattn)
                             Canonical choices: fp16, kitty, shadowkv,
                             qlutattn, kivi, kivi_star, llamacpp_q40,
                             llamacpp_q40_star, custom
  --tasks all|CSV            Tasks (default: all 13 NVIDIA RULER tasks)
                             niah_single_1/2/3, niah_multikey_1/2/3,
                             niah_multivalue, niah_multiquery, vt, cwe, fwe,
                             qa_1, qa_2
  --lengths CSV              Nominal lengths (default: 4096,8192,16384,32768)
  --data-root PATH           Prepared RULER data root
  --out-root PATH            Output root (default: ruler_out)
  --max-samples N            Per-pair limit; N > 0 selects smoke layout,
                             -1 selects full/resumable evaluation
  --max-model-len N          Hard model context limit (default: 32768)
  --dtype NAME               float16/fp16/bfloat16/bf16/float32/fp32
  --local-files-only         Disable model/tokenizer network access (default)
  --no-local-files-only      Allow normal Hugging Face resolution
  --force                    Overwrite selected full-run pairs

GPU scheduling:
  --gpu N                    Run every method serially on physical GPU N
                             (default: physical GPU 1)
  --gpus CSV                 For each method, shard tasks across GPU slots.
                             Methods run sequentially; each GPU loads one model
                             and runs all lengths for its assigned tasks.
  --gpu and --gpus are mutually exclusive. A failed task worker does not stop
  sibling workers or later methods; the aggregate command exits nonzero.

Output:
  full:  <out-root>/<model>_<method>/{pred,logs}
  smoke: <out-root>/smoke/<model>_<method>/{pred,logs}
  Smoke clears an arm once before any task worker starts. Full runs resume
  only pairs whose manifests match the CPU-preflight hashes. Every arm is
  scored once after all task workers finish and keeps per-worker run reports.

Environment (explicit CLI > process environment > repo .env):
  PYTHON_BIN, MODEL, MODEL_PATH, MODEL_TAG, MODEL_FAMILY
  VARIANTS_CSV, TASKS, LENGTHS, RULER_DATA_ROOT, OUT_ROOT
  MAX_SAMPLES, MAX_MODEL_LEN, DTYPE (or TORCH_DTYPE), LOCAL_FILES_ONLY, FORCE
  GPU_OVERRIDE, GPUS_OVERRIDE, GPU_ID, GPU_IDS_CSV
  QLUT_CB_MASK (required by qlutattn), plus the standard LongBench variant
  knobs KBITS, VBITS, PROMOTE_BIT, PROMOTE_RATIO, PROMOTE_RATIO_CONFIG,
  SHADOWKV_BUDGET, SHADOWKV_RANK, SHADOWKV_CHUNK

Examples:
  bash scripts/run_ruler.sh --gpu 1 --variants fp16,qlutattn \
    --tasks all --lengths 4096 --max-samples 2

  bash scripts/run_ruler.sh --gpus 0,1,2 \
    --variants qlutattn --tasks all --max-samples 2

  bash scripts/run_ruler.sh --model /models/MiniCPM5-1B \
    --model-path /models/MiniCPM5-1B --model-tag minicpm5-1b \
    --model-family minicpm --gpu 1 --max-samples 2
USAGE
}

die() {
  echo "[run-ruler] ERROR: $*" >&2
  exit 2
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

GPU_FLAG_SEEN=0
GPUS_FLAG_SEEN=0
set_value_option() {
  local option="$1" value="$2"
  [[ -n "${value}" ]] || die "${option} requires a non-empty value"
  case "${option}" in
    --model) MODEL="${value}" ;;
    --model-path) MODEL_PATH="${value}" ;;
    --model-tag) MODEL_TAG="${value}" ;;
    --model-family) MODEL_FAMILY="${value}" ;;
    --variants) VARIANTS="${value}" ;;
    --tasks) TASKS="${value}" ;;
    --lengths) LENGTHS="${value}" ;;
    --data-root) DATA_ROOT="${value}" ;;
    --out-root) OUT_ROOT="${value}" ;;
    --max-samples) MAX_SAMPLES="${value}" ;;
    --max-model-len) MAX_MODEL_LEN="${value}" ;;
    --dtype) DTYPE="${value}" ;;
    --gpu)
      [[ "${GPUS_FLAG_SEEN}" == "0" ]] || die "--gpu and --gpus are mutually exclusive"
      [[ "${value}" != *,* ]] || die "--gpu accepts one id; use --gpus for a CSV list"
      GPU_FLAG_SEEN=1
      GPU_CSV="${value}"
      ;;
    --gpus)
      [[ "${GPU_FLAG_SEEN}" == "0" ]] || die "--gpu and --gpus are mutually exclusive"
      GPUS_FLAG_SEEN=1
      GPU_CSV="${value}"
      ;;
    *) die "unknown option: ${option}" ;;
  esac
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --local-files-only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --no-local-files-only)
      LOCAL_FILES_ONLY=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --model|--model-path|--model-tag|--model-family|--variants|--tasks|--lengths|--data-root|--out-root|--max-samples|--max-model-len|--dtype|--gpu|--gpus)
      [[ "$#" -ge 2 ]] || die "$1 requires a value"
      set_value_option "$1" "$2"
      shift 2
      ;;
    --model=*|--model-path=*|--model-tag=*|--model-family=*|--variants=*|--tasks=*|--lengths=*|--data-root=*|--out-root=*|--max-samples=*|--max-model-len=*|--dtype=*|--gpu=*|--gpus=*)
      option="${1%%=*}"
      value="${1#*=}"
      set_value_option "${option}" "${value}"
      shift
      ;;
    *) die "unknown argument: $1 (use --help)" ;;
  esac
done

normalize_bool() {
  local name="$1" raw="${2,,}"
  case "${raw}" in
    1|true|yes|on) printf '1' ;;
    0|false|no|off|'') printf '0' ;;
    *) die "${name} must be one of 1/0, true/false, yes/no, on/off (got '$2')" ;;
  esac
}

parse_csv() {
  local raw="$1" output_name="$2"
  local -n output_ref="${output_name}"
  local -a pieces=()
  local piece cleaned
  [[ -n "${raw}" ]] || die "CSV value cannot be empty"
  [[ "${raw}" != ,* && "${raw}" != *, && "${raw}" != *,,* ]] || die "invalid CSV value: '${raw}'"
  IFS=',' read -r -a pieces <<< "${raw}"
  output_ref=()
  for piece in "${pieces[@]}"; do
    cleaned="$(trim "${piece}")"
    [[ -n "${cleaned}" ]] || die "invalid empty CSV item in '${raw}'"
    output_ref+=("${cleaned}")
  done
}

join_csv() {
  local array_name="$1"
  local -n array_ref="${array_name}"
  local IFS=','
  printf '%s' "${array_ref[*]}"
}

LOCAL_FILES_ONLY="$(normalize_bool LOCAL_FILES_ONLY "${LOCAL_FILES_ONLY}")"
FORCE="$(normalize_bool FORCE "${FORCE}")"
[[ "${MAX_SAMPLES}" =~ ^-?[0-9]+$ ]] || die "--max-samples must be an integer"
(( MAX_SAMPLES >= -1 )) || die "--max-samples must be -1 or non-negative"
[[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ ]] || die "--max-model-len must be a positive integer"
(( MAX_MODEL_LEN > 0 )) || die "--max-model-len must be positive"
[[ -n "${MODEL}" ]] || die "--model cannot be empty"
[[ -n "${DATA_ROOT}" ]] || die "--data-root cannot be empty"
OUT_ROOT="${OUT_ROOT%/}"
[[ -n "${OUT_ROOT}" && "${OUT_ROOT}" != "/" ]] || die "--out-root must name a directory below the filesystem root"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"

parse_csv "${VARIANTS}" VARIANT_ARR
VARIANTS="$(join_csv VARIANT_ARR)"
if [[ "${TASKS}" != "all" ]]; then
  parse_csv "${TASKS}" TASK_ARR
  TASKS="$(join_csv TASK_ARR)"
fi
parse_csv "${LENGTHS}" LENGTH_ARR
for length in "${LENGTH_ARR[@]}"; do
  [[ "${length}" =~ ^[0-9]+$ ]] || die "RULER lengths must be positive integers (got '${length}')"
  (( length > 0 )) || die "RULER lengths must be positive (got '${length}')"
done
LENGTHS="$(join_csv LENGTH_ARR)"
parse_csv "${GPU_CSV}" GPU_ARR
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU ids must be non-negative integers (got '${gpu}')"
done

SMOKE=0
if (( MAX_SAMPLES > 0 )); then
  SMOKE=1
  LAYOUT_ROOT="${OUT_ROOT}/smoke"
else
  LAYOUT_ROOT="${OUT_ROOT}"
fi

TMP_ROOT=""
declare -A ACTIVE_PIDS=()
cleanup() {
  local rc=$?
  trap - EXIT
  local pid
  for pid in "${!ACTIVE_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${!ACTIVE_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  if [[ -n "${TMP_ROOT}" && -d "${TMP_ROOT}" ]]; then
    rm -rf -- "${TMP_ROOT}"
  fi
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/kitty-ruler.XXXXXX")"

declare -a PREFLIGHT_FILES=()
declare -a PREFLIGHT_LOGS=()
declare -a METHOD_SLUGS=()
declare -a MODEL_SLUGS=()
declare -a PREFLIGHT_HASHES=()
declare -a TASK_COUNTS=()
declare -a VALID_INDICES=()

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
  return 0
}

build_common_args() {
  local variant="$1" output_name="$2"
  local -n output_ref="${output_name}"
  output_ref=(
    "${MODEL}"
    --model-path "${MODEL_PATH}"
    --variant "${variant}"
    --tasks "${TASKS}"
    --seq-lens "${LENGTHS}"
    --data-root "${DATA_ROOT}"
    --max-samples "${MAX_SAMPLES}"
    --max-model-len "${MAX_MODEL_LEN}"
    --torch-dtype "${DTYPE}"
  )
  [[ -n "${MODEL_TAG}" ]] && output_ref+=(--model-tag "${MODEL_TAG}")
  [[ -n "${MODEL_FAMILY}" ]] && output_ref+=(--model-family "${MODEL_FAMILY}")
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    output_ref+=(--local-files-only)
  fi
  append_variant_args "${output_name}"
}

preflight_variant() {
  local index="$1" variant="$2"
  local json_file="${TMP_ROOT}/preflight-${index}.json"
  local log_file="${TMP_ROOT}/preflight-${index}.log"
  local -a args=()
  build_common_args "${variant}" args
  local -a cmd=(
    env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -m kitty_sim.cli.preflight_ruler
    "${args[@]}" --json
  )

  echo "[run-ruler] CPU preflight: variant=${variant}"
  if ! "${cmd[@]}" >"${json_file}" 2>"${log_file}"; then
    echo "[run-ruler] preflight FAILED: variant=${variant}" >&2
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
values = [payload["method_slug"], payload["model_slug"], payload["preflight_hash"], payload["canonical_variant"], len(payload["tasks"])]
if not payload.get("pairs"):
    raise SystemExit("preflight returned no task/length pairs")
print("\t".join(str(value) for value in values))
' "${json_file}" 2>>"${log_file}")"; then
    echo "[run-ruler] invalid preflight JSON: variant=${variant}" >&2
    while IFS= read -r line; do
      printf '[preflight %s] %s\n' "${variant}" "${line}" >&2
    done < "${log_file}"
    return 1
  fi

  local method_slug model_slug preflight_hash canonical_variant task_count
  IFS=$'\t' read -r method_slug model_slug preflight_hash canonical_variant task_count <<< "${metadata}"
  [[ "${method_slug}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "unsafe method slug from preflight: '${method_slug}'"
  [[ "${model_slug}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "unsafe model slug from preflight: '${model_slug}'"
  [[ -n "${preflight_hash}" && -n "${canonical_variant}" ]] || die "preflight omitted canonical variant/hash for '${variant}'"
  [[ "${task_count}" =~ ^[0-9]+$ && "${task_count}" -gt 0 ]] || die "preflight returned invalid task count '${task_count}'"

  PREFLIGHT_FILES[index]="${json_file}"
  PREFLIGHT_LOGS[index]="${log_file}"
  METHOD_SLUGS[index]="${method_slug}"
  MODEL_SLUGS[index]="${model_slug}"
  PREFLIGHT_HASHES[index]="${preflight_hash}"
  TASK_COUNTS[index]="${task_count}"
  echo "[run-ruler] preflight ok: variant=${variant} canonical=${canonical_variant} arm=${model_slug}_${method_slug}"
}

OVERALL_RC=0
declare -A SEEN_ARMS=()
for index in "${!VARIANT_ARR[@]}"; do
  variant="${VARIANT_ARR[index]}"
  if ! preflight_variant "${index}" "${variant}"; then
    OVERALL_RC=1
    continue
  fi
  arm_name="${MODEL_SLUGS[index]}_${METHOD_SLUGS[index]}"
  if [[ -n "${SEEN_ARMS[${arm_name}]:-}" ]]; then
    echo "[run-ruler] ERROR: variants '${SEEN_ARMS[${arm_name}]}' and '${variant}' resolve to the same arm '${arm_name}'" >&2
    OVERALL_RC=1
    continue
  fi
  SEEN_ARMS["${arm_name}"]="${variant}"
  VALID_INDICES+=("${index}")
done

run_task_worker() {
  local index="$1" slot="$2" shard_count="$3" gpu="$4" pred_dir="$5" logs_dir="$6"
  local variant="${VARIANT_ARR[index]}"
  local preflight_hash="${PREFLIGHT_HASHES[index]}"
  local worker_name="worker-${slot}-of-${shard_count}-gpu${gpu}"
  local -a args=()
  build_common_args "${variant}" args
  local -a eval_cmd=(
    "${PYTHON_BIN}" -m kitty_sim.cli.eval_ruler
    "${args[@]}"
    --output-dir "${pred_dir}"
    --task-shard-index "${slot}"
    --task-shard-count "${shard_count}"
    --require-cuda-visible-devices "${gpu}"
    --report-json "${logs_dir}/${worker_name}.report.json"
    --expected-preflight-hash "${preflight_hash}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    eval_cmd+=(--overwrite)
  fi
  local -a env_cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "GPU_ID=${gpu}"
    "GPU_IDS_CSV=${gpu}"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONUNBUFFERED=1"
    "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
  )

  echo "[run-ruler] launch GPU${gpu}: variant=${variant} shard=${slot}/${shard_count}"
  local -a pipe_status=()
  local status
  set +e
  "${env_cmd[@]}" "${eval_cmd[@]}" 2>&1 \
    | tee "${logs_dir}/${worker_name}.run.log" \
    | sed -u "s/^/[gpu${gpu} shard${slot} ${variant}] /"
  pipe_status=("${PIPESTATUS[@]}")
  set -e
  for status in "${pipe_status[@]}"; do
    if [[ "${status}" -ne 0 ]]; then
      echo "[run-ruler] FAILED GPU${gpu}: variant=${variant} shard=${slot}/${shard_count} rc=${status}" >&2
      return "${status}"
    fi
  done
  echo "[run-ruler] done GPU${gpu}: variant=${variant} shard=${slot}/${shard_count}"
}

run_arm() {
  local index="$1"
  local variant="${VARIANT_ARR[index]}"
  local method_slug="${METHOD_SLUGS[index]}"
  local model_slug="${MODEL_SLUGS[index]}"
  local task_count="${TASK_COUNTS[index]}"
  local arm_dir="${LAYOUT_ROOT}/${model_slug}_${method_slug}"
  local pred_dir="${arm_dir}/pred"
  local logs_dir="${arm_dir}/logs"
  local worker_count="${#GPU_ARR[@]}"
  if (( worker_count > task_count )); then
    worker_count="${task_count}"
  fi

  [[ "${arm_dir}" == "${LAYOUT_ROOT}/"* && "${arm_dir}" != "${LAYOUT_ROOT}/" ]] \
    || die "refusing unsafe arm path: ${arm_dir}"
  if [[ "${SMOKE}" == "1" && -d "${arm_dir}" ]]; then
    echo "[run-ruler] smoke clean slate: ${arm_dir}"
    rm -rf -- "${arm_dir}"
  fi
  mkdir -p "${pred_dir}" "${logs_dir}"
  rm -f -- \
    "${logs_dir}"/worker-*-of-*-gpu*.report.json \
    "${logs_dir}"/worker-*-of-*-gpu*.run.log \
    "${logs_dir}/report.json" \
    "${logs_dir}/run.log"
  cp -- "${PREFLIGHT_FILES[index]}" "${logs_dir}/preflight.json"
  cp -- "${PREFLIGHT_LOGS[index]}" "${logs_dir}/preflight.log"

  echo "[run-ruler] task parallelism: variant=${variant} tasks=${task_count} workers=${worker_count}"
  local -a worker_pids=()
  local -A pid_description=()
  local slot gpu pid status worker_rc=0
  for ((slot = 0; slot < worker_count; slot++)); do
    gpu="${GPU_ARR[slot]}"
    run_task_worker "${index}" "${slot}" "${worker_count}" "${gpu}" "${pred_dir}" "${logs_dir}" &
    pid=$!
    ACTIVE_PIDS["${pid}"]=1
    worker_pids+=("${pid}")
    pid_description["${pid}"]="GPU${gpu} shard=${slot}/${worker_count}"
  done

  for pid in "${worker_pids[@]}"; do
    set +e
    wait "${pid}"
    status=$?
    set -e
    unset 'ACTIVE_PIDS['"${pid}"']'
    if [[ "${status}" -ne 0 ]]; then
      echo "[run-ruler] task worker failed: variant=${variant} ${pid_description[${pid}]} rc=${status}" >&2
      worker_rc=1
    fi
  done

  local -a score_cmd=(
    env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -m kitty_sim.cli.score_ruler
    "${pred_dir}"
    --tasks "${TASKS}"
    --seq-lens "${LENGTHS}"
    --heatmap-dir "${logs_dir}"
    --title "${model_slug} ${method_slug}"
  )
  local -a pipe_status=()
  local score_rc=0
  set +e
  "${score_cmd[@]}" 2>&1 \
    | tee "${logs_dir}/score.log" \
    | sed -u "s/^/[score ${variant}] /"
  pipe_status=("${PIPESTATUS[@]}")
  set -e
  for status in "${pipe_status[@]}"; do
    if [[ "${status}" -ne 0 ]]; then
      score_rc="${status}"
      break
    fi
  done

  if [[ "${worker_rc}" -ne 0 || "${score_rc}" -ne 0 ]]; then
    echo "[run-ruler] FAILED variant=${variant}: worker_rc=${worker_rc} score_rc=${score_rc}" >&2
    return 1
  fi
  echo "[run-ruler] done variant=${variant}"
}

for index in "${VALID_INDICES[@]}"; do
  if ! run_arm "${index}"; then
    OVERALL_RC=1
  fi
done

if [[ "${OVERALL_RC}" -eq 0 ]]; then
  echo "[run-ruler] all requested variants completed successfully"
else
  echo "[run-ruler] one or more variants failed" >&2
fi
exit "${OVERALL_RC}"
