#!/usr/bin/env bash
# Launch cache-aware PPL for several model registries in parallel.
# Each model receives a disjoint physical-GPU group and delegates method
# scheduling, preflight, scoring, and comparison to scripts/run_ppl.sh.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=accuracy_simulation/env.sh
source "${REPO_ROOT}/accuracy_simulation/env.sh"
cd "${REPO_ROOT}"

MODELS="${PPL_MODELS_CSV:-llama32-1b,llama32-3b,minicpm5-1b}"
VARIANTS="${VARIANTS_CSV:-fp16,shadowkv,kivi,qlutattn}"
DATA_PATH="${PPL_DATA_PATH:-${KITTY_WIKITEXT2_TEST_PATH:-}}"
OUT_ROOT="${OUT_ROOT:-ppl_out}"
PREFILL_TOKENS="${PREFILL_TOKENS:-4096}"
SCORE_TOKENS="${SCORE_TOKENS:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-${TORCH_DTYPE:-float16}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FORCE="${FORCE:-${OVERWRITE:-0}}"
GPU_CSV=""
MODEL_RUNNER="${REPO_ROOT}/scripts/run_ppl.sh"
if [[ "${PPL_TESTING:-0}" == "1" && -n "${PPL_MODEL_RUNNER_TEST_ONLY:-}" ]]; then
  MODEL_RUNNER="${PPL_MODEL_RUNNER_TEST_ONLY}"
fi
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_ppl_models.sh [OPTIONS]

Run several registered models concurrently. The physical GPU pool is split into
non-overlapping groups, one group per model. Each model delegates its configured
method list to scripts/run_ppl.sh, which schedules those methods within the
model's GPU group.

Registered models:
  llama32-1b   meta-llama/Llama-3.2-1B-Instruct
  llama32-3b   meta-llama/Llama-3.2-3B-Instruct
  minicpm5-1b  openbmb/MiniCPM5-1B

Selection:
  --models CSV               Model aliases (default: all three above)
  --variants CSV             Methods; must contain exactly one fp16 baseline
                             and at least one comparison method
                             (default: fp16,shadowkv,kivi,qlutattn)
  --gpus CSV                 Physical GPU pool. GPU count must be >= model count.
                             GPUs are assigned round-robin into disjoint groups.

Shared PPL configuration:
  --data-path PATH           Local document-level WikiText-2 test parquet
  --out-root PATH            Output root (default: ppl_out)
  --prefill-tokens N         Dense prefill length (default: 4096)
  --score-tokens N           One-token decode targets/window (default: 256)
  --max-samples N            Window limit (default: 2 smoke; <=0 means full)
  --max-model-len N          Hard context cap (default: 32768)
  --dtype NAME               float16/fp16/bfloat16/bf16/float32/fp32
  --local-files-only         Disable model/tokenizer network access (default)
  --no-local-files-only      Allow normal Hugging Face resolution
  --force                    Overwrite selected outputs
  --dry-run                  Validate registry/paths and print the launch plan

Required .env paths for the default matrix:
  KITTY_LLAMA32_1B_PATH, KITTY_LLAMA32_3B_PATH, KITTY_MINICPM5_1B_PATH
  KITTY_LLAMA32_1B_QLUTATTN_MASK, KITTY_LLAMA32_3B_QLUTATTN_MASK,
  KITTY_MINICPM5_1B_QLUTATTN_MASK, PPL_DATA_PATH

Examples:
  bash scripts/run_ppl_models.sh \
    --models llama32-1b,llama32-3b,minicpm5-1b \
    --variants fp16,shadowkv,kivi,qlutattn \
    --gpus 0,1,2,3,4,5 --max-samples 2

  bash scripts/run_ppl_models.sh \
    --models llama32-1b,minicpm5-1b \
    --variants fp16,kivi --gpus 1,2 --max-samples -1
USAGE
}

die() {
  echo "[run-ppl-models] ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) [[ $# -ge 2 ]] || die "--models needs a value"; MODELS="$2"; shift 2 ;;
    --variants) [[ $# -ge 2 ]] || die "--variants needs a value"; VARIANTS="$2"; shift 2 ;;
    --gpus) [[ $# -ge 2 ]] || die "--gpus needs a value"; GPU_CSV="$2"; shift 2 ;;
    --data-path) [[ $# -ge 2 ]] || die "--data-path needs a value"; DATA_PATH="$2"; shift 2 ;;
    --out-root) [[ $# -ge 2 ]] || die "--out-root needs a value"; OUT_ROOT="$2"; shift 2 ;;
    --prefill-tokens) [[ $# -ge 2 ]] || die "--prefill-tokens needs a value"; PREFILL_TOKENS="$2"; shift 2 ;;
    --score-tokens) [[ $# -ge 2 ]] || die "--score-tokens needs a value"; SCORE_TOKENS="$2"; shift 2 ;;
    --max-samples) [[ $# -ge 2 ]] || die "--max-samples needs a value"; MAX_SAMPLES="$2"; shift 2 ;;
    --max-model-len) [[ $# -ge 2 ]] || die "--max-model-len needs a value"; MAX_MODEL_LEN="$2"; shift 2 ;;
    --dtype) [[ $# -ge 2 ]] || die "--dtype needs a value"; DTYPE="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --no-local-files-only) LOCAL_FILES_ONLY=0; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${GPU_CSV}" ]] || die "--gpus is required for multi-model scheduling"
[[ -n "${DATA_PATH}" && -f "${DATA_PATH}" ]] \
  || die "local WikiText-2 parquet does not exist: ${DATA_PATH:-<unset>}"
[[ -f "${MODEL_RUNNER}" ]] || die "per-model PPL runner does not exist: ${MODEL_RUNNER}"
command -v setsid >/dev/null 2>&1 || die "setsid is required for process-group-safe model workers"
[[ "${PREFILL_TOKENS}" =~ ^[0-9]+$ && "${PREFILL_TOKENS}" -gt 1 ]] \
  || die "prefill tokens must be >1"
[[ "${SCORE_TOKENS}" =~ ^[0-9]+$ && "${SCORE_TOKENS}" -gt 0 ]] \
  || die "score tokens must be >0"
[[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ && "${MAX_MODEL_LEN}" -gt 0 ]] \
  || die "max model length must be positive"
[[ "${MAX_SAMPLES}" =~ ^-?[0-9]+$ ]] || die "max samples must be an integer"
[[ "${LOCAL_FILES_ONLY}" == "0" || "${LOCAL_FILES_ONLY}" == "1" ]] \
  || die "LOCAL_FILES_ONLY must be 0 or 1"
[[ "${FORCE}" == "0" || "${FORCE}" == "1" ]] \
  || die "FORCE/OVERWRITE must be 0 or 1"
case "${DTYPE}" in
  float16|fp16|bfloat16|bf16|float32|fp32) ;;
  *) die "unsupported dtype: ${DTYPE}" ;;
esac

IFS=',' read -r -a RAW_MODELS <<< "${MODELS}"
declare -a MODEL_ALIASES=()
declare -A SEEN_MODELS=()
for alias in "${RAW_MODELS[@]}"; do
  alias="${alias//[[:space:]]/}"
  [[ -n "${alias}" ]] || continue
  [[ -z "${SEEN_MODELS[${alias}]:-}" ]] || die "duplicate model alias '${alias}'"
  SEEN_MODELS["${alias}"]=1
  MODEL_ALIASES+=("${alias}")
done
[[ "${#MODEL_ALIASES[@]}" -gt 0 ]] || die "--models resolved to an empty list"

IFS=',' read -r -a RAW_VARIANTS <<< "${VARIANTS}"
declare -a VARIANT_ARR=()
declare -A SEEN_VARIANTS=()
FP16_COUNT=0
NEEDS_QLUT=0
for variant in "${RAW_VARIANTS[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -n "${variant}" ]] || continue
  [[ -z "${SEEN_VARIANTS[${variant}]:-}" ]] || die "duplicate variant '${variant}'"
  SEEN_VARIANTS["${variant}"]=1
  VARIANT_ARR+=("${variant}")
  [[ "${variant}" == "fp16" ]] && FP16_COUNT=$((FP16_COUNT + 1))
  [[ "${variant}" == "qlutattn" ]] && NEEDS_QLUT=1
done
[[ "${#VARIANT_ARR[@]}" -gt 0 ]] || die "--variants resolved to an empty list"
[[ "${#VARIANT_ARR[@]}" -ge 2 ]] \
  || die "--variants must contain fp16 and at least one comparison method"
[[ "${FP16_COUNT}" -eq 1 ]] \
  || die "--variants must contain exactly one fp16 baseline for comparison"
VARIANTS="$(IFS=','; echo "${VARIANT_ARR[*]}")"

IFS=',' read -r -a RAW_GPUS <<< "${GPU_CSV}"
declare -a GPU_ARR=()
declare -A SEEN_GPUS=()
for gpu in "${RAW_GPUS[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id '${gpu}'"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate GPU id '${gpu}'"
  SEEN_GPUS["${gpu}"]=1
  GPU_ARR+=("${gpu}")
done
[[ "${#GPU_ARR[@]}" -ge "${#MODEL_ALIASES[@]}" ]] \
  || die "GPU count (${#GPU_ARR[@]}) must be >= model count (${#MODEL_ALIASES[@]})"

declare -a MODEL_IDS=()
declare -a MODEL_PATHS=()
declare -a MODEL_TAGS=()
declare -a MODEL_FAMILIES=()
declare -a MODEL_MASKS=()

resolve_model() {
  local index="$1" alias="${MODEL_ALIASES[index]}"
  case "${alias}" in
    llama32-1b)
      MODEL_IDS[index]="meta-llama/Llama-3.2-1B-Instruct"
      MODEL_PATHS[index]="${KITTY_LLAMA32_1B_PATH:-}"
      MODEL_TAGS[index]="llama32-1b-instruct"
      MODEL_FAMILIES[index]="llama3.2"
      MODEL_MASKS[index]="${KITTY_LLAMA32_1B_QLUTATTN_MASK:-}"
      ;;
    llama32-3b)
      MODEL_IDS[index]="meta-llama/Llama-3.2-3B-Instruct"
      MODEL_PATHS[index]="${KITTY_LLAMA32_3B_PATH:-}"
      MODEL_TAGS[index]="llama32-3b-instruct"
      MODEL_FAMILIES[index]="llama3.2"
      MODEL_MASKS[index]="${KITTY_LLAMA32_3B_QLUTATTN_MASK:-}"
      ;;
    minicpm5-1b)
      MODEL_IDS[index]="openbmb/MiniCPM5-1B"
      MODEL_PATHS[index]="${KITTY_MINICPM5_1B_PATH:-}"
      MODEL_TAGS[index]="minicpm5-1b"
      MODEL_FAMILIES[index]="minicpm"
      MODEL_MASKS[index]="${KITTY_MINICPM5_1B_QLUTATTN_MASK:-}"
      ;;
    *) die "unknown model alias '${alias}'" ;;
  esac
  [[ -n "${MODEL_PATHS[index]}" && -d "${MODEL_PATHS[index]}" ]] \
    || die "${alias} model directory does not exist: ${MODEL_PATHS[index]:-<unset>}"
  if [[ "${NEEDS_QLUT}" == "1" ]]; then
    [[ -n "${MODEL_MASKS[index]}" && -f "${MODEL_MASKS[index]}" ]] \
      || die "${alias} QLUTATTN mask does not exist: ${MODEL_MASKS[index]:-<unset>}"
  fi
}

for index in "${!MODEL_ALIASES[@]}"; do
  resolve_model "${index}"
done

declare -a MODEL_GPU_GROUPS=()
for index in "${!MODEL_ALIASES[@]}"; do
  MODEL_GPU_GROUPS[index]=""
done
for gpu_index in "${!GPU_ARR[@]}"; do
  model_index=$((gpu_index % ${#MODEL_ALIASES[@]}))
  if [[ -n "${MODEL_GPU_GROUPS[model_index]}" ]]; then
    MODEL_GPU_GROUPS[model_index]+=","
  fi
  MODEL_GPU_GROUPS[model_index]+="${GPU_ARR[gpu_index]}"
done

print_plan() {
  echo "[run-ppl-models] models=${MODELS} variants=${VARIANTS}"
  echo "[run-ppl-models] data=${DATA_PATH} max_samples=${MAX_SAMPLES}"
  for index in "${!MODEL_ALIASES[@]}"; do
    echo "[run-ppl-models] plan model=${MODEL_ALIASES[index]} tag=${MODEL_TAGS[index]} gpus=${MODEL_GPU_GROUPS[index]} path=${MODEL_PATHS[index]}"
  done
}
print_plan
[[ "${DRY_RUN}" == "0" ]] || exit 0

declare -A ACTIVE_PIDS=()
cleanup() {
  local rc=$? pid
  trap - EXIT INT TERM
  for pid in "${!ACTIVE_PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${!ACTIVE_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

MODEL_LAUNCHED_PID=""
launch_model() {
  local index="$1"
  local alias="${MODEL_ALIASES[index]}"
  local -a command=(
    bash "${MODEL_RUNNER}"
    --model "${MODEL_IDS[index]}"
    --model-path "${MODEL_PATHS[index]}"
    --model-tag "${MODEL_TAGS[index]}"
    --model-family "${MODEL_FAMILIES[index]}"
    --data-path "${DATA_PATH}"
    --variants "${VARIANTS}"
    --gpus "${MODEL_GPU_GROUPS[index]}"
    --out-root "${OUT_ROOT}"
    --prefill-tokens "${PREFILL_TOKENS}"
    --score-tokens "${SCORE_TOKENS}"
    --max-samples "${MAX_SAMPLES}"
    --max-model-len "${MAX_MODEL_LEN}"
    --dtype "${DTYPE}"
  )
  [[ "${LOCAL_FILES_ONLY}" == "1" ]] && command+=(--local-files-only)
  [[ "${LOCAL_FILES_ONLY}" == "0" ]] && command+=(--no-local-files-only)
  [[ "${FORCE}" == "1" ]] && command+=(--force)

  setsid env \
    "QLUT_CB_MASK=${MODEL_MASKS[index]}" \
    "PPL_MODEL_ALIAS=${alias}" \
    "PPL_MODEL_GPU_GROUP=${MODEL_GPU_GROUPS[index]}" \
    "${command[@]}" &
  MODEL_LAUNCHED_PID=$!
}

declare -a MODEL_PIDS=()
declare -A PID_MODEL=()
for index in "${!MODEL_ALIASES[@]}"; do
  echo "[run-ppl-models] launch model=${MODEL_ALIASES[index]} gpus=${MODEL_GPU_GROUPS[index]}"
  launch_model "${index}"
  pid="${MODEL_LAUNCHED_PID}"
  ACTIVE_PIDS["${pid}"]=1
  MODEL_PIDS+=("${pid}")
  PID_MODEL["${pid}"]="${MODEL_ALIASES[index]}"
done

OVERALL_RC=0
for pid in "${MODEL_PIDS[@]}"; do
  set +e
  wait "${pid}"
  status=$?
  set -e
  unset 'ACTIVE_PIDS['"${pid}"']'
  if [[ "${status}" -ne 0 ]]; then
    echo "[run-ppl-models] FAILED model=${PID_MODEL[${pid}]} rc=${status}" >&2
    OVERALL_RC=1
  else
    echo "[run-ppl-models] done model=${PID_MODEL[${pid}]}"
  fi
done

if [[ "${OVERALL_RC}" -ne 0 ]]; then
  echo "[run-ppl-models] one or more model matrices failed" >&2
  exit "${OVERALL_RC}"
fi
echo "[run-ppl-models] all model matrices completed successfully"
