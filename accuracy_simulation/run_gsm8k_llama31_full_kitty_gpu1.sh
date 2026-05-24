#!/usr/bin/env bash
set -euo pipefail

# Reproduce only GSM8K full/FP16 and Kitty results for LLaMA3.1-8B-Instruct.
# Hard restriction: use physical GPU1 only. Inside the process it appears as cuda:0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=accuracy_simulation/env.sh
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-kitty}"
MODEL_PATH="${MODEL_PATH:-${KITTY_LLAMA31_8B_PATH:-}}"
TASK_NAME="${TASK_NAME:-gsm8k_cot_llama}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/eval_results_gsm8k_gpu1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval_logs_gsm8k_gpu1}"
NUM_REPEATS="${NUM_REPEATS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
# Paper Table 3 label "Kitty" uses 12.5% promoted key-cache channels.
# Override PROMOTE_RATIO=0.25 if you intentionally want Kitty-Pro.
PROMOTE_RATIO="${PROMOTE_RATIO:-0.125}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TORCH_CUDA_ARCH_LIST="8.0"

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Conda env: %s\n' "${CONDA_ENV}"
printf 'Model path: %s\n' "${MODEL_PATH}"
printf 'Task: %s\n' "${TASK_NAME}"
printf 'Physical GPU restriction: CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Results dir: %s\n' "${RESULTS_DIR}"
printf 'Logs dir: %s\n' "${LOG_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is unset or does not exist: ${MODEL_PATH:-<empty>}" >&2
  echo "Set MODEL_PATH or KITTY_LLAMA31_8B_PATH in the ignored .env file." >&2
  exit 2
fi

conda run --no-capture-output -n "${CONDA_ENV}" python -c "import os, torch; print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch.cuda.device_count() =', torch.cuda.device_count()); assert torch.cuda.device_count() == 1, 'Expected exactly one visible CUDA device after CUDA_VISIBLE_DEVICES=1'; print('visible cuda:0 name =', torch.cuda.get_device_name(0))"

run_eval() {
  local label="$1"
  shift
  local log_file="${LOG_DIR}/${label}.log"
  echo "================================================================"
  echo "Running ${label}; log: ${log_file}"
  echo "================================================================"
  conda run --no-capture-output -n "${CONDA_ENV}" "$@" 2>&1 | tee "${log_file}"
}

# Full / FP16 baseline only.
run_eval "llama31_gsm8k_full_fp16_gpu1" \
  eval_kitty "${MODEL_PATH}" \
    --task "${TASK_NAME}" \
    --num_repeats "${NUM_REPEATS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --results_dir "${RESULTS_DIR}"

# Kitty only: no KIVI-2 or KIVI*-2 runs.
run_eval "llama31_gsm8k_kitty_gpu1" \
  eval_kitty "${MODEL_PATH}" \
    --task "${TASK_NAME}" \
    --eval_kitty \
    --sink_length 32 \
    --buffer_length 128 \
    --group_size 128 \
    --kbits 2 \
    --vbits 2 \
    --promote_bit 4 \
    --promote_ratio "${PROMOTE_RATIO}" \
    --channel_selection 1 \
    --num_repeats "${NUM_REPEATS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --results_dir "${RESULTS_DIR}"

echo "Done. Summary files:"
find "${RESULTS_DIR}" -name '*summary.json' -print | sort
