#!/usr/bin/env bash
# Offline-first RULER data preparation wrapper.  This installs no packages.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# accuracy_simulation/env.sh predates the RULER variables.  Preserve explicit
# caller values across its optional .env load so CLI/env precedence stays intact.
declare -A _ruler_was_set=()
declare -A _ruler_saved=()
_ruler_preserve_vars=(
  TOKENIZER_PATH RULER_DATA_ROOT RULER_SOURCE_ROOT TASKS LENS LENGTHS
  NUM_SAMPLES MARGIN SEED DOWNLOAD_SOURCES
)
for _ruler_name in "${_ruler_preserve_vars[@]}"; do
  if [[ -v ${_ruler_name} ]]; then
    _ruler_was_set["${_ruler_name}"]=1
    _ruler_saved["${_ruler_name}"]="${!_ruler_name}"
  fi
done
# shellcheck source=/dev/null
source "${REPO_ROOT}/accuracy_simulation/env.sh"
for _ruler_name in "${_ruler_preserve_vars[@]}"; do
  if [[ "${_ruler_was_set[${_ruler_name}]:-0}" == "1" ]]; then
    printf -v "${_ruler_name}" '%s' "${_ruler_saved[${_ruler_name}]}"
    export "${_ruler_name}"
  fi
done
unset _ruler_was_set _ruler_saved _ruler_preserve_vars _ruler_name

cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${KITTY_LLAMA32_1B_PATH:-${HOME}/models/Llama-3.2-1B-Instruct}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
MODEL_FAMILY="${MODEL_FAMILY:-llama3}"
MODEL_TAG="${MODEL_TAG:-${MODEL_SLUG:-}}"
RULER_DATA_ROOT="${RULER_DATA_ROOT:-}"
RULER_SOURCE_ROOT="${RULER_SOURCE_ROOT:-${HOME}/data/ruler_sources}"
TASKS="${TASKS:-all}"
LENGTHS="${LENGTHS:-${LENS:-4096,8192,16384,32768}}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MARGIN="${MARGIN:-256}"
SEED="${SEED:-42}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/lm-evaluation-harness${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --model-path "${MODEL_PATH}"
  --model-family "${MODEL_FAMILY}"
  --source-root "${RULER_SOURCE_ROOT}"
  --tasks "${TASKS}"
  --lengths "${LENGTHS}"
  --num-samples "${NUM_SAMPLES}"
  --margin "${MARGIN}"
  --seed "${SEED}"
)
if [[ -n "${MODEL_TAG}" ]]; then
  args+=(--model-tag "${MODEL_TAG}")
fi
if [[ -n "${RULER_DATA_ROOT}" ]]; then
  args+=(--data-root "${RULER_DATA_ROOT}")
fi
if [[ -n "${TOKENIZER_PATH}" ]]; then
  args+=(--tokenizer-path "${TOKENIZER_PATH}")
fi

case "${LOCAL_FILES_ONLY:-1}" in
  1|true|TRUE|yes|YES|on|ON) args+=(--local-files-only) ;;
  0|false|FALSE|no|NO|off|OFF) args+=(--no-local-files-only) ;;
  *) echo "[prepare-ruler] LOCAL_FILES_ONLY must be a boolean" >&2; exit 2 ;;
esac
case "${FORCE:-0}" in
  1|true|TRUE|yes|YES|on|ON) args+=(--force) ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *) echo "[prepare-ruler] FORCE must be a boolean" >&2; exit 2 ;;
esac
case "${DOWNLOAD_SOURCES:-0}" in
  1|true|TRUE|yes|YES|on|ON) args+=(--download-sources) ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *) echo "[prepare-ruler] DOWNLOAD_SOURCES must be a boolean" >&2; exit 2 ;;
esac

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_ruler_data.py" "${args[@]}" "$@"
