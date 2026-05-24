#!/usr/bin/env bash
# Load repo-local environment defaults from an ignored `.env` file.
#
# This file is sourced by scheduler scripts. Existing environment variables
# (for example `GPU_IDS_CSV=1 bash ...`) intentionally take precedence over
# values from `.env`.

if [[ -z "${REPO_ROOT:-}" ]]; then
  _kitty_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_kitty_env_script_dir}/.." && pwd)"
fi

_kitty_env_file="${KITTY_ENV_FILE:-${REPO_ROOT}/.env}"

if [[ -f "${_kitty_env_file}" ]]; then
  _kitty_env_preserve_vars=(
    PYTHON_BIN CONDA_ENV
    MODEL MODEL_PATH MODEL_TAG MODEL_FAMILY
    DATA_ROOT LONGBENCH_DATA_ROOT OUTPUT_DIR LOG_DIR
    MAX_SAMPLES MAX_MODEL_LEN MAX_GEN PROMPT_TOKEN_RESERVE TORCH_DTYPE
    LOCAL_FILES_ONLY OVERWRITE STRICT_COMPLETE VARIANTS_CSV DATASETS_CSV
    GPU_IDS_CSV GPU_ID
    TASK_NAME RESULTS_DIR NUM_REPEATS BATCH_SIZE MAX_NEW_TOKENS PROMOTE_RATIO
    KITTY_PYTHON_BIN KITTY_CONDA_ENV KITTY_LLAMA31_8B_PATH KITTY_QWEN3_8B_PATH
  )
  _kitty_env_saved_names=()
  for _kitty_env_var in "${_kitty_env_preserve_vars[@]}"; do
    if [[ -v ${_kitty_env_var} ]]; then
      _kitty_env_saved_name="_kitty_env_saved_${_kitty_env_var}"
      printf -v "${_kitty_env_saved_name}" '%s' "${!_kitty_env_var}"
      _kitty_env_saved_names+=("${_kitty_env_var}")
    fi
  done

  _kitty_env_restore_nounset=0
  if [[ $- == *u* ]]; then
    _kitty_env_restore_nounset=1
    set +u
  fi
  set -a
  # shellcheck source=/dev/null
  source "${_kitty_env_file}"
  set +a
  if [[ "${_kitty_env_restore_nounset}" == "1" ]]; then
    set -u
  fi

  for _kitty_env_var in "${_kitty_env_saved_names[@]}"; do
    _kitty_env_saved_name="_kitty_env_saved_${_kitty_env_var}"
    printf -v "${_kitty_env_var}" '%s' "${!_kitty_env_saved_name}"
    export "${_kitty_env_var}"
    unset "${_kitty_env_saved_name}"
  done
fi

if [[ -z "${PYTHON_BIN:-}" && -n "${KITTY_PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN="${KITTY_PYTHON_BIN}"
fi
if [[ -z "${CONDA_ENV:-}" && -n "${KITTY_CONDA_ENV:-}" ]]; then
  export CONDA_ENV="${KITTY_CONDA_ENV}"
fi

unset _kitty_env_file _kitty_env_preserve_vars _kitty_env_saved_names \
  _kitty_env_var _kitty_env_saved_name _kitty_env_restore_nounset \
  _kitty_env_script_dir
