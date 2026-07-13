#!/usr/bin/env bash
# Generate RULER-NIAH jsonl data for the Kitty NIAH evaluation.
#
# Runs the (user-forked) NVIDIA RULER data generator per task x context length.
# The generator needs `wonderwords` + `nltk` (punkt) which are NOT part of the
# core kitty stack; run this once on a machine that has them, then rsync the
# output NIAH_DATA_ROOT to the eval host. Data depends only on the *tokenizer*
# (Llama-3.2 1B/3B share it), so one generation serves the model family.
#
# Layout produced:  ${NIAH_DATA_ROOT}/<LEN>/<task>/validation.jsonl
# <LEN> is the NOMINAL eval context (4096, ...); generation uses LEN-MARGIN so
# the eval-time chat template can never push the prompt past max_model_len
# (the Kitty NIAH runner refuses to truncate).
#
# Usage:
#   bash scripts/prepare_niah_data.sh
#   TASKS="niah_single_2" LENS="4096" NUM_SAMPLES=2 bash scripts/prepare_niah_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/accuracy_simulation/env.sh"

RULER_REPO_ROOT="${RULER_REPO_ROOT:-${HOME}/Code/RULER}"
NIAH_DATA_ROOT="${NIAH_DATA_ROOT:-${HOME}/data/ruler_niah/llama3}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${KITTY_LLAMA32_1B_PATH:-${HOME}/models/Llama-3.2-1B-Instruct}}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
TASKS="${TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1}"
LENS="${LENS:-4096 8192 16384 32768}"
# Reserve for the eval-time chat template (+ answer prefix + safety).
MARGIN="${MARGIN:-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"

prepare_py="${RULER_REPO_ROOT}/scripts/data/prepare.py"
if [[ ! -f "${prepare_py}" ]]; then
  echo "[error] RULER prepare.py not found: ${prepare_py} (set RULER_REPO_ROOT)" >&2
  exit 1
fi
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "[error] tokenizer path not found: ${TOKENIZER_PATH} (set TOKENIZER_PATH)" >&2
  exit 1
fi

export NLTK_DATA="${NLTK_DATA:-${RULER_REPO_ROOT}/nltk_data}"

echo "[prepare-niah] RULER=${RULER_REPO_ROOT} tokenizer=${TOKENIZER_PATH}"
echo "[prepare-niah] out=${NIAH_DATA_ROOT} tasks='${TASKS}' lens='${LENS}' n=${NUM_SAMPLES} margin=${MARGIN}"

for LEN in ${LENS}; do
  gen_len=$((LEN - MARGIN))
  out_dir="${NIAH_DATA_ROOT}/${LEN}"
  mkdir -p "${out_dir}"
  for TASK in ${TASKS}; do
    target="${out_dir}/${TASK}/validation.jsonl"
    if [[ -f "${target}" ]]; then
      rows="$(grep -c . "${target}" || true)"
      if [[ "${rows}" == "${NUM_SAMPLES}" ]]; then
        echo "[skip] ${target} already has ${rows} rows"
        continue
      fi
      echo "[regen] ${target} has ${rows} rows, expected ${NUM_SAMPLES}"
      rm -f "${target}"
    fi
    echo "[gen] task=${TASK} len=${LEN} (gen_seq_len=${gen_len})"
    (cd "${RULER_REPO_ROOT}/scripts/data" && \
      "${PYTHON_BIN}" prepare.py \
        --save_dir "${out_dir}" \
        --benchmark synthetic \
        --task "${TASK}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --tokenizer_type hf \
        --max_seq_length "${gen_len}" \
        --model_template_type base \
        --num_samples "${NUM_SAMPLES}")
    if [[ ! -f "${target}" ]]; then
      echo "[error] generation did not produce ${target}" >&2
      exit 1
    fi
  done
done

echo "[prepare-niah] done."
find "${NIAH_DATA_ROOT}" -name validation.jsonl | sort
