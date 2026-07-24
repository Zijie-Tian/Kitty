#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "[autoresearch] failed at line ${LINENO} (exit ${status})" >&2; exit "${status}"' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Load ignored, host-local model/data defaults without overriding explicit values.
# shellcheck source=accuracy_simulation/env.sh
source "${REPO_ROOT}/accuracy_simulation/env.sh"

PYTHON_BIN="${PYTHON_BIN:-${KITTY_PYTHON_BIN:-}}"
MODEL_PATH="${LLAMA32_MODEL_PATH:-${KITTY_LLAMA32_1B_PATH:-}}"
DATA_ROOT="${DATA_ROOT:-${LONGBENCH_DATA_ROOT:-${HOME}/data/LongBench}}"
CANONICAL_MASK="${QLUT_CB_MASK:-${KITTY_LLAMA32_1B_QLUTATTN_MASK:-}}"
CANDIDATE_MASK="${REPO_ROOT}/autoresearch_current_mask.pt"
if [[ -f "${CANDIDATE_MASK}" ]]; then
  MASK_PATH="${CANDIDATE_MASK}"
else
  MASK_PATH="${CANONICAL_MASK}"
fi

[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: set KITTY_PYTHON_BIN in .env to an executable Python" >&2
  exit 2
}
[[ -n "${MODEL_PATH}" && -d "${MODEL_PATH}" ]] || {
  echo "ERROR: set KITTY_LLAMA32_1B_PATH in .env to the local Llama-3.2-1B checkpoint" >&2
  exit 2
}
[[ -f "${MASK_PATH}" ]] || {
  echo "ERROR: no QLUTATTN mask found; set QLUT_CB_MASK or KITTY_LLAMA32_1B_QLUTATTN_MASK" >&2
  exit 2
}
for dataset in qasper trec; do
  [[ -f "${DATA_ROOT}/data/${dataset}.jsonl" ]] || {
    echo "ERROR: missing LongBench dataset ${DATA_ROOT}/data/${dataset}.jsonl" >&2
    exit 2
  }
done

# Fixed workload: first 20 rows of qasper and trec, greedily decoded at the
# production 32K context cap. The two datasets run concurrently on GPUs 0 and 1.
readonly VARIANT="qlutattn"
readonly MODEL_SLUG="llama32-1b-autoresearch"
readonly DATASETS="qasper,trec"
readonly SAMPLE_COUNT=20

export PYTHON_BIN DATA_ROOT
export LLAMA32_MODEL_PATH="${MODEL_PATH}"
export LLAMA32_MODEL_SLUG="${MODEL_SLUG}"
export LLAMA32_MAX_GEN=256
export QLUT_CB_MASK="${MASK_PATH}"
export MAX_MODEL_LEN=32768
export DATASETS_CSV="${DATASETS}"
export RUN_MODE=smoke
export MAX_SAMPLES="${SAMPLE_COUNT}"
export LOCAL_FILES_ONLY=1
export PYTHONHASHSEED=0

# Prevent caller state from silently changing the fixed QLUTATTN algorithm.
unset KBITS VBITS PROMOTE_BIT PROMOTE_RATIO PROMOTE_RATIO_CONFIG
unset QUEST_KERNEL QUEST_TRITON SIM_QUEST QUEST_SIM QUEST_TOKEN_BUDGET QUEST_BUDGET QUEST_SKIP_LAYERS

METHOD_SLUG="$(bash scripts/run_exp.sh llama32 --variant "${VARIANT}" --print-method-slug)"
bash scripts/run_exp.sh llama32 --gpus 0,1 --variant "${VARIANT}" --max-samples "${SAMPLE_COUNT}"

PRED_DIR="${REPO_ROOT}/longbench_out/smoke/${MODEL_SLUG}_${METHOD_SLUG}/pred"
RESULT_PATH="${PRED_DIR}/result.json"
[[ -f "${RESULT_PATH}" ]] || {
  echo "ERROR: benchmark did not produce ${RESULT_PATH}" >&2
  exit 3
}

"${PYTHON_BIN}" - "${PRED_DIR}" "${SAMPLE_COUNT}" <<'PY'
import json
import math
import sys
from pathlib import Path

pred_dir = Path(sys.argv[1])
expected_samples = int(sys.argv[2])
datasets = ("qasper", "trec")

result = json.loads((pred_dir / "result.json").read_text(encoding="utf-8"))
if set(result) != set(datasets):
    raise SystemExit(
        f"result datasets mismatch: expected {sorted(datasets)}, got {sorted(result)}"
    )

scores = {}
for dataset in datasets:
    value = result[dataset]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{dataset} score is not numeric: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"{dataset} score is not finite: {value!r}")
    scores[dataset] = value

    manifest_path = pred_dir / f"{dataset}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise SystemExit(f"{dataset} manifest status is not ok")
    if manifest.get("expected_samples") != expected_samples:
        raise SystemExit(
            f"{dataset} expected_samples={manifest.get('expected_samples')} != {expected_samples}"
        )
    if manifest.get("written_samples") != expected_samples:
        raise SystemExit(
            f"{dataset} written_samples={manifest.get('written_samples')} != {expected_samples}"
        )
    variant = manifest.get("variant") or {}
    if variant.get("k_codebook") != "qlut":
        raise SystemExit(f"{dataset} did not use the QLUT K path: {variant!r}")
    if not manifest.get("mask_sha256") or not manifest.get("run_config_hash"):
        raise SystemExit(f"{dataset} lacks mask/run provenance")
    engagement = manifest.get("engagement") or {}
    if (
        engagement.get("last_v_quant_mode") != "tile16_rescued"
        or int(engagement.get("v_tile_blocks", 0)) <= 0
        or int(engagement.get("v_quantized_tokens", 0)) <= 0
    ):
        raise SystemExit(f"{dataset} QLUTATTN V path did not engage: {engagement!r}")

mean_score = sum(scores.values()) / len(scores)
print(f"METRIC focused_longbench_mean={mean_score:.8f}")
print(f"METRIC qasper={scores['qasper']:.8f}")
print(f"METRIC trec={scores['trec']:.8f}")
PY
