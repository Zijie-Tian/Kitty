#!/usr/bin/env bash
# Calibrate WikiText-2 ρ=0.62 top-p reference masks for the M2 figure red line.
#
# Writes (gitignored under /outputs):
#   outputs/sensitivity_overlap/ref/llama32-1b_stats.pt
#   outputs/sensitivity_overlap/ref/llama32-1b_topp_p0p62.pt
#   outputs/sensitivity_overlap/ref/llama32-3b_stats.pt
#   outputs/sensitivity_overlap/ref/llama32-3b_topp_p0p62.pt
#
# Protocol matches the sensitivity_overlap experiment / prior paper figure:
#   num_samples=128, sample_len=2048, skip_first=32, group_size=128, seed=0, top_p=0.62
#
# Usage:
#   bash experiments/sensitivity_overlap/make_ref_masks.sh          # 1B then 3B on GPU1
#   MODELS=llama32-1b bash experiments/sensitivity_overlap/make_ref_masks.sh
#   GPU=1 bash experiments/sensitivity_overlap/make_ref_masks.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=accuracy_simulation/env.sh
source "${REPO_ROOT}/accuracy_simulation/env.sh"

GPU="${GPU:-1}"
MODELS="${MODELS:-llama32-1b,llama32-3b}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/sensitivity_overlap/ref}"
PY="${PYTHON_BIN:-${KITTY_PYTHON_BIN:-python}}"
WIKI="${KITTY_WIKITEXT2_TRAIN_PATH:-}"
mkdir -p "${OUT_DIR}"

if [[ -z "${WIKI}" || ! -f "${WIKI}" ]]; then
  echo "ERROR: set KITTY_WIKITEXT2_TRAIN_PATH in .env to the wikitext-2 train parquet." >&2
  exit 2
fi

calib_one() {
  local tag="$1" model="$2"
  local stats="${OUT_DIR}/${tag}_stats.pt"
  local mask="${OUT_DIR}/${tag}_topp_p0p62.pt"
  if [[ -z "${model}" || ! -d "${model}" ]]; then
    echo "ERROR: model path missing for ${tag}" >&2
    exit 2
  fi
  echo "[ref] calibrating ${tag} on physical GPU ${GPU} -> ${mask}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${REPO_ROOT}/src" PYTHONDONTWRITEBYTECODE=1 \
    "${PY}" "${REPO_ROOT}/scripts/calibrate_qlutattn_mask.py" \
    --model "${model}" \
    --calib-data "${WIKI}" \
    --num-samples 128 --sample-len 2048 \
    --skip-first 32 --group-size 128 --seed 0 \
    --top-p 0.62 \
    --stats-output "${stats}" \
    --output "${mask}"
  (cd "${OUT_DIR}" && sha256sum "$(basename "${stats}")" "$(basename "${mask}")" \
    > "SHA256SUMS_${tag}.txt")
  echo "[ref] done ${tag}"
}

IFS=',' read -r -a tags <<< "${MODELS}"
for tag in "${tags[@]}"; do
  tag="$(echo "${tag}" | tr -d '[:space:]')"
  [[ -z "${tag}" ]] && continue
  case "${tag}" in
    llama32-1b) calib_one llama32-1b "${KITTY_LLAMA32_1B_PATH:-}" ;;
    llama32-3b) calib_one llama32-3b "${KITTY_LLAMA32_3B_PATH:-}" ;;
    *)
      echo "ERROR: unknown model tag ${tag} (want llama32-1b or llama32-3b)" >&2
      exit 2
      ;;
  esac
done

echo "[ref] all finished under ${OUT_DIR}"
