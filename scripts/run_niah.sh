#!/usr/bin/env bash
# RULER-NIAH driver for the sign/SNF x PT2/vtile16 V2 study (+ fp16 ceiling).
#
# Serial over variants on ONE GPU; each variant loads the model once and loops
# tasks x lengths internally, then the pred dir is scored and depth x length
# heatmaps are rendered into <arm>/logs/.
#
# Usage:
#   bash scripts/run_niah.sh --gpu 0 --variants fp16,qlutattn_k125v2_pt,qlutattn_k125v2_pt_vtile16,qlutattn_k188v2_pt,qlutattn_k188v2_pt_vtile16
#   bash scripts/run_niah.sh --gpu 0 --variants fp16 --max-samples 2          # smoke
#
# Env (explicit env/CLI > .env):
#   LLAMA32_MODEL_PATH  model checkpoint (default ~/models/Llama-3.2-1B-Instruct)
#   MODEL_SLUG          output <model> segment (default llama32-1b-instruct)
#   NIAH_DATA_ROOT      data root from scripts/prepare_niah_data.sh
#   QLUT_CB_MASK        sign,nf2 f=0.5 mask (required by k188* arms)
#   V_TILE_CHANNELS_CLI is not used here; pass --v-tile-channels (default 64 for *_vtile16)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/accuracy_simulation/env.sh"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU_ID:-1}"   # repo default is the GPU1-only rule; override with --gpu
VARIANTS="fp16,qlutattn_k125v2_pt,qlutattn_k125v2_pt_vtile16,qlutattn_k188v2_pt,qlutattn_k188v2_pt_vtile16"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1}"
LENS="${LENS:-4096,8192,16384,32768}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
V_TILE_C="${V_TILE_C:-64}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_PATH="${LLAMA32_MODEL_PATH:-${KITTY_LLAMA32_1B_PATH:-${HOME}/models/Llama-3.2-1B-Instruct}}"
MODEL_SLUG="${MODEL_SLUG:-llama32-1b-instruct}"
# Same family the LongBench vtile study used (llama3 = raw prompt, matches
# RULER's base template protocol).
MODEL_FAMILY="${MODEL_FAMILY:-llama3}"
NIAH_DATA_ROOT="${NIAH_DATA_ROOT:-${HOME}/data/ruler_niah/llama3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)             GPU="$2"; shift 2 ;;
    --variants)        VARIANTS="$2"; shift 2 ;;
    --variant)         VARIANTS="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --lens)            LENS="$2"; shift 2 ;;
    --max-samples)     MAX_SAMPLES="$2"; shift 2 ;;
    --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
    --v-tile-channels) V_TILE_C="$2"; shift 2 ;;
    *) echo "[run-niah] unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[run-niah] model path not found: ${MODEL_PATH} (set LLAMA32_MODEL_PATH)" >&2
  exit 1
fi

method_slug() {
  local variant="$1"
  case "${variant}" in
    fp16)                        echo "fp16" ;;
    qlutattn_k125v2_pt)          echo "qlutattn-k125v2-pt" ;;
    qlutattn_k188v2_pt)          echo "qlutattn-k188v2-pt" ;;
    qlutattn_k125v2_pt_vtile16)  echo "qlutattn-k125v2-pt-vtile16c${V_TILE_C}-rv1" ;;
    qlutattn_k188v2_pt_vtile16)  echo "qlutattn-k188v2-pt-vtile16c${V_TILE_C}-rv1" ;;
    llamacpp_q40)                echo "llamacpp-q40" ;;
    llamacpp_q40_star)           echo "llamacpp-q40-star" ;;
    *) echo "" ;;
  esac
}

layout_root="niah_out"
if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
  layout_root="niah_out/smoke"
fi

overall_rc=0
IFS=',' read -ra VARIANT_ARR <<< "${VARIANTS}"
for VARIANT in "${VARIANT_ARR[@]}"; do
  VARIANT="$(echo "${VARIANT}" | xargs)"
  slug="$(method_slug "${VARIANT}")"
  if [[ -z "${slug}" ]]; then
    echo "[run-niah] unsupported variant for this driver: ${VARIANT}" >&2
    exit 1
  fi
  arm_dir="${layout_root}/${MODEL_SLUG}_${slug}"
  pred_dir="${arm_dir}/pred"
  logs_dir="${arm_dir}/logs"
  mkdir -p "${pred_dir}" "${logs_dir}"
  if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
    # smoke is never resumed: always a clean slate
    rm -rf "${pred_dir}" && mkdir -p "${pred_dir}"
  fi

  # Per-arm env guards (mirrors the AGENTS.md sign/SNF V2 test method).
  declare -a env_kv=(
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "GPU_ID=${GPU}" "GPU_IDS_CSV=${GPU}"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}"
    "NIAH_DATA_ROOT=${NIAH_DATA_ROOT}"
    "PERTOKEN_BLOCK=1" "PROMOTE_RATIO_CONFIG=" "PROMPT_TOKEN_RESERVE=0"
    "QUEST_KERNEL=0" "QUEST_TRITON=0" "SIM_QUEST=0" "QUEST_SIM=0"
  )
  declare -a extra_args=()
  case "${VARIANT}" in
    fp16)
      env_kv+=("VBITS=" "QLUT_BIN_CODEBOOKS=" "QLUT_CB_MASK=" "V_TILE_CHANNELS=")
      ;;
    qlutattn_k125v2_pt)
      env_kv+=("VBITS=2" "QLUT_BIN_CODEBOOKS=sign" "QLUT_CB_MASK=" "V_TILE_CHANNELS=")
      ;;
    qlutattn_k125v2_pt_vtile16)
      env_kv+=("VBITS=2" "QLUT_BIN_CODEBOOKS=sign" "QLUT_CB_MASK=")
      extra_args+=(--v-tile-channels "${V_TILE_C}")
      ;;
    qlutattn_k188v2_pt)
      : "${QLUT_CB_MASK:?k188 arms need QLUT_CB_MASK=/path/to/sign-nf2 f0.5 mask}"
      env_kv+=("VBITS=2" "QLUT_BIN_CODEBOOKS=" "QLUT_CB_MASK=${QLUT_CB_MASK}" "V_TILE_CHANNELS=")
      ;;
    qlutattn_k188v2_pt_vtile16)
      : "${QLUT_CB_MASK:?k188 arms need QLUT_CB_MASK=/path/to/sign-nf2 f0.5 mask}"
      env_kv+=("VBITS=2" "QLUT_BIN_CODEBOOKS=" "QLUT_CB_MASK=${QLUT_CB_MASK}")
      extra_args+=(--v-tile-channels "${V_TILE_C}")
      ;;
    llamacpp_q40|llamacpp_q40_star)
      # Q4_0 fixes the bit-width in the codebook; KBITS/VBITS/QLUT_* are
      # ignored by the variant, so pass explicit-unset sentinels only.
      env_kv+=("VBITS=" "QLUT_BIN_CODEBOOKS=" "QLUT_CB_MASK=" "V_TILE_CHANNELS=")
      ;;
  esac

  echo "==================================================================="
  echo "[run-niah] variant=${VARIANT} gpu=${GPU} -> ${arm_dir}"
  echo "[run-niah] tasks=${TASKS} lens=${LENS} max_samples=${MAX_SAMPLES}"
  echo "==================================================================="
  set +e
  env "${env_kv[@]}" "${PYTHON_BIN}" -m kitty_sim.cli.eval_niah "${MODEL_ID}" \
    --model-path "${MODEL_PATH}" \
    --model-tag "${MODEL_SLUG}" \
    --model-family "${MODEL_FAMILY}" \
    --variant "${VARIANT}" \
    --tasks "${TASKS}" \
    --seq-lens "${LENS}" \
    --data-root "${NIAH_DATA_ROOT}" \
    --output-dir "${pred_dir}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --torch-dtype float16 \
    --local-files-only \
    --require-cuda-visible-devices "${GPU}" \
    --report-json "${logs_dir}/report.json" \
    "${extra_args[@]}" 2>&1 | tee "${logs_dir}/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "[run-niah] variant ${VARIANT} FAILED rc=${rc} (continuing to next)" >&2
    overall_rc=1
    continue
  fi

  env "PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" -m kitty_sim.cli.score_niah \
    "${pred_dir}" \
    --heatmap-dir "${logs_dir}" \
    --title "${MODEL_SLUG} ${slug}" 2>&1 | tee "${logs_dir}/score.log"
done

echo "[run-niah] all variants done (rc=${overall_rc})."
exit "${overall_rc}"
