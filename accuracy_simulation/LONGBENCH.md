# LongBench Evaluation (GPU1-only)

This repo provides a Kitty-native LongBench path that mirrors the LUTAttn LongBench flow:

- local JSONL data under `<data-root>/data/{dataset}.jsonl`
- LUTAttn-compatible prompt templates and generation lengths
- LUTAttn-compatible output files: `longbench_out/pred/<model-tag>/<dataset>.jsonl`
- strict scoring to `longbench_out/pred/<model-tag>/result.json`

## Hard GPU constraint

For this workflow, all evaluation and smoke commands are **physical GPU1 only**. The scheduler validates this and launches Python with `CUDA_VISIBLE_DEVICES=1`. Do not use GPU0 or multi-GPU examples for this workflow.

## Local paths via `.env`

Do not hardcode host-local model, Python, or dataset paths in tracked files. Copy `.env.example` to `.env` and fill in local values there:

```bash
cp .env.example .env
```

The scheduler explicitly sources the ignored `.env` file. Existing command-line environment variables still take precedence over `.env` values.

Tracked LongBench config does not contain a model-to-local-path map. Use `MODEL`
for a Hugging Face id and use `MODEL_PATH` or `.env` variables such as
`KITTY_QWEN3_8B_PATH` only for local filesystem paths.

## Smoke example: all LongBench subtasks, 2 samples each

```bash
GPU_IDS_CSV=1 \
MODEL="Qwen/Qwen3-8B" \
MODEL_TAG="qwen3-8b-gpu1-smoke2" \
MODEL_FAMILY="qwen" \
VARIANTS_CSV="kitty" \
MAX_SAMPLES=2 \
MAX_MODEL_LEN=3500 \
LOCAL_FILES_ONLY=1 \
OVERWRITE=1 \
bash accuracy_simulation/run_longbench.sh
```

Use `VARIANTS_CSV="fp16,kitty"` only when you explicitly want both baseline and Kitty smoke runs on GPU1; this doubles runtime.

## Variants

- `fp16`: HuggingFace default KV cache.
- `kitty`: paper-style Kitty, K2V2 with 12.5% Key channels promoted to INT4 (`sink=32`, `buffer=128`, `group=128`).
- `kitty_pro`: K2V2 with 25% Key channels promoted to INT4.
- `kivi_2`: K2V2 without sink or promoted channels.
- `kivi_star_2`: K2V2 with first 32 sink tokens kept in full precision, no promoted channels.
- `custom`: use CLI Kitty parameters directly.

## Direct single-dataset command

```bash
set -a
source .env
set +a
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src "${KITTY_PYTHON_BIN:-python}" -m kitty_sim.cli.eval_longbench \
  "${KITTY_QWEN3_8B_PATH:-Qwen/Qwen3-8B}" \
  --model-tag qwen3-8b-gpu1-smoke2 \
  --model-family qwen \
  --variant kitty \
  --dataset trec \
  --data-root "${LONGBENCH_DATA_ROOT}" \
  --max-samples 2 \
  --max-model-len 3500 \
  --require-gpu1 \
  --overwrite
```

Then score:

```bash
PYTHONPATH=src "${KITTY_PYTHON_BIN:-python}" -m kitty_sim.cli.score_longbench \
  --model longbench_out/pred/qwen3-8b-gpu1-smoke2-kitty_g128_b128_s32_sel1_k2_v2_pb4_pr0p125
```

## Completeness checks

Every dataset writes a sibling `<dataset>.manifest.json` with expected and actual row counts. The scorer is strict by default: incomplete outputs create `result.partial.json` and fail instead of silently writing `result.json`.
