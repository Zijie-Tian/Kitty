# LongBench Evaluation

`scripts/run_exp.sh` is the **sole entry point** for LongBench in this repo. Do
not launch LongBench through any other script or by calling
`kitty_sim.cli.eval_longbench` directly. The script owns:

- local JSONL data under `<data-root>/data/{dataset}.jsonl`
- LUTAttn-compatible prompt templates and generation lengths
- a deterministic, smoke/full-separated output layout
- strict scoring to `<pred-dir>/result.json`

It sources the repo-root `.env` automatically (via
`accuracy_simulation/env.sh`), so `KITTY_PYTHON_BIN`, `KITTY_*_PATH`, and
`LONGBENCH_DATA_ROOT` are picked up without any manual `source`.

## Output layout (deterministic, smoke vs full)

The layout is decided by the sample count, and the two never mix:

- **full** (no `--max-samples`, or `--max-samples -1`):
  `longbench_out/<model>_<method>/{pred,logs}`
- **smoke** (`--max-samples N`, N > 0):
  `longbench_out/smoke/<model>_<method>/{pred,logs}`

Within each base directory:

- `pred/` holds the predictions: `<dataset>.jsonl`, `<dataset>.manifest.json`,
  and the scored `result.json`.
- `logs/` holds the per-dataset run metadata: `report_<dataset>.json`.

`<model>` and `<method>` are the slugs from
`src/kitty_sim/longbench/runner.py` (`model_layout_slug` / `method_layout_slug`)
joined by an underscore, e.g. `llama31-8b-instruct_kitty`,
`qwen3-8b_quest-kitty`, `glm4-9b-chat-1m_kivi-star-2`.

To force the layout independently of the sample count (e.g. a few-sample
correctness check that should still land in the full tree), set
`RUN_MODE=smoke|full`.

## GPU selection

Pass `--gpu N` (or set the per-target `*_GPU`). When the GPU is `1`, the script
adds `--require-gpu1` so the run aborts unless `CUDA_VISIBLE_DEVICES=1`. The
default `all` target fans out llama/qwen/glm across GPU 0/1/2; use `SERIAL=1`
to run them one at a time on a single GPU.

## Local paths via `.env`

Do not hardcode host-local model, Python, or dataset paths in tracked files.
Copy `.env.example` to `.env` and fill in local values there:

```bash
cp .env.example .env
```

Per-target overrides (`LLAMA32_MODEL_PATH`, `QWEN_MODEL_PATH`, ...) and the
shared `DATA_ROOT` still take precedence over `.env` values.

## Smoke example: LLaMA 3.2 1B, all subtasks, 2 samples each (GPU1)

```bash
bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2
```

This writes under `longbench_out/smoke/llama32-1b-instruct_quest-kitty/{pred,logs}`.

## Full example: paper-style Kitty, LLaMA 3.1 8B (GPU1)

```bash
bash scripts/run_exp.sh llama --gpu 1
# -> longbench_out/llama31-8b-instruct_kitty/{pred,logs}
```

## Targets and variants

Targets: `llama` (LLaMA3.1-8B), `llama32` (LLaMA3.2-1B), `qwen` (Qwen3-8B),
`glm` (GLM-4-9B-Chat-1M), `deepseek` (R1-Distill-Llama-8B), and `all`.

The variant defaults per target (`llama32` → `quest_kitty_page16_sim`, others → `kitty`)
and is overridden with `--variant`:

- `fp16`: HuggingFace default KV cache.
- `kitty`: paper-style Kitty, K2V2 with 12.5% Key channels promoted to INT4 (`sink=32`, `buffer=128`, `group=128`).
- `quest_kitty_page16_sim`: pure-PyTorch (no-Triton) QUEST + Kitty. The sim KittyKVCache applies page16 fake-quant (`sink=32`, `buffer=16`, `group=16`) and a per-arch attention hook runs the gather-based QUEST oracle on decode (budget 2048 → 128 pages). Architecture-portable accuracy + relative-timing proxy; not a kernel-speed proof.
- `quest_kitty_page16_kernel`: real Triton QUEST + Kitty decode kernel (Llama/Qwen only). The genuine speed path.
- `kitty_pro`: K2V2 with 25% Key channels promoted to INT4.
- `kivi_2`: K2V2 without sink or promoted channels.
- `kivi_star_2`: K2V2 with first 32 sink tokens kept in full precision, no promoted channels.
- `custom`: use CLI Kitty parameters directly.

## Scoping datasets

By default all 21 LongBench datasets run. Restrict the set with `DATASETS_CSV`:

```bash
DATASETS_CSV=trec,samsum bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2
```

## Completeness checks

Every dataset writes a sibling `<dataset>.manifest.json` with expected and
actual row counts. Scoring is strict: incomplete predictions create
`result.partial.json` and fail instead of silently writing `result.json`. A
dataset whose `pred/<dataset>.jsonl` already has the expected row count is
skipped; a partial file is deleted and rerun. The script refuses to shrink an
output that already has **more** rows than the current target (e.g. a full
result when you ask for `--max-samples 2`) unless you pass `FORCE=1`.
