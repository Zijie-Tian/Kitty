# Kitty Agent Notes

This file captures durable, repo-local guidance for agents working in this Kitty checkout. It should contain stable operational constraints and reproducible commands, not transient run logs.

## GPU1-only evaluation rule

When a task mentions the current GPU1-only reproduction/evaluation setup, restrict execution to physical GPU1.

Required pattern:

```bash
CUDA_VISIBLE_DEVICES=1
```

For scheduler scripts in this repo, prefer `GPU_IDS_CSV=1` and reject GPU0 or multi-GPU defaults unless the user explicitly changes the hardware constraint.

Host-local paths belong in the ignored repo-root `.env` file, not in tracked
docs, scripts, or source. Copy `.env.example` to `.env` and fill in the local
values there.

```bash
KITTY_CONDA_ENV=kitty
KITTY_PYTHON_BIN=python
KITTY_LLAMA31_8B_PATH=/path/to/Llama-3.1-8B-Instruct
KITTY_QWEN3_8B_PATH=/path/to/Qwen3-8B
LONGBENCH_DATA_ROOT=/path/to/LongBench
GPU_IDS_CSV=1
GPU_ID=1
```

### Path/config hygiene

- `.env` is intentionally ignored by git. Keep machine-specific Python, model,
  and dataset locations there only.
- `.env` is not automatically loaded by Python or the shell. The repo scheduler
  scripts source `accuracy_simulation/env.sh`, which loads repo-root `.env`.
  Direct Python commands must either source `.env` first or pass paths explicitly.
- Explicit command-line environment variables take precedence over `.env`
  values. Example: `MODEL_PATH=/some/model bash accuracy_simulation/run_longbench.sh`
  must override `.env`.
- `.env.example` is the only tracked template for local path variables. Keep it
  generic; never add host-specific paths to it.
- `src/kitty_sim/longbench/config/model2path.json` must not contain local model
  paths. It is intentionally an empty JSON object (`{}`); local models are
  selected via `MODEL_PATH`, `.env` variables such as `KITTY_QWEN3_8B_PATH`, or
  explicit CLI arguments.
- LongBench tracked config files under `src/kitty_sim/longbench/config/` are
  for portable benchmark metadata only:
  - `dataset2prompt.json`: LongBench prompt templates.
  - `dataset2maxlen.json`: generation length per dataset.
  - `model2maxlen.json`: optional context-length metadata by model id.
  - `model2path.json`: must remain free of local filesystem paths.
- The LongBench runner treats `MODEL` as a Hugging Face/Transformers model id
  unless `MODEL_PATH` or `--model-path` is provided. Do not reintroduce tracked
  model-to-local-path maps.

The `kitty` conda env has the compatible stack used here:

- Python 3.10
- PyTorch 2.4.1 + CUDA 12.1
- Transformers 4.53.2 from `third_party/transformers` branch `hf-4.53.2`
- lm-evaluation-harness from `third_party/lm-evaluation-harness` branch `kitty`
- editable Kitty install

## Kitty paper-style quantization defaults

Use these settings for the paper-style `Kitty` variant unless the user asks for a different variant:

```text
sink_length=32
buffer_length=128
group_size=128
kbits=2
vbits=2
promote_bit=4
promote_ratio=0.125
channel_selection=1  # magnitude-based Key-channel selection
```

Use `promote_ratio=0.25` only for an intentional Kitty-Pro run.

## GSM8K LLaMA3.1-8B-Instruct GPU1 reproduction

Set `MODEL_PATH` or `KITTY_LLAMA31_8B_PATH` in the ignored `.env` file.

Clean local GSM8K dataset path used by the lm-evaluation-harness task:

```text
data/gsm8k_local_clean
```

The GSM8K-only GPU1 script is:

```bash
bash accuracy_simulation/run_gsm8k_llama31_full_kitty_gpu1.sh
```

Properties expected from that script:

- Exports `CUDA_VISIBLE_DEVICES=1`.
- Verifies PyTorch sees exactly one CUDA device.
- Uses task `gsm8k_cot_llama` only.
- Runs exactly two evaluations: FP16/full baseline and paper-style Kitty.
- Does not run KIVI-2 or KIVI*-2.

Smoke tests previously performed with physical GPU1, local LLaMA3.1 model, local GSM8K data, `--debug`, `batch_size=1`, and `max_new_tokens=16` passed for both FP16/full and Kitty.

Monitoring commands for a background GSM8K run:

```bash
cat eval_logs_gsm8k_gpu1/full_run_driver.pid
ps -fp $(cat eval_logs_gsm8k_gpu1/full_run_driver.pid)
nvidia-smi -i 1
tail -f eval_logs_gsm8k_gpu1/full_run_driver.log
tail -f eval_logs_gsm8k_gpu1/llama31_gsm8k_full_fp16_gpu1.log
find eval_results_gsm8k_gpu1 -name '*summary.json' -print | sort
```

## LongBench GPU1 workflow

Use the Kitty-native LongBench runner added in this checkout. Full LongBench on GPU1 for LLaMA3.1-8B-Instruct:

```bash
GPU_IDS_CSV=1 \
MODEL=meta-llama/Llama-3.1-8B-Instruct \
MODEL_TAG=llama31-8b-instruct-gpu1-full \
MODEL_FAMILY=llama3 \
VARIANTS_CSV=kitty \
MAX_SAMPLES=-1 \
MAX_MODEL_LEN=3500 \
LOCAL_FILES_ONLY=1 \
OVERWRITE=1 \
bash accuracy_simulation/run_longbench.sh
```

For Qwen3-8B full LongBench on GPU1, use `MODEL=Qwen/Qwen3-8B`, `MODEL_FAMILY=qwen`, and an appropriate Qwen model tag. Local model paths should come from `.env` (`KITTY_QWEN3_8B_PATH`) or an explicit `MODEL_PATH` environment variable.

## LongBench smoke output naming convention

Example output directory:

```text
qwen3-8b-gpu1-smoke2-kitty_g128_b128_s32_sel1_k2_v2_pb4_pr0p125
```

Meaning:

- `qwen3-8b`: the evaluated model is Qwen3-8B.
- `gpu1`: the run is constrained to physical GPU1.
- `smoke2`: smoke test mode; each LongBench subtask runs 2 samples.
- `kitty`: the Kitty KV-cache quantization variant is used.
- `g128`: `group_size = 128`.
- `b128`: `buffer_length = 128`.
- `s32`: `sink_length = 32`.
- `sel1`: `channel_selection = 1`, meaning magnitude-based channel selection.
- `k2`: Key cache uses 2-bit quantization.
- `v2`: Value cache uses 2-bit quantization.
- `pb4`: promoted Key-cache channels use 4-bit precision.
- `pr0p125`: `promote_ratio = 0.125`, meaning 12.5% of Key-cache channels are promoted to 4-bit.

In short, this name means: LongBench smoke testing on GPU1 with 2 samples per subtask, using paper-style Kitty K2V2 (`group=128`, `buffer=128`, `sink=32`) and magnitude-based selection to promote 12.5% of Key-cache channels to INT4.

## 32k memory probe evidence

A GPU1 probe with Qwen3-8B + paper-style Kitty, 32768-token input, and `max_new_tokens=1` completed without OOM.

Observed peak:

- `nvidia-smi` peak memory: about `25947 MiB` / `25.34 GiB`.
- PyTorch peak allocated: about `23.03 GiB`.
- PyTorch peak reserved: about `24.83 GiB`.

Treat these as environment-specific smoke numbers, not a formal benchmark.
