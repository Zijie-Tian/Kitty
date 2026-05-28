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
KITTY_LLAMA32_1B_PATH=/path/to/Llama-3.2-1B-Instruct
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

The `kitty` conda env has the compatible stack used here. The detailed
package snapshot below was read from the active `kitty` env on 2026-05-25.
Re-check this table after upgrading packages, because cache APIs can change
across Transformers releases.

| Component | Observed version / detail |
| --- | --- |
| Python | `3.10.20` (`GCC 14.3.0`) |
| PyTorch | `torch==2.4.1+cu121` |
| PyTorch CUDA ABI | `torch.version.cuda == 12.1` |
| Triton | `triton==3.0.0` |
| Transformers | `transformers==4.57.6` |
| Tokenizers | `tokenizers==0.22.2` |
| Accelerate | `accelerate==1.13.0` |
| Datasets | `datasets==3.6.0` |
| Hugging Face Hub | `huggingface-hub==0.36.2` |
| Safetensors | `safetensors==0.7.0` |
| lm-evaluation-harness | `lm-eval==0.4.9.1` |
| Evaluate | `evaluate==0.4.6` |
| NumPy | `numpy==2.2.6` |
| SciPy | `scipy==1.15.3` |
| scikit-learn | `scikit-learn==1.7.2` |
| Pandas | `pandas==2.3.3` |
| PyArrow | `pyarrow==24.0.0` |
| tqdm | `tqdm==4.67.3` |
| einops | `einops==0.8.2` |
| rouge-score | `rouge-score==0.1.2` |
| NLTK | `nltk==3.9.4` |
| regex | `regex==2026.5.9` |
| requests | `requests==2.34.2` |
| filelock | `filelock==3.29.0` |
| packaging | `packaging==26.2` |
| psutil | `psutil==7.2.2` |
| Kitty package | `kitty==1.0.0` |
| torchvision | not installed |
| torchaudio | not installed |
| sentencepiece | not installed |
| protobuf | not installed |
| jieba | not installed |
| fuzzywuzzy | not installed |

Important compatibility note: this env currently uses Transformers `4.57.6`,
not the older `third_party/transformers` `hf-4.53.2` checkout. In this version,
`transformers.cache_utils.CacheConfig` is no longer importable, so Kitty
simulation code must keep its local compatibility shim unless the environment
is pinned back to an older Transformers API.

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

## Kitty page16 experiment configuration

Keep paper-style Kitty defaults at 128-token pages unless the task explicitly
asks for the QUEST-aligned page16 experiment. Page16 is opt-in and should be
reported as an experimental variant, not as the default Kitty setting.

Real Triton Kitty path:

```python
from kitty.kvcache import get_kvcache_kitty

kv_cache = get_kvcache_kitty(
    config,
    max_batch_size=max_batch_size,
    max_length=max_length,
    page_size=16,
)
```

Latency benchmark path:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python latency_benchmarking/benchmark_kitty.py \
  --cache_implementation 0 \
  --page_size 16 \
  --max_seq_len 4096 \
  --batch_size 1 \
  --warmup_runs 1 \
  --repeat_runs 2
```

LongBench fake-quant accuracy proxy:

```bash
GPU_IDS_CSV=1 \
VARIANTS_CSV=kitty_page16 \
MAX_MODEL_LEN=32768 \
MAX_GEN=256 \
LOCAL_FILES_ONLY=1 \
OVERWRITE=1 \
bash accuracy_simulation/run_longbench.sh
```

Important reporting caveat: `kitty_page16` in `kitty_sim` is a fake-quant
accuracy proxy (`sink=32`, `buffer=16`, `group=16`). It is useful for quick
accuracy smoke testing, but it is not by itself a proof of real Triton page16
accuracy. Real page16 correctness must be validated with GPU1 kernel/cache
smoke tests, and latency claims must come from the real Triton path.

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

Use the Kitty-native LongBench runner added in this checkout. Unless a task explicitly asks for a shorter smoke/proxy run, LongBench runs in this checkout must use `MAX_MODEL_LEN=32768` (32k context) and `MAX_GEN=256`. Do not use the old `MAX_MODEL_LEN=3500` default for full LongBench commands. LongBench outputs must use a flat path convention `longbench_out/pref/<model>-<tag>/`; do not nest under extra model-specific `.../pred/` subdirectories such as `longbench_out/<model>/pred/<model>-<tag>/`. Set each script's `*_OUTPUT_DIR=longbench_out/pref` and choose a unique `*_MODEL_TAG` / `*_MODEL_TAG_PREFIX` so the final directory is exactly one `<model>-<tag>` leaf. Full LongBench on GPU1 for LLaMA3.1-8B-Instruct:

```bash
GPU_IDS_CSV=1 \
MODEL=meta-llama/Llama-3.1-8B-Instruct \
MODEL_TAG=llama31-8b-instruct-gpu1-full \
MODEL_FAMILY=llama3 \
VARIANTS_CSV=kitty \
MAX_SAMPLES=-1 \
MAX_MODEL_LEN=32768 \
MAX_GEN=256 \
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
