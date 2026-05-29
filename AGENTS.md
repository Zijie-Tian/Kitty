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

## True QUEST + Kitty page16 kernel usage

The true QUEST + Kitty path is the real Qwen3 Kitty decode path with 16-token
pages, query-aware page selection, and Triton sparse QK/SV kernels. It is not
the same as the `kitty_sim` fake proxy.

Naming rules:

- `quest_proxy_kitty_page16` is a fake-quant/dense LongBench accuracy proxy.
  Do not use it for kernel-speed claims.
- A real `quest+kitty` / `quest_kitty_page16_kernel` result must show
  `last_quest_path` values like `triton_sparse_reduced_budget` or
  `triton_sparse_forced_all_pages`.
- `python_sparse_debug` is an internal correctness/debug path only and does not
  count as true QUEST kernel evidence.

Default true QUEST settings:

```text
page_size=16
promote_ratio=0.125
quest_enabled=True
quest_token_budget=2048  # default when no explicit QUEST budget is supplied
quest_skip_layers=0      # default: every decode layer uses QUEST sparse selection
```

`quest_skip_layers=0` is the current default: all decode layers use query-aware
QUEST page selection. Older builds defaulted to `2` (the QUEST-paper convention
of keeping the first two layers dense). The skip fallback is still available via
`--quest-skip-layers N` / `quest_skip_layers=N`, but each skipped layer runs
dense full attention with an O(context) cost, and the fallback has not been
accuracy-validated in this repo.

Budget mapping for page16:

| QUEST token budget | Selected logical pages |
| ---: | ---: |
| 512 | 32 |
| 1024 | 64 |
| 2048 | 128 |

### GPU decode speed gate

Use this command shape to prove the real QUEST + Kitty implementation on a
32k input. It compares pure Kitty page16 against QUEST + Kitty page16 using
only decode-token timing; prefill is excluded from the reported ms/token.

Note: the benchmark loads the model with `attn_implementation="flash_attention_2"`,
which is used only for prefill — decode runs the Triton Kitty kernel, so the
reported decode ms/token is independent of the prefill backend. The documented
`kitty` conda env does not ship `flash_attn`; either install it, or override the
prefill backend to `sdpa` (decode numbers are unchanged).

Default GPU1 command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python latency_benchmarking/benchmark_kitty.py \
  --model /mnt/data/tzj/models/Qwen3-8B \
  --cache_implementation 0 \
  --page_size 16 \
  --promote_ratio 0.125 \
  --max_seq_len 32768 \
  --max_new_tokens 32 \
  --batch_size 1 \
  --warmup_runs 1 \
  --repeat_runs 3 \
  --compare-quest-kitty \
  --quest-enabled \
  --quest-token-budget 2048 \
  --quest-skip-layers 0
```

Use GPU0 only when the user explicitly overrides the GPU1-only evaluation rule:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python latency_benchmarking/benchmark_kitty.py \
  --model /mnt/data/tzj/models/Qwen3-8B \
  --cache_implementation 0 \
  --page_size 16 \
  --promote_ratio 0.125 \
  --max_seq_len 32768 \
  --max_new_tokens 32 \
  --batch_size 1 \
  --warmup_runs 1 \
  --repeat_runs 3 \
  --compare-quest-kitty \
  --quest-enabled \
  --quest-token-budget 2048 \
  --quest-skip-layers 0
```

Expected evidence in output:

```text
paths={'triton_sparse_reduced_budget': <eligible decode layer-steps>}
selected_pages=128
selected_tokens=2048
decode_speedup=<pure Kitty page16 ms/token / QUEST+Kitty ms/token>
```

With `quest_skip_layers=0` there should be no `'dense'` skip-layer entries; a
`'dense_full_budget'` entry only appears when the context has fewer logical
pages than the budget (nothing to drop). A real implementation should be
materially faster than pure Kitty page16 on a long decode (e.g. roughly 3x at
16k and ~20x at 128k with budget 2048). If `decode_speedup` is unexpectedly low
(e.g. `< 1.5` at 32k+), first suspect dense fallback, Python debug fallback,
selector-side all-page dequantization, or unselected pages still being loaded by
the sparse kernels.

Recent local GPU0 sample evidence with Qwen3-8B, page16, `quest_token_budget=2048`,
`quest_skip_layers=0`, `max_new_tokens=32`, `warmup_runs=1`, `repeat_runs=2`,
decode-only ms/token across context lengths:

| Context | QUEST ms/token | QUEST tok/s | Shared pages |
| ---: | ---: | ---: | ---: |
| 8k | 40.41 | 24.75 | ~509 |
| 16k | 41.47 | 24.11 | ~1021 |
| 32k | 42.84 | 23.34 | ~2045 |
| 64k | 45.16 | 22.14 | ~4093 |
| 96k | 47.15 | 21.21 | ~6141 |
| 128k | 49.18 | 20.33 | ~8189 |

With `quest_skip_layers=0` the decode cost is nearly flat in context length
(linear fit roughly `40.3 ms + 0.07 ms per 1k context tokens`); the small
residual is the O(context) page-selection scan over all logical pages, not the
budget-bound sparse attention. For reference, pure Kitty page16 decode is about
`139.5 ms/token` at 16k and about `1013 ms/token` at 128k, so QUEST+Kitty decode
speedup grows with context (about 3.4x at 16k and about 20x at 128k).

Treat these as environment-specific smoke numbers, not a formal benchmark.

### GPU0-only correctness tests for true sparse kernels

When a task explicitly says to restrict this QUEST + Kitty work to GPU0, use:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m unittest tests.test_kitty_quest_sparse -v
```

These tests require `CUDA_VISIBLE_DEVICES=0` and exactly one visible CUDA
device. They assert true Triton sparse path labels, budget behavior, dense
full-budget fallback behavior, and that `python_sparse_debug` is not accepted as
real kernel evidence. The documented `kitty` conda env has no `pytest`, so use
`unittest` (or run from an env that provides `pytest`).

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

Use the Kitty-native LongBench runner added in this checkout. Unless a task explicitly asks for a shorter smoke/proxy run, LongBench runs in this checkout must use `MAX_MODEL_LEN=32768` (32k context) and `MAX_GEN=256`. Do not use the old `MAX_MODEL_LEN=3500` default for full LongBench commands.

### Normalized LongBench output layout

- Do not write new LongBench predictions under `longbench_out/pred`.
- Smoke tests must write under `longbench_out/smoke/<model>-<method>/pred`.
- Full/non-smoke runs must write directly under
  `longbench_out/<model>-<method>/pred`.
- The `pred` directory is the prediction directory itself: write
  `<dataset>.jsonl`, `<dataset>.manifest.json`, and `result.json` directly
  inside it. Do not create an extra `<model-tag>-<variant>/` subdirectory under
  `pred`.
- For the QUEST-aligned Kitty LongBench proxy, use method name `quest-kitty`
  and variant `kitty_page16`; for paper-style Kitty, use method name `kitty`
  and variant `kitty`.

Full QUEST-aligned Kitty LongBench on GPU1 for LLaMA3.1-8B-Instruct:

```bash
GPU_IDS_CSV=1 \
MODEL=meta-llama/Llama-3.1-8B-Instruct \
MODEL_TAG=llama31-8b-instruct-gpu1-quest-kitty-full-32k \
MODEL_FAMILY=llama3 \
VARIANTS_CSV=kitty_page16 \
MAX_SAMPLES=-1 \
MAX_MODEL_LEN=32768 \
MAX_GEN=256 \
LOCAL_FILES_ONLY=1 \
OVERWRITE=1 \
bash accuracy_simulation/run_longbench.sh
```

With the normalized default layout, that command writes directly to:

```text
longbench_out/llama31-8b-instruct-quest-kitty/pred
```

For Qwen3-8B full LongBench on GPU1, use `MODEL=Qwen/Qwen3-8B`,
`MODEL_FAMILY=qwen`, and an appropriate Qwen model tag. Paper-style Kitty
outputs should go directly under `longbench_out/qwen3-8b-kitty/pred`; local
model paths should come from `.env` (`KITTY_QWEN3_8B_PATH`) or an explicit
`MODEL_PATH` environment variable.

Smoke example layout:

```text
longbench_out/smoke/llama31-8b-instruct-quest-kitty/pred
```

## 32k memory probe evidence

A GPU1 probe with Qwen3-8B + paper-style Kitty, 32768-token input, and `max_new_tokens=1` completed without OOM.

Observed peak:

- `nvidia-smi` peak memory: about `25947 MiB` / `25.34 GiB`.
- PyTorch peak allocated: about `23.03 GiB`.
- PyTorch peak reserved: about `24.83 GiB`.

Treat these as environment-specific smoke numbers, not a formal benchmark.
