# Kitty Agent Notes

This file captures durable, repo-local guidance for agents working in this Kitty checkout. It should contain stable operational constraints and reproducible commands, not transient run logs.

## Notion 笔记路由

When recording notes to Notion for this project, route by note type:

- **研究笔记 / Research notes** (方法调研 / 文献综述 / 结论与观察 / 技术报告) → database
  `📚 研究笔记 | Research Notes`, id `819846320d954eeea661638174068076`
  (data source `35fd3ce0-5640-4963-979f-4c1bad6d2639`).
- **实验测试笔记 / Experiment & test notes** (跑了哪些实验、数据表、复现命令、精度评测)
  → database `测试笔记 | Test Notes`, id `fa575373aeaa494d957a90fe07d5c8e0`
  (data source `ee4db7c4-cbd7-4b35-bd6e-fbc04a9c8310`).

Use the `ntn` CLI with `NOTION_KEYRING=0` (headless / file-based auth). Create a
page under a database via `parent.type=data_source_id`. Note: ntn currently can
NOT attach a `file_upload` to an image block — save figures locally and drag them
into Notion manually if embedding is needed.

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
  values. Example: `MAX_MODEL_LEN=4096 bash scripts/run_exp.sh llama32`
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

LongBench QUEST accuracy proxy (sole entry point is `scripts/run_exp.sh`;
a smoke run uses `--max-samples N`, a full run omits it):

```bash
bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2   # default variant: quest_kitty_page16_sim
```

The temporary `kitty_page16` / `quest_proxy_kitty_page16` fake-quant proxy
(dense KV quant, NO query-aware selection) has been REMOVED. There are now two
QUEST variants, both genuinely query-aware:

- `quest_kitty_page16_sim`: pure-PyTorch (no-Triton) QUEST + Kitty. The sim
  `KittyKVCache` applies page16 fake-quant (`sink=32`, `buffer=16`, `group=16`)
  and a per-arch attention hook (`kitty_sim/sim_quest.py`) runs the gather-based
  QUEST oracle (`kitty_sim/quest_sparse.py`) on decode (budget 2048 -> 128
  pages). Architecture-portable accuracy + relative-timing proxy; it does NOT
  save KV memory and is not a kernel-speed proof. Verify it is real QUEST (not
  pure Kitty) by comparing 16k vs 128k decode ms/token: sim QUEST stays
  near-flat (bounded to budget) while pure dense Kitty grows with context.
- `quest_kitty_page16_kernel`: the real Triton kernel path (below), Llama/Qwen
  only, the genuine speed proof.

## True QUEST + Kitty page16 kernel usage

The true QUEST + Kitty kernel path is the real Qwen3/Llama Kitty decode path with
16-token pages, query-aware page selection, and Triton sparse QK/SV kernels. It is
not the same as the `quest_kitty_page16_sim` pure-torch proxy.

Naming rules:

- `quest_kitty_page16_sim` is the pure-PyTorch QUEST accuracy/relative-timing
  proxy. It performs real query-aware selection but on a dense gather (no Triton);
  do not use it for kernel-speed claims.
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

QUEST budget rule: always set the QUEST token budget explicitly to `2048` for
QUEST + Kitty experiments and commands. Do not rely on an implicit default, do
not substitute `MAX_GEN`/generation length for the QUEST budget, and do not use
other QUEST budgets unless the user explicitly requests a budget sweep or a
different budget. For CLI paths, pass the interface-specific equivalent such as
`--quest-token-budget 2048` or `QUEST_BUDGET=2048` when that path supports true
QUEST selection.

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

`scripts/run_exp.sh` is the **sole entry point** for LongBench in this repo. Do
not launch LongBench through any other script. It sources `.env` automatically
and owns a deterministic, smoke/full-separated output layout.

### Command reference: targets, flags, and environment

`scripts/run_exp.sh [TARGET] [flags]` is the only supported way to launch
LongBench. `TARGET` is positional and defaults to `all`. CLI flags always win
over environment variables, and explicit env/CLI always wins over `.env`.

**Targets.** Each target ships a default GPU, model id/path, model family,
generation cap, and variant; every one of these is overridable (see the env
table). `qwen` is an alias of `qwen3`, `glm` of `glm4`, `llama32` of `llama3.2`.

| Target | Default GPU | Model id | Family | Default variant | Default max-gen |
| --- | ---: | --- | --- | --- | --- |
| `llama` | 0 | meta-llama/Llama-3.1-8B-Instruct | llama3 | `kitty` | per-dataset |
| `llama32` | 1 | meta-llama/Llama-3.2-1B-Instruct | llama3 | `quest_kitty_page16_sim` | 256 |
| `qwen` | 1 | Qwen/Qwen3-8B | qwen | `kitty` | 2048 |
| `glm` | 2 | THUDM/GLM-4-9B-Chat-1M | glm4 | `kitty` | per-dataset |
| `deepseek` | 0 | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | llama3 | `kitty` | 1024 |
| `all` | 0/1/2 | llama+qwen+glm concurrently | — | per-target | — |

`all` runs llama (GPU0), qwen (GPU1), and glm (GPU2) concurrently, one process
each; `SERIAL=1` runs them one after another. `deepseek` is opt-in and is never
part of `all`.

**Flags.**

| Flag | Meaning |
| --- | --- |
| `--gpu N` | Run the target on a single physical GPU N. |
| `--gpus G0,G1,...` | Fan the target's datasets across several GPUs (one dataset per GPU at a time; a GPU that finishes steals the next pending dataset). |
| `--max-samples N` | `N>0` = smoke (N samples/dataset, output under `smoke/`); `N<=0` = full (all rows). A bare trailing integer is also taken as `--max-samples` (e.g. `run_exp.sh llama32 2`). |
| `--max-model-len N` | Context cap (default `32768`). Full runs must keep `32768` unless a smoke/proxy is explicitly requested. |
| `--variant NAME` | Override the target's default variant. |

`--gpu` vs `--gpus` is purely shell-level task parallelism — the Python eval
code is identical; `--gpus` just dispatches one dataset per free GPU. This is
opt-in: per the GPU1-only rule above, stay on the per-target default GPU unless
the user explicitly widens the hardware constraint.

With `TARGET=all`, `--gpu/--gpus` is rejected unless `SERIAL=1` (the
llama/qwen/glm loops already occupy GPU0/1/2 concurrently). To fan one model's
datasets across GPUs, use a single target, e.g. `run_exp.sh llama32 --gpus 0,1,2`.

**Variants** (`--variant` / `RUN_VARIANT`). The method slug is the output dir's
`<method>` suffix:

| Variant | Method slug | What it is |
| --- | --- | --- |
| `kitty` | `kitty` | Paper-style 2-bit Kitty, 128-token pages (sim fake-quant). |
| `kitty_pro` | `kitty-pro` | Kitty with `promote_ratio=0.25`. |
| `kitty_k1v4` | `kitty-k1v4` | Sub-2-bit K: 1-bit base + 2-bit magnitude channel boost (`kbits=1, promote_bit=2`); V per-token **4-bit**. Scalar default `promote_ratio=0.25`; the boost fraction is **per-layer configurable** via `--promote-ratio-config` / `PROMOTE_RATIO_CONFIG` (JSON; only this variant accepts it). The old fixed-pr variants `kitty_k1v2*` and `kitty_k1v4_pr50/_pr75` were removed — express them as JSON (`{"default":0.5}` etc.; K1V2 needs `custom --vbits 2`). |
| `fp16` | `fp16` | Full-precision baseline — no Kitty cache, HF dense fp16 KV. |
| `kivi_2` | `kivi-2` | KIVI-2 baseline (sim fake-quant; no promote, no channel-select, no sink). |
| `kivi_star_2` | `kivi-star-2` | KIVI*-2 — same as `kivi_2` but `sink_length=32`. |
| `quest_kitty_page16_sim` | `quest-kitty-sim` | Pure-torch QUEST+Kitty page16 accuracy proxy (real query-aware selection, no Triton, any arch). |
| `quest_kitty_page16_kernel` | `quest-kitty-kernel` | Real Triton QUEST+Kitty page16 sparse decode (speed proof; Llama/Qwen/GLM). |
| `custom` | `custom-kitty` | Custom Kitty config. |

The `fp16`, `kivi_2`, and `kivi_star_2` baselines keep dense fp16 KV, so they do
not save KV memory; only `kitty` / `*_kernel` actually compress the cache.

**Environment overrides.**

Run control:
- `MAX_SAMPLES` (= `--max-samples`), `MAX_MODEL_LEN` (= `--max-model-len`, default `32768`), `RUN_VARIANT` (= `--variant`).
- `RUN_MODE=auto|smoke|full` — force the smoke/full *layout* independently of the
  sample count (e.g. `RUN_MODE=full ... --max-samples 2` = a few-sample run that
  still writes to the full dir).
- `DATASETS_CSV=ds1,ds2,...` — restrict to a subset of the default 21 datasets.
- `SERIAL=1` — for `all`, run the three models serially instead of concurrently.
- `FORCE=1` — allow overwriting an existing output that is *more* complete than
  the requested run (otherwise refused, so results are never silently shrunk).

GPU selection (precedence `--gpus` > `--gpu` > `GPU_IDS_CSV` > per-target default):
- `GPU_OVERRIDE` (= `--gpu`), `GPUS_OVERRIDE` (= `--gpus`), `GPU_IDS_CSV`.

QUEST controls (only used by `quest_kitty_page16_kernel`; ignored by other variants):
- `QUEST_BUDGET` (= `--quest-token-budget`, always `2048` for QUEST+Kitty),
  `QUEST_SKIP_LAYERS` (= `--quest-skip-layers`, default `0`).

Per-layer promote_ratio (only used by `kitty_k1v4`; any other variant rejects it):
- `PROMOTE_RATIO_CONFIG` (= `--promote-ratio-config`) — path to a JSON schedule
  `{"default": r, "layers": {"idx": r}}` or a bare list `[r0, r1, ...]`. Omitted
  => scalar default 0.25 for every layer (historical kitty_k1v4 behaviour).
  Effective K bits = `1 + mean_l(pr_l)`.

Per-target overrides — `<T>` is one of `LLAMA`, `LLAMA32`, `QWEN`, `GLM`, `DEEPSEEK`:
- `<T>_GPU`, `<T>_MODEL_ID`, `<T>_MODEL_PATH`, `<T>_MODEL_SLUG`, `<T>_MAX_GEN`, `<T>_DEFAULT_VARIANT`.
- `<T>_MODEL_PATH` falls back to the matching `.env` `KITTY_*_PATH`.
- **`<T>_MODEL_SLUG` sets the output dir's `<model>` segment.** When running a
  target with a *non-default* model (e.g. `qwen` with Qwen3-4B instead of the
  default Qwen3-8B), set `<T>_MODEL_SLUG` too, so results are labeled correctly
  and do not get mislabeled into / collide with the default model's dir.

**Resume / clean-slate.**
- smoke: the target's previous smoke dir is wiped on every launch (smoke is never resumed).
- full: completed datasets are kept; partial/missing ones are rerun, so an
  interrupted full run resumes and fills in the rest.
- When the datasets finish, the pred dir is scored automatically
  (`kitty_sim.cli.score_longbench`) into `pred/result.json`.

**Multi-GPU example** — fan one model's 21 datasets across GPUs. Set the slug
because this is a non-default model:

```bash
QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
QWEN_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
QWEN_MODEL_SLUG=qwen3-4b-instruct-2507 \
QWEN_MAX_GEN=512 MAX_MODEL_LEN=32768 \
bash scripts/run_exp.sh qwen --gpus 0,1,2 --variant quest_kitty_page16_kernel
# -> longbench_out/qwen3-4b-instruct-2507_quest-kitty-kernel/{pred,logs}
```

Two non-overlapping GPU groups run different variants at once (distinct output
dirs, safe to launch in two terminals):

```bash
# terminal 1 — FP16 baseline on GPUs 0,1,2
QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 QWEN_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
QWEN_MODEL_SLUG=qwen3-4b-instruct-2507 QWEN_MAX_GEN=512 MAX_MODEL_LEN=32768 \
bash scripts/run_exp.sh qwen --gpus 0,1,2 --variant fp16

# terminal 2 — KIVI-2 baseline on GPUs 3,4,5
QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 QWEN_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
QWEN_MODEL_SLUG=qwen3-4b-instruct-2507 QWEN_MAX_GEN=512 MAX_MODEL_LEN=32768 \
bash scripts/run_exp.sh qwen --gpus 3,4,5 --variant kivi_2
```

### Output layout

- full (no `--max-samples`, or `--max-samples -1`):
  `longbench_out/<model>_<method>/{pred,logs}`
- smoke (`--max-samples N`, N > 0):
  `longbench_out/smoke/<model>_<method>/{pred,logs}`
- `pred/` holds `<dataset>.jsonl`, `<dataset>.manifest.json`, and `result.json`;
  `logs/` holds the per-dataset `report_<dataset>.json`.
- `<model>` / `<method>` are the slugs from `src/kitty_sim/longbench/runner.py`
  (`model_layout_slug` / `method_layout_slug`) joined by an underscore, e.g.
  `llama31-8b-instruct_kitty`, `qwen3-8b_quest-kitty`.

Unless a task explicitly asks for a shorter smoke/proxy run, full LongBench runs
must use `MAX_MODEL_LEN=32768` (32k context) and the per-target generation
length. Use `--max-samples N` only for smoke runs. Do not use the old
`MAX_MODEL_LEN=3500` default for full runs.

LongBench command-answer rule: when the user asks for LongBench test commands, always provide both a smoke-test command and a full-test command. Both commands must be complete, directly runnable shell blocks with all relevant environment variables included; do not abbreviate with phrases like "change MAX_SAMPLES to -1" or omit paths, model tags, output dirs, report prefixes, GPU selection, variant, `MAX_MODEL_LEN`, `MAX_GEN`/runner-specific generation cap, or QUEST budget settings.

- The canonical QUEST + Kitty LongBench accuracy test is the pure-torch sim
  variant `quest_kitty_page16_sim` (real query-aware page selection on Kitty
  fake-quant, architecture-portable, no Triton). When asked for "QUEST + Kitty"
  LongBench test commands, default to this variant and provide both smoke and
  full forms. The real Triton variant `quest_kitty_page16_kernel` is the
  latency/speed proof only (Llama/Qwen) and need not be the default test command.
- For any QUEST + Kitty LongBench/runtime path, the QUEST budget must be
  explicitly fixed at `2048` tokens (`QUEST_BUDGET=2048` or
  `--quest-token-budget 2048`, depending on the runner). The `MAX_GEN=256`
  generation cap is separate and must not be confused with the QUEST budget.

Canonical QUEST + Kitty LongBench test commands (sim variant, Llama-3.2-1B; the
`.env` here has no `KITTY_LLAMA32_1B_PATH`, so pass `LLAMA32_MODEL_PATH=`):

```bash
# smoke (2 samples/dataset)
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 QUEST_BUDGET=2048 QUEST_SKIP_LAYERS=0 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant quest_kitty_page16_sim --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_quest-kitty-sim/{pred,logs}
```

```bash
# full (all 21 datasets, 32k context)
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 QUEST_BUDGET=2048 QUEST_SKIP_LAYERS=0 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant quest_kitty_page16_sim
# -> longbench_out/llama32-1b-instruct_quest-kitty-sim/{pred,logs}
```

Full paper-style Kitty LongBench on GPU1 for LLaMA3.1-8B-Instruct:

```bash
bash scripts/run_exp.sh llama --gpu 1
# -> longbench_out/llama31-8b-instruct_kitty/{pred,logs}
```

Pure-torch QUEST + Kitty (no Triton) for Qwen3-8B, full on GPU1:

```bash
bash scripts/run_exp.sh qwen --gpu 1 --variant quest_kitty_page16_sim
# -> longbench_out/qwen3-8b_quest-kitty-sim/{pred,logs}
```

Smoke example (LLaMA3.2-1B, 2 samples each, GPU1; llama32 defaults to quest_kitty_page16_sim):

```bash
bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_quest-kitty-sim/{pred,logs}
```

Scope datasets with `DATASETS_CSV`, and force the layout independently of the
sample count with `RUN_MODE=smoke|full` (e.g. a few-sample correctness check
that still writes to the full layout:
`RUN_MODE=full bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2`).
Local model paths come from `.env` (`KITTY_*_PATH`) or per-target `*_MODEL_PATH`
overrides; never hardcode host paths in tracked files.

### Real QUEST + Kitty kernel on LongBench (variant `quest_kitty_page16_kernel`)

`--variant quest_kitty_page16_kernel` is the REAL Triton Kitty + QUEST sparse
decode path on LongBench, NOT the `kitty_page16` fake-quant proxy. Decode runs the
Triton sparse QK/SV kernels with query-aware page selection over the real paged
`kitty.kvcache` cache.

Supported model families: `llama`, `qwen`, and `glm`.
- `llama` / `qwen`: the runner loads the architecture-specific `*_Kitty` model
  class (`kitty.models.llama` `LlamaForCausalLM_Kitty` / `kitty.models.qwen3`
  `Qwen3ForCausalLM_Kitty`, stock HF model with only the attention forward
  replaced) and builds the real paged cache per sample via `past_key_values`.
- `glm`: ChatGLM-4 loads its own remote code with a legacy tuple cache that can
  NOT thread an HF cache via `past_key_values`, so the runner loads the stock
  remote-code model (`AutoModelForCausalLM`, NOT a `*_Kitty` class) and installs
  the kernel post-load with `kitty_sim.glm_kitty_patch.install_glm_real_kitty_kernel`
  (per-layer 1-layer `KittyCache` on each `SelfAttention`; reuses the GLM
  de-frag + generate shims). GLM must run in **fp16** (the GLM target already
  passes `--torch-dtype float16`; the kitty cache/kernel buffers are fp16 while
  GLM weights are bf16). The runner sizes each layer's cache per sample via
  `set_glm_real_kitty_sample_length`, and the guardrail validates
  `kitty_stats["paths"]` (printed as `[glm-quest-kernel] ... paths=...`).

Always pass the QUEST budget explicitly: `QUEST_BUDGET=2048` (page16 -> 128
logical pages). `QUEST_SKIP_LAYERS` defaults to 0 (every decode layer uses QUEST
selection). The runner has a first-sample guardrail that refuses to proceed
unless decode shows real kernel evidence (`last_quest_path` =
`triton_sparse_reduced_budget` / `triton_sparse_forced_all_pages`, or
`dense_full_budget` / `dense_no_shared_pages` when a context is smaller than the
budget). A bare `dense` with `quest_skip_layers=0`, `python_sparse_debug`, or
`unknown` is rejected as a silent degradation to dense fp16.

Real QUEST+Kitty smoke for Llama-3.2-1B on GPU0 (scoped to long-context
datasets so the sparse path is exercised):

```bash
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
QUEST_BUDGET=2048 QUEST_SKIP_LAYERS=0 \
DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 0 --variant quest_kitty_page16_kernel --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_quest-kitty-kernel/{pred,logs}
```

Full real QUEST+Kitty for Llama-3.2-1B on GPU0 (all 21 datasets, 32k context):

```bash
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
QUEST_BUDGET=2048 QUEST_SKIP_LAYERS=0 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant quest_kitty_page16_kernel
# -> longbench_out/llama32-1b-instruct_quest-kitty-kernel/{pred,logs}
```

The `.env` here has no `KITTY_LLAMA32_1B_PATH`, so the model path must be passed
explicitly via `LLAMA32_MODEL_PATH=` (otherwise it falls back to a non-existent
`$HOME/models/...`). For Qwen3-8B, use `qwen --variant quest_kitty_page16_kernel`
(the `kitty` env already resolves `KITTY_QWEN3_8B_PATH`).

Real QUEST+Kitty for GLM-4-9B-Chat-1M on GPU0 (remote code, fp16 forced by the GLM
target; `.env` has no `KITTY_GLM4_9B_1M_PATH`, so pass `GLM_MODEL_PATH=`; GLM-9B is
memory-heavy, so smoke at `MAX_MODEL_LEN=8192` first, then scale to 32768 watching
`nvidia-smi`):

```bash
cd /mnt/data/tzj/Code/Kitty
GLM_MODEL_PATH=/mnt/data/tzj/models/GLM-4-9B-Chat-1M \
MAX_MODEL_LEN=8192 QUEST_BUDGET=2048 QUEST_SKIP_LAYERS=0 \
DATASETS_CSV=multifieldqa_en,hotpotqa \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/run_exp.sh glm --variant quest_kitty_page16_kernel --gpu 0 --max-samples 2
# -> longbench_out/smoke/glm4-9b-chat-1m_quest-kitty-kernel/{pred,logs}
# full: drop --max-samples and raise MAX_MODEL_LEN to 32768 (watch GPU0 memory).
```

GLM evidence lines read `[glm-quest-kernel] ... paths={'triton_sparse_reduced_budget': N}`
on long-context samples (short samples legitimately show `dense_full_budget`).

### Pure-torch QUEST + Kitty on LongBench (variant `quest_kitty_page16_sim`)

`quest_kitty_page16_sim` is the no-Triton QUEST accuracy proxy and the default
for the `llama32` target. The sim `KittyKVCache` supplies the Kitty page16
fake-quant; a per-arch attention hook (`kitty_sim/sim_quest.py`, covers Llama and
Qwen3 via a `q_norm` attribute check) runs the gather-based QUEST oracle
(`kitty_sim/quest_sparse.py`) on decode. It does genuine query-aware page
selection (unlike the removed `kitty_page16`), is architecture-portable, but does
not save KV memory and is not a kernel-speed proof.

Smoke for Llama-3.2-1B on GPU0 (`llama32` already defaults to this variant):

```bash
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
QUEST_BUDGET=2048 QUEST_SKIP_LAYERS=0 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant quest_kitty_page16_sim --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_quest-kitty-sim/{pred,logs}
```

Full (all 21 datasets, 32k context):

```bash
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
QUEST_BUDGET=2048 QUEST_SKIP_LAYERS=0 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant quest_kitty_page16_sim
# -> longbench_out/llama32-1b-instruct_quest-kitty-sim/{pred,logs}
```

To confirm it is real QUEST (not pure Kitty), compare decode ms/token at 16k vs
128k: sim QUEST stays near-flat (attention bounded to the 2048-token budget)
while pure dense Kitty grows roughly linearly with context. The runner's
first-sample guardrail also prints `[sim-quest] ... decode_calls=... last_selected_pages=...`
and refuses to proceed if the QUEST hook never ran.

## Experimental low-bit K-cache exploration (kitty_k1v4 + per-layer promote_ratio)

This is an ongoing exploration of how far the **K cache can be pushed below
Kitty's 2-bit floor**. The surviving variant is **`kitty_k1v4`**: K = 1-bit base
+ a magnitude-selected fraction (`promote_ratio`, "pr") of channels promoted to
2-bit; V per-token 4-bit. Shared K config: `kbits=1, promote_bit=2,
sink_length=32, buffer_length=128, group_size=128, channel_selection=1`.
Effective K bits = `1 + mean_l(pr_l)`. Research probes, not a production setting.

The K boost fraction is **per-layer configurable**: pass
`--promote-ratio-config <json>` (env `PROMOTE_RATIO_CONFIG`) with
`{"default": r, "layers": {"idx": r}}` or a bare list `[r0, ..., r_{L-1}]`;
layers absent from `layers` fall back to `default`. Only `kitty_k1v4` accepts
this flag (any other variant fail-fasts). Without it the scalar default 0.25
applies — byte-for-byte the historical behaviour. pr granularity is per LAYER;
within a layer all KV heads share the same pr (each head promotes its own
magnitude-top `head_dim*pr` channels per 128-token buffer).

The old fixed-pr variants (`kitty_k1v2`, `kitty_k1v2_pr50/_pr75`,
`kitty_k1v4_pr50/_pr75`) were **removed**: K1V4 points are now JSON one-liners
(`{"default":0.5}`), K1V2 needs `custom --kbits 1 --vbits 2 --promote_bit 2`.

Everything runs on the pure-torch sim fake-quant path (`kitty_sim`):
**accuracy proxy only, no KV memory savings** — the real Triton kernel hardcodes
2-bit/4-bit packing and is not built for a 1-bit K base.

### Results: uniform-pr sweep (LLaMA-3.2-1B, full LongBench, 21 datasets, 32k, sim)

| `promote_ratio` | eff. K bitwidth | K1V2 (V 2-bit) | K1V4 (V 4-bit) |
| ---: | ---: | ---: | ---: |
| 0.25 | ~1.25-bit | 10.46 | 10.76 |
| 0.5 | ~1.5-bit | 15.96 | 17.49 |
| 0.75 | ~1.75-bit | 23.36 | 24.31 |

Baselines (same harness): fp16 27.59 / kitty (2-bit base, 4-bit boost) 26.25 /
kivi (2-bit) 24.24.

**Finding (uniform sweep, final).** Accuracy tracks the **effective K bitwidth**
almost monotonically; a 1-bit K base collapses when too many channels stay at
1-bit and recovers toward the 2-bit floor as the boost fraction rises (K1V4
0.75 → 24.31 ≈ kivi 24.24). Raising V from 2-bit to 4-bit buys only +0.3~1.5 at
every K point — the collapse is **K-driven, not V-limited**. **2-bit is the
practical K floor on 1B**; long-range retrieval (hotpotqa/musique/qasper) fails
first and recovers first.

### Results: per-layer pr probes (LLaMA-3.2-1B, trec/qasper/hotpotqa/multifieldqa_en, full, sim)

- Reproduction anchor: `{"default":0.6875}` → trec 64.0 / qasper 20.27,
  bit-identical to the old scalar path (fp16: 65.5 / 24.28).
- **Per-layer K sensitivity** (drop one layer's K to 1-bit, others at ref;
  consistent across 0.6875 and 0.875 refs): **L10 is by far the most sensitive**
  (qasper 20.4 → 10.1 at the 0.875 ref), then **L14**, then L9; the early layers
  **L0–L3 are the least sensitive**. trec is saturated and carries no signal.
- **Placement at equal average bits: uniform ≥ back-loaded > front-loaded.**
  Front-boosting (L0–L3 → 1.0) loses accuracy even with a NET budget increase.
- **Naive calibration loses to uniform.** Allocating pr proportional to the
  single-layer sensitivities (water-filling) at equal avg bits LOST to uniform at
  1.5-bit (qasper 8.90 vs 13.15) — single-layer probes ignore the cross-layer
  error accumulation when many layers are compressed together. **Uniform pr is a
  strong baseline**; per-layer gains likely need operating-point-consistent
  sensitivities, larger models (3B/8B), or per-head granularity.

### Reproduction

Canonical GPU1 single-card form; the pr schedule lives in a JSON file (ignored
`/configs/` dir or any path); model path resolves via `.env`/`LLAMA32_MODEL_PATH`:

```bash
# smoke (2 samples/dataset); {"default":0.5} reproduces the old kitty_k1v4_pr50
cd /home/zijie/Code/Kitty
printf '{ "default": 0.5 }\n' > /tmp/pr_sched.json
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
PROMOTE_RATIO_CONFIG=/tmp/pr_sched.json \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kitty_k1v4 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_kitty-k1v4/{pred,logs}
```

```bash
# full (all 21 datasets, 32k context), per-layer example
cd /home/zijie/Code/Kitty
printf '{ "default": 0.5, "layers": { "10": 1.0, "14": 1.0 } }\n' > /tmp/pr_sched.json
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
PROMOTE_RATIO_CONFIG=/tmp/pr_sched.json \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kitty_k1v4
# -> longbench_out/llama32-1b-instruct_kitty-k1v4/{pred,logs}
```

Direct `eval_longbench` runs take `--promote-ratio-config /path/sched.json`.
Unit tests: `PYTHONPATH=src python -m unittest tests.test_kitty_per_layer_promote_ratio`.
To fan one variant's 21 datasets across several GPUs for speed (an explicit
override of the GPU1-only rule), use e.g. `--gpus 0,1,2,3,4,5` instead of `--gpu 1`.

## 32k memory probe evidence

A GPU1 probe with Qwen3-8B + paper-style Kitty, 32768-token input, and `max_new_tokens=1` completed without OOM.

Observed peak:

- `nvidia-smi` peak memory: about `25947 MiB` / `25.34 GiB`.
- PyTorch peak allocated: about `23.03 GiB`.
- PyTorch peak reserved: about `24.83 GiB`.

Treat these as environment-specific smoke numbers, not a formal benchmark.

## Low-bit-K full-LongBench harness + recommended `pb2` configs

Model-agnostic harness and the recommended low-bit-K "type" from the low-bit-K
study. It extends the K1V4 family: a **1-bit K base + magnitude-selected channels
promoted to 2-bit** (V fixed 4-bit), swept over the boost fraction `promote_ratio`.
All on the pure-torch `kitty_sim` fake-quant path: **accuracy proxy only, no memory
/ speed savings**; the real Triton kernel does NOT support a 1-bit base (hardcoded
2/4-bit packing).

### The recommended configs ("pb2" type)

`--variant custom` with `kbits=1 vbits=4 promote_bit=2 channel_selection=1
sink_length=32 buffer_length=128 group_size=128`, varying `promote_ratio`:

| config | promote_ratio | effective K bits | what |
| --- | ---: | ---: | --- |
| `pb2_pr6875` | 0.6875 | **1.6875** | cheapest config clearing the bars (trec>=62 AND qasper>19) on 1B |
| `pb2_pr875` | 0.875 | 1.875 | safer-margin pick |
| `fp16` | — | 16 | dense baseline (ceiling) |

`promote_ratio` sets how many of the `head_dim` K channels keep 2-bit; the rest
stay 1-bit. head_dim is 64 on Llama-3.2-1B and 128 on Llama-3.2-3B / phi-4-mini /
Llama-3.1-8B, so pr0.6875 = 44/64 or 88/128 channels @2-bit.

### Harness (`autoresearch/loop-260607/`)

- `driver_ds.sh TAG KBITS VBITS PBIT PRATIO CHANSEL [MAXSAMPLES]` — one config on one
  LongBench dataset. Env: `DATASET`, `GPU`, `MODEL_FAMILY`, `MODEL_TAG`,
  `LLAMA32_MODEL_PATH` (= model path, any model), `MAXS`. Writes `res_<dataset>/<TAG>.tsv`.
- `full_lb_sweep.sh [CFG]` — every config in CFG x all 21 LongBench subtasks, parallel
  across a GPU pool (list a GPU N times in `GPUS` for N workers on it). `MAXS=-1` = full.
- `sweep_ds_mgpu.sh [CFG]` — every config x ONE `DATASET`, multi-GPU round-robin.
- `aggregate.py TAG ...` — per-config full-LongBench average over `res_*/<TAG>.tsv`.
- Run artifacts (`out_*/ res_*/ logs/ *.out DONE*`) are gitignored (regenerable).

### Canonical: run the three cases (fp16 + pb2_pr6875 + pb2_pr875), full LongBench

Use distinct tags per model so `res_*/` never collide (`<m>` = model slug):

```bash
cd <repo>/autoresearch/loop-260607 && conda activate kitty
cat > configs_3cases.txt <<'EOF'
fp16_<m>        fp16 fp16 0  0      1
pb2_pr6875_<m>  1    4    2  0.6875 1
pb2_pr875_<m>   1    4    2  0.875  1
EOF
LLAMA32_MODEL_PATH=/path/to/Model MODEL_FAMILY=<family> MODEL_TAG=<m> \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GPUS=<layout> MAXS=-1 \
setsid bash full_lb_sweep.sh configs_3cases.txt > fulllb_<m>.out 2>&1 &
# then, per config: python3 aggregate.py fp16_<m> ; python3 aggregate.py pb2_pr6875_<m> ; ...
```

`MODEL_FAMILY` ∈ `llama3` (Llama-3.x), `qwen`, `glm4`, `phi` (Phi-3/4; added here for
the chat template). For a new arch, extend `infer_model_family` / `build_chat` in
`src/kitty_sim/longbench/templates.py`.

### GPU parallelism vs context (24GB 3090; per-process peak @32k input)

Each worker loads its own weights (no cross-process sharing), so workers/card =
`floor(24GB / per-process-peak)`:

| model | weights | ~peak @32k | workers / 24GB card | `GPUS` layout |
| --- | ---: | ---: | ---: | --- |
| Llama-3.2-1B | 2.5GB | ~5GB | 3 | `0,0,0,1,1,1,...` |
| Llama-3.2-3B | 6.5GB | ~12GB | 2 | `0,0,1,1,...` |
| phi-4-mini (3.8B) | 7.7GB | 15.4GB | **1** (2 OOMs at 32k) | `0,1,2,3,4,5` |
| Llama-3.1-8B | ~16GB | ~22GB | **1** (tight on 24GB) | `0,1,2,3,4,5` |

### Findings so far (full LongBench, sim)

| model | fp16 avg | pb2_pr6875 (eff 1.69) | gap |
| --- | ---: | ---: | ---: |
| Llama-3.2-1B | 27.59 | 23.95 | ~ -13% |
| Llama-3.2-3B | ~38 | ~37 (matched subset) | **~ -3%** |

Bigger model = far more robust to the aggressive 1-bit-base K. Mechanism: accuracy
is set by the FRACTION of K channels kept >=2-bit (~2/3 needed), not their per-channel
precision; 2-bit boost is bit-optimal (1-bit+f16 needs ~6x the bits, 3-bit already
matches the f16 upper bound). qasper>19 is the binding bar (needs more 2-bit channels
than trec); long-summarization tasks (qmsum/vcsum/multi_news) are the first to collapse.

### Llama-3.1-8B recipe (other machine)

```bash
cd <repo>/autoresearch/loop-260607 && conda activate kitty
cat > configs_8b.txt <<'EOF'
fp16_8b        fp16 fp16 0  0      1
pb2_pr6875_8b  1    4    2  0.6875 1
pb2_pr875_8b   1    4    2  0.875  1
EOF
LLAMA32_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct MODEL_FAMILY=llama3 MODEL_TAG=llama31-8b \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GPUS=0,1,2,3,4,5 MAXS=-1 \
setsid bash full_lb_sweep.sh configs_8b.txt > fulllb_8b.out 2>&1 &
python3 aggregate.py pb2_pr6875_8b   # full-LB avg per config
```

8B @32k needs ~22GB/process -> **1 worker/card** on a 24GB GPU (list each GPU once in
`GPUS`); prefer >=40GB cards for headroom / 2-per-card. `head_dim=128` so pr0.6875 =
88/128 channels @2-bit.
