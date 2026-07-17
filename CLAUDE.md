# Kitty Agent Notes

This file captures durable, repo-local guidance for agents working in this Kitty checkout. It should contain stable operational constraints and reproducible commands, not transient run logs.

`AGENTS.md` must stay a byte-identical copy of this file: after editing
`CLAUDE.md`, run `cp CLAUDE.md AGENTS.md`.

## Notion 笔记路由

When recording notes to Notion for this project, route by note type:

- **研究笔记 / Research notes** (方法调研 / 文献综述 / 结论与观察 / 技术报告) → database
  `📚 研究笔记 | Research Notes`, id `819846320d954eeea661638174068076`
  (data source `35fd3ce0-5640-4963-979f-4c1bad6d2639`).
- **实验测试笔记 / Experiment & test notes** (跑了哪些实验、数据表、复现命令、精度评测)
  → database `测试笔记 | Test Notes`, id `fa575373aeaa494d957a90fe07d5c8e0`
  (data source `ee4db7c4-cbd7-4b35-bd6e-fbc04a9c8310`).

Use the `ntn` CLI with `NOTION_KEYRING=0` (headless / file-based auth). Create a
page under a database via `parent.type=data_source_id`.

Embedding a local figure (verified 2026-06-23; supersedes the old "ntn can't
attach file_upload" note): `ntn files create < fig.png` returns a file-upload id,
then append it as an image block with
`ntn api v1/blocks/<page_id>/children -X PATCH -d '{"children":[{"object":"block","type":"image","image":{"type":"file_upload","file_upload":{"id":"<id>"}}}]}'`.
Wrap ntn in `timeout`: `ntn api` GET on a big page's `/children` can hang, but the
PATCH write works. The Notion MCP `notion-update-page insert_content` path only
accepts URL images in Markdown (`![](url)`), not file_upload ids — so use MCP
`insert_content` for text/tables and the ntn api PATCH above for local figures.

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

The `kitty` variant fixes the Kitty machinery (magnitude channel-select
`channel_selection=1` + `sink_length=32`, `buffer/group=128`) and makes the four
bit/ratio knobs tunable via `KBITS` / `PROMOTE_BIT` / `PROMOTE_RATIO` / `VBITS`
(or `--promote-ratio-config` / `PROMOTE_RATIO_CONFIG` for a per-layer ratio
schedule). Defaults reproduce the paper-style Kitty:

```text
kbits=2              # KBITS       — K base bit
promote_bit=4        # PROMOTE_BIT — bit of the magnitude-boosted K channels
promote_ratio=0.125  # PROMOTE_RATIO — boosted-channel fraction
vbits=2              # VBITS       — V bit
sink_length=32, buffer_length=128, group_size=128, channel_selection=1   # fixed
```

The output slug encodes all four: `kitty-k{kbits}b{promote_bit}v{vbits}-pr{ratio}`
(default → `kitty-k2b4v2-pr0p125`). This single parametric `kitty` **subsumes the
removed `kitty_pro`** (`PROMOTE_RATIO=0.25` → `kitty-k2b4v2-pr0p25`) **and
`kitty_k1v4`** (`KBITS=1 PROMOTE_BIT=2 VBITS=4 PROMOTE_RATIO=0.25` →
`kitty-k1b2v4-pr0p25`).

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

### LongBench result-scope rule

When summarizing, comparing, or reporting completed LongBench results, only
consider result drops that currently exist under the repo-root `longbench_out/`
directory. Treat `archieve/longbench_out/` as archived history: do not include
those archived runs in scoreboards, tables, recommendations, or "current result"
answers unless the user explicitly asks to inspect archived results.

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
| `llama32` | 1 | meta-llama/Llama-3.2-1B-Instruct | llama3 | `fp16` | 256 |
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
| `kitty` | `kitty-k{kbits}b{promote_bit}v{vbits}-pr{ratio}` | Kitty machinery (magnitude channel-select + sink=32) with tunable K base (`KBITS`), boost bit (`PROMOTE_BIT`), boost fraction (`PROMOTE_RATIO`, or `--promote-ratio-config` for per-layer), V (`VBITS`). Defaults = paper Kitty (k2/b4/v2/pr0.125 → `kitty-k2b4v2-pr0p125`). Subsumes the removed `kitty_pro` (`PROMOTE_RATIO=0.25`) and `kitty_k1v4` (`KBITS=1 PROMOTE_BIT=2 VBITS=4 PROMOTE_RATIO=0.25`). |
| `qlutattn` | `qlutattn` | The single canonical QLUTATTN variant (pure-torch sim fake-quant accuracy proxy; see the dedicated section below and `docs/qlutattn.md`). Q stays FP16 — never quantized. K: post-RoPE **per-token** quant along head_dim — a per-channel mean `μ_d` is self-calibrated at prefill and subtracted (free for attention: `q·μ` cancels in softmax); the residual is quantized under an OFFLINE per-channel codebook mask (`scripts/calibrate_qlutattn_mask.py`, wikitext): the 50% lowest-σ² channels → sign (1-bit + per-token mean-|r| scale, ~1.25b), the other 50% → the fixed symmetric NF2 LUT `symnf2-v1` (`{-1, -c, +c, +1}`, `c = 0.25256848`, per-token absmax scale, ~2.25b); nominal K = **1.75 bit/value**. V: rescued 2-bit tile16c64, algo `rht-pcaff-mse1-bias-v1`. sink=32, recent-128 fp16 window, group=128. Fixed algorithm — `QLUT_CB_MASK` is the ONLY runtime input; FP16 model only, head_dim a power of two divisible by 64, GLM family fail-fast. |
| `fp16` | `fp16` | Full-precision baseline — no Kitty cache, HF dense fp16 KV. |
| `kivi` | `kivi-k{kbits}v{vbits}` | KIVI-style uniform quant (no promote, no channel-select, no sink). K/V bit-width is set via `KBITS`/`VBITS` (default 2/2 = the old `kivi_2`); the slug encodes the bits so each combo gets its own dir (e.g. `kivi-k2v4`). |
| `kivi_star` | `kivi-star-k{kbits}v{vbits}` | Same as `kivi` but `sink_length=32` (the old `kivi_star_2`). |
| `llamacpp_q40` | `llamacpp-q40` | llama.cpp/ggml **Q4_0** KV cache, faithful port (sim fake-quant): K and V per-token, 32-channel symmetric absmax blocks (`d = signed_max/-8`, fp16 scale → **4.5 bit/value** each), quantize-on-write — **no sink, no fp16 recent window** (`buffer=0`; unique among variants here). head_dim must be a multiple of 32. Known deviations vs llama.cpp (documented, not simulated): PostQuant lets the current step read pre-quant values (one-token difference), and Q stays fp16 (llama.cpp quantizes Q to Q8_0 for the integer dot), so scores are slightly optimistic. |
| `llamacpp_q40_star` | `llamacpp-q40-star` | Q4_0 codebook under the Kitty protection policy (`sink=32` + recent-128 fp16 window) — ablates codebook vs no-sink/no-recent effects. |
| `shadowkv` | `shadowkv` | ShadowKV pure-torch sim (accuracy proxy; no memory/speed savings). |
| `custom` | `custom-kitty` | Custom Kitty config. |

All LongBench variants here run on the pure-torch sim fake-quant path (accuracy
proxy, no real KV-memory savings); `fp16`/`kivi`/`kivi_star` keep dense fp16 KV.

### KIVI K/V bit-width sweep (variants `kivi` / `kivi_star`)

`kivi` (no sink) and `kivi_star` (sink=32) are KIVI-style uniform quant (no
promote, no channel-select); the **K and V bit-widths are free**, set via
`KBITS` / `VBITS` (default 2/2 = the old `kivi_2` / `kivi_star_2`). The output
slug encodes the bits (`kivi-k{KBITS}v{VBITS}`), so different combinations never
collide and no `LLAMA32_MODEL_SLUG` is needed. `KBITS`/`VBITS` accept 1–16
(≥16 = no quant).

Smoke one combo (2 samples, long-context datasets):

```bash
KBITS=2 VBITS=4 \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kivi --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_kivi-k2v4/{pred,logs}
```

Full K×V sweep (all 21 datasets, 32k):

```bash
for kb in 1 2 4; do for vb in 2 4; do
  KBITS=$kb VBITS=$vb \
  LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 --variant kivi
  # -> longbench_out/llama32-1b-instruct_kivi-k${kb}v${vb}/{pred,logs}
done; done
# kivi_star (sink=32): same loop with --variant kivi_star -> ..._kivi-star-k{kb}v{vb}
```

### llama.cpp Q4_0 KV cache (variants `llamacpp_q40` / `llamacpp_q40_star`)

Faithful sim port of the llama.cpp/ggml `-ctk q4_0 -ctv q4_0` KV cache. Q4_0
quantizes **both K and V per-token along head_dim in 32-channel blocks**
(`QK4_0=32`): symmetric absmax scale `d = signed_max/-8` (stored fp16), codes
`min(15, int(x/d + 8.5))`, dequant `(q-8)*d` → **K = V = 4.5 bit/value**.

- **`llamacpp_q40`** — the faithful port: quantize-on-write, **no sink, no fp16
  recent window** (`sink=0`, `buffer_length=0`; the only variant allowed to run
  `buffer=0`, and only in per_token mode). Prefill quantizes the whole prompt;
  decode quantizes each newest token immediately.
- **`llamacpp_q40_star`** — same codebook under the Kitty protection policy
  (`sink=32` + recent-128 fp16 window): ablates codebook vs protection effects.

Usage constraints:

- **No calibration step, no env knobs.** `KBITS`/`VBITS`/`QLUT_*` are ignored;
  the bit-width is fixed by the codebook (kbits/vbits=4 in the config are
  bookkeeping only).
- head_dim must be a multiple of 32 (Llama-3.2 1B/3B 64, Llama-3.1-8B 128 OK).
- Known deviations vs llama.cpp (accepted, not simulated): PostQuant lets the
  current step read pre-quant values (one-token difference), and Q stays fp16
  (llama.cpp quantizes Q to Q8_0 for the integer dot) → slightly optimistic.
- Implementation map: quant core `fake_quant_q4_0_lastdim` in
  `src/kitty_sim/utils_quant.py` (bit-exact vs the C reference incl. the fp16
  rounding of the stored d); cache branches `KittyKVCache._quant_k_pertoken`
  (`k_codebook="q4_0"`) + `_quant_v` (`v_codebook="q4_0"`) and the `buffer=0`
  schedule in `update()`; variants in `runner.py::build_variant()`. GLM gets
  `k_codebook`/`v_codebook` via `glm_kitty_patch.cache_config_from_variant`
  (GLM untested for q4_0).

Verified 2026-07-08 (GPU1, Llama-3.2-1B): 12/12 unit tests pass (bit-exact
C-oracle; prefill + N decode steps == one-shot quantization, bit-identical);
smoke completed for `llamacpp_q40`, `llamacpp_q40_star`, and `fp16`.

```bash
# 0-GPU unit tests (C oracle + schedule invariants)
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src python -m unittest tests.test_q4_0_fakequant -v
```

```bash
# smoke (2 samples/dataset, long-context datasets so the K path is exercised)
cd "$(git rev-parse --show-toplevel)"
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant llamacpp_q40 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_llamacpp-q40/{pred,logs}
# ablation: --variant llamacpp_q40_star -> .../llama32-1b-instruct_llamacpp-q40-star
```

```bash
# full (all 21 datasets, 32k context)
cd "$(git rev-parse --show-toplevel)"
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant llamacpp_q40
# -> longbench_out/llama32-1b-instruct_llamacpp-q40/{pred,logs}
# (multi-GPU fan-out, explicit override of the GPU1-only rule: --gpus 0,0,0,1,1,1)

# ablation full run (Q4_0 codebook + Kitty policy: sink=32, recent-128 fp16)
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant llamacpp_q40_star
# -> longbench_out/llama32-1b-instruct_llamacpp-q40-star/{pred,logs}

# 4-bit reference points: fp16 (ceiling) and per-channel-K KIVI-4 (4.25b):
KBITS=4 VBITS=4 \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kivi
# -> longbench_out/llama32-1b-instruct_kivi-k4v4/{pred,logs}
```

The old low-bit-K variants (`kitty_k1v2*`, `kitty_k1v4*`, `kitty_pro`) were
REMOVED and folded into the parametric `kitty`: K base / boost bit / V are
`KBITS`/`PROMOTE_BIT`/`VBITS`, and the boost fraction is `PROMOTE_RATIO` (or
`--promote-ratio-config` for a per-layer schedule). The slug now encodes
`pr{ratio}`, so different scalar ratios auto-split into separate dirs (no
`LLAMA32_MODEL_SLUG` needed). Only a per-layer *schedule* still needs
`LLAMA32_MODEL_SLUG` to disambiguate, since the slug's `pr` reflects only the
default ratio, not the full schedule.

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
bash scripts/run_exp.sh qwen --gpus 0,1,2 --variant kitty
# -> longbench_out/qwen3-4b-instruct-2507_kitty-k2b4v2-pr0p125/{pred,logs}
```

Two non-overlapping GPU groups run different variants at once (distinct output
dirs, safe to launch in two terminals):

```bash
# terminal 1 — FP16 baseline on GPUs 0,1,2
QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 QWEN_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
QWEN_MODEL_SLUG=qwen3-4b-instruct-2507 QWEN_MAX_GEN=512 MAX_MODEL_LEN=32768 \
bash scripts/run_exp.sh qwen --gpus 0,1,2 --variant fp16

# terminal 2 — KIVI baseline (K2V2) on GPUs 3,4,5
QWEN_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 QWEN_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
QWEN_MODEL_SLUG=qwen3-4b-instruct-2507 QWEN_MAX_GEN=512 MAX_MODEL_LEN=32768 \
bash scripts/run_exp.sh qwen --gpus 3,4,5 --variant kivi
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
  `llama31-8b-instruct_kitty-k2b4v2-pr0p125`, `qwen3-8b_qlutattn`.

Unless a task explicitly asks for a shorter smoke/proxy run, full LongBench runs
must use `MAX_MODEL_LEN=32768` (32k context) and the per-target generation
length. Use `--max-samples N` only for smoke runs. Do not use the old
`MAX_MODEL_LEN=3500` default for full runs.

LongBench command-answer rule: when the user asks for LongBench test commands, always provide both a smoke-test command and a full-test command. Both commands must be complete, directly runnable shell blocks with all relevant environment variables included; do not abbreviate with phrases like "change MAX_SAMPLES to -1" or omit paths, model tags, output dirs, report prefixes, GPU selection, variant, `MAX_MODEL_LEN`, and `MAX_GEN`/runner-specific generation cap.

Canonical LongBench test commands (Llama-3.2-1B, canonical `qlutattn`
variant; set `KITTY_LLAMA32_1B_PATH` in `.env` or pass `LLAMA32_MODEL_PATH=`
explicitly; the offline mask must exist first — see the QLUTATTN section):

```bash
# smoke (2 samples/dataset)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn/{pred,logs}
```

```bash
# full (all 21 datasets, 32k context)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn
# -> longbench_out/llama32-1b-instruct_qlutattn/{pred,logs}
```

Full paper-style Kitty LongBench on GPU1 for LLaMA3.1-8B-Instruct:

```bash
bash scripts/run_exp.sh llama --gpu 1
# -> longbench_out/llama31-8b-instruct_kitty-k2b4v2-pr0p125/{pred,logs}
```

Smoke example (LLaMA3.2-1B, 2 samples each, GPU1; llama32 defaults to fp16):

```bash
bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_fp16/{pred,logs}
```

Scope datasets with `DATASETS_CSV`, and force the layout independently of the
sample count with `RUN_MODE=smoke|full` (e.g. a few-sample correctness check
that still writes to the full layout:
`RUN_MODE=full bash scripts/run_exp.sh llama32 --gpu 1 --max-samples 2`).
Local model paths come from `.env` (`KITTY_*_PATH`) or per-target `*_MODEL_PATH`
overrides; never hardcode host paths in tracked files.

## QLUTATTN(唯一 canonical 变体)

`qlutattn` 是本仓库**唯一**的公开 QLUTATTN 变体(variant 名 = method slug = 输出
slug = `qlutattn`)。算法**完全固定**:不支持码本选择、sign 比例、K-block、旋转、
V-tile 尺寸等任何旋钮,也不支持 QUEST 叠加 —— `QUEST_KERNEL` / `QUEST_TRITON` /
`SIM_QUEST` / `QUEST_SIM` 与 `--variant qlutattn` 组合会直接硬报错(QUEST 仍可用于
kitty/kivi)。历史 QLUTATTN 调参环境变量已全部退役:任何残留非空值都会硬报错
(空串视为未设置);`VBITS` 对 qlutattn 只接受 unset/空串/`2`。唯一的 QLUTATTN
运行时输入是 `QLUT_CB_MASK`。完整设计文档:`docs/qlutattn.md`。

### 语义

- **Q 保持 FP16**,永不量化。
- **K**:post-RoPE **per-token**(沿 head_dim)量化。prefill 时自标定 per-channel
  均值 `μ_d` 并减去(对 attention 免费:`q·μ` 是 per-query 常数,在 softmax 中
  抵消);残差按**离线 per-channel 码本掩码**量化
  (`scripts/calibrate_qlutattn_mask.py`,wikitext 残差 σ² 排序):
  - 50% 最低 σ² 通道 → **sign**:1-bit 码字 + per-token mean-|r| scale
    (~1.25 bit/value);
  - 50% 最高 σ² 通道 → **nf2(`symnf2-v1`)**:固定对称 NF2 LUT
    `{-1, -c, +c, +1}`,`c = 0.25256848`,per-token absmax scale,无二次均值
    (~2.25 bit/value)。
  - 名义 K 位宽 = `0.5×1.25 + 0.5×2.25` = **1.75 bit/value**。
  - 明确**不包含**:Hadamard/FWHT 旋转、SmoothAttention、在线 σ² 分箱、per-token
    自适应码本拟合、outlier 稠密-稀疏侧路、跨 token 的 block 共享码本(block 固定
    为 1)。
- **V**:rescued 2-bit **tile16c64**,算法 `rht-pcaff-mse1-bias-v1`(固定 RHT,
  seed 20260711;1 轮 MSE 迭代):每 tile = 16 个连续 token × 64 通道;码字 2-bit,
  per-tile scale 与 per-channel side-info 使理论有效位宽略高于 2;sink / recent
  窗口 / 不足 16 token 的尾部保持 FP16。
- **保护窗口**:`sink_length=32`,recent FP16 窗口 `buffer_length=128`,
  `group_size=128`。

### 离线掩码(唯一运行时输入)

每个模型一份掩码,命名约定 `<MODEL_PATH>.qlutattn_mask.pt`;`.env.example` 提供
`KITTY_WIKITEXT2_TRAIN_PATH` / `KITTY_LLAMA32_1B_QLUTATTN_MASK` 两个可选变量存放
本机路径。加载时严格校验,不满足即拒绝运行:

| 字段 | 要求 |
| --- | --- |
| `codebooks` | 恰为 `["sign", "nf2"]` |
| `low_frac` | 恰为 `0.5` |
| `codebook_mask` | `uint8`,`ndim == 3`,取值恰为 `{0, 1}`(0=sign,1=nf2) |
| 实际 sign 占比 | 恰为 `0.5` |
| shape | `[num_layers, num_key_value_heads, head_dim]`(目标模型) |

legacy 的 `target_bits` / `nominal_bits` 元数据一律忽略。掩码文件的 SHA-256 进入
变体语义哈希与每个 run manifest。

### 标定 → smoke → full

```bash
# 1) 离线标定(每模型一次,~1min/卡;固定 sign/nf2 50/50,无可调参数)
cd "$(git rev-parse --show-toplevel)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_qlutattn_mask.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --output /path/to/Llama-3.2-1B-Instruct.qlutattn_mask.pt
```

```bash
# 2) smoke(2 samples/dataset,long-context 数据集确保走到 K 路径)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn/{pred,logs}
```

```bash
# 3) full(全部 21 数据集,32k;多卡 fan-out 需显式放宽 GPU1-only 规则,
#    例如 --gpus 0,1,2)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn
# -> longbench_out/llama32-1b-instruct_qlutattn/{pred,logs}
```

输出布局:full → `longbench_out/<model>_qlutattn/{pred,logs}`;smoke →
`longbench_out/smoke/<model>_qlutattn/{pred,logs}`。

### run 验收(manifest + engagement)

一个 qlutattn run 只有在每个数据集都满足以下条件时才有效:

- `manifest.status == "ok"`,`run_config_hash` 非空(shell preflight 与 Python
  worker 各自重算且必须一致);
- `variant.name == "qlutattn"`,`nf2_impl == "symnf2-v1"`,`mask_sha256` 非空;
- engagement 证据证明 V 路径真实执行:
  `engagement.last_v_quant_mode == "tile16_rescued"`,`last_v_tile_channels == 64`,
  `v_tile_blocks > 0`,`v_quantized_tokens > 0`。

### 限制

- 纯 torch **fake-quant 精度代理**:量化后立即重建回 FP16 cache,没有打包 KV
  存储,不省真实显存、不加速 —— 这些 run 只测精度。
- 模型必须是 FP16;`head_dim` 必须是 2 的幂且可被 64 整除;GLM 家族直接
  fail-fast 拒绝(不做静默回退)。

## KV-cache 可视化 skill (`kv-cache-viz`)

仓库里所有 KV-cache **探测/离线可视化** 脚本已整理成 Claude Code skill：

- 路径：`.claude/skills/kv-cache-viz/`
- 统一入口：`bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh <cmd>`
- 也可直接用 slash 命令：`/kv-cache-viz <cmd>`

### 何时触发

在 Kitty 仓库内，用户出现以下意图时**应该调用** `kv-cache-viz`：

- "画一下 KV cache 分布" / "看一下 K/V 的 channel distribution"
- "probe 一下 sign scale" / "跑一下 mu2sigma2"
- "画 heatmap" / "画 e2e latency/memory 柱状图"
- "分析 KV cache 的能量 / DC share / sigma2"
- "dump layer 8 的 K/V" 用于离线可视化

**注意**：如果用户只是要"跑 LongBench 精度对比"，应使用 `lutdecoding-acc-bench` skill，
而不是 `kv-cache-viz`。

### 子命令

#### `probe` — 需要 GPU + LongBench 数据

| 命令 | 原脚本 | 作用 | 输出 |
| --- | --- | --- | --- |
| `probe channel-energy` | `dump_channel_energy_csv.py` | 每层 K channel 能量统计 CSV | `.csv` |
| `probe mu2sigma2` | `dump_kv_mu2_sigma2_dist.py` | K/V 每通道 μ²/σ² 分布 | `.pt`, `.json`, `.png` |
| `probe sigma2-block-concentration` | `dump_sigma2_block_concentration.py` | σ² 在 128-token block 内/跨 block 集中度 | `.json`, `.png` |
| `probe signpt-dequant` | `dump_signpt_dequant_dist.py` | sign 反量化前后 channel 分布对比 | `.png` |
| `probe sign-scale` | `dump_sign_scale_dist.py` | sign-group scale 分布 + 16-token 共享实验 | `.pt`, `.json`, `.png` |

#### `dump-layer` — 为离线画图准备 layer K/V

```bash
bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh dump-layer \
  --layer 8 --model /path/to/Llama-3.2-1B-Instruct \
  --longbench-dir /path/to/LongBench/data \
  --tag llama32-1b --outdir probe_out
```

生成 `probe_out/quant_kvcache_analysis/layer8_{K,V}_fp16.pt`，离线 viz 默认会读它。

#### `viz` — 离线画图（纯 CPU / 可选 GPU）

| 命令 | 原脚本 | 作用 |
| --- | --- | --- |
| `viz channel-dist` | `plot_kv_channel_dist.py` | layer/head 8 个 channel 的 K/V 分布 |
| `viz channel-dist-multi` | `plot_kcache_channel_dist_multi.py` | 每 channel 单独一张 PNG |
| `viz dcshare-heatmap` | `plot_kcache_dcshare_heatmap.py` | DC-energy share 2D heatmap |
| `viz decomp` | `plot_k_decomp_steps.py` | K = μ + residual + sign 重建的 3D 图 |
| `viz kv-3d-submean` | `plot_kv_3d_submean.py` | K = μ + residual 3D 图 |
| `viz kv-seg-mu2sigma2-3d` | `plot_kv_seg_mu2sigma2_3d.py` | 分段 μ²/σ² 3D 图 |
| `viz codebook-mu2sigma2` | `plot_codebook_mu2_sigma2.py` | sign/nf2 codebook 在 μ²/σ² 通道上的示意 |
| `viz why-sign-beats-minmax` | `plot_why_sign_beats_minmax.py` | sign 打败 minmax 的理论+实证图 |
| `viz kitty-kv-heatmap` | `plot_kitty_kv_heatmap.py` | Kitty K×V bit sweep heatmap |
| `viz kivistar-heatmap` | `plot_kivistar_heatmap.py` | KIVI* K×V bit sweep heatmap |
| `viz kv-channel-dist-layers` | `plot_kv_channel_dist_layers.py` | 跨层 K/V 分布汇总 |
| `viz e2e-perf` | `plot_e2e_perf_bars.py` | end-to-end 性能柱状图 |
| `viz kv-memory-bars` | `plot_kv_memory_bars.py` | KV memory 柱状图 |
| `viz attn-op-latency` | `plot_attn_op_latency_bars.py` | attention op latency 柱状图 |
| `viz qlutattn-energy-quest1024` | `plot_qlutattn_energy_quest1024.py` | Quest 预算 1024 下的能量对比 |

### 公共参数

probe / dump-layer：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 模型路径 |
| `--longbench-dir` | 必填 | LongBench `data/` 目录 |
| `--seq-len` | 32768 | prefill 长度 |
| `--sink` | 32 | sink token 数 |
| `--recent` | 128 | recent token 数 |
| `--chunk` | 4096 | chunked prefill 步长 |
| `--device` | `cuda:0` | 设备 |
| `--tag` | `model` | 输出文件名标签 |
| `--outdir` | `probe_out/sign_scale` | 输出目录 |

viz：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--layer` | 8 | 目标层 |
| `--head` | 0 | 目标 head |
| `--pt` | 自动 | K/V dump 路径；默认 `outdir/quant_kvcache_analysis/layer{L}_{K,V}_fp16.pt` |
| `--outdir` | `probe_out` | 输出目录前缀 |
| `--tag` | `llama32-1b` | 输出文件名标签 |
| `--device` | `cuda:0` | 需要 GPU 的 viz 命令使用 |

### 典型流程示例

```bash
cd "$(git rev-parse --show-toplevel)"

# 1) 探测（GPU1-only 规则：CUDA_VISIBLE_DEVICES=1）
CUDA_VISIBLE_DEVICES=1 bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh probe mu2sigma2 \
  --model /path/to/Llama-3.2-1B-Instruct \
  --longbench-dir /path/to/LongBench/data \
  --seq-len 32768 --tag llama32-1b --outdir probe_out/sign_scale

# 2) dump layer 8 的 K/V 用于离线画图
CUDA_VISIBLE_DEVICES=1 bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh dump-layer \
  --layer 8 --model /path/to/Llama-3.2-1B-Instruct \
  --longbench-dir /path/to/LongBench/data \
  --seq-len 32768 --tag llama32-1b --outdir probe_out

# 3) 离线画图（纯 CPU）
bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh viz channel-dist \
  --layer 8 --head 0 --tag llama32-1b --outdir probe_out
```

### 注意事项

- 所有 `probe` 和 `dump-layer` 命令都会加载模型到 GPU，必须遵守 **GPU1-only evaluation rule**，
  默认 `CUDA_VISIBLE_DEVICES=1`，除非用户显式放宽硬件约束。
- `viz` 命令默认读 `probe_out/quant_kvcache_analysis/layer{L}_{K,V}_fp16.pt`；
  如果路径不同，用 `--pt` 指定。
- 输出目录若不存在会自动创建。
- 本 skill 只用于分析和可视化，不替代 LongBench 精度 benchmark。

## Experimental low-bit K-cache exploration (K1V2 / K1V4 families)

This is an ongoing exploration of how far the **K cache can be pushed below
Kitty's 2-bit floor**, and whether spending precision on the **V cache** instead
compensates. All variants keep the Kitty machinery (sink tokens, magnitude
channel selection, 128-token groups) and change only the K base bitwidth, the
2-bit boost fraction, and the V bitwidth. Paper-style Kitty is a 2-bit K base +
4-bit boost; everything here is a deliberately more aggressive, sub-2-bit-K
regime — research probes, not a recommended production setting.

### Motivation

Kitty's KIVI base quantizes K per-channel and V per-token to 2-bit, then promotes
a magnitude-selected fraction of K channels to a higher precision (4-bit in the
paper). The open question we explore: **how much of the K cache can drop to 1-bit
before accuracy collapses, and does a higher-precision V buy any of it back?** We
attack it along two axes:

1. **K boost fraction** — keep the K base at **1-bit** and promote a fraction
   (`promote_ratio` ∈ {0.25, 0.5, 0.75}) of channels to **2-bit**. This sets the
   *effective average K bitwidth*: 0.25 → ~1.25-bit, 0.5 → ~1.5-bit, 0.75 →
   ~1.75-bit.
2. **V precision** — run the same K sweep twice, once with V at **2-bit** (K1V2)
   and once with V at **4-bit** (K1V4). V has no channel boost (per the paper's
   per-token V quantization); only its bitwidth changes between families.

Shared K config for every variant: `kbits=1, promote_bit=2, sink_length=32,
buffer_length=128, group_size=128, channel_selection=1`.

### Variant matrix

| Family | V cache | How to run |
| --- | --- | --- |
| **K1V2** | per-token **2-bit** | REMOVED from code (results below retained for reference). |
| **K1V4** | per-token **4-bit** | `KBITS=1 PROMOTE_BIT=2 VBITS=4 PROMOTE_RATIO=<r> --variant kitty` (flat ratio); per-layer schedule via `PROMOTE_RATIO_CONFIG` JSON. |

All of these run on the pure-torch sim fake-quant path (`kitty_sim`), so they are
an **accuracy proxy only and save no KV memory** — the real Triton kernel
hardcodes 2-bit/4-bit packing and is not built for a 1-bit K base (nor for
per-head-variable boosted-channel counts). Treat these as accuracy research, not
a kernel-speed or memory-savings claim.

### Results so far (LLaMA-3.2-1B, full LongBench, 21 datasets, 32k context, sim)

| `promote_ratio` | eff. K bitwidth | K1V2 (V 2-bit) | K1V4 uniform |
| ---: | ---: | ---: | ---: |
| 0.25 | 1.250 | **10.46** | **10.76** |
| 0.5 | 1.500 | **15.96** | **17.49** |
| 0.625 | 1.625 | — | **21.89** |
| 0.75 | 1.750 | **23.36** | **24.31** |
| 0.875 | 1.88 | — | **25.20** |

Baselines (same harness): fp16 27.59 / kitty (2-bit base, 4-bit boost) 26.25 /
kivi (2-bit) 24.24.

**Finding (K1V2, final).** Accuracy tracks the **effective K bitwidth** almost
monotonically. A 1-bit K base collapses when too many channels stay at 1-bit
(`promote_ratio=0.25` → 10.46, well below every 2-bit baseline), but recovers
steadily as the boost fraction rises, nearly reaching the true 2-bit floor by 0.75
(23.36 ≈ kivi 24.24). Only a small fraction of 1-bit channels is tolerable;
**2-bit is the practical K floor on 1B**, and the cliff is steepest at low boost
fractions (long-range retrieval such as hotpotqa/musique is the first to fail and
the first to recover).

**Finding (K1V4, final).** Relaxing V to 4-bit lifts every K operating point only
modestly (+0.3 to +1.5 over K1V2) — the collapse is K-driven, not V-limited. The
uniform K1V4 dose-response is cleanly monotone in effective K bits
(10.76 → 17.49 → 21.89 → 24.31 → 25.20).

A per-layer `promote_ratio` schedule (via `PROMOTE_RATIO_CONFIG`) does not beat a
flat uniform ratio at equal bits on the uniform arm (replicating the layer-diff
"calibration doesn't beat uniform" negative result). The cross-head allocation
strategy and its per-layer-schedule interaction were studied under the
now-removed `kitty_k1v4_xhead` variant; that exploration is archived in memory,
not here.

### Reproduction

Canonical GPU1 single-card form; model path resolves via
`.env`/`LLAMA32_MODEL_PATH`. The boost fraction comes from a JSON config
(`{"default": 0.5}` here). When running several ratios, encode the ratio in the
output dir via `LLAMA32_MODEL_SLUG` (e.g. `llama32-1b-instruct-pr625`):

```bash
# smoke (2 samples/dataset): K1V4 with a flat boost ratio = the old kitty_k1v4
cd "$(git rev-parse --show-toplevel)"
KBITS=1 PROMOTE_BIT=2 VBITS=4 PROMOTE_RATIO=0.5 \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kitty --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_kitty-k1b2v4-pr0p5/{pred,logs}
```

```bash
# full (all 21 datasets, 32k context). Per-layer schedule instead of a flat
# ratio: drop PROMOTE_RATIO and pass PROMOTE_RATIO_CONFIG=$PWD/configs/sched.json
# (+ LLAMA32_MODEL_SLUG to keep different schedules in separate dirs).
cd "$(git rev-parse --show-toplevel)"
KBITS=1 PROMOTE_BIT=2 VBITS=4 PROMOTE_RATIO=0.5 \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kitty
# -> longbench_out/llama32-1b-instruct_kitty-k1b2v4-pr0p5/{pred,logs}
```

Set `PROMOTE_RATIO_CONFIG` to apply a per-layer ratio schedule — an arm launched
without it silently runs the built-in `promote_ratio=0.25` default and still
writes to the same output dir name. When sweeping several schedules, encode the
schedule in the output dir via `LLAMA32_MODEL_SLUG` (e.g.
`llama32-1b-instruct-s1sens`) so arms never share a dir.

To fan one variant's 21 datasets across several GPUs for speed (an explicit
override of the GPU1-only rule), use e.g. `--gpus 0,1,2,3,4,5` instead of `--gpu 1`.
A 1B model at 32k needs ~5 GB/worker, so two concurrent launches can share the
same 24 GB card (list the card in both launches' `--gpus`) for 2 workers/card.

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
| `pb2_pr875` | 0.875 | 1.88 | safer-margin pick |
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

## RULER-NIAH (捞针) evaluation

Standalone needle-in-a-haystack harness following the RULER protocol (NVIDIA,
arXiv 2404.06654): needles are "One of the special magic numbers/uuids for
{key} is: {value}", scoring is `string_match_all` (case-insensitive substring
of each gold value in the greedy 128-token continuation). It complements the
LongBench accuracy studies with an exact-retrieval stress test.
`scripts/run_niah.sh` is the sole NIAH entry point; do not launch
`kitty_sim.cli.eval_niah` by hand for real runs.

### Data generation (offline, one-time per tokenizer family)

`scripts/prepare_niah_data.sh` shells into a RULER checkout
(`RULER_REPO_ROOT`, default `~/Code/RULER` — the local fork whose `niah.py`
emits `answer_prefix` and `token_position_answer`) and writes
`${NIAH_DATA_ROOT}/<LEN>/<task>/validation.jsonl` (default
`~/data/ruler_niah/llama3`). Generation needs `pip install wonderwords nltk`
(+ `NLTK_DATA=$RULER_REPO_ROOT/nltk_data`) — pure-Python, only for datagen.
Data is generated at `LEN - 256` tokens so the eval-time chat template can
never overflow `max_model_len`: the NIAH runner hard-errors instead of
truncating (a middle-truncate could silently delete the needle). Llama-3.2
1B/3B share one llama3-tokenizer dataset; seed 42 makes all arms see identical
samples.

```bash
cd "$(git rev-parse --show-toplevel)"
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
RULER_REPO_ROOT=/path/to/RULER \
NIAH_DATA_ROOT=/path/to/ruler_niah/llama3 \
bash scripts/prepare_niah_data.sh
# TASKS / LENS / NUM_SAMPLES env override the 4-task x 4-len x 50 default.
```

### Eval + scoring

First-party modules: `src/kitty_sim/niah/{data,runner,scorer}.py`,
`kitty_sim.cli.eval_niah`, `kitty_sim.cli.score_niah`,
`scripts/plot_niah_montage.py`, tests in `tests/test_niah_wiring.py` (0-GPU).
The runner reuses `build_variant` / `load_model_and_tokenizer` /
`_cache_factory` from the LongBench runner: fresh `KittyKVCache` per sample via
`past_key_values`, greedy `max_new_tokens=128`, prompt =
`build_chat(input) + answer_prefix` (family `llama3` = raw text, matching the
RULER base template). Manifests carry `run_config_hash` + the `engagement`
evidence (`last_v_quant_mode`/`v_tile_blocks`); full runs resume, smoke wipes.
Output: `niah_out/[smoke/]<model>_<method>/{pred,logs}`; scoring writes
`pred/result.json` and depth x length heatmap PNGs into `logs/`
(`token_position_answer/length` binned into 10 depth bins — the classic NIAH
heatmap, no controlled-depth regeneration needed).

The driver default is `--variants fp16,qlutattn`. The `qlutattn` arm needs the
model-specific offline mask via `QLUT_CB_MASK` (the same
`<MODEL_PATH>.qlutattn_mask.pt` used for LongBench, from
`scripts/calibrate_qlutattn_mask.py`); `llamacpp_q40` / `llamacpp_q40_star`
are also wired into `run_niah.sh` and need no mask or extra env. GPU default
is 1 (GPU1-only rule); any other GPU is an explicit user override.

```bash
# smoke (2 samples, 1 task, 2 lens)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
NIAH_DATA_ROOT=/path/to/ruler_niah/llama3 \
bash scripts/run_niah.sh --gpu 1 --variants fp16,qlutattn \
  --tasks niah_single_2 --lens 4096,32768 --max-samples 2
# -> niah_out/smoke/llama32-1b-instruct_<method>/{pred,logs}
```

```bash
# full (4 tasks x 4 lens x 50 samples, both arms serially)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/path/to/Llama-3.2-1B-Instruct.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
NIAH_DATA_ROOT=/path/to/ruler_niah/llama3 \
bash scripts/run_niah.sh --gpu 1 --variants fp16,qlutattn
# -> niah_out/llama32-1b-instruct_<method>/{pred,logs}; per-arm result.json + heatmaps auto-written

# comparison heatmap montage (shared color scale):
PYTHONPATH=src python scripts/plot_niah_montage.py \
  --arms niah_out/llama32-1b-instruct_fp16 \
         niah_out/llama32-1b-instruct_qlutattn \
  --labels "fp16" "qlutattn" \
  --task pooled --output niah_out/niah_heatmap_montage_pooled.png
```

### Reference points (verified 2026-07-12, A100-80GB, Llama-3.2-1B, 50 samples/cell)

Mean `string_match_all` over niah_single_1/2/3 + niah_multikey_1: fp16 scores
**97.62** overall (99.0/98.5/96.5/96.5 at 4k/8k/16k/32k) and shows no
lost-in-the-middle at these lengths; llama.cpp Q4_0 (K=V=4.5b, no sink/no
recent window) is a near-fp16 reference at **97.00**, its only visible dent
being multikey_1@32k (78 vs fp16 86) — exact retrieval is essentially intact
at 4.5 bit even without any protection policy. General lesson: exact retrieval
degrades long before LongBench averages do, so always pair a LongBench score
with a NIAH check when evaluating aggressive K-cache quantization.
