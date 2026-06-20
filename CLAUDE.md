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
| `qlutattn_k1v4` | `qlutattn-k1v4` | σ²-binned mixed-codebook K quant (sim fake-quant). Per-layer channels are binned by residual σ²; low-σ² bins use cheap codebooks (sign), high-σ² bins use richer ones (nf2). Winner `["sign","sign","sign","tern","nf2","nf2"]` ≈ K **1.68 bit**, V per-token 4-bit. See `docs/qlutattn_k1v4.md`. |
| `qlutattn_k184v4` | `qlutattn-k184v4` | Uniform-tern K (all channels tern) + V per-token 4-bit, K ≈ 1.84 bit — the iso-tern baseline `qlutattn-k1v4` is compared against. |
| `qlutattn_k125v4` | `qlutattn-k125v4` | Uniform-sign K (all channels sign) + V per-token 4-bit, K ≈ 1.25 bit — cheapest member of the qlutattn-k<bits>v4 family; the pure-sign 1-bit codebook isolated for ablation against σ²-mix (1.68) and tern (1.84). |
| `qlutattn_pertoken` | `qlutattn-pertoken` | Per-token K quant (head_dim-axis grouping), single codebook (`QLUT_BIN_CODEBOOKS`, default `nf2`), V per-token 4-bit. |
| `qlutattn_k125v4_pt` | `qlutattn-k125v4-pt` | **Per-token sign with per-CHANNEL mean removal** (the fixed submean): subtract a per-channel μ (cached at prefill, free for attention — `q·μ` cancels in softmax), then pure 1-bit binary on the residual. ~1.25 bit K, V per-token 4-bit. Per-token form of `qlutattn-k125v4`; fixes the broken per-TOKEN submean (full LongBench 10.99→**21.68**). `QLUT_BIN_CODEBOOKS=tern` switches codebook. |
| `qlutattn_k185v4_pt` | `qlutattn-k185v4-pt` | Per-token **tern** + per-channel mean removal. ~1.84 bit K, V 4-bit. Per-token form of `qlutattn-k184v4`. |
| `qlutattn_k168v4_pt` | `qlutattn-k168v4-pt` | Per-token **σ²-binned mixed codebook** + per-channel mean removal. Channels binned by per-channel residual σ²; per-bin codebook via `QLUT_BIN_CODEBOOKS` (default k1v4 winner `sign,sign,sign,tern,nf2,nf2`). Per-token form of `qlutattn-k1v4`. **NOTE**: per-token side-info (one scale per σ²-bin per token) makes the real eff. bit ~3 (6 bins), not 1.68 — the name follows the k1v4 lineage, not the per-token bit. |
| `fp16` | `fp16` | Full-precision baseline — no Kitty cache, HF dense fp16 KV. |
| `kivi` | `kivi-k{kbits}v{vbits}` | KIVI-style uniform quant (no promote, no channel-select, no sink). K/V bit-width is set via `KBITS`/`VBITS` (default 2/2 = the old `kivi_2`); the slug encodes the bits so each combo gets its own dir (e.g. `kivi-k2v4`). |
| `kivi_star` | `kivi-star-k{kbits}v{vbits}` | Same as `kivi` but `sink_length=32` (the old `kivi_star_2`). |
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
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kivi --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_kivi-k2v4/{pred,logs}
```

Full K×V sweep (all 21 datasets, 32k):

```bash
for kb in 1 2 4; do for vb in 2 4; do
  KBITS=$kb VBITS=$vb \
  LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpu 1 --variant kivi
  # -> longbench_out/llama32-1b-instruct_kivi-k${kb}v${vb}/{pred,logs}
done; done
# kivi_star (sink=32): same loop with --variant kivi_star -> ..._kivi-star-k{kb}v{vb}
```

### QLUT-Attn k1v4 (σ²-binned K) test (variants `qlutattn-k1v4` / `qlutattn_k184v4`)

`qlutattn-k1v4` is a per-channel mixed-codebook K quant: channels are binned by residual σ²
and given different codebooks (low σ² → cheap `sign`, high σ² → richer `nf2`), so the
average K bit-width (≈1.68) drops below uniform tern (≈1.84) while accuracy is kept
or improved. Full design + usage: `docs/qlutattn_k1v4.md`. The matched baseline is
`qlutattn_k184v4` (uniform tern K, ≈1.84 bit); `fp16` is the ceiling. V is per-token
4-bit for both quantized variants. Override the policy without code edits via
`QLUT_BIN_CODEBOOKS=sign,sign,sign,tern,nf2,nf2`.

Validated on full LongBench (21 datasets, 32k); full test guide + per-dataset results:
`docs/qlutattn_k1v4_testing.md`. Mean over 21:

| model | fp16 | qlutattn_k184v4 (1.84b) | qlutattn-k1v4 (1.68b) | k1v4 retains |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.2-1B | 27.59 | 22.96 | 24.88 | 90.2% |
| Llama-3.2-3B | 36.33 | 30.72 | 34.28 | 94.4% |

qlutattn-k1v4 Pareto-beats uniform tern at both scales (fewer bits + higher score;
k1v4−tern = +1.91 on 1B, +3.56 on 3B), the gap widening with model size; gains
concentrate on retrieval tasks (3B multifieldqa_en +13.1, qasper +10.0, hotpotqa +7.6).
`nf2` (per-group Lloyd) is the runtime bottleneck. **3B needs 1 worker/GPU at 32k
(2/card OOMs); 1B can use 3/card.**

```bash
# smoke (2 samples/dataset, long-context datasets so the K path is exercised)
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_k1v4 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn-k1v4/{pred,logs}
```

```bash
# full comparison (all 21 datasets, 32k context): qlutattn-k1v4 vs its baselines
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_k1v4         # -> longbench_out/llama32-1b-instruct_qlutattn-k1v4
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_k184v4  # -> longbench_out/llama32-1b-instruct_qlutattn-k184v4
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant fp16          # -> longbench_out/llama32-1b-instruct_fp16
```

Fan one variant's 21 datasets across several GPUs (faster, explicit override of the
GPU1-only rule): replace `--gpu 0` with `--gpus 0,1,3`. A fast offline overlap proxy
for policy search (no LongBench) lives in `scripts/build_kq_cache.py` +
`scripts/eval_qlut_policy.py` (see `docs/qlutattn_k1v4.md` §5b).

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
  `llama31-8b-instruct_kitty-k2b4v2-pr0p125`, `qwen3-8b_qlutattn-k1v4`.

Unless a task explicitly asks for a shorter smoke/proxy run, full LongBench runs
must use `MAX_MODEL_LEN=32768` (32k context) and the per-target generation
length. Use `--max-samples N` only for smoke runs. Do not use the old
`MAX_MODEL_LEN=3500` default for full runs.

LongBench command-answer rule: when the user asks for LongBench test commands, always provide both a smoke-test command and a full-test command. Both commands must be complete, directly runnable shell blocks with all relevant environment variables included; do not abbreviate with phrases like "change MAX_SAMPLES to -1" or omit paths, model tags, output dirs, report prefixes, GPU selection, variant, `MAX_MODEL_LEN`, and `MAX_GEN`/runner-specific generation cap.

Canonical LongBench test commands (Llama-3.2-1B; the `.env` here has no
`KITTY_LLAMA32_1B_PATH`, so pass `LLAMA32_MODEL_PATH=`):

```bash
# smoke (2 samples/dataset)
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_k1v4 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn-k1v4/{pred,logs}
```

```bash
# full (all 21 datasets, 32k context)
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_k1v4
# -> longbench_out/llama32-1b-instruct_qlutattn-k1v4/{pred,logs}
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

## qlutattn* per-token variants with per-channel submean (the submean fix)

The naive per-token submean codebook subtracts the wrong mean: `apply_codebook`
for `sign`/`tern` removes a **per-token** mean (averaging across the head_dim
channels of one token), which doesn't center any individual channel and leaves
the per-channel DC heterogeneity that makes per-token sign collapse. The fix is
to subtract a **per-channel** mean `μ_d` instead (each channel's mean over
tokens). This is **free for attention**: `q·(K−μ) = q·K − q·μ`, and `q·μ` is a
per-query constant identical for every key, so it cancels in softmax / top-k.

Implemented in `kitty_simulate.KittyKVCache._quant_k_pertoken` via two flags on
`KittyKVCacheConfig`: `pertoken_pc_submean` (per-channel center + per-token PURE
binary/ternary on the residual — no second per-token submean) and
`pertoken_mixed` (per-channel center + σ²-binned mixed codebook). The per-channel
`μ_d` (and σ²-bin ids) are computed once at prefill (`k_pc_mean` / `k_mix_bins`)
and reused at decode. The three named variants below are pre-wired; all keep
V per-token 4-bit and need **no** smoothed checkpoint (the per-channel center is
self-calibrated from the prompt at prefill).

| variant | codebook on residual | eff. K bit | per-channel form |
| --- | --- | ---: | --- |
| `qlutattn_k125v4_pt` | pure sign (1-bit) | ~1.25 | `qlutattn-k125v4` |
| `qlutattn_k185v4_pt` | pure tern | ~1.84 | `qlutattn-k184v4` |
| `qlutattn_k168v4_pt` | σ²-binned mixed (`QLUT_BIN_CODEBOOKS`) | ~3 (see table note) | `qlutattn-k1v4` |

Impact (Llama-3.2-1B, full LongBench 21 datasets, 32k): fixing the submean
dimension lifts per-token sign from **10.99** (per-token submean bug) to
**21.68** (`qlutattn-k125v4-pt`, ~1.25 bit) — within 2.1 of the 2.5-bit
`qlutattn_pertoken` nf2 baseline (23.82); fp16 is 27.59.

```bash
# smoke (2 samples/dataset, long-context datasets exercise the K path)
cd /home/zijie/Code/Kitty
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_k125v4_pt --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn-k125v4-pt/{pred,logs}
# swap --variant for qlutattn_k185v4_pt (tern) or qlutattn_k168v4_pt (σ²-mix)
```

```bash
# full (all 21 datasets, 32k) — fan across 4 GPUs (GPU0×6 + GPU1/2/3×2 = 12 workers)
cd /home/zijie/Code/Kitty
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,0,0,0,1,1,2,2,3,3 --variant qlutattn_k125v4_pt
# -> longbench_out/llama32-1b-instruct_qlutattn-k125v4-pt/{pred,logs}
# k125v4_pt / k185v4_pt are fully vectorized (fast); k168v4_pt has a per-head
# σ²-bin loop (slower, like the dense-and-sparse champion).
```

Notes: `qlutattn_k168v4_pt` also accepts the alias `qlutattn-k1.68v4-pt`; its
`QLUT_BIN_CODEBOOKS` is a comma-separated per-σ²-bin policy (default the k1v4
winner `sign,sign,sign,tern,nf2,nf2`). `qlutattn_k125v4_pt` / `qlutattn_k185v4_pt`
take a single `QLUT_BIN_CODEBOOKS` name (default `sign` / `tern` respectively).

## qlutattn-k1v4 per-token exploration (reorder + SmoothAttention)

Research probes pushing qlutattn-k1v4's K quant from **per-channel** to
**per-token** (KIVI-V-style, decode-friendly head_dim-axis grouping). All on the
pure-torch `kitty_sim` fake-quant path: accuracy proxy only, no memory/speed
savings. Full design doc + roadmap: `docs/qlutattn_reorder_smooth.md`.

Three orthogonal pieces:

- **per-token K quant** (`k_quant_mode=per_token`, variant `qlutattn_pertoken`):
  K grouped along head_dim, uniform, no promote/channel-select. The per-token
  loss is codebook-driven, NOT axis-driven — per-token nf2 (Lloyd, self-adaptive)
  ≈ per-channel KIVI, while per-token uniform collapses. `qlutattn_pertoken` is
  single-codebook (`QLUT_BIN_CODEBOOKS`, one name, default `nf2`); V per-token
  4-bit. (A uniform-codebook per-token run is reachable via `--variant custom
  --k_quant_mode per_token`.)
- **SmoothAttention** (`scripts/calibrate_smooth_qk.py`, Llama-family only):
  QServe-style `λ=max(absmax_K pair)^0.5` with the RoPE rotate-half pair
  constraint (`λ_i==λ_{i+D/2}`), folded offline into `W_q*=λ` / `W_k/=λ`.
  Flattens per-channel K outliers. q_norm/k_norm models (Qwen3) raise (the norm
  renormalizes the fold away).
- **channel reorder** (`scripts/preprocess_qlutattn_model.py` + loader
  `kitty_sim.qk_reorder.apply_qk_reorder`): offline reorder Q/K_proj output
  channels by post-RoPE K σ² so same-energy channels are contiguous on head_dim
  (prerequisite for per-token segmented mixed-codebook quant). A SINGLE GLOBAL
  RoPE-pair permutation (model-level inv_freq is shared across layers), folded
  into every layer's W_q/W_k; `inv_freq[pair_perm]` is re-applied at load
  (`persistent=False`, not saved). Mathematically identity (fp16 logits top-1
  99.76%); the loader prints `[qk-reorder] applied` per worker.

### Results (Llama-3.2-1B, full LongBench, 21 datasets, 32k, sim)

| K quant | codebook | K bit/value | score | +smooth |
| --- | --- | ---: | ---: | ---: |
| per-token | uniform 2-bit | 2.5 | 12.95 | 16.61 |
| per-token | qlut nf2 (Lloyd) | 2.5 | 23.50 | 24.08 |
| per-channel (ref) | KIVI-2 | 2.25 | 24.24 | — |
| per-channel (ref) | qlut σ²-mix | 1.68 | 24.88 | — |
| reorder + per-channel qlut σ²-mix | identity check | 1.68 | **24.82** (Δ−0.06 vs 24.88) | — |

fp16 ceiling 27.59. smooth rescues uniform (+3.66, outlier-dominated) but barely
helps nf2 (+0.58, Lloyd already self-absorbs outliers). The reorder row should
reproduce the non-reorder per-channel 24.88, confirming route-a reorder is an
identity for LongBench.

### Usage

Offline reorder model (single card ~5min; default `--output` writes to the
gitignored `reorder/<name>`, prefer an explicit `~/models` path so the reordered
checkpoint sits next to the source model):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/preprocess_qlutattn_model.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --output /path/to/models/Llama-3.2-1B-Instruct-reorder
```

Offline SmoothAttention calibration (single card ~5min; Llama-family only):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_smooth_qk.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --alpha 0.5 --output /path/to/models/Llama-3.2-1B-Instruct-smooth
```

LongBench (point `LLAMA32_MODEL_PATH` at the reorder/smooth checkpoint; set
`LLAMA32_MODEL_SLUG` so output dirs don't collide). smoke adds `--max-samples 2`
(+ `DATASETS_CSV=multifieldqa_en,hotpotqa` to exercise the K path), full omits it:

```bash
# reorder + qlutattn-k1v4 per-channel (verify identity ≈ 24.88), full 6-card×3
LLAMA32_MODEL_PATH=/path/to/models/Llama-3.2-1B-Instruct-reorder \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-reorder \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --variant qlutattn_k1v4
# -> longbench_out/llama32-1b-instruct-reorder_qlutattn-k1v4

# per-token qlut-nf2 + smooth (full): model = smoothed checkpoint
LLAMA32_MODEL_PATH=/path/to/models/Llama-3.2-1B-Instruct-smooth \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-smooth \
QLUT_BIN_CODEBOOKS=nf2 MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --variant qlutattn_pertoken
```

`reorder/` and `calib/` checkpoint dirs are gitignored; use `--output ~/models/...`.

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
| 0.875 | 1.875 | — | **25.20** |

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
cd /home/zijie/Code/Kitty
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
cd /home/zijie/Code/Kitty
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
