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
| `--v-tile-channels C` | Required channel block for rescued `*_vtile16` V2 variants. `C>0` and must divide `head_dim`; there is deliberately no silent default. |

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
| `qlutattn_k168v4_pt` | `qlutattn-k168v4-pt` | Per-token K with an **OFFLINE per-channel sign/tern codebook**. Each post-RoPE K channel is fixed offline to sign (1.25b, low σ²) or tern (1.85b, high σ²) by `scripts/calibrate_k168v4_pt.py` on wikitext (default ~28% sign → nominal **1.68b**); the mask is given via `QLUT_CB_MASK` and used unchanged — **no online σ² binning, no nf2**. Per-channel mean still self-calibrated at prefill (free for attention). V per-token 4-bit. Corrected impl: the old online-σ²+nf2 path was ~3b, scored 1B-full **14.39** and collapsed summarization (gov_report 0.64); offline sign/tern restores it (smoke gov_report→14.5). |
| `qlutattn_k188v4_pt` ⭐ | `qlutattn-k188v4-pt` | **默认推荐优化算法.** k168v4-pt 的姊妹方法,rich 码本由 tern 换成 **nf2(per-token Lloyd)**:离线 per-channel **sign/nf2** 掩码(`scripts/calibrate_k168v4_pt.py --codebooks sign,nf2 --sign-frac <f>`),sign 占比 `f` = 实际 K bit 旋钮(`f·1.25+(1−f)·2.5`)。per-channel 均值 prefill 自标定;V 4-bit;**无旋转、纯 per-token**。1B 上 Pareto 实用最优:**f=0.5(1.875b)=25.03 > per-channel k1v4 24.88**,降到 f=0(纯 nf2,2.5b)=25.65。详见下方「默认推荐优化算法」段。 |
| `qlutattn_k125v2_pt` | `qlutattn-k125v2-pt` | Sign K identical to `qlutattn_k125v4_pt`, but V is named whole-head per-token asymmetric 2-bit. This is the matched sign baseline for rescued V tiles. |
| `qlutattn_k125v2_pt_vtile16` | `qlutattn-k125v2-pt-vtile16c{C}-rv1` | Same sign K; rescued V2 uses fixed token tile 16 and explicit channel tile `C`: fixed RHT + prompt per-channel mean/RMS + affine MSE1/fallback + inverse RHT + bias correction. |
| `qlutattn_k188v2_pt` | `qlutattn-k188v2-pt` | SNF K identical to `qlutattn_k188v4_pt`, with whole-head per-token asymmetric V2. Requires the model-specific offline `sign,nf2` mask through `QLUT_CB_MASK`. |
| `qlutattn_k188v2_pt_vtile16` | `qlutattn-k188v2-pt-vtile16c{C}-rv1` | Same SNF K/mask; rescued tile16cC V2. This is the matched SNF tile arm. |
| `qlutattn_rotated_k125v4_pt` | `qlutattn-rotated-k125v4-pt` | **Rotated** per-token sign (k125 + Hadamard): per-channel mean, FWHT-rotate residual → isotropic, per-token sign, de-rotate (FWHT self-inverse). ~1.25 bit K, V 4-bit. Rotation hidden inside K quant (free for attention). 1B full **22.14** (vs sign 21.39). See the rotated-k*v4-pt section for the no-offline-fold / cost rationale. |
| `qlutattn_rotated_k185v4_pt` | `qlutattn-rotated-k185v4-pt` | **Rotated** per-token tern (k185 + Hadamard). ~1.84 bit K. 1B full **24.21** (vs tern 22.54) — approaches per-channel k1v4 24.88 using only sign/tern. |
| `fp16` | `fp16` | Full-precision baseline — no Kitty cache, HF dense fp16 KV. |
| `kivi` | `kivi-k{kbits}v{vbits}` | KIVI-style uniform quant (no promote, no channel-select, no sink). K/V bit-width is set via `KBITS`/`VBITS` (default 2/2 = the old `kivi_2`); the slug encodes the bits so each combo gets its own dir (e.g. `kivi-k2v4`). |
| `kivi_star` | `kivi-star-k{kbits}v{vbits}` | Same as `kivi` but `sink_length=32` (the old `kivi_star_2`). |
| `llamacpp_q40` | `llamacpp-q40` | llama.cpp/ggml **Q4_0** KV cache, faithful port (sim fake-quant): K and V per-token, 32-channel symmetric absmax blocks (`d = signed_max/-8`, fp16 scale → **4.5 bit/value** each), quantize-on-write — **no sink, no fp16 recent window** (`buffer=0`; unique among variants here). head_dim must be a multiple of 32. Known deviations vs llama.cpp (documented, not simulated): PostQuant lets the current step read pre-quant values (one-token difference), and Q stays fp16 (llama.cpp quantizes Q to Q8_0 for the integer dot), so scores are slightly optimistic. |
| `llamacpp_q40_star` | `llamacpp-q40-star` | Q4_0 codebook under the Kitty protection policy (`sink=32` + recent-128 fp16 window) — ablates codebook vs no-sink/no-recent effects. |
| `shadowkv` | `shadowkv` | ShadowKV pure-torch sim (accuracy proxy; no memory/speed savings). |
| `custom` | `custom-kitty` | Custom Kitty config. |

All LongBench variants here run on the pure-torch sim fake-quant path (accuracy
proxy, no real KV-memory savings); `fp16`/`kivi`/`kivi_star` keep dense fp16 KV.

### Sign / SNF rescued V2 `tile16c64` test method

This is the canonical GPU1-only Llama-3.2-1B comparison. Keep the K arm fixed
within each pair and change only V:

| K arm | Per-token V2 baseline | Rescued V2 tile arm |
| --- | --- | --- |
| sign | `qlutattn_k125v2_pt` | `qlutattn_k125v2_pt_vtile16 --v-tile-channels 64` |
| SNF (`f=0.5`) | `qlutattn_k188v2_pt` | `qlutattn_k188v2_pt_vtile16 --v-tile-channels 64` |

Both arms keep `sink=32`, recent FP16 window `128`, `MAX_MODEL_LEN=32768`, and
`LLAMA32_MAX_GEN=256`. The tile arm quantizes only complete 16-token settled
blocks, so at most 15 pending tokens remain FP16. `C=64` is one whole-head tile
on Llama-3.2-1B (`head_dim=64`); on a model with `head_dim=128`, it means two
channel tiles per 16-token block. These are dense-FP16 pure-torch fake-quant
accuracy proxies: the bit accounting is theoretical and no packed kernel or
real KV-memory saving is implemented yet.

The corresponding implementation diagrams are tracked at:

- `docs/figures/vcache_2bit_quantization_strategies.png` — grouping strategies;
- `docs/figures/vcache_tile16c64_rescue_explainer-v2.png` — the rescued tile16c64 pipeline;
- `docs/figures/vcache_rht_formula_explainer.png` — RHT formulas and
  accumulate-then-inverse equivalence.

Configure the ignored repo-root `.env` from `.env.example` before running:

```bash
KITTY_LLAMA32_1B_PATH=/path/to/Llama-3.2-1B-Instruct
LONGBENCH_DATA_ROOT=/path/to/LongBench
KITTY_WIKITEXT2_TRAIN_PATH=/path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet
KITTY_LLAMA32_1B_SNF_MASK=/path/to/Llama-3.2-1B-Instruct.snf-f50.pt
GPU_IDS_CSV=1
GPU_ID=1
```

Do not reuse a `sign,tern` k168 mask for SNF. The current runtime trusts the
mask blob's `codebooks` metadata, so an incorrectly supplied mask can silently
change the K method while retaining an SNF output slug. Generate and validate a
model-specific `sign,nf2`, `f=0.5` mask first. Calibration uses logical
`cuda:0` after `CUDA_VISIBLE_DEVICES=1`, which is physical GPU1:

```bash
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
: "${KITTY_LLAMA32_1B_PATH:?set KITTY_LLAMA32_1B_PATH in .env}"
: "${KITTY_WIKITEXT2_TRAIN_PATH:?set KITTY_WIKITEXT2_TRAIN_PATH in .env}"
: "${KITTY_LLAMA32_1B_SNF_MASK:?set KITTY_LLAMA32_1B_SNF_MASK in .env}"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src "${PYTHON_BIN:-python}" \
  scripts/calibrate_k168v4_pt.py \
  --model "${KITTY_LLAMA32_1B_PATH}" \
  --calib-data "${KITTY_WIKITEXT2_TRAIN_PATH}" \
  --codebooks sign,nf2 --sign-frac 0.5 --target-bits 1.875 \
  --device cuda:0 \
  --output "${KITTY_LLAMA32_1B_SNF_MASK}"

QLUT_CB_MASK="${KITTY_LLAMA32_1B_SNF_MASK}" \
  "${PYTHON_BIN:-python}" - <<'PY'
import os
import torch

payload = torch.load(os.environ["QLUT_CB_MASK"], map_location="cpu", weights_only=True)
mask = payload["codebook_mask"]
assert payload["codebooks"] == ["sign", "nf2"], payload["codebooks"]
assert tuple(mask.shape) == (16, 8, 64), tuple(mask.shape)
assert mask.dtype == torch.uint8 and set(mask.unique().tolist()) == {0, 1}
assert abs(float(payload["low_frac"]) - 0.5) < 1e-9, payload["low_frac"]
print("SNF mask OK:", os.environ["QLUT_CB_MASK"], tuple(mask.shape))
PY
```

Run the 0-GPU core/schedule/wiring and accumulate-then-inverse tests before any
LongBench launch. Expected result for the current suite is 72 tests passing:

```bash
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=src "${PYTHON_BIN:-python}" -m unittest \
  tests.test_vcache_2bit_core \
  tests.test_vcache_2bit_schedule \
  tests.test_vcache_2bit_wiring \
  tests.test_vcache_rht_accumulate_then_restore -v
```

#### Sign: smoke and full

Smoke runs both matched sign arms on two long-context datasets. `V_TILE_CHANNELS=`
is an explicit unset sentinel for the non-tile baseline; do not omit it, because
a stale value in `.env` must not leak into PT2. `--gpus 1` is a single GPU1 slot
and overrides any stale multi-GPU scheduler environment.

```bash
# sign smoke: PT2 baseline, then rescued tile16c64
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
: "${KITTY_LLAMA32_1B_PATH:?set KITTY_LLAMA32_1B_PATH in .env}"
: "${LONGBENCH_DATA_ROOT:?set LONGBENCH_DATA_ROOT in .env}"

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS=sign QLUT_CB_MASK= PROMOTE_RATIO_CONFIG= \
PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=smoke MAX_SAMPLES=2 \
DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k125v2_pt --max-samples 2

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS=sign QLUT_CB_MASK= PROMOTE_RATIO_CONFIG= \
PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=smoke MAX_SAMPLES=2 \
DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k125v2_pt_vtile16 \
  --v-tile-channels 64 --max-samples 2

# -> longbench_out/smoke/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k125v2-pt/{pred,logs}
# -> longbench_out/smoke/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k125v2-pt-vtile16c64-rv1/{pred,logs}
# report prefix in each arm: <arm>/logs/report_<dataset>.json
```

```bash
# sign full: all 21 datasets, 32k, GPU1 only
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
: "${KITTY_LLAMA32_1B_PATH:?set KITTY_LLAMA32_1B_PATH in .env}"
: "${LONGBENCH_DATA_ROOT:?set LONGBENCH_DATA_ROOT in .env}"

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS=sign QLUT_CB_MASK= PROMOTE_RATIO_CONFIG= \
PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=full MAX_SAMPLES=-1 DATASETS_CSV= \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k125v2_pt

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS=sign QLUT_CB_MASK= PROMOTE_RATIO_CONFIG= \
PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=full MAX_SAMPLES=-1 DATASETS_CSV= \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k125v2_pt_vtile16 \
  --v-tile-channels 64

# -> longbench_out/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k125v2-pt/{pred,logs}
# -> longbench_out/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k125v2-pt-vtile16c64-rv1/{pred,logs}
# report prefix in each arm: <arm>/logs/report_<dataset>.json
```

#### SNF: smoke and full

Use the same validated `f=0.5` mask for the PT2 and tile arms so only V changes.

```bash
# SNF smoke: PT2 baseline, then rescued tile16c64
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
: "${KITTY_LLAMA32_1B_PATH:?set KITTY_LLAMA32_1B_PATH in .env}"
: "${LONGBENCH_DATA_ROOT:?set LONGBENCH_DATA_ROOT in .env}"
: "${KITTY_LLAMA32_1B_SNF_MASK:?set KITTY_LLAMA32_1B_SNF_MASK in .env}"

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS= PROMOTE_RATIO_CONFIG= PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QLUT_CB_MASK="${KITTY_LLAMA32_1B_SNF_MASK}" \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=smoke MAX_SAMPLES=2 \
DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k188v2_pt --max-samples 2

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS= PROMOTE_RATIO_CONFIG= PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QLUT_CB_MASK="${KITTY_LLAMA32_1B_SNF_MASK}" \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=smoke MAX_SAMPLES=2 \
DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k188v2_pt_vtile16 \
  --v-tile-channels 64 --max-samples 2

# -> longbench_out/smoke/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k188v2-pt/{pred,logs}
# -> longbench_out/smoke/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k188v2-pt-vtile16c64-rv1/{pred,logs}
# report prefix in each arm: <arm>/logs/report_<dataset>.json
```

```bash
# SNF full: all 21 datasets, 32k, GPU1 only
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
: "${KITTY_LLAMA32_1B_PATH:?set KITTY_LLAMA32_1B_PATH in .env}"
: "${LONGBENCH_DATA_ROOT:?set LONGBENCH_DATA_ROOT in .env}"
: "${KITTY_LLAMA32_1B_SNF_MASK:?set KITTY_LLAMA32_1B_SNF_MASK in .env}"

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS= PROMOTE_RATIO_CONFIG= PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QLUT_CB_MASK="${KITTY_LLAMA32_1B_SNF_MASK}" \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=full MAX_SAMPLES=-1 DATASETS_CSV= \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k188v2_pt

CUDA_VISIBLE_DEVICES=1 GPU_IDS_CSV=1 V_TILE_CHANNELS= VBITS=2 PERTOKEN_BLOCK=1 \
QLUT_BIN_CODEBOOKS= PROMOTE_RATIO_CONFIG= PROMPT_TOKEN_RESERVE=0 FORCE=0 \
QLUT_CB_MASK="${KITTY_LLAMA32_1B_SNF_MASK}" \
QUEST_KERNEL=0 QUEST_TRITON=0 SIM_QUEST=0 QUEST_SIM=0 \
LLAMA32_MODEL_PATH="${KITTY_LLAMA32_1B_PATH}" \
LLAMA32_MODEL_SLUG=llama32-1b-instruct-vtile-gpu1-rv1 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 RUN_MODE=full MAX_SAMPLES=-1 DATASETS_CSV= \
bash scripts/run_exp.sh llama32 --gpus 1 --variant qlutattn_k188v2_pt_vtile16 \
  --v-tile-channels 64

# -> longbench_out/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k188v2-pt/{pred,logs}
# -> longbench_out/llama32-1b-instruct-vtile-gpu1-rv1_qlutattn-k188v2-pt-vtile16c64-rv1/{pred,logs}
# report prefix in each arm: <arm>/logs/report_<dataset>.json
```

Acceptance checks after smoke/full:

- every requested dataset has an `ok` manifest with
  `written_samples == expected_samples` and the expected `run_config_hash`;
- PT2 manifests report `last_v_quant_mode="per_token2"` and
  `v_quantized_tokens>0`;
- tile manifests report `last_v_quant_mode="tile16_rescued"`,
  `last_v_tile_channels=64`, `v_tile_blocks>0`, and `v_quantized_tokens>0`;
- all GPU1-only manifests report `cuda_visible_devices="1"`;
- full runs have 21 dataset JSONL files plus `pred/result.json`; compare the
  unweighted 21-dataset means from the two `result.json` files within each K
  pair. Do not compare smoke means with full means.

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
  bookkeeping only). Do NOT set `PERTOKEN_BLOCK` with these variants: the shell
  `method_slug()` would append `-blk{N}` to the output dir while the Python
  config ignores it (mislabeled dir).
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
cd /mnt/data/tzj/Code/Kitty
PYTHONPATH=src python -m unittest tests.test_q4_0_fakequant -v
```

```bash
# smoke (2 samples/dataset, long-context datasets so the K path is exercised)
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant llamacpp_q40 --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_llamacpp-q40/{pred,logs}
# ablation: --variant llamacpp_q40_star -> .../llama32-1b-instruct_llamacpp-q40-star
```

```bash
# full (all 21 datasets, 32k context)
cd /mnt/data/tzj/Code/Kitty
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant llamacpp_q40
# -> longbench_out/llama32-1b-instruct_llamacpp-q40/{pred,logs}
# (multi-GPU fan-out, explicit override of the GPU1-only rule: --gpus 0,0,0,1,1,1)

# ablation full run (Q4_0 codebook + Kitty policy: sink=32, recent-128 fp16)
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant llamacpp_q40_star
# -> longbench_out/llama32-1b-instruct_llamacpp-q40-star/{pred,logs}

# 4-bit reference points: fp16 (ceiling) and per-channel-K KIVI-4 (4.25b):
KBITS=4 VBITS=4 \
LLAMA32_MODEL_PATH=/mnt/data/tzj/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 1 --variant kivi
# -> longbench_out/llama32-1b-instruct_kivi-k4v4/{pred,logs}
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

Implemented in `kitty_simulate.KittyKVCache._quant_k_pertoken` via flags on
`KittyKVCacheConfig`: `pertoken_pc_submean` (per-channel center + per-token PURE
binary/ternary on the residual — no second per-token submean) and
`pertoken_cb_mask` (per-channel center + an **OFFLINE per-channel sign/tern codebook
mask** — each channel fixed to sign or tern by its wikitext σ², loaded once and used
unchanged; this is the corrected k168v4-pt). A legacy `pertoken_mixed` flag (ONLINE
σ²-binned mixed codebook incl. nf2) still exists but is **superseded** — it cost ~3b
and scored only 14.39 on 1B. The per-channel `μ_d` is computed once at prefill
(`k_pc_mean`) and reused at decode; the codebook mask is offline. All keep V
per-token 4-bit and need **no** smoothed checkpoint (the per-channel center is
self-calibrated from the prompt). Decode is **vectorized across heads**
(`_pt_codebook_masked` / `_masked_lloyd_lastdim`): per-step 41→7.66ms, end-to-end lcc
333→29 s/it, numerically identical to the per-head loop (max diff 2.4e-7).

| variant | codebook on residual | eff. K bit | offline calib | per-channel form |
| --- | --- | ---: | --- | --- |
| `qlutattn_k125v4_pt` | pure sign (1-bit) | ~1.25 | no (single codebook) | `qlutattn-k125v4` |
| `qlutattn_k185v4_pt` | pure tern | ~1.84 | no (single codebook) | `qlutattn-k184v4` |
| `qlutattn_k168v4_pt` | **offline** sign/tern per channel | ~1.68 (nominal) | **yes** (`QLUT_CB_MASK`) | `qlutattn-k1v4` |

Impact (Llama-3.2-1B, full LongBench 21 datasets, 32k): fixing the submean
dimension lifts per-token sign from **10.99** (per-token submean bug) to
**21.68** (`qlutattn-k125v4-pt`, ~1.25 bit) — within 2.1 of the 2.5-bit
`qlutattn_pertoken` nf2 baseline (23.82); fp16 is 27.59. For **k168v4-pt** the
first (ONLINE σ²+nf2) impl was wrong — it binned per-channel σ² into 6 bins incl.
nf2, paid ~3b of per-token side-info, scored only **14.39**, and collapsed
summarization (gov_report 0.64 / multi_news 0.34 / vcsum 0.15). The corrected impl
— **OFFLINE** σ²-ranked sign/tern (no nf2, ~1.68b) — restores summarization (smoke
5-sample: gov_report 0.64→**14.5** / multi_news 0.34→**14.7** / vcsum 0.15→**10.6**);
full 21-dataset mean pending.

Offline calibration (k168v4-pt only; ~1min/card; ranks post-RoPE K residual σ²
per layer over wikitext, low σ²→sign / high σ²→tern, `--target-bits` sets the mix):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_k168v4_pt.py \
  --model /home/zijie/models/Llama-3.2-1B-Instruct \
  --calib-data /home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --target-bits 1.68 \
  --output /home/zijie/models/Llama-3.2-1B-Instruct.k168v4pt_cbmask.pt
# -> mask [n_layers, n_kv, head_dim] uint8 (0=sign, 1=tern); prints realized sign frac + nominal bits
```

```bash
# smoke (2 samples/dataset, long-context datasets exercise the K path)
cd /home/zijie/Code/Kitty
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_k125v4_pt --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn-k125v4-pt/{pred,logs}
# k185v4_pt (tern): same. k168v4_pt (offline sign/tern): prepend
#   QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.k168v4pt_cbmask.pt
```

```bash
# full (all 21 datasets, 32k). 1B runs 3 workers/24GB card -> 18 on 6 cards.
cd /home/zijie/Code/Kitty
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --variant qlutattn_k125v4_pt
# -> longbench_out/llama32-1b-instruct_qlutattn-k125v4-pt/{pred,logs}
# k168v4_pt: prepend QLUT_CB_MASK=.../...k168v4pt_cbmask.pt (run calibrate_k168v4_pt.py first).
# All three are fully vectorized across heads (no nf2 -> no per-head Lloyd loop).
```

Notes: `qlutattn_k168v4_pt` also accepts the alias `qlutattn-k1.68v4-pt`. It needs an
OFFLINE mask via `QLUT_CB_MASK` (from `scripts/calibrate_k168v4_pt.py`) and no longer
uses `QLUT_BIN_CODEBOOKS`. `qlutattn_k125v4_pt` / `qlutattn_k185v4_pt` take a single
`QLUT_BIN_CODEBOOKS` name (default `sign` / `tern`). The legacy online-σ²+nf2 mixed
path is still reachable via `--variant custom` with `pertoken_mixed`, but is
superseded (14.39 on 1B).

## ⭐ 默认推荐优化算法: `qlutattn_k188v4_pt`(per-token sign/nf2 σ²-mix,非旋转)

> **nf2 语义变更(2026-07-14,symnf2-v1)。** 全仓库的 `nf2` 码本已从"每组在线
> Lloyd-Max(k-means,10 轮)"替换为 **IR-QLoRA 对称 NF2 固定 LUT**:归一化电平
> `{−1, −c, +c, +1}`,`c = 0.25256848`。两条路径的均值口径:per-token 掩码路径
> (k188/k168/rotated 系)在 per-channel μ_d 中心化残差上**只做 absmax scale**
> (无二次均值,side=1×fp16 → nf2 名义 **2.25b**,旧 Lloyd 记 2.5b),k188 旋钮公式
> 改为 `bit = f·1.25 + (1−f)·2.25`(f=0.5 → **1.75b**);per-channel 分组路径
> (`apply_codebook`,qlutattn-k1v4/pertoken)= 组内减 μ + absmax + LUT(side=2×fp16,
> 2.25b@G=128 记账不变)。实现:`qlut_quant.nf2_symmetric_lastdim` /
> `KittyKVCache._masked_nf2sym_lastdim`(闭式,无迭代,原 Lloyd 运行时瓶颈消失);
> 0-GPU 测试 `tests/test_nf2_symmetric.py`。哈希隔离:`VariantConfig.nf2_impl="symnf2-v1"`
> 进入 run_config_hash,旧 Lloyd manifest 的 resume/复用会响亮报错,非 nf2 变体哈希
> 逐字节不变。**下文所有含 nf2 的分数(k188 全线、k1v4、pertoken-nf2、rotated-snf、
> SNF V2/NIAH)均为 Lloyd 时代 legacy**,原始结果目录已移至
> `archieve/longbench_out/`(含 smoke)与 `archieve/niah_out/`,对称 NF2 重测 pending。
> 现有 `QLUT_CB_MASK` 掩码(σ² 排序)全部继续有效,无需重新标定;
> `calibrate_k168v4_pt.py` 的 `--target-bits` 反解已按 nf2=2.25 更新。
> **`qlutattn_pertoken` 同步改轴**:该变体现在也走 per-channel μ_d
> (`pertoken_pc_submean=True`,与 k125/k188 同配方;旧的整头 per-token 均值是错误轴),
> 该字段进语义哈希 → 它所有旧结果(含 sign/tern 码本)一并失效;遗留
> `PERTOKEN_OUTLIER_K` 稠密-稀疏组合已显式拒绝(ValueError,不再接线)。另修复:
> `QLUT_BIN_CODEBOOKS` 空字符串现在等价于未设置(run_exp.sh preflight 注入空串,
> 旧行为会让 qlutattn_pertoken/k125v4_pt/rotated 系在 preflight 与 worker 间算出
> 不同 bin_codebooks,触发 "Worker run_config_hash disagrees with shell preflight")。

经过 sign/tern vs sign/nf2 的**全量 Pareto 扫描**(Llama-3.2-1B,21 数据集,32k,
横轴 = K-cache 实际 bit/value 按比例算,纵轴 = LongBench 平均分),**非旋转的
per-token sign/nf2 σ²-混合码本 `qlutattn_k188v4_pt` 是当前默认推荐的优化算法**:
它在「K-cache 实际 bit/value」与「LongBench 平均分」之间给出最优的实用折中,且
decode 友好(纯 per-token、无在线 Hadamard、无 per-channel)。

**它是什么.** `qlutattn_k168v4_pt` 的姊妹方法,把 "rich" 码本从 tern 换成
**nf2(per-token Lloyd,自适应)**,其余完全一致:post-RoPE K 先减 **per-channel
均值 μ**(prefill 自标定、对 attention 免费,`q·μ` 在 softmax 抵消),再按**离线 σ²
标定的 per-channel 码本掩码**把每个通道固定为 sign(1.25b,低 σ²)或 nf2(2.5b,高
σ²),V 走 per-token 4-bit。掩码自带 codebooks,运行时覆盖 `bin_codebooks`。

**bit 旋钮.** 混合比例(sign 通道占比 `f`)是 K bit/value 的**唯一旋钮**:
`bit = f·1.25 + (1−f)·2.5`,在离线标定时由 `--sign-frac` 设定。方法名里的 "188" 只是
`f=0.5`(1.875b)默认点的命名,**没有意义,实际 bit 必须按比例算**。

### 为什么是非旋转(而不是 rotated sign/nf2)

同一条 sign/nf2 混合线,非旋转 vs Hadamard 旋转的全量对比(21 集均分):

| K bit/value(按比例) | sign 占比 f | 非旋转 sign/nf2 | rotated sign/nf2 |
| ---: | ---: | ---: | ---: |
| 1.2500 | 1.00 (=纯 sign) | 21.39 | 22.14 |
| 1.5625 | 0.75 | **24.07** | 23.44 |
| 1.8750 | 0.50 | **25.03** | 24.19 |
| 2.1875 | 0.25 | **25.42** | 25.16 |
| 2.5000 | 0.00 (=纯 nf2) | 25.65 | **26.04** |

基线:fp16 **27.59** / per-channel `qlutattn-k1v4` **24.88** / per-channel KIVI-2 24.24。

- **实用中段(1.56–2.19b)非旋转全面胜过 rotated**:旋转把基底各向同性化,反而破坏了
  σ²-mix「低 σ²→sign / 高 σ²→nf2」的分工(nf2 本就自适应、不需要旋转)。
- **1.875b(f=0.5)的 25.03 是首个超过 per-channel k1v4(24.88)的 per-token 方法**,
  且 decode 更友好(无 per-channel)。
- 仅在纯 nf2 的 2.5b 极端角,旋转才反超(+0.39 → 26.04),那是高 bit 角、不是甜点;为
  保持 decode 简单(无在线 Hadamard)默认选非旋转。需要那 0.39 时再用
  `qlutattn_rotated_snf_pt --sign-frac 0.0`(见下方 rotated 段)。

→ **默认操作点:`f=0.5`(1.875b)的 `qlutattn_k188v4_pt`,全量 25.03,留存 fp16 的
90.7%**;要更高精度就降 `--sign-frac`(更多 nf2 通道)沿 Pareto 线上移到 2.5b/25.65。

### 测试流程(offline 标定 → smoke → full)

第一步永远是离线 σ² 标定生成 per-channel 码本掩码(`--codebooks sign,nf2`,
`--sign-frac` 设定实际 bit)。掩码必须先生成,未设 `QLUT_CB_MASK` 会直接报
`FileNotFoundError`。

```bash
# 1) 离线标定(~1min/卡):sign,nf2 混合,sign-frac 设定实际 K bit/value
#    f=0.5 -> 1.875b(默认 k188 点);要扫 Pareto 就遍历 f∈{1.0,0.75,0.5,0.25,0.0}
cd /home/zijie/Code/Kitty
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_k168v4_pt.py \
  --model /home/zijie/models/Llama-3.2-1B-Instruct \
  --calib-data /home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --codebooks sign,nf2 --sign-frac 0.5 \
  --output /home/zijie/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt
# -> mask [n_layers, n_kv, head_dim] uint8 (0=sign, 1=nf2);打印实际 sign 占比 + nominal bits
```

```bash
# 2) smoke(2 samples/dataset,long-context 数据集确保走到 K 路径)
cd /home/zijie/Code/Kitty
QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_k188v4_pt --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn-k188v4-pt/{pred,logs}
```

```bash
# 3) full(全部 21 数据集,32k;1B 每卡 3 worker -> 6 卡 18)
cd /home/zijie/Code/Kitty
QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --variant qlutattn_k188v4_pt
# -> longbench_out/llama32-1b-instruct_qlutattn-k188v4-pt/{pred,logs}
```

扫不同 bit:改 `--sign-frac`(1.0/0.75/0.5/0.25/0.0),给每个掩码与
`LLAMA32_MODEL_SLUG` 不同后缀(如 `llama32-1b-snf-f25`),避免输出目录碰撞。

注意:
- `qlutattn_k188v4_pt` 与 `qlutattn_k168v4_pt` 共用 `scripts/calibrate_k168v4_pt.py`
  与 `QLUT_CB_MASK`,区别只在 `--codebooks`(`sign,nf2` vs `sign,tern`)。
- nf2 通道走 per-token Lloyd,比纯 sign/tern 略慢,但已向量化(无 per-head 循环),
  1B 每卡仍可 3 worker;empty-mask 行的 NaN 已修(commit a1433d3)。
- 这是纯 torch sim fake-quant 路径(精度代理,不省真实 KV 显存),与其余 qlutattn 一致。

### per-token block 共用码本(`PERTOKEN_BLOCK`)

per-token 量化默认给**每个 token 各自一套码本**(nf2 的 4 个 Lloyd levels / sign 的
per-token mag),side-info 是 per-token 的——这正是 per-token nf2 有效 bit 被抬到 ~2.5
(而非纯 2-bit codeword)的原因。`PERTOKEN_BLOCK=N`(默认 1)把码本粒度放粗成**每 N 个
连续 token 共用一套码本**:量化轴**不变**(仍沿 head_dim),只是把一个 block 的
`(N, head_dim)` 展平进同一量化组,side-info 摊薄 ~N 倍。N=1 即现状(per-token),
完全向后兼容。

- **旋钮 / 接入**:环境变量 `PERTOKEN_BLOCK`(跟随 `PERTOKEN_OUTLIER_K` 命名),仅在
  **offline per-token** 变体上生效:`qlutattn_k188v4_pt`(默认推荐,sign/nf2)、
  `qlutattn_k168v4_pt`(sign/tern)、`qlutattn_rotated_st_pt`、`qlutattn_rotated_snf_pt`。
  **纯 nf2 + block** 用 `k188v4_pt` + 标定 `--sign-frac 0`(全 nf2 掩码),同走 offline
  路径。`pertoken_pc_submean` 系(k125/k185/rotated_k125/k185)与 `qlutattn_pertoken`
  本次**不接** block。
- **block 对所有码本生效**:block 内 sign 通道也共用一个 mag、nf2 通道共用一套 levels
  (在 `_pt_codebook_blocked` 里把 `(block,D)` flatten 进最后一维、复用现有 core
  `_pt_codebook_masked` / `_masked_lloyd_lastdim`,sign/tern/nf2/uni 自动 per-block)。
- **decode 严格 block-aligned**:`k_pt_quant_end[layer]` 指针(prefill 设、decode 推进),
  decode 攒够 `block` 个滑出 recent 窗口的 token 才量化一块,与 prefill 分组一致;尾部
  不满 block 的 token 暂留 fp16(生成结束最多 `block-1` 个 fp16,轻微乐观)。
- **输出目录**:`method_layout_slug` 在 `block>1` 时自动追加 `-blk{N}`(如
  `qlutattn-k188v4-pt-blk16`),`tag` 同加 `_blk{N}`,sweep 不同 block 不会撞目录。
- **bit 账(sim 不自动算)**:codeword 仍 2-bit,side-info 摊薄 block 倍 →
  有效 nf2 bit ≈ `2.0 + (side·16)/(block·D_nf2)`;按既有口径(per-token nf2 ≈2.5,即
  +0.5 side),block=16 → ≈ `2.0 + 0.5/16 ≈ 2.03` bit。**sim 分数只反映精度损失,bit
  收益手算填表。** 0-GPU 已验证 `block=1` 与 per-token core **bit-identical**(diff 0.0)、
  `block=16` 与手工分块逐块一致(含尾块)。

**代码改动落点.** 这次 block 共用码本由三层一起接通:配置层
`KittyKVCacheConfig.pertoken_block` / `get_kvcache_kitty(...pertoken_block)`;量化层
`KittyKVCache._pt_codebook_blocked()` 将 `(block,D)` 展平后复用
`_pt_codebook_masked` / `_masked_lloyd_lastdim`,并用 `k_pt_quant_end[layer]` 做严格
block-aligned decode;运行层 `VariantConfig.pertoken_block` 从 `PERTOKEN_BLOCK` 读取,
同时 `runner.py::method_layout_slug()` 和 `scripts/run_exp.sh::method_slug()` 都在
`block>1` 时追加 `-blk{N}`。因此实际 LongBench 入口仍然只用
`scripts/run_exp.sh`,不要绕过它手写输出目录。

**如何调用.** 最小调用只需在 offline per-token 变体前设置
`PERTOKEN_BLOCK=N` 和对应 `QLUT_CB_MASK`:默认推荐 f50 用
`qlutattn_k188v4_pt` + `--codebooks sign,nf2 --sign-frac 0.5` 标定出的 mask;纯 nf2 用同
variant 但 mask 全置 nf2(等价标定 `--sign-frac 0`)。`PERTOKEN_BLOCK=1` 或不设置即原始
per-token;`PERTOKEN_BLOCK=16` 写入
`longbench_out/..._qlutattn-k188v4-pt-blk16/{pred,logs}`。block sweep 的结果图统一放在
`docs/figures/`: `pareto_llama32-1b.png`, `pareto_minicpm5-1b.png`,
`pareto_llama32-3b.png`, `pareto_qwen3-4b.png`。

**全量结果(Llama-3.2-1B,21 数据集,32k,block=16;GPU 各 3 卡并行)**:

| 掩码 | block1(per-token)| block16 | Δ | 实际 bit(估,side÷16)|
| --- | ---: | ---: | ---: | ---: |
| f50 sign/nf2(50/50)| 25.03 | **24.47** | −0.56 | 1.875 → ~1.52 |
| f00 纯 nf2 | 25.65 | **24.82** | −0.83 | 2.5 → ~2.03 |

fp16 27.59 / per-channel k1v4 24.88(1.68b)。**block16 用 side-info 摊薄 16× 把有效 bit
显著压低,精度只掉 0.5–0.8**:f00 block16 @~2.03b 得 24.82 ≈ per-channel k1v4;f50
block16 @~1.52b 仍 24.47。验证 block 共用码本可行、bit-精度权衡良性。注意 LongBench 输出
目录由 run_exp.sh 的 shell `method_slug()`(非 python)决定,已同步加 `-blk{N}` 后缀。

**多模型全量(block16,21 集,32k;f50=1.875b sign/nf2,f00=纯 nf2 2.5b;3B/Qwen/MiniCPM gen256)**:

| model | fp16 | k188 block1 | block16 f50 | block16 f00 |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.2-1B | 27.59 | 25.03 | 24.47 | 24.82 |
| MiniCPM5-1B | 20.96 | 18.43 | 17.54 | 17.34 |
| Llama-3.2-3B | 36.33 | 33.99 | 32.91 | 31.80 |
| Qwen3-4B-2507 | 45.44 | 44.02 | 42.62 | 41.41 |

block16−block1 损失随模型增大单调递增(−0.56/−0.89/−1.08/−1.40),留存 fp16 的 83.7–93.8%;
**除 1B 外 f50(1.875b)≥ f00(纯 nf2 2.5b)**(sign/nf2 σ²-mix 比纯 nf2 更划算)。3B/MiniCPM 走
llama32 target(MiniCPM5-1B=LlamaForCausalLM)、Qwen3-4B 走 qwen;f00 掩码可由 f50 复制
codebook_mask 全置 nf2 秒生(等价 `--sign-frac 0`)。

```bash
# smoke(2 samples/dataset;同一 mask,block=16 vs 默认 block=1 靠 slug 自动分目录)
cd /home/zijie/Code/Kitty
QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt PERTOKEN_BLOCK=16 \
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_k188v4_pt --max-samples 2
# -> longbench_out/smoke/llama32-1b-instruct_qlutattn-k188v4-pt-blk16/{pred,logs}
```

```bash
# full(全部 21 数据集,32k;扫 PERTOKEN_BLOCK ∈ {1,8,16,32,64} 画 bit-精度曲线)
cd /home/zijie/Code/Kitty
for blk in 1 8 16 32 64; do
  QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt PERTOKEN_BLOCK=$blk \
  LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
  MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
  bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5 --variant qlutattn_k188v4_pt
done
# -> longbench_out/llama32-1b-instruct_qlutattn-k188v4-pt[-blk{N}]/{pred,logs}
# 纯 nf2+block:先用 --sign-frac 0 标定一个全-nf2 掩码,再以同样命令跑。
```

## qlutattn-rotated-k*v4-pt (Hadamard-rotated per-token K)

`qlutattn_rotated_k125v4_pt` (rotated sign) and `qlutattn_rotated_k185v4_pt` (rotated
tern) add a **normalized Hadamard rotation** to the per-token K quant. On the
`pertoken_pc_submean` path: subtract the per-channel mean, **FWHT-rotate the residual**
into an isotropic basis, per-token sign/tern quantize, then **de-rotate** (FWHT is
self-inverse). Rotation spreads the post-RoPE K channel outliers (the KIVI finding) so a
single per-token scale fits every channel.

**de-rotate trick (why attention is untouched).** The stored key is
`k_hat = μ + H·quant(H·(k−μ))`, kept in the ORIGINAL post-RoPE basis. Then
`q·k_hat = q·μ + (Hq)·quant(...)` — `q·μ` is a per-query constant (cancels in softmax) and
`(Hq)·quant(...)` is exactly the rotated-basis dot. So the rotation lives entirely inside
`_quant_k_pertoken` (`_fwht_lastdim` = a vectorized self-inverse FWHT; `pertoken_rotate`
flag); the attention/query path and `q` are NOT modified. Fake-quant accuracy proxy (no
real KV-memory saving), like the rest of qlutattn.

**Results (Llama-3.2-1B, full LongBench 21 datasets, 32k).** Rotation lifts the whole
sign/tern line; tern benefits more than sign:

| variant | K bit | no-rotation | rotated | Δ |
| --- | ---: | ---: | ---: | ---: |
| k125 (sign) | 1.25 | 21.39 | **22.14** | +0.75 |
| k185 (tern) | 1.85 | 22.54 | **24.21** | +1.67 |

rotated-tern @1.85b (24.21) approaches per-channel k1v4 (24.88) and sign/nf2 @1.875b
(25.03) using only sign/tern (no nf2 per-token Lloyd). nf2 still wins the high-bit end
(rotation does NOT beat sign/nf2). sign's real gain is modest vs the synthetic oracle
(14→47% attn recovery) because real post-RoPE outliers are milder than the synthetic 12×
and 1-bit is intrinsically limited.

**Engineering notes (deciding facts for productionizing — kept on purpose).**
- The rotation MATRIX is offline / zero-calibration (a fixed Hadamard needs no data).
  "online" here means *applying* the FWHT at runtime, not calibrating it.
- The rotation **cannot be folded into the weights** in the general case. We rotate the
  **post-RoPE** K, and RoPE is a position-dependent runtime op sitting between `W_k` and the
  attention dot — nothing after RoPE absorbs into a static weight. Folding a dense Hadamard
  *pre-RoPE* makes the score `qᵀHᵀR_Δ H k`, which equals the true `qᵀR_Δ k` only if H commutes
  with RoPE; a dense Hadamard does NOT, so pre-RoPE folding **breaks attention** (not "loses a
  little accuracy"). Only a RoPE-commuting grouped-head rotation (RotateKV) folds, but it
  decorrelates within-head weakly and flattens pre-RoPE (not the harmful post-RoPE) outliers,
  so it cannot match 24.21. Hence "offline-fold + plain existing method + same accuracy" is
  **impossible for the K cache** (QuaRot/KVLinC concur: K needs an online Hadamard; only V,
  which has no RoPE, folds fully).
- **Cost split.** Only the per-channel mean μ is truly prefill-only (computed once, cached,
  reused at decode). The rotation is applied to every token's value, so it runs at BOTH
  prefill (whole region) AND decode (one token/step; real deployment also rotates the one
  query/step). But the decode increment is tiny: one `O(d·log d)` FWHT per token (d=64),
  <0.1% of a decode step's attention+MLP and fusible into the attention kernel — decode
  throughput is effectively unaffected, though NOT literally zero. Verdict: keep it online
  (current behavior); there is no offline-fold variant that preserves the accuracy.

No checkpoint/calibration needed — run directly:

```bash
# smoke (2 samples/dataset)
cd /home/zijie/Code/Kitty
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,gov_report \
bash scripts/run_exp.sh llama32 --gpu 0 --variant qlutattn_rotated_k125v4_pt --max-samples 2
# swap --variant for qlutattn_rotated_k185v4_pt (rotated tern)
```

```bash
# full (all 21 datasets, 32k)
cd /home/zijie/Code/Kitty
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2 --variant qlutattn_rotated_k125v4_pt
# -> longbench_out/llama32-1b-instruct_qlutattn-rotated-k125v4-pt/{pred,logs}
```

Invariants/oracle: `scripts/verify_rotated_oracle.py` (0-GPU) checks orthogonality /
self-inverse / FWHT==matmul(H) all <1e-6 and reproduces the synthetic trend (sign
14→47%, tern 25→67% attn recovery).

### Rotated σ²-mix variants: `qlutattn_rotated_st_pt` / `qlutattn_rotated_snf_pt`

Two MIX variants combine the online Hadamard with an offline codebook mask (k168/k188 +
rotation): `qlutattn_rotated_st_pt` (sign/tern) and `qlutattn_rotated_snf_pt` (sign/nf2).
The offline mask sets each channel's codebook; the **fraction of cheap (sign) channels
controls the effective K bit/value**. At runtime the per-channel-centered residual is
Hadamard-rotated ONLINE, per-bin quantized, then de-rotated (rotation never touches the
offline step). They reuse the `pertoken_offline` path with `pertoken_rotate=True`; the mask
carries its own codebooks.

**Key fact — rotation turns the mask into a pure ratio.** After the Hadamard the basis is
isotropic, so the per-channel σ² heterogeneity the mask sorted by is gone: WHICH channels
get sign vs tern/nf2 no longer matters, only HOW MANY (the ratio). Therefore:
- the offline calibration does NOT need to know about rotation (rotation is online-only);
- reuse the plain `calibrate_k168v4_pt.py` (original-basis σ², **no `--rotate` needed**);
- `--sign-frac` is the single knob → the K bit/value: `bit = f·1.25 + (1−f)·1.85` (sign/tern)
  or `f·1.25 + (1−f)·2.5` (sign/nf2).

**Flow — offline ratio (→bit), then online rotation:**
```bash
# 1) offline: pick ratio f -> bit, generate the mask (NO rotation here; ~1min/card)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_k168v4_pt.py \
  --model /home/zijie/models/Llama-3.2-1B-Instruct \
  --calib-data /home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --codebooks sign,nf2 --sign-frac 0.5 \
  --output /home/zijie/models/Llama-3.2-1B-Instruct.rot_snf_f50.pt
# sign,tern for the st variant; sweep --sign-frac in {1.0,0.75,0.5,0.25,0.0} for the bit axis
```
```bash
# 2) online: QLUT_CB_MASK points at that mask; the FWHT rotation is applied at runtime
QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.rot_snf_f50.pt \
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct LLAMA32_MODEL_SLUG=llama32-1b-rot-snf-f50 \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus 0,0,0,1,1,1,2,2,2 --variant qlutattn_rotated_snf_pt
# -> longbench_out/llama32-1b-rot-snf-f50_qlutattn-rotated-snf-pt/{pred,logs}
```
Set `LLAMA32_MODEL_SLUG` per ratio so a sweep's dirs don't collide. The mix endpoints
(f=1.0 pure sign, f=0.0 pure tern/nf2) coincide with the single-codebook rotated variants
(`qlutattn_rotated_k125v4_pt` / `_k185v4_pt`).

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

## RULER-NIAH (捞针) evaluation for the sign/SNF V2 arms

Standalone needle-in-a-haystack harness following the RULER protocol (NVIDIA,
arXiv 2404.06654): needles are "One of the special magic numbers/uuids for
{key} is: {value}", scoring is `string_match_all` (case-insensitive substring
of each gold value in the greedy 128-token continuation). It complements the
LongBench vtile study with an exact-retrieval stress test. `scripts/run_niah.sh`
is the sole NIAH entry point; do not launch `kitty_sim.cli.eval_niah` by hand
for real runs.

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
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
RULER_REPO_ROOT=/home/zijie/Code/RULER \
NIAH_DATA_ROOT=/home/zijie/data/ruler_niah/llama3 \
bash scripts/prepare_niah_data.sh
# TASKS / LENS / NUM_SAMPLES env override the 4-task x 4-len x 50 default.
```

### Eval + scoring

New first-party modules: `src/kitty_sim/niah/{data,runner,scorer}.py`,
`kitty_sim.cli.eval_niah`, `kitty_sim.cli.score_niah`,
`scripts/plot_niah_montage.py`, tests in `tests/test_niah_wiring.py` (0-GPU).
The runner reuses `build_variant` / `load_model_and_tokenizer` /
`_cache_factory` from the LongBench runner: fresh `KittyKVCache` per sample via
`past_key_values`, greedy `max_new_tokens=128`, prompt =
`build_chat(input) + answer_prefix` (family `llama3` = raw text, matching the
RULER base template and the LongBench vtile-study family). Manifests carry
`run_config_hash` + the NEW_V2 `engagement` evidence
(`last_v_quant_mode`/`v_tile_blocks`); full runs resume, smoke wipes.
Output: `niah_out/[smoke/]<model>_<method>/{pred,logs}`; scoring writes
`pred/result.json` and depth x length heatmap PNGs into `logs/`
(`token_position_answer/length` binned into 10 depth bins — the classic NIAH
heatmap, no controlled-depth regeneration needed).

```bash
# smoke (2 samples, 1 task, 2 lens; ~3 min total on an A100)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
NIAH_DATA_ROOT=/home/zijie/data/ruler_niah/llama3 \
bash scripts/run_niah.sh --gpu 0 \
  --variants fp16,qlutattn_k125v2_pt,qlutattn_k125v2_pt_vtile16,qlutattn_k188v2_pt,qlutattn_k188v2_pt_vtile16 \
  --tasks niah_single_2 --lens 4096,32768 --max-samples 2
# -> niah_out/smoke/llama32-1b-instruct_<method>/{pred,logs}
```

```bash
# full (4 tasks x 4 lens x 50 samples, all 5 arms serially; ~3.5 h on one A100)
cd "$(git rev-parse --show-toplevel)"
QLUT_CB_MASK=/home/zijie/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
LLAMA32_MODEL_PATH=/home/zijie/models/Llama-3.2-1B-Instruct \
NIAH_DATA_ROOT=/home/zijie/data/ruler_niah/llama3 \
bash scripts/run_niah.sh --gpu 0 \
  --variants fp16,qlutattn_k125v2_pt,qlutattn_k125v2_pt_vtile16,qlutattn_k188v2_pt,qlutattn_k188v2_pt_vtile16
# -> niah_out/llama32-1b-instruct_<method>/{pred,logs}; per-arm result.json + heatmaps auto-written

# 5-arm comparison heatmap montage (shared color scale):
PYTHONPATH=src python scripts/plot_niah_montage.py \
  --arms niah_out/llama32-1b-instruct_fp16 \
         niah_out/llama32-1b-instruct_qlutattn-k125v2-pt \
         niah_out/llama32-1b-instruct_qlutattn-k125v2-pt-vtile16c64-rv1 \
         niah_out/llama32-1b-instruct_qlutattn-k188v2-pt \
         niah_out/llama32-1b-instruct_qlutattn-k188v2-pt-vtile16c64-rv1 \
  --labels "fp16" "sign K + V2 PT2" "sign K + V2 tile16c64" "SNF K + V2 PT2" "SNF K + V2 tile16c64" \
  --task pooled --output niah_out/niah_heatmap_montage_pooled.png
```

The k188 arms need the model-specific `sign,nf2` `f=0.5` mask via
`QLUT_CB_MASK` (same file as the LongBench SNF study); tile arms default to
`--v-tile-channels 64` (`V_TILE_C` env). GPU default is 1 (GPU1-only rule);
`--gpu 0` on the A100 host was an explicit user override.

### Results (verified 2026-07-12, A100-80GB GPU0, Llama-3.2-1B, 50 samples/cell)

Mean `string_match_all` over niah_single_1/2/3 + niah_multikey_1
(`llamacpp_q40` is also wired into `run_niah.sh` — Q4_0 needs no masks/env,
same command with `--variants llamacpp_q40`):

| arm | 4k | 8k | 16k | 32k | overall | LongBench-21 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 99.0 | 98.5 | 96.5 | 96.5 | **97.62** | 27.59 |
| llama.cpp Q4_0 (K=V=4.5b) | 99.0 | 98.0 | 97.5 | 93.5 | **97.00** | 26.92 |
| sign K + V2 PT2 | 3.0 | 2.0 | 1.5 | 1.5 | **2.00** | 21.25 |
| sign K + V2 tile16c64 | 3.0 | 2.0 | 1.5 | 1.5 | **2.00** | 21.37 |
| SNF K + V2 PT2 | 29.0 | 23.5 | 17.5 | 12.0 | **20.50** | 24.95 |
| SNF K + V2 tile16c64 | 36.0 | 31.5 | 21.0 | 18.5 | **26.75** | 25.04 |

Findings:

- **Exact retrieval collapses long before LongBench does.** sign-K (~1.25b)
  scores 21+ on LongBench but ~2 on NIAH at every length — K-driven (the two V
  variants tie; per-task cell scores are identical, predictions differ on 14/50
  rows). The only green NIAH cells are the 90-100% depth bin, i.e. needles
  inside/near the recent-128 fp16 window — a window artifact, not retrieval.
- **SNF K (~1.875b) retrieves partially and degrades with length**
  (29 -> 12 for PT2), consistent with "retrieval fails first" from the
  low-bit-K study.
- **tile16c64 V2 beats whole-head PT2 V2 by +6.25 NIAH points at equal K**
  (20.50 -> 26.75), while the same pair is a tie on LongBench (24.95 vs
  25.04) — NIAH separates the V-cache designs that LongBench cannot.
- **Q4_0 (4.5b, no sink/no recent window) is a near-fp16 NIAH reference**
  (97.0 vs 97.62); its only visible dent is multikey_1@32k (78 vs fp16 86),
  so exact retrieval is essentially intact at 4.5 bit even without any
  protection policy.
- fp16 shows no lost-in-the-middle at these lengths (uniformly green).
