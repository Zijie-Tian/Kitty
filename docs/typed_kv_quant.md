# Typed σ²-binned K-cache Quantization

A per-channel **mixed-codebook** K-cache quantization strategy: within a layer,
different K channels are quantized with **different codebooks**, chosen by the
channel's residual energy σ². Bits are spent where quantization actually hurts —
high-σ² channels get a richer codebook, low-σ² channels get a cheaper one — so the
**average** K bit-width drops below uniform tern while accuracy is preserved (or
improved). It is a post-RoPE, per-channel-affine scheme (no de-RoPE), so the
int-accumulator / scale-fold structure of attention is preserved.

This variant was discovered by an autoresearch bit-minimization loop and validated
on LongBench (Llama-3.2-1B). It is an **accuracy proxy** on the pure-torch
`kitty_sim` fake-quant path: it does not save real memory or run a kernel; the real
kernel would need mixed-codebook packing.

## 1. Why σ² is the difficulty signal

`kitty_sim` quantizes K per-channel along the token axis in 128-token groups, with
`sink` + `recent` tokens kept fp16. The submean codebooks (sign / tern) store each
group's mean **μ in fp16 exactly**, so the quantization error lands entirely on the
residual and is proportional to **σ² = E[(k−μ)²]**. Two consequences:

- **μ²-dominant channels ("red", low σ²)** — energy lives in the exactly-preserved
  mean; the residual is tiny. A 1-bit codebook (sign) already reconstructs them well;
  spending more bits here buys almost nothing.
- **σ²-dominant channels ("blue", high σ²)** — energy lives in the residual; these are
  the error hot-spots, and empirically the channels that retrieval-heavy attention
  depends on most. They need a richer codebook.

Giving every channel the *same* codebook (e.g. uniform tern) is therefore doubly
wasteful: over-spending on red channels and under-spending on blue ones. `typed`
fixes the allocation.

> Note on selector choice: Kitty's default `channel_selection=1` promotes channels by
> magnitude E|K|. Under submean codebooks E|K| ≈ |μ|, so magnitude promotes the
> *red* channels whose residual is already small — the wrong target. The right signal
> is σ² (variance). Measured magnitude-vs-variance top-k overlap is ~9% (nearly
> orthogonal), and the σ² selector tracks the per-channel MSE oracle.

## 2. Design

### Channel binning (static, task-independent)

For each layer, compute every channel's residual σ² over the quantization region and
sort channels into **`n_bins` equal-count quantile bins** (`bin 0` = lowest σ²,
`bin n_bins-1` = highest). This classification is a stable model property: per-channel
σ² profiles are nearly identical across LongBench tasks (pairwise r ≥ 0.99) and an
offline σ² calibration on out-of-domain text (wikitext) captures ~97% of each task's
own oracle channel-selection energy — so the bins can be fixed offline.

In the LongBench cache the bins are computed **once** from the prompt's quant region
at prefill and reused for every buffer flush and decode step.

### Codebook policy

A policy is a list `bin_codebooks` mapping each σ²-bin to a codebook. The autoresearch
winner (6 bins, Llama-3.2-1B):

| bin | σ² | codebook | bits |
| ---: | --- | --- | ---: |
| 0 | lowest | `sign` | 1.25 |
| 1 |  | `sign` | 1.25 |
| 2 |  | `sign` | 1.25 |
| 3 | mid | `tern` | 1.83 |
| 4 |  | `nf2` | 2.25 |
| 5 | highest | `nf2` | 2.25 |

`bin_codebooks = ["sign","sign","sign","tern","nf2","nf2"]`

### Codebooks

All post-RoPE, per 128-token group; effective bit = codeword bits + fp16 side-info
(16 bit each) / group. With `group=128`, side-info = `16·n_side/128`.

| codebook | levels | side info | eff. bits | note |
| --- | --- | --- | ---: | --- |
| `meanonly` | μ only (0 codeword) | μ | 0.125 | store mean, drop residual |
| `sign` | μ ± m (1 bit) | μ, m | 1.25 | 1-bit Lloyd-optimal residual |
| `tern` | μ−m, μ, μ+m (log₂3) | μ, m | 1.83 | dead-zone three-level (τ=0.5) |
| `uni2` | 4 (2 bit) | min, scale | 2.25 | uniform min-max 2-bit |
| `nf2` | 4 (2 bit) | scale, zp | 2.25 | per-group Lloyd-optimal 4 levels |
| `uni3` | 8 (3 bit) | min, scale | 3.25 | uniform min-max 3-bit |

`nf2` runs a per-group 1-D Lloyd-Max (k-means, 10 iters) — the optimal 4-level scalar
quantizer, which fits the bimodal (arcsine) distribution of fast-RoPE blue channels
far better than uniform/tern. It is the runtime bottleneck; a deployable form is an
offline-calibrated NF4-style fixed-shape codebook at the same 2.25 bit.

### Bit accounting (winner, Llama-3.2-1B, 6 equal bins ≈ 16.7% each)

| channels | uniform tern | typed | Δ |
| --- | ---: | ---: | --- |
| red (bins 0–2, 50%) | 1.83 | sign 1.25 | −0.58 |
| mid (bin 3, ~17%) | 1.83 | tern 1.83 | 0 |
| blue (bins 4–5, ~33%) | 1.83 | nf2 2.25 | +0.42 |
| **average** | **1.83** | **1.68** | **−0.15** |

The bits saved on the (near-lossless) red channels are reinvested into the blue
channels where the marginal accuracy gain is large — a rate-distortion bit-allocation
("water-filling": more bits where the channel is harder).

## 3. Mechanism (why it wins)

- Marginal gain of adding bits to a red channel ≈ 0 (residual already tiny); marginal
  gain on a blue channel is large. The optimal allocation moves bits red→blue, which is
  exactly what `typed` does.
- Fast-RoPE blue channels have a bimodal arcsine post-RoPE distribution (the rotation
  sweeps cos/sin within a group). sign/tern/uniform all mismatch it; `nf2`'s free
  4 levels fit the two modes — so blue channels specifically get `nf2`.
- Everything stays post-RoPE per-channel affine `k̂ = s·c + z`, so the per-channel
  scale folds into the query side and the QK inner loop stays integer (de-RoPE, which
  is more accurate on rotation-blue channels, is intentionally **not** used because it
  breaks that int structure).

## 4. Implementation

- `src/kitty_sim/typed_quant.py` — codebooks (`apply_codebook`), Lloyd (`_lloyd`),
  σ² binning (`channel_sigma2`, `sigma2_bins`, `compute_sigma_bins`), buffer quant
  (`fake_quant_typed_buffer`), and offline-proxy helpers (`apply_typed`,
  `effective_bits`).
- `src/kitty_sim/kitty_simulate.py` — `KittyKVCacheConfig`/`KittyKVCache` gain
  `k_codebook` (`"kivi"` default = existing min-max path; `"typed"`), `bin_codebooks`,
  `n_bins`. The typed branch (`_ensure_k_bins` + `_quant_k_buffer`) computes per-layer
  σ²-bins once at prefill and applies per-bin codebooks at every K flush. V is
  unchanged (per-token, `vbits`).
- `src/kitty_sim/longbench/runner.py` — variants `typed` (winner policy, override via
  `TYPED_BIN_CODEBOOKS`) and `tern_uniform` (all-tern K baseline), both V 4-bit.

The `k_codebook="kivi"` default means every existing variant is byte-for-byte
unchanged.

## 5. Usage

### 5a. LongBench accuracy test (the deliverable)

`typed` (winner, K≈1.68 bit) vs `tern_uniform` (uniform tern K≈1.83 bit, the iso-tern
baseline) vs `fp16` (ceiling). V is per-token 4-bit for both quantized variants.

```bash
cd <repo>
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpu 0 --variant typed          # -> longbench_out/llama32-1b-instruct_typed
bash scripts/run_exp.sh llama32 --gpu 0 --variant tern_uniform   # -> longbench_out/llama32-1b-instruct_tern-uniform
bash scripts/run_exp.sh llama32 --gpu 0 --variant fp16           # ceiling
```

Smoke (2 samples/dataset; scope to long-context datasets so the K path is exercised):

```bash
LLAMA32_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpu 0 --variant typed --max-samples 2
```

Fan one variant's 21 datasets across several GPUs (faster, opt-in override of the
GPU1-only rule): replace `--gpu 0` with `--gpus 0,1,3`. Override the policy without
code edits via `TYPED_BIN_CODEBOOKS=sign,sign,sign,tern,nf2,nf2`.

Scoring is automatic when the run finishes; for a partial/preliminary score over the
datasets already finished, use `--no-strict-complete`:

```bash
python -m kitty_sim.cli.score_longbench \
  --model longbench_out/llama32-1b-instruct_typed/pred --no-strict-complete
```

### 5b. Offline overlap proxy + policy search (no LongBench)

A fast (~seconds) top-32 attention-overlap proxy over a cached 32k doc — used to
search policies before paying for LongBench.

```bash
# 1) one-time cache (post-RoPE K + real queries + RoPE periods)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/build_kq_cache.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --longbench-dir /path/to/LongBench/data --tag llama32-1b
# -> probe_out/kq_cache_llama32-1b.pt

# 2) evaluate a policy: prints eff_bits + overlap, writes probe_out/typed_policy_eval.json
printf '{"group_size":128,"bin_codebooks":["sign","sign","sign","tern","nf2","nf2"]}' > /tmp/pol.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/eval_typed_policy.py \
  --cache probe_out/kq_cache_llama32-1b.pt --policy /tmp/pol.json
```

The autoresearch metric is `eff_bits` (minimize) with guard `overlap ≥ 0.774`
(iso-tern). Baseline all-tern = 1.835 bit / overlap 0.774.

### Policy JSON

```json
{ "group_size": 128, "bin_codebooks": ["sign","sign","sign","tern","nf2","nf2"] }
```

`n_bins` is `len(bin_codebooks)`. `bin_codebooks[i]` is the codebook for the i-th σ²
quantile bin (bin 0 = lowest σ²). Codebook names: see §2.

## 6. Results (Llama-3.2-1B)

**Overlap proxy** (single 32k doc, top-32 attention overlap):

| policy | eff. K bits | overlap |
| --- | ---: | ---: |
| all-tern (uniform baseline) | 1.835 | 0.774 |
| **typed `[s,s,s,t,nf2,nf2]`** | **1.680** | **0.777** |

`typed` Pareto-dominates uniform tern (lower bits **and** higher overlap). Below
1.68 bit overlap drops below the 0.774 iso-tern wall.

**LongBench** (full 21 datasets, Llama-3.2-1B, 32k context):

| | fp16 | tern_uniform (1.83b) | typed (1.68b) |
| --- | ---: | ---: | ---: |
| avg (21) | 27.63 | 23.44 | **25.26** |
| retain vs fp16 | 100% | 84.8% | **91.4%** |

typed (fewer bits) beats uniform tern by **+1.82** on the full 21, closing the gap to
fp16 from tern's 84.8% to 91.4%. Gains are concentrated on the K-fidelity-sensitive
retrieval tasks (trec 65.00 = fp16 vs tern 56.00; multifieldqa_en +6.1, qasper +4.4,
hotpotqa +3.9, lsht +2.8); the only dataset where typed slightly trails tern is
repobench-p (−0.26). The full-21 margin (+1.82) is smaller than the 9-hardest-dataset
margin (+2.66 → 22.82 vs 20.16) because saturated easy datasets (passage_count,
passage_retrieval) dilute the average — the direction is unchanged. The real-score win
exceeds the overlap proxy because typed spends bits exactly on the retrieval-critical
high-σ² channels.

## 7. Caveats

- Pure-torch `kitty_sim` fake-quant: accuracy proxy only, **no memory/speed savings**;
  a real kernel needs mixed-codebook packing.
- `nf2` is measured as the per-group Lloyd upper bound; deployable form is an
  offline-calibrated NF4-style fixed-shape codebook (same 2.25 bit). Lloyd k-means is
  the runtime bottleneck for the LongBench eval.
- Validated on Llama-3.2-1B only. Qwen3-8B has a different σ² mix (more genuine
  heavy-tail "blue" channels) and may favor a different policy.
- Post-RoPE only by design (preserves the int path). de-RoPE is more accurate on
  rotation-blue channels but breaks per-channel affine structure and is excluded here.
