# QLUTATTN — the canonical KV-cache quantization variant

`qlutattn` is the single public QLUTATTN variant in this repository. Its
quantizers and runtime schedule are fixed: Q remains FP16, K uses the
sign/`symnf2-v1` pair, V uses rescued tile16c64, and QUEST is forbidden.
`QLUT_CB_MASK` is still the only QLUTATTN-specific runtime input.

The default artifact remains the canonical fixed-65/35 selector. A versioned
research artifact may instead carry a static layer-channel top-p selector. The
artifact is generated offline, loaded once, and never rescored or reassigned at
inference. Canonical artifacts with no research metadata retain their historical
behavior, slug, and semantic hash.

```text
variant name: qlutattn
canonical method slug: qlutattn
top-p method slug: qlutattn-topp-p<threshold>-m<full-mask-sha256>
```

## 1. Algorithm

### Q — untouched

Queries stay FP16. QLUTATTN never quantizes Q.

### K-cache — per-token sign/nf2 on the mean-centered residual

Post-RoPE keys are quantized **per token** along `head_dim`:

1. **Per-channel mean removal.** At prefill, a per-channel mean `mu_d`
   (`[n_kv_heads, head_dim]`) is computed from the prompt's quantized region
   and subtracted before quantization; decode reuses the cached `mu_d`. This
   is free for attention: `q · mu` is a per-query constant that cancels in
   softmax.
2. **Offline per-channel codebook mask.** Each channel is assigned ONE of two
   codebooks by an offline calibration
   (`scripts/calibrate_qlutattn_mask.py`, one wikitext pass collecting both
   statistics). The ranking signal is **`sigma^2 × E|q|`** — residual variance
   (how hard the channel is to quantize) times mean absolute post-RoPE query
   activation, query heads averaged per GQA group (how much attention actually
   reads it). The canonical artifact ranks channels per layer across all KV
   heads jointly:
   - the 65% lowest-ranked channels → **sign**: 1-bit codeword, one
     per-token scale = mean |residual| over the sign channels (~1.25
     bit/value nominal);
   - the 35% highest-ranked channels → **nf2** (`symnf2-v1`): fixed
     symmetric NF2 LUT `{-1, -c, +c, +1}`, `c = 0.25256848...` (IR-QLoRA
     appendix B.2), one per-token absmax scale, **no second mean** (~2.25
     bit/value nominal).
   A versioned `layer_channel_top_p` research artifact instead selects, in
   every layer, the shortest score-descending prefix whose FP64 cumulative
   score mass reaches the recorded threshold. A deterministic head-local
   permutation makes each head's NF2/sign segments contiguous; inference
   gathers, quantizes with the same kernels, and inverse-gathers before
   attention. Every assignment remains frozen per model and prompt-independent.
3. **K width:** the canonical artifact has nominal width
   `0.65 × 1.25 + 0.35 × 2.25 = 1.60 bit/value` under its original
   head-dimension convention. Research artifacts record exact packed
   codeword-plus-scale cost; sink/recent FP16 windows and `mu_d` are excluded
   from that selector-only comparison.

Explicitly **not** part of the algorithm: Hadamard/FWHT rotation,
SmoothAttention, online sigma^2 binning, per-token Lloyd fitting, outlier
dense-and-sparse side paths, block-shared per-token codebooks (the block size
is fixed at 1), per-head fraction equalization, RoPE-pair binding, per-layer
fraction schedules, and token-level tiering (all measured non-positive under
this signal).

### V-cache — rescued 2-bit tile16c64 (`rht-pcaff-mse1-bias-v1`)

Values are quantized in tiles of **16 consecutive tokens × 64 channels**:

1. Fixed Rademacher-sign RHT (seed `20260711`) rotates each head's values.
2. Frozen per-channel affine (mean/RMS calibrated once on the prompt's
   settled region) normalizes the rotated values.
3. Each tile gets an asymmetric 2-bit min-max quantizer, refined by exactly
   **one** alternating least-squares MSE iteration with a per-tile SSE
   fallback to the FP16 min-max baseline.
4. De-normalization, inverse RHT, and a frozen per-channel bias correction
   reconstruct the cached FP16 values.

The V **codeword is 2-bit**; per-tile scales and per-channel side statistics
push the theoretical effective width slightly above 2 bit/value
(`2 + 32/(16·C) + 48/T_quantized` in the quantized region at `C = 64`).

### Protection windows

- `sink_length = 32` — the first 32 tokens stay FP16.
- `buffer_length = 128` — the most recent 128 tokens stay FP16.
- `group_size = 128` (bookkeeping for the recent-window schedule).
- V tiles flush strictly in 16-token blocks: a trailing partial tile (fewer
  than 16 settled tokens) stays FP16 until it fills.

### Exact theoretical K storage accounting

Selector-region cost and full-cache cost are different quantities. For head
`(l,h)` with `k_lh` NF2 channels and head dimension `D`:

```text
code_bits_lh/token  = 2*k_lh + (D-k_lh)
scale_bits_lh/token = 16*I(k_lh>0) + 16*I(k_lh<D)
B_region            = sum_lh(code_bits_lh + scale_bits_lh)
```

For total sequence length `T`, `N=L*H_kv*D` K values/token, protected FP16
tokens `P=sink+recent`, quantized tokens `T_q=T-P`, and packed metadata
`B_meta`:

```text
full_cache_K_bpv = (T_q*B_region + P*16N + B_meta) / (T*N)
```

The 32K Llama-3.2-1B accounting uses `L=16`, `H_kv=8`, `D=64`,
`N=8192`, `T=32768`, `P=160`, and `T_q=32608`. Packed metadata assumes one
FP16 prompt mean/channel, a 1-bit mask/channel, and—for top-p artifacts—both
permutations at `ceil(log2 64)=6` bits/index. This is 17 KiB for canonical and
29 KiB for top-p, counted once conservatively for one batch-1 cache. Calibration
statistics and provenance digests are excluded because they are not packed
hot-path state.

These figures remain theoretical: the fake-quant implementation stores
reconstructed K in FP16 and prompt means in FP32. The artifact mask is uint8,
but cache construction converts it to int64 `k_cb_mask`; reorder/inverse
indices are int64 and `nf2_count_per_head` is int32. It does not realize the
packed memory footprint.

## 2. Offline mask: formats and validation

Generate one canonical mask per model (model-intrinsic; recalibrate per model):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_qlutattn_mask.py \
  --model      /path/to/model \
  --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --stats-output /path/to/model.qlutattn_stats.pt \
  --output     /path/to/model.qlutattn_mask.pt
```

Generate additional top-p artifacts on CPU from the same saved statistics;
this performs no second model pass:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/calibrate_qlutattn_mask.py \
  --stats-input /path/to/model.qlutattn_stats.pt \
  --top-p 0.62 \
  --output /path/to/model.qlutattn_topp_p62.pt
```


The runtime validates the artifact strictly **before any model/GPU work** and
refuses to run otherwise:

| artifact | required invariants |
| --- | --- |
| common | `codebooks == ["sign", "nf2"]`; `codebook_mask` is CPU `uint8`, rank 3, binary, and matches `[num_layers, num_key_value_heads, head_dim]`; `low_frac` matches the actual mask |
| canonical | every layer has exactly `round(0.65 × n_kv × head_dim)` sign entries |
| top-p format v3 | fixed selector/tie/dtype metadata; strictly recomputed FP64 score, per-layer cumulative boundary, mask, per-head reorder/inverse, counts, ratios, exact packed K cost, and calibration provenance |

The mask file's full SHA-256 is embedded in the variant semantic hash and every
run manifest. Top-p output slugs also use the full digest, so two different
artifacts cannot share a result directory through a short-prefix collision.
Legacy `target_bits` / `nominal_bits` metadata on a canonical artifact remains
ignored. The retired uniform fixed-top-k control format v2 and its
`--uniform-top-k-control-from` / `--control-kind` CLI flags are hard errors,
not legacy canonical metadata. Regenerate a canonical or top-p v3 artifact;
there is no compatibility shim.

## 3. Running it

`scripts/run_exp.sh` is the only LongBench entry point. `QLUT_CB_MASK` is the
only QLUTATTN-specific runtime input.

```bash
# smoke (2 samples/dataset, wiped every launch)
QLUT_CB_MASK=/path/to/model.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/model LLAMA32_MODEL_SLUG=<model-slug> \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
RUN_MODE=smoke MAX_SAMPLES=2 DATASETS_CSV=multifieldqa_en,hotpotqa \
bash scripts/run_exp.sh llama32 --gpus <ids> --variant qlutattn --max-samples 2
```

```bash
# full (all 21 datasets; do not lower MAX_MODEL_LEN)
QLUT_CB_MASK=/path/to/model.qlutattn_mask.pt \
LLAMA32_MODEL_PATH=/path/to/model LLAMA32_MODEL_SLUG=<model-slug> \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 \
bash scripts/run_exp.sh llama32 --gpus <ids> --variant qlutattn
```

Model constraints (validated config-only, before weights load): FP16 model;
`head_dim` a power of two divisible by 64; GLM-family models are rejected
(fail-fast — no silent KIVI fallback). NVIDIA RULER uses the same canonical
variant via `bash scripts/run_ruler.sh --variants fp16,qlutattn`.

### Output layout

```text
canonical full  -> longbench_out/<model>_qlutattn/{pred,logs}
top-p full      -> longbench_out/<model>_qlutattn-topp-p<P>-m<SHA256>/{pred,logs}
smoke           -> the same method-qualified layout under longbench_out/smoke/
```

### Retired knobs

The algorithm being fixed, the historical tuning surface is gone. The retired
environment knobs are hard errors when non-empty (empty string = unset
sentinel), `VBITS` accepts only unset/empty/`2`, `PERTOKEN_BLOCK` only
unset/empty/`1`, and combining any QUEST overlay flag with
`--variant qlutattn` is a hard error (QUEST remains available for the
kitty/kivi variants).

## 4. Run acceptance (manifest + engagement)

A qlutattn run is valid only when, for every dataset:

- `manifest.status == "ok"`, `written_samples == expected_samples`, JSONL row
  count matches;
- `run_config_hash` non-empty (shell preflight and Python worker recompute and
  must agree before model loading);
- `variant.name == "qlutattn"`, `nf2_impl == "symnf2-v1"`, `mask_sha256`
  non-empty; top-p artifacts also carry their selector metadata;
- engagement evidence proves the K path actually executed:
  `k_quant_calls > 0`, `k_quantized_tokens > 0`,
  `k_prompt_mean_layers > 0`, `last_k_quant_mode == "per_token:qlut"`;
- engagement evidence proves the V path actually executed:
  `last_v_quant_mode == "tile16_rescued"`, `last_v_tile_channels == 64`,
  `v_tile_blocks > 0`, `v_quantized_tokens > 0`;
- the worker log shows the offline mask load:
  `[qlutattn-offline] loaded codebook mask ... codebooks=['sign', 'nf2']`.

The runner hard-fails the first sample if either required cache path did not
engage; it never accepts dense-FP16 execution mislabeled as QLUTATTN.

## 5. Limitations

This is a **pure-Torch fake-quant accuracy proxy**: tensors are quantized and
immediately reconstructed to FP16 in the cache. There is no packed KV-cache
storage, so no real memory savings and no speedup is demonstrated by these
runs — they measure accuracy only.

The layer-channel top-p format is a research selector, not a new packed kernel
or a new default. Results must report its artifact digest, exact K cost, and
calibration/model scope. Evidence from one `head_dim=64` model is model-specific
and is not a cross-model claim.

## 6. Llama-3.2-1B top-p research result

The 2026-07-23 study used Llama-3.2-1B only. Two stable thresholds produced
quality/bit tradeoffs against canonical fixed-65/35:

|selector|quant-region K bpv|full-cache K bpv @32K|full-cache saving|test PPL|full LongBench|
|---|---:|---:|---:|---:|---:|
|canonical fixed-65/35|1.849609|1.919222|0%|13.767101|24.529048|
|top-p `p=0.62`|1.741089|1.811597|5.608%|13.742994|24.300476|
|top-p `p=0.60`|1.719238|1.789854|6.741%|13.777823|24.120476|

Compared with the canonical mask, `p=0.62` saves `5.867%` in the quantized
region and `5.608%` in the theoretical full 32K K cache while losing `0.229`
LongBench points; `p=0.60` saves `7.049%` in the region and `6.741%` full-cache
while losing `0.409`. Classification: **single-model quality/bit tradeoff versus
canonical**. It does not justify changing the default.

Follow-up robustness probes did not change the selected artifact. Across three
calibration seeds, focused means were `49.963±0.774` at `p=0.60`,
`51.027±0.251` at `p=0.62`, and `50.693±0.467` at `p=0.65` (population
standard deviation). Seed1's slightly higher `p=0.62` focused score did not
transfer to full PPL (`2.620574766` average NLL versus seed0
`2.620529133`). Neighbor thresholds `0.618/0.622`, 256-sample calibration,
4096-token calibration windows, and `skip_first=128` all scored below the
seed0 `p=0.62` focused result. Seed0 remains fixed; no multi-objective selector
was introduced.


## 7. Figures

- `docs/figures/vcache_tile16c64_rescue_explainer-v2.png` — the rescued
  tile16c64 V pipeline.
- `docs/figures/vcache_rht_formula_explainer.png` — the RHT normalization
  formula used by the V path.
