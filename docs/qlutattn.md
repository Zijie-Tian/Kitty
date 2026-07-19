# QLUTATTN — the canonical KV-cache quantization variant

`qlutattn` is the single public QLUTATTN variant in this repository. The
algorithm is **fully fixed**: there are no codebook, sign-ratio, K-block,
rotation, or V-tile-size knobs, and no QUEST overlay. Anything that needs a
different algorithm is a different (new) variant, not a configuration of this
one.

```text
variant name / method slug / output slug:  qlutattn
```

## 1. Algorithm

### Q — untouched

Queries stay FP16. QLUTATTN never quantizes Q.

### K-cache — per-token sign/nf2 on the mean-centered residual (nominal ~1.60 bit)

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
   reads it). Channels are ranked per layer across all kv heads jointly:
   - the 65% lowest-ranked channels → **sign**: 1-bit codeword, one
     per-token scale = mean |residual| over the sign channels (~1.25
     bit/value nominal);
   - the 35% highest-ranked channels → **nf2** (`symnf2-v1`): fixed
     symmetric NF2 LUT `{-1, -c, +c, +1}`, `c = 0.25256848...` (IR-QLoRA
     appendix B.2), one per-token absmax scale, **no second mean** (~2.25
     bit/value nominal).
   The mask is loaded once and used unchanged — never recomputed per prompt.
   The 65/35 split is the measured accuracy optimum: nf2 beyond ~35% LOSES
   accuracy because sign + mean-|r| fits low-difficulty channels better than
   the absmax-scaled fixed LUT (mixing is a better quantizer, not merely a
   bit saving).
3. **Nominal K width:** `0.65 × 1.25 + 0.35 × 2.25 = 1.60 bit/value`.

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

## 2. Offline mask: format and validation

Generate one mask per model (model-intrinsic; recalibrate per model):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_qlutattn_mask.py \
  --model      /path/to/model \
  --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --output     /path/to/model.qlutattn_mask.pt
```

The runtime validates the mask strictly **before any model/GPU work** and
refuses to run otherwise:

| field | requirement |
| --- | --- |
| `codebooks` | exactly `["sign", "nf2"]` |
| `codebook_mask` | `torch.uint8`, `ndim == 3`, values exactly `{0, 1}` (0=sign, 1=nf2) |
| per-layer sign count | exactly `round(0.65 × n_kv × head_dim)` in every layer |
| `low_frac` | equal to the mask's ACTUAL sign fraction (±1e-6; integer rounding puts it near but not exactly on 0.65) |
| shape | `[num_layers, num_key_value_heads, head_dim]` of the target model |

Legacy metadata keys (`target_bits`, `nominal_bits`, research-era keys) are
ignored — the mask content, not its historical labels, is what is validated.
The mask file's SHA-256 is embedded in the variant semantic hash and every
run manifest.

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
(fail-fast — no silent KIVI fallback). RULER-NIAH uses the same variant via
`bash scripts/run_niah.sh --variants fp16,qlutattn`.

### Output layout

```text
full  -> longbench_out/<model>_qlutattn/{pred,logs}
smoke -> longbench_out/smoke/<model>_qlutattn/{pred,logs}
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
  non-empty;
- engagement evidence proves the V path actually executed:
  `last_v_quant_mode == "tile16_rescued"`, `last_v_tile_channels == 64`,
  `v_tile_blocks > 0`, `v_quantized_tokens > 0`;
- the worker log shows the offline mask load:
  `[qlutattn-offline] loaded codebook mask ... codebooks=['sign', 'nf2']`.

The runner also hard-fails the first sample if the KV quantization path never
engaged (no silent dense-FP16 mislabelling).

## 5. Limitations

This is a **pure-Torch fake-quant accuracy proxy**: tensors are quantized and
immediately reconstructed to FP16 in the cache. There is no packed KV-cache
storage, so no real memory savings and no speedup is demonstrated by these
runs — they measure accuracy only.

## 6. Figures

- `docs/figures/vcache_tile16c64_rescue_explainer-v2.png` — the rescued
  tile16c64 V pipeline.
- `docs/figures/vcache_rht_formula_explainer.png` — the RHT normalization
  formula used by the V path.
