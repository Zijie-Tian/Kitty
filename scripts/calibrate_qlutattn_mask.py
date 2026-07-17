# -*- coding: utf-8 -*-
"""Offline codebook-mask calibration for the canonical qlutattn variant.

qlutattn quantizes the post-RoPE K cache per token on the per-channel-mean-
centered residual. Each channel is assigned ONE of two codebooks OFFLINE by
its residual sigma^2 on a calibration corpus:
  - the 50% lowest-sigma^2 channels  -> sign (1.25 bit nominal)
  - the 50% highest-sigma^2 channels -> nf2  (symnf2-v1, 2.25 bit nominal)
The split is FIXED at 50/50, giving a nominal K width of
0.5 * 1.25 + 0.5 * 2.25 = 1.75 bit/value. The runtime loads the mask and uses
it unchanged (NOT recomputed per prompt); there is no online sigma^2 binning.

The per-channel MEAN is NOT calibrated here -- the runtime subtracts a
per-channel mean self-calibrated from each prompt at prefill (free for
attention, q.mu cancels in softmax). This script only fixes the sign-vs-nf2
CODEBOOK assignment, which is a model-intrinsic property.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_qlutattn_mask.py \
    --model /path/to/model \
    --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
    --output /path/to/model.qlutattn_mask.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from kitty_sim.qlut_quant import channel_sigma2  # noqa: E402

# The canonical qlutattn codebook pair and split are fixed; they are not knobs.
CODEBOOKS = ("sign", "nf2")
SIGN_FRACTION = 0.5
# nominal per-token bits: sign = 1-bit codeword + one fp16 scale per 64-channel
# token group; nf2 = 2-bit symnf2-v1 codeword + one fp16 absmax scale (no
# second mean).
BITS = {"sign": 1.25, "nf2": 2.25}
NOMINAL_K_BITS = SIGN_FRACTION * BITS["sign"] + (1 - SIGN_FRACTION) * BITS["nf2"]  # 1.75
NF2_IMPL = "symnf2-v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calib-data", required=True, help="wikitext train parquet")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--sample-len", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128, help="sigma^2 submean group along tokens")
    p.add_argument("--skip-first", type=int, default=32, help="skip the sink window when measuring")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_calib_token_batches(tokenizer, parquet_path, num_samples, sample_len, seed):
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": parquet_path})["train"]
    texts = [t for t in ds["text"] if t and not t.isspace()]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]
    need = num_samples * sample_len
    if ids.numel() < need:
        raise RuntimeError(f"corpus too small: {ids.numel()} < {need}")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - sample_len, (num_samples,), generator=g)
    return torch.stack([ids[s:s + sample_len] for s in starts])


def cache_layer_keys(cache, layer_idx):
    try:
        keys, _ = cache[layer_idx]
        return keys
    except Exception:
        pass
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx]
    raise TypeError(f"Unsupported cache type: {type(cache)}")


@torch.no_grad()
def collect_per_channel_sigma2(model, batches, skip_first, group_size, device):
    """Per-layer per-kv-head per-channel post-RoPE K residual sigma^2, averaged
    over calib samples. Returns [n_layers, n_kv, D]."""
    nl = model.config.num_hidden_layers
    accum = None
    n = 0
    for bi in range(batches.shape[0]):
        ids = batches[bi:bi + 1].to(device)
        cache = model(ids, use_cache=True).past_key_values
        per_layer = []
        for li in range(nl):
            keys = cache_layer_keys(cache, li)                  # [1, n_kv, T, D] post-RoPE
            x = keys[0, :, skip_first:, :].transpose(1, 2)      # [n_kv, D, T]
            per_layer.append(channel_sigma2(x.float().cpu(), group_size))  # [n_kv, D]
        per_layer = torch.stack(per_layer)                      # [nl, n_kv, D]
        accum = per_layer if accum is None else accum + per_layer
        n += 1
        del cache
        if (bi + 1) % 16 == 0:
            print(f"  [calib] {bi + 1}/{batches.shape[0]} samples")
    return accum / max(n, 1)


def build_mask(sigma2):
    """sigma2:[nl,n_kv,D] -> codebook mask [nl,n_kv,D] uint8 (0=sign, 1=nf2).
    Per layer, the lowest-sigma^2 half of the nh*D channels -> sign, rest -> nf2."""
    nl, n_kv, D = sigma2.shape
    mask = torch.ones(nl, n_kv, D, dtype=torch.uint8)           # default nf2(1)
    for li in range(nl):
        flat = sigma2[li].reshape(-1)                           # [n_kv*D]
        k = int(round(SIGN_FRACTION * flat.numel()))
        if k > 0:
            idx = torch.argsort(flat)[:k]
            m = mask[li].reshape(-1)
            m[idx] = 0
            mask[li] = m.reshape(n_kv, D)
    return mask


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    print(f"[calib] model={args.model} layers={cfg.num_hidden_layers} q={n_q} kv={n_kv} D={head_dim}")
    print(f"[calib] codebooks low=sign({BITS['sign']}b)/high=nf2({BITS['nf2']}b, {NF2_IMPL}) "
          f"sign_fraction={SIGN_FRACTION} -> nominal K ~{NOMINAL_K_BITS} bit/value")

    batches = load_calib_token_batches(tok, args.calib_data, args.num_samples, args.sample_len, args.seed)
    sigma2 = collect_per_channel_sigma2(model, batches, args.skip_first, args.group_size, args.device)
    mask = build_mask(sigma2)

    n_lo = int((mask == 0).sum())
    tot = int(mask.numel())
    frac = n_lo / tot
    print(f"[calib] sign={n_lo}/{tot} ({frac:.1%}) nf2={tot - n_lo}")
    print(f"[calib] sigma^2 global range {sigma2.min():.4e}..{sigma2.max():.4e}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "codebook_mask": mask,                  # [nl, n_kv, D] uint8: 0=sign, 1=nf2
        "codebooks": list(CODEBOOKS),
        "low_frac": SIGN_FRACTION,
        "nominal_bits": NOMINAL_K_BITS,
        "nf2_impl": NF2_IMPL,
        "group_size": args.group_size,
        "skip_first": args.skip_first,
        "model": args.model,
        "n_layers": cfg.num_hidden_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "sigma2": sigma2,                       # [nl, n_kv, D] for inspection
    }, out)
    print(f"[calib] wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
