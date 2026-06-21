# -*- coding: utf-8 -*-
"""Offline calibration for qlutattn-k168v4-pt (sign/tern per-channel mix).

Design (per user): k168v4-pt is a per-token K quant where each post-RoPE K
channel is assigned ONE of two codebooks OFFLINE by its residual sigma^2:
  - low-sigma^2 channels  -> sign  (~1.25 bit, k125v4-pt)
  - high-sigma^2 channels -> tern  (~1.85 bit, k185v4-pt)
The split fraction is fixed here at calibration time (default targets a nominal
~1.68 bit average), producing a per-(layer, kv-head, channel) codebook mask that
the runtime loads and uses unchanged (NOT recomputed per prompt). No nf2, no
online sigma^2 binning.

The per-channel MEAN is NOT calibrated here -- the runtime subtracts a per-channel
mean self-calibrated from each prompt at prefill (free for attention, q.mu cancels
in softmax), exactly like k125v4-pt/k185v4-pt. This script only fixes the cheap
vs rich CODEBOOK assignment, which is a model-intrinsic property.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_k168v4_pt.py \
    --model /home/zijie/models/Llama-3.2-1B-Instruct \
    --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
    --target-bits 1.68 \
    --output /home/zijie/models/Llama-3.2-1B-Instruct.k168v4pt_cbmask.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from kitty_sim.qlut_quant import channel_sigma2  # noqa: E402

# nominal per-token effective bits of each codebook (matches the k125/k185 names)
BITS = {"sign": 1.25, "tern": 1.85, "nf2": 2.5}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calib-data", required=True, help="wikitext train parquet")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--sample-len", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128, help="sigma^2 submean group along tokens")
    p.add_argument("--skip-first", type=int, default=32, help="skip the sink window when measuring")
    p.add_argument("--codebooks", default="sign,tern",
                   help="two codebooks 'low_sigma,high_sigma' from sign/tern/nf2 "
                        "(k168v4-pt=sign,tern; k188v4-pt=sign,nf2)")
    p.add_argument("--target-bits", type=float, default=1.68,
                   help="nominal avg bits -> low-codebook fraction reverse-solved from the two codebooks' bits")
    p.add_argument("--sign-frac", type=float, default=None,
                   help="override: fraction of lowest-sigma^2 channels assigned to the LOW codebook")
    p.add_argument("--per-head", action="store_true",
                   help="rank sigma^2 within each (layer,kv-head); default ranks per-layer over nh*D")
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


def build_mask(sigma2, sign_frac, per_head):
    """sigma2:[nl,n_kv,D] -> codebook mask [nl,n_kv,D] uint8 (0=sign, 1=tern).
    Lowest-sigma^2 `sign_frac` channels -> sign, the rest -> tern."""
    nl, n_kv, D = sigma2.shape
    mask = torch.ones(nl, n_kv, D, dtype=torch.uint8)           # default tern(1)
    for li in range(nl):
        if per_head:
            for h in range(n_kv):
                flat = sigma2[li, h]                            # [D]
                k = int(round(sign_frac * D))
                if k > 0:
                    idx = torch.argsort(flat)[:k]
                    mask[li, h, idx] = 0
        else:
            flat = sigma2[li].reshape(-1)                       # [n_kv*D]
            k = int(round(sign_frac * flat.numel()))
            if k > 0:
                idx = torch.argsort(flat)[:k]
                m = mask[li].reshape(-1)
                m[idx] = 0
                mask[li] = m.reshape(n_kv, D)
    return mask


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cbs = [c.strip() for c in args.codebooks.split(",")]
    if len(cbs) != 2 or any(c not in BITS for c in cbs):
        raise ValueError(f"--codebooks must be two of {list(BITS)} (low_sigma,high_sigma); got {args.codebooks!r}")
    lo, hi = cbs
    if args.sign_frac is not None:
        lo_frac = args.sign_frac
    else:
        lo_frac = (BITS[hi] - args.target_bits) / (BITS[hi] - BITS[lo])
    lo_frac = float(min(max(lo_frac, 0.0), 1.0))

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
    print(f"[calib] codebooks low={lo}({BITS[lo]}b)/high={hi}({BITS[hi]}b) "
          f"target_bits={args.target_bits} -> low_frac={lo_frac:.3f} "
          f"({'per-head' if args.per_head else 'per-layer'} sigma^2 ranking)")

    batches = load_calib_token_batches(tok, args.calib_data, args.num_samples, args.sample_len, args.seed)
    sigma2 = collect_per_channel_sigma2(model, batches, args.skip_first, args.group_size, args.device)
    mask = build_mask(sigma2, lo_frac, args.per_head)

    n_lo = int((mask == 0).sum())
    tot = int(mask.numel())
    frac = n_lo / tot
    nominal = frac * BITS[lo] + (1 - frac) * BITS[hi]
    print(f"[calib] {lo}={n_lo}/{tot} ({frac:.1%}) {hi}={tot - n_lo} "
          f"-> nominal eff bit ~{nominal:.3f}")
    print(f"[calib] sigma^2 global range {sigma2.min():.4e}..{sigma2.max():.4e}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "codebook_mask": mask,                  # [nl, n_kv, D] uint8: 0=low(lo), 1=high(hi)
        "codebooks": [lo, hi],
        "low_frac": lo_frac,
        "target_bits": args.target_bits,
        "nominal_bits": nominal,
        "per_head": args.per_head,
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
