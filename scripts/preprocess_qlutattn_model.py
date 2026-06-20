#!/usr/bin/env python
"""Offline preprocessing for qlutattn-k1v4: reorder Q/K_proj channels by post-RoPE K sigma^2 (route a).

Goal: make same-energy K channels CONTIGUOUS along head_dim, so that later
per-token quantization can apply different codebooks to contiguous segments.

How (mathematically an identity -- QK^T unchanged up to fp16 summation order):
  1. Collect post-RoPE K per-channel sigma^2 over a calib corpus.
  2. RoPE rotate-half pairs (i, i+D/2) must move together (so the permutation
     commutes with the rotation). Score each pair by pair_sigma2 =
     max(sigma2[i], sigma2[i+D/2]), aggregated over all layers & kv-heads, and
     sort -> a SINGLE GLOBAL pair-permutation (low sigma^2 first).
  3. Fold the permutation into EVERY layer's W_q / W_k output channels
     (pair-as-unit), and reorder RoPE inv_freq[pair_perm] to match.

HF's LlamaRotaryEmbedding.inv_freq is model-level and persistent=False, so it is
NOT saved with the checkpoint. This script saves reorder_qk.pt {pair_perm,
channel_perm, head_dim}; the loader (kitty_sim.qk_reorder.apply_qk_reorder)
re-applies inv_freq = inv_freq[pair_perm] after from_pretrained.

per-channel qlut on the reordered checkpoint MUST reproduce the original score
(identity check). This is the prerequisite for per-token segmented quant.

Example (GPU0):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/preprocess_qlutattn_model.py \
    --model /home/zijie/models/Llama-3.2-1B-Instruct \
    --calib-data /home/zijie/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kitty_sim.qlut_quant import channel_sigma2  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Local path of the source model")
    p.add_argument("--output", default=None, help="Output dir (default: reorder/<model basename>)")
    p.add_argument("--calib-data", required=True, help="Parquet with a 'text' column (wikitext-2 train)")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--sample-len", type=int, default=2048)
    p.add_argument("--skip-first", type=int, default=32, help="Per-seq tokens excluded from stats (Kitty sink)")
    p.add_argument("--group-size", type=int, default=128, help="sigma^2 submean group along tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-equivalence", action="store_true")
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
def collect_pair_sigma2(model, batches, skip_first, group_size, device, head_dim):
    """Aggregated pair-sigma^2 [head_dim//2] over all layers and kv-heads."""
    nl = model.config.num_hidden_layers
    half = head_dim // 2
    accum = torch.zeros(half)
    n = 0
    for bi in range(batches.shape[0]):
        ids = batches[bi:bi + 1].to(device)
        cache = model(ids, use_cache=True).past_key_values
        for li in range(nl):
            keys = cache_layer_keys(cache, li)            # [1, n_kv, T, D] post-RoPE
            x = keys[0, :, skip_first:, :].transpose(1, 2)  # [n_kv, D, T]
            sig2 = channel_sigma2(x.float().cpu(), group_size)  # [n_kv, D]
            pair = torch.maximum(sig2[:, :half], sig2[:, half:])  # [n_kv, half]
            accum += pair.mean(dim=0)  # mean over kv-heads
            n += 1
        del cache
    return accum / max(n, 1)  # [half]


@torch.no_grad()
def reorder_weights(model, channel_perm, head_dim):
    """Permute every layer's W_q / W_k output channels per head (pair-as-unit)."""
    cp = channel_perm
    for layer in model.model.layers:
        attn = layer.self_attn
        for proj in (attn.q_proj, attn.k_proj):
            W = proj.weight.data
            nh = W.shape[0] // head_dim
            W = W.view(nh, head_dim, -1)[:, cp, :].reshape(nh * head_dim, -1).contiguous()
            proj.weight.data = W
            if getattr(proj, "bias", None) is not None:
                b = proj.bias.data.view(nh, head_dim)[:, cp].reshape(-1).contiguous()
                proj.bias.data = b


@torch.no_grad()
def equivalence_check(orig_path, reorder_model, pair_perm, batches, device):
    """fp16 logits: original model vs reordered model (with patched inv_freq)."""
    from transformers import AutoModelForCausalLM
    ids = batches[:2, :1024].to(device)
    orig = AutoModelForCausalLM.from_pretrained(orig_path, torch_dtype=torch.float16,
                                                local_files_only=True).to(device).eval()
    la = orig(ids).logits.float().cpu()
    del orig; torch.cuda.empty_cache()
    # reordered model already has W_q/W_k permuted; patch its inv_freq
    re = reorder_model.model.rotary_emb
    re.inv_freq = re.inv_freq[pair_perm].contiguous()
    re.original_inv_freq = re.inv_freq
    lb = reorder_model.half().to(device).eval()(ids).logits.float().cpu()
    diff = (la - lb).abs()
    return {"max_abs_logit_diff": diff.max().item(),
            "mean_abs_logit_diff": diff.mean().item(),
            "top1_agreement": (la.argmax(-1) == lb.argmax(-1)).float().mean().item()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output) if args.output else REPO_ROOT / "reorder" / Path(args.model).name
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    print(f"[reorder] model={args.model} layers={cfg.num_hidden_layers} q={n_q} kv={n_kv} D={head_dim}")

    batches = load_calib_token_batches(tok, args.calib_data, args.num_samples, args.sample_len, args.seed)
    pair_sig2 = collect_pair_sigma2(model, batches, args.skip_first, args.group_size, args.device, head_dim)
    half = head_dim // 2
    pair_perm = torch.argsort(pair_sig2)                       # low sigma^2 first
    channel_perm = torch.cat([pair_perm, pair_perm + half])    # keep pair structure
    print(f"[reorder] pair sigma^2 range {pair_sig2.min():.4f}..{pair_sig2.max():.4f}; "
          f"pair_perm[:8]={pair_perm[:8].tolist()}")

    reorder_weights(model, channel_perm, head_dim)
    print("[reorder] W_q/W_k output channels permuted (pair-as-unit, all layers)")

    torch.save({"pair_perm": pair_perm, "channel_perm": channel_perm, "head_dim": head_dim},
               out_dir / "reorder_qk.pt")
    # mark the checkpoint so the loader knows to patch inv_freq
    cfg.qk_reorder = True
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    equivalence = None
    if not args.skip_equivalence:
        equivalence = equivalence_check(args.model, model, pair_perm, batches, args.device)
        print(f"[reorder] fp16 equivalence: {equivalence}")

    meta = {"source_model": str(args.model), "head_dim": head_dim,
            "pair_perm": pair_perm.tolist(), "aggregation": "max-over-pair, mean-over-(layers,kv_heads)",
            "rope": "inv_freq[pair_perm] patched at load (persistent=False)",
            "equivalence_fp16": equivalence, "wall_seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(out_dir / "reorder_meta.json", "w"), indent=2)
    print(f"[reorder] saved reordered checkpoint + reorder_qk.pt to {out_dir}")


if __name__ == "__main__":
    main()
