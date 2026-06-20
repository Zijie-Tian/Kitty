#!/usr/bin/env python
"""QServe-style SmoothAttention offline calibration for Kitty.

Collects post-RoPE Key absmax statistics per (kv_head, channel) on a
calibration corpus, computes per-channel smoothing scales with the RoPE
rotate-half pair constraint (lambda_i == lambda_{i+D/2}, so diag(lambda)
commutes with the rotation and can be folded into the pre-RoPE projection
weights):

    lam[h, i] = lam[h, i + D/2] = max(absmax_K[h, i], absmax_K[h, i+D/2]) ** alpha

and folds them offline into the attention projections:

    W_q <- lam_q * W_q   (output-channel dim, GQA-broadcast over query heads)
    W_k <- W_k / lam_k   (output-channel dim)

The smoothed checkpoint is mathematically equivalent (QK^T unchanged in exact
arithmetic; bf16 weight rounding only), but its post-RoPE K channels are
flattened toward their geometric mean -- which is exactly what per-token
(head_dim-axis) low-bit K quantization needs (--variant qlutattn_pertoken).

Outputs under --output (default calib/<model basename>/):
  - full save_pretrained checkpoint + tokenizer (drop-in for <T>_MODEL_PATH)
  - smooth_scales.pt        {layer_idx: lam tensor [n_kv_heads, head_dim]}
  - calib_meta.json         args, flatness stats, quant-error proxy, equivalence

Example (GPU1):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/calibrate_smooth_qk.py \
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

from kitty_sim.utils_quant import fake_quant_groupwise_lastdim  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Local path of the source model")
    p.add_argument("--output", default=None,
                   help="Output dir for the smoothed checkpoint (default: calib/<model basename>)")
    p.add_argument("--calib-data", required=True,
                   help="Parquet file with a 'text' column (e.g. local wikitext-2-raw-v1 train split)")
    p.add_argument("--alpha", type=float, default=0.5, help="Smoothing strength: lam = absmax(K)^alpha")
    p.add_argument("--num-samples", type=int, default=32, help="Number of calibration sequences")
    p.add_argument("--sample-len", type=int, default=2048, help="Tokens per calibration sequence")
    p.add_argument("--skip-first", type=int, default=32,
                   help="Per-sequence tokens excluded from absmax stats (Kitty keeps sink_length=32 fp16)")
    p.add_argument("--kbits", type=int, default=2, help="K bits for the quant-error proxy report")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-equivalence", action="store_true",
                   help="Skip the fp16 logits equivalence check (saves time/VRAM)")
    return p.parse_args()


def load_calib_token_batches(tokenizer, parquet_path: str, num_samples: int, sample_len: int,
                             seed: int) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={"train": parquet_path})["train"]
    texts = [t for t in ds["text"] if t and not t.isspace()]
    corpus = "\n\n".join(texts)
    needed = num_samples * sample_len
    enc = tokenizer(corpus, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]
    if ids.numel() < needed:
        raise RuntimeError(f"Calibration corpus too small: {ids.numel()} tokens < {needed} needed")
    g = torch.Generator().manual_seed(seed)
    # Random non-overlapping-ish windows over the corpus for diversity.
    starts = torch.randint(0, ids.numel() - sample_len, (num_samples,), generator=g)
    return torch.stack([ids[s : s + sample_len] for s in starts])  # [N, L]


def cache_layer_keys(cache, layer_idx: int) -> torch.Tensor:
    """Post-RoPE keys for one layer from a transformers DynamicCache (4.5x-compatible)."""
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
def collect_key_stats(model, batches: torch.Tensor, skip_first: int, device: str,
                      keep_sample_tokens: int = 512):
    """Run calibration forwards; return per-layer post-RoPE K absmax and a K sample.

    absmax[layer]: [n_kv_heads, head_dim] over all batches/tokens (skip_first
    tokens of each sequence excluded, mirroring Kitty's fp16 sink window).
    k_sample[layer]: [1, n_kv_heads, keep_sample_tokens, head_dim] from the
    first batch, for the quantization-error proxy report.
    """
    num_layers = model.config.num_hidden_layers
    absmax = [None] * num_layers
    k_sample = [None] * num_layers
    for bi in range(batches.shape[0]):
        ids = batches[bi : bi + 1].to(device)
        out = model(ids, use_cache=True)
        cache = out.past_key_values
        for li in range(num_layers):
            keys = cache_layer_keys(cache, li)  # [1, n_kv, T, D], post-RoPE
            stat = keys[:, :, skip_first:, :].abs().amax(dim=(0, 2)).float().cpu()  # [n_kv, D]
            absmax[li] = stat if absmax[li] is None else torch.maximum(absmax[li], stat)
            if bi == 0:
                k_sample[li] = keys[:, :, skip_first : skip_first + keep_sample_tokens, :].detach().float().cpu()
        del out, cache
    return absmax, k_sample


def compute_smooth_scales(absmax: torch.Tensor, alpha: float) -> torch.Tensor:
    """lam [n_kv, D] with the RoPE rotate-half pair constraint lam_i == lam_{i+D/2}."""
    half = absmax.shape[1] // 2
    paired = torch.maximum(absmax[:, :half], absmax[:, half:])  # [n_kv, D/2]
    lam_half = paired.clamp(min=1e-8).pow(alpha)
    # Channels that are exactly zero across the whole corpus get lam=1 (no-op).
    lam_half = torch.where(paired > 0, lam_half, torch.ones_like(lam_half))
    return torch.cat([lam_half, lam_half], dim=1)  # [n_kv, D]


@torch.no_grad()
def fold_scales(model, scales: dict[int, torch.Tensor]) -> None:
    """Fold lam into q_proj/k_proj output channels (fp32 math, cast back)."""
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    rep = n_q // n_kv
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, "q_norm") or hasattr(attn, "k_norm"):
            raise RuntimeError(
                "Model has per-head q_norm/k_norm after the projections (e.g. Qwen3); "
                "folding a per-channel scale into W_k would be renormalized away and "
                "break equivalence. SmoothAttention fusion is not supported here."
            )
        lam = scales[li].to(torch.float32)  # [n_kv, D]
        lam_k = lam.reshape(n_kv * head_dim)
        lam_q = lam.unsqueeze(1).expand(n_kv, rep, head_dim).reshape(n_q * head_dim)
        for proj, vec, op in ((attn.q_proj, lam_q, "mul"), (attn.k_proj, lam_k, "div")):
            v = vec.to(proj.weight.device)
            w32 = proj.weight.data.float()
            w32 = w32 * v[:, None] if op == "mul" else w32 / v[:, None]
            proj.weight.data = w32.to(proj.weight.dtype)
            if proj.bias is not None:
                b32 = proj.bias.data.float()
                b32 = b32 * v if op == "mul" else b32 / v
                proj.bias.data = b32.to(proj.bias.dtype)


def quant_error_proxy(k_sample: torch.Tensor, lam: torch.Tensor, kbits: int,
                      group_size: int = 128) -> dict[str, float]:
    """Per-token K fake-quant error in the ORIGINAL K domain, before vs after smoothing.

    before: quant(K) vs K. after: quant(K/lam)*lam vs K -- i.e. the deployed
    configuration where the checkpoint emits K/lam and attention consumes the
    smooth-domain values (errors mapped back via *lam for comparability).
    """
    k = k_sample  # [1, n_kv, T, D] float32
    lam_b = lam.view(1, lam.shape[0], 1, lam.shape[1]).to(k.dtype)
    q_before = fake_quant_groupwise_lastdim(k.clone(), group_size, kbits)
    q_after = fake_quant_groupwise_lastdim((k / lam_b).clone(), group_size, kbits) * lam_b
    denom = k.pow(2).mean().item() or 1e-12
    mse_b = (q_before - k).pow(2).mean().item()
    mse_a = (q_after - k).pow(2).mean().item()
    return {
        "nmse_before": mse_b / denom,
        "nmse_after": mse_a / denom,
        "improvement_x": (mse_b / mse_a) if mse_a > 0 else float("inf"),
    }


def flatness(absmax: torch.Tensor) -> float:
    """Mean over heads of (channel absmax max / median) -- 1.0 is perfectly flat."""
    return (absmax.amax(dim=1) / absmax.median(dim=1).values.clamp(min=1e-8)).mean().item()


@torch.no_grad()
def equivalence_check(orig_path: str, smooth_path: str, batches: torch.Tensor, device: str) -> dict:
    """fp16 logits comparison on held-out sequences (matches the eval pipeline dtype)."""
    from transformers import AutoModelForCausalLM

    ids = batches[:2, :1024].to(device)
    results = []
    for path in (orig_path, smooth_path):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                                 local_files_only=True).to(device).eval()
        results.append(m(ids).logits.float().cpu())
        del m
        torch.cuda.empty_cache()
    a, b = results
    diff = (a - b).abs()
    return {
        "max_abs_logit_diff": diff.max().item(),
        "mean_abs_logit_diff": diff.mean().item(),
        "top1_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output) if args.output else REPO_ROOT / "calib" / Path(args.model).name
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).eval()
    cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    print(f"[calib] model={args.model} layers={cfg.num_hidden_layers} "
          f"q_heads={n_q} kv_heads={n_kv} head_dim={head_dim}")

    batches = load_calib_token_batches(tokenizer, args.calib_data, args.num_samples,
                                       args.sample_len, args.seed)
    print(f"[calib] corpus={args.calib_data} batches={tuple(batches.shape)}")

    absmax, k_sample = collect_key_stats(model, batches, args.skip_first, args.device)
    print(f"[calib] stats collected in {time.time() - t0:.1f}s")

    scales: dict[int, torch.Tensor] = {}
    report_layers = []
    for li in range(cfg.num_hidden_layers):
        lam = compute_smooth_scales(absmax[li], args.alpha)
        scales[li] = lam
        proxy = quant_error_proxy(k_sample[li], lam, args.kbits)
        entry = {
            "layer": li,
            "flatness_before": round(flatness(absmax[li]), 3),
            "flatness_after": round(flatness(absmax[li] / lam), 3),
            "lam_min": round(lam.min().item(), 4),
            "lam_max": round(lam.max().item(), 4),
            **{k: round(v, 5) for k, v in proxy.items()},
        }
        report_layers.append(entry)
        print(f"[calib] L{li:02d} flat {entry['flatness_before']:>7.2f} -> {entry['flatness_after']:>5.2f}"
              f"  k{args.kbits} NMSE {entry['nmse_before']:.4f} -> {entry['nmse_after']:.4f}"
              f"  ({entry['improvement_x']:.2f}x)")

    fold_scales(model, scales)
    print("[calib] scales folded into q_proj/k_proj")

    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    torch.save({li: s for li, s in scales.items()}, out_dir / "smooth_scales.pt")
    del model
    torch.cuda.empty_cache()

    equivalence = None
    if not args.skip_equivalence:
        equivalence = equivalence_check(args.model, str(out_dir), batches, args.device)
        print(f"[calib] fp16 equivalence: {equivalence}")

    meta = {
        "source_model": str(args.model),
        "alpha": args.alpha,
        "num_samples": args.num_samples,
        "sample_len": args.sample_len,
        "skip_first": args.skip_first,
        "calib_data": str(args.calib_data),
        "seed": args.seed,
        "kbits_proxy": args.kbits,
        "rope_pair_constraint": "lam_i == lam_{i+D/2} (HF rotate-half)",
        "fold": "W_q *= lam_q (GQA-broadcast); W_k /= lam_k; output-channel dim, fp32 math",
        "layers": report_layers,
        "equivalence_fp16": equivalence,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "calib_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[calib] saved smoothed checkpoint + meta to {out_dir}")


if __name__ == "__main__":
    main()
