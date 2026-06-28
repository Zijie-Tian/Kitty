"""Shared helpers for the kv-cache-viz skill.

All command modules should import from here to avoid duplicating LongBench doc
picking, model prefill, cache extraction, and matplotlib boilerplate.
"""
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch


def setup_matplotlib():
    """Set matplotlib to a headless backend."""
    import matplotlib
    matplotlib.use("Agg")


def save_fig(fig, path: str, dpi: int = 120, tight: bool = True, **save_kwargs):
    """Save and close a matplotlib figure, creating parent dirs as needed."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, **save_kwargs)
    plt.close(fig)
    print("[fig]", path, flush=True)


def rope_period(D: int, base: float = 500000.0) -> np.ndarray:
    """RoPE period per channel (pairs i, i+D/2 share period)."""
    j = np.arange(D) % (D // 2)
    return 2 * np.pi * base ** (2.0 * j / D)


def pick_doc(
    lb_dir: str,
    tok,
    seq_len: int,
    lb_files: Optional[List[str]] = None,
) -> torch.Tensor:
    """Pick the first LongBench document that reaches seq_len tokens.

    Returns a [1, seq_len] tensor of input ids.
    """
    if lb_files is None:
        lb_files = [
            "narrativeqa.jsonl",
            "gov_report.jsonl",
            "2wikimqa.jsonl",
            "hotpotqa.jsonl",
            "multifieldqa_en.jsonl",
        ]
    best: Optional[Tuple[int, torch.Tensor]] = None
    for name in lb_files:
        p = os.path.join(lb_dir, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                ctx = json.loads(line).get("context", "")
                if len(ctx) < seq_len * 2:
                    continue
                ids = tok(ctx, return_tensors="pt").input_ids
                if ids.shape[1] >= seq_len:
                    return ids[:, :seq_len]
                if best is None or ids.shape[1] > best[0]:
                    best = (ids.shape[1], ids[:, :seq_len])
    if best is None:
        raise RuntimeError("no long-enough LongBench doc")
    return best[1]


def load_model(model_path: str, device: str = "cuda:0"):
    """Load tokenizer and model in fp16 with SDPA attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()
    return model, tok


def prefill_and_extract(
    model,
    tok,
    lb_dir: str,
    seq_len: int = 32768,
    sink: int = 32,
    recent: int = 128,
    chunk: int = 4096,
    device: str = "cuda:0",
    lb_files: Optional[List[str]] = None,
):
    """Prefill one LongBench doc and return the [sink, T-recent) region per layer.

    Returns a dict with:
      K, V: list[nl] of [H, T_region, D] float32 CPU tensors
      T, sink, recent, region, layers, D, H
    """
    from transformers import DynamicCache
    ids = pick_doc(lb_dir, tok, seq_len, lb_files=lb_files)
    T = ids.shape[1]
    cache = DynamicCache()
    with torch.inference_mode():
        for s in range(0, T, chunk):
            cache = model(
                input_ids=ids[:, s:s + chunk].to(device),
                past_key_values=cache,
                use_cache=True,
            ).past_key_values
    q0, q1 = sink, T - recent
    nl = model.config.num_hidden_layers
    D = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    H = model.config.num_key_value_heads
    Ks, Vs = [], []
    for li in range(nl):
        if hasattr(cache, "layers"):
            K = cache.layers[li].keys
            V = cache.layers[li].values
        else:
            K = cache.key_cache[li]
            V = cache.value_cache[li]
        Ks.append(K[0, :, q0:q1, :].float().cpu())
        Vs.append(V[0, :, q0:q1, :].float().cpu())
    return {
        "K": Ks,
        "V": Vs,
        "T": T,
        "sink": sink,
        "recent": recent,
        "region": (q0, q1),
        "layers": nl,
        "D": D,
        "H": H,
    }


def save_layer_kv(
    Ks: List[torch.Tensor],
    Vs: List[torch.Tensor],
    outdir: str,
    layer: int,
) -> Tuple[str, str]:
    """Save a single layer's K and V as fp16 .pt files."""
    os.makedirs(outdir, exist_ok=True)
    path_k = os.path.join(outdir, f"layer{layer}_K_fp16.pt")
    path_v = os.path.join(outdir, f"layer{layer}_V_fp16.pt")
    torch.save(Ks[layer].half(), path_k)
    torch.save(Vs[layer].half(), path_v)
    print("[dump]", path_k, path_v, flush=True)
    return path_k, path_v


def default_layer_pt(outdir: str, layer: int, kv: str) -> str:
    """Default path used by offline plotters: outdir/quant_kvcache_analysis/layer{L}_{K,V}_fp16.pt"""
    return os.path.join(outdir, "quant_kvcache_analysis", f"layer{layer}_{kv}_fp16.pt")


def add_prefill_args(parser):
    """Add the common args used by model-based probe commands."""
    parser.add_argument("--model", required=True)
    parser.add_argument("--longbench-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--sink", type=int, default=32)
    parser.add_argument("--recent", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="model")
    parser.add_argument("--outdir", default="probe_out/sign_scale")


def add_viz_args(parser):
    """Add the common args used by offline viz commands."""
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--pt", default=None,
                        help="Path to the K or V dump .pt; defaults to quant_kvcache_analysis/ layer dump")
    parser.add_argument("--outdir", default="probe_out")
    parser.add_argument("--tag", default="llama32-1b")
    parser.add_argument("--device", default="cuda:0")
