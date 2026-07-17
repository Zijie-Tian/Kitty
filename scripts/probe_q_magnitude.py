# -*- coding: utf-8 -*-
"""Collect post-RoPE per-channel query magnitudes E|q_d| for research masks.

MixKVQ-style signal (arXiv 2512.19206): a K channel deserves the 2-bit nf2
budget when it is BOTH hard to quantize (residual sigma^2) AND actually read
by the queries (mean |q_d|). This probe measures the second factor on the
same wikitext corpus the sigma^2 calibration uses.

Post-RoPE q is captured by wrapping transformers' apply_rotary_pos_emb; query
heads are averaged within each GQA group so the output aligns with the
per-kv-head codebook mask: q_absmean [n_layers, n_kv_heads, head_dim].

Run (one GPU, a few minutes):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/probe_q_magnitude.py \
    --model ~/models/Llama-3.2-1B-Instruct \
    --calib-data ~/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
    --output ~/models/Llama-3.2-1B-Instruct.rm_qstats.pt
"""
import argparse
import time
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calib-data", required=True, help="wikitext train parquet")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sample-len", type=int, default=2048)
    p.add_argument("--skip-first", type=int, default=32, help="skip the sink window")
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.llama.modeling_llama as ml

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").to(args.device).eval()
    cfg = model.config
    nl = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    group = n_q // n_kv
    print(f"[q-probe] model={args.model} layers={nl} q={n_q} kv={n_kv} D={D} gqa_group={group}")

    # Wrap apply_rotary_pos_emb: calls occur once per layer in layer order per
    # forward, so layer index = call counter % n_layers.
    orig_rope = ml.apply_rotary_pos_emb
    state = {"call": 0}
    accum = torch.zeros(nl, n_kv, D, dtype=torch.float64)

    def wrapped_rope(q, k, cos, sin, *a, **kw):
        q_emb, k_emb = orig_rope(q, k, cos, sin, *a, **kw)
        li = state["call"] % nl
        state["call"] += 1
        qa = q_emb[0, :, args.skip_first:, :].abs().mean(dim=1)      # [n_q, D]
        accum[li] += qa.reshape(n_kv, group, D).mean(dim=1).double().cpu()
        return q_emb, k_emb

    batches = load_calib_token_batches(tok, args.calib_data, args.num_samples, args.sample_len, args.seed)
    ml.apply_rotary_pos_emb = wrapped_rope
    try:
        with torch.no_grad():
            for bi in range(batches.shape[0]):
                model(batches[bi:bi + 1].to(args.device), use_cache=False)
                if (bi + 1) % 16 == 0:
                    print(f"  [q-probe] {bi + 1}/{batches.shape[0]} samples")
    finally:
        ml.apply_rotary_pos_emb = orig_rope

    expected_calls = args.num_samples * nl
    if state["call"] != expected_calls:
        raise RuntimeError(
            f"rope call count {state['call']} != expected {expected_calls}; "
            "layer attribution would be wrong (model not plain per-layer rope?)")

    q_absmean = (accum / args.num_samples).float()                   # [nl, n_kv, D]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "q_absmean": q_absmean,
        "model": args.model,
        "num_samples": args.num_samples,
        "sample_len": args.sample_len,
        "skip_first": args.skip_first,
        "seed": args.seed,
        "note": "post-RoPE mean|q| over (batch,tokens); query heads averaged per GQA group",
    }, out)
    rng = (q_absmean.min().item(), q_absmean.max().item())
    print(f"[q-probe] wrote {out}  range {rng[0]:.4f}..{rng[1]:.4f}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
