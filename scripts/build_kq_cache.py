#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-time builder: prefill Llama-3.2-1B on a single 32k LongBench doc, cache the
post-RoPE K (full sequence) + real last-N queries + RoPE periods, so the
autoresearch loop's per-iteration eval is pure tensor math (no model reload).

Output: probe_out/kq_cache_<tag>.pt  with
  k_post: list[nl] of [H,D,T] fp16  (post-RoPE K, full sequence)
  q_real: list[nl] of [H, n_rep*NQ, D] fp16  (real queries, post q_norm + RoPE)
  period_chan: [D]   RoPE period per channel
  meta: dict(group, sink, recent, topk, H, D, nl, src, seq_len)

Usage (GPU0 on the compute host):
  CUDA_VISIBLE_DEVICES=0 python scripts/build_kq_cache.py \
      --model /mnt/data/tzj/models/Llama-3.2-1B-Instruct \
      --longbench-dir /mnt/data/tzj/data/LongBench/data --tag llama32-1b
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

LB_FILES = ["narrativeqa.jsonl", "gov_report.jsonl", "musique.jsonl", "qmsum.jsonl",
            "2wikimqa.jsonl", "hotpotqa.jsonl", "multifieldqa_en.jsonl", "qasper.jsonl"]


def pick_single_doc(lb_dir, tok, seq_len):
    best = None
    for name in LB_FILES:
        p = os.path.join(lb_dir, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f):
                ctx = json.loads(line).get("context", "")
                if len(ctx) < seq_len * 3:
                    continue
                ids = tok(ctx, return_tensors="pt").input_ids
                if ids.shape[1] >= seq_len:
                    return ids[:, :seq_len], f"{name}#{ln}"
                if best is None or ids.shape[1] > best[0]:
                    best = (ids.shape[1], ids[:, :seq_len], f"{name}#{ln}")
    if best is None:
        raise RuntimeError("no long enough doc")
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--longbench-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--sink", type=int, default=32)
    ap.add_argument("--recent", type=int, default=128)
    ap.add_argument("--num-queries", type=int, default=256)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tag", default="llama32-1b")
    ap.add_argument("--outdir", default="probe_out")
    args = ap.parse_args()

    dev = "cuda:0"
    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids, src = pick_single_doc(os.path.expanduser(args.longbench_dir), tok, args.seq_len)
    T = ids.shape[1]
    print(f"[input] {src}: {T} tok")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    cfg = model.config
    H = cfg.num_key_value_heads
    nqh = cfg.num_attention_heads
    n_rep = nqh // H
    D = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    nl = cfg.num_hidden_layers
    NQ = args.num_queries
    base = getattr(cfg, "rope_theta", 10000.0)

    qcap, handles = {}, []

    def mk(li):
        def hook(m, i, o):
            qcap[li] = o.detach()
        return hook

    cache = DynamicCache()
    starts = list(range(0, T, args.chunk))
    with torch.inference_mode():
        for si, s in enumerate(starts):
            if si == len(starts) - 1:
                for li, lyr in enumerate(model.model.layers):
                    handles.append(lyr.self_attn.q_proj.register_forward_hook(mk(li)))
            out = model(input_ids=ids[:, s:s + args.chunk].to(dev),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    for h in handles:
        h.remove()
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    posQ = torch.arange(T - NQ, T, device=dev)[None]
    dq = torch.zeros(1, NQ, D, device=dev, dtype=torch.float16)
    cosQ, sinQ = model.model.rotary_emb(dq, posQ)

    k_post, q_real = [], []
    for li in range(nl):
        K = cache.layers[li].keys if hasattr(cache, "layers") else cache.key_cache[li]
        k_post.append(K[0].permute(0, 2, 1).to(torch.float16).cpu())          # [H,D,T]
        q = qcap[li][0, -NQ:, :].view(NQ, nqh, D)
        attn = model.model.layers[li].self_attn
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        q = q.permute(1, 0, 2)[None]
        q, _ = apply_rotary_pos_emb(q, q, cosQ, sinQ)
        q_real.append(q[0].reshape(H, n_rep * NQ, D).to(torch.float16).cpu())  # [H,n_rep*NQ,D]

    j = torch.arange(D // 2)
    theta = base ** (-2.0 * j / D)
    period = (2 * np.pi / theta).float()
    period_chan = torch.cat([period, period])                                 # [D]

    art = {"k_post": k_post, "q_real": q_real, "period_chan": period_chan,
           "meta": {"group": args.group, "sink": args.sink, "recent": args.recent,
                    "topk": args.topk, "H": H, "D": D, "nl": nl, "n_rep": n_rep,
                    "src": src, "seq_len": T, "rope_base": base, "model": args.model}}
    out = os.path.join(args.outdir, f"kq_cache_{args.tag}.pt")
    torch.save(art, out)
    sz = os.path.getsize(out) / 2**20
    print(f"[saved] {out}  ({sz:.0f} MiB)  H={H} D={D} nl={nl} fast-pairs={int((period<args.group).sum())}/{D//2}")


if __name__ == "__main__":
    main()
