#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoresearch verify harness — **per-token K-cache** quantization search.

Goal: maximize top-32 attention overlap (quant K vs fp16 K, real queries) at
**≤ bit-ceiling effective bits/element** (default 2.5, side-info included), using
**per-token** K quantization (group along head_dim, one scale per token; KIVI-V axis).

Loads the cached post-RoPE K/Q (probe_out/kq_cache_<tag>.pt from build_kq_cache.py),
runs every method in REGISTRY over all layers, reports per method:
  - overlap  : mean top-32 attended-key overlap vs fp16 (THE metric, higher better)
  - nmse     : K reconstruction NMSE over the quant region (recorded)
  - eff_bits : codeword bits + fp16 side-info / group (the guard, ≤ ceiling)
The CHAMPION = max-overlap among `per_token` methods with eff_bits ≤ ceiling.
Prints `METRIC=<overlap>` (champion) last; writes <out> leaderboard json.

Unified method contract: `fn(x[H,D,T]) -> x_rec[H,D,T]` reconstructs the K in the
ORIGINAL post-RoPE space (rotation methods rotate, quantize per-token, rotate back),
so overlap/nmse are computed identically by `q·x`. Only the quant region
[sink, T-recent) is quantized; sink+recent stay fp16 (matches the real Kitty cache).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/eval_pertoken_quant.py \
      --cache probe_out/kq_cache_llama32-1b.pt --bit-ceiling 2.5
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from scipy.linalg import hadamard

# ----------------------------- codebooks (last axis = group) ---------------- #
# Each returns reconstruction with the SAME shape; side-info cost is bookkept
# separately by the method's (cw_bits, n_side) declaration, not here.

def cb_sign(xg):                       # 1-bit submean: {mu-mag, mu+mag}
    mu = xg.mean(-1, keepdim=True); r = xg - mu
    mag = r.abs().mean(-1, keepdim=True)
    return mu + torch.sign(r) * mag

def cb_tern(xg):                       # ternary submean (dead-zone tau=0.5)
    mu = xg.mean(-1, keepdim=True); r = xg - mu
    thr = 0.5 * r.abs().mean(-1, keepdim=True)
    mask = r.abs() > thr
    mag = (r.abs() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
    return mu + torch.sign(r) * mag * mask

def cb_uni(xg, L):                     # asym uniform min-max, L levels
    mn = xg.min(-1, keepdim=True).values; mx = xg.max(-1, keepdim=True).values
    s = (mx - mn).clamp(min=1e-8) / (L - 1)
    return ((xg - mn) / s).round().clamp(0, L - 1) * s + mn

def cb_lloyd(xg, L=4, iters=12):       # per-group Lloyd-Max (optimal L-level scalar)
    lo = xg.min(-1, keepdim=True).values; hi = xg.max(-1, keepdim=True).values
    lev = lo + (hi - lo) * (torch.arange(L, device=xg.device) + 0.5) / L
    for _ in range(iters):
        d = (xg.unsqueeze(-1) - lev.unsqueeze(-2)).abs(); a = d.argmin(-1)
        oh = torch.nn.functional.one_hot(a, L).to(xg.dtype)
        cnt = oh.sum(-2); summ = (oh * xg.unsqueeze(-1)).sum(-2)
        lev = torch.where(cnt > 0, summ / cnt.clamp(min=1), lev)
    a = (xg.unsqueeze(-1) - lev.unsqueeze(-2)).abs().argmin(-1)
    return torch.gather(lev, -1, a)

# Fixed normal-float style codebooks (per-token std-scaled; 1 side = scale).
_NF_LEVELS = {  # MSE-optimal Lloyd-Max levels for N(0,1)
    2: torch.tensor([-0.7979, 0.7979]),
    4: torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),
    8: torch.tensor([-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519]),
}
def cb_nf_fixed(xg, L):                # normalize by per-group std, snap to fixed levels
    s = xg.std(-1, keepdim=True).clamp(min=1e-8)
    z = xg / s
    lev = _NF_LEVELS[L].to(xg.device)
    a = (z.unsqueeze(-1) - lev).abs().argmin(-1)
    return lev[a] * s

# ----------------------------- axis drivers --------------------------------- #
def quant_per_token(xHDT, cb):
    """group along head_dim: per (head, token) one group of D channels."""
    x = xHDT.transpose(1, 2)           # [H,T,D]
    return cb(x).transpose(1, 2)       # back to [H,D,T]

def quant_per_channel(xHDT, cb, G):
    """group along token axis: per (head, channel) groups of G tokens."""
    H, D, T = xHDT.shape; ng = T // G
    xg = xHDT[:, :, :ng * G].reshape(H, D, ng, G)
    rec = cb(xg).reshape(H, D, ng * G)
    out = xHDT.clone(); out[:, :, :ng * G] = rec
    return out

_HCACHE = {}
def hadamard_R(D, dev):
    if (D, dev) not in _HCACHE:
        H = torch.tensor(hadamard(D), dtype=torch.float32, device=dev) / math.sqrt(D)
        _HCACHE[(D, dev)] = H          # symmetric, orthogonal: R==R.T==R^-1
    return _HCACHE[(D, dev)]

def rotate_D(xHDT, R):                  # rotate head_dim axis: x'[:,i,:] = sum_j R[i,j] x[:,j,:]
    return torch.einsum("ij,hjt->hit", R, xHDT)

def smooth_lambda(xHDT, alpha=0.5):     # QServe per-channel: lam=absmax(K_chan)^alpha
    return (xHDT.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)) ** alpha  # [H,D,1]

# ----------------------------- method wrappers ------------------------------ #
# A method: dict(name, axis, bits, fn) where fn(x_region_or_full)->x_rec_full.
# All operate on the quant region only; sink/recent kept fp16 by the runner.

def make_method(name, axis, cw_bits, n_side, recon, note=""):
    return dict(name=name, axis=axis, cw_bits=cw_bits, n_side=n_side, recon=recon, note=note)


def build_registry():
    reg = []
    # ---- per-channel references (NOT eligible as per-token champion) -------- #
    reg.append(make_method("ref/per-chan sign", "per_channel", 1.0, 2,
                           lambda x, G: quant_per_channel(x, cb_sign, G), "KIVI-K 风格 1-bit"))
    reg.append(make_method("ref/per-chan tern", "per_channel", math.log2(3), 2,
                           lambda x, G: quant_per_channel(x, cb_tern, G), "iso-tern"))
    reg.append(make_method("ref/per-chan KIVI uni2", "per_channel", 2.0, 2,
                           lambda x, G: quant_per_channel(x, lambda g: cb_uni(g, 4), G), "per-channel 2-bit min-max"))
    # ---- per-token baselines (iteration 0) --------------------------------- #
    reg.append(make_method("pt/sign", "per_token", 1.0, 2,
                           lambda x, G: quant_per_token(x, cb_sign), "崩溃基线"))
    reg.append(make_method("pt/tern", "per_token", math.log2(3), 2,
                           lambda x, G: quant_per_token(x, cb_tern), ""))
    reg.append(make_method("pt/uni2 min-max", "per_token", 2.0, 2,
                           lambda x, G: quant_per_token(x, lambda g: cb_uni(g, 4)), "QuaRot 无旋转"))
    reg.append(make_method("pt/nf2 Lloyd", "per_token", 2.0, 2,
                           lambda x, G: quant_per_token(x, lambda g: cb_lloyd(g, 4)), "已知较好 per-token"))
    reg.append(make_method("pt/nf2 fixed(std)", "per_token", 2.0, 1,
                           lambda x, G: quant_per_token(x, lambda g: cb_nf_fixed(g, 4)), "Gaussian NF, 1 side"))
    return reg


# ----------------------------- eval ----------------------------------------- #
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="probe_out/kq_cache_llama32-1b.pt")
    ap.add_argument("--bit-ceiling", type=float, default=2.5)
    ap.add_argument("--out", default="probe_out/pertoken_eval.json")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    art = torch.load(args.cache, map_location=dev, weights_only=False)
    meta = art["meta"]
    G, sink, recent, topk = meta["group"], meta["sink"], meta["recent"], meta["topk"]
    nl, D = meta["nl"], meta["D"]
    reg = build_registry()

    # preload tensors
    Ks = [art["k_post"][li].to(dev).float() for li in range(nl)]   # [H,D,T]
    Qs = [art["q_real"][li].to(dev).float() for li in range(nl)]   # [H,Q,D]

    def eff_bits(m):
        group = D if m["axis"] == "per_token" else G
        return m["cw_bits"] + 16.0 * m["n_side"] / group

    def quant_region(m, x):
        """apply method to region [sink, T-recent); keep rest fp16."""
        T = x.shape[-1]; q0, q1 = sink, T - recent
        reg_x = x[:, :, q0:q1]
        rec = m["recon"](reg_x, G) if m["axis"] == "per_channel" else m["recon"](reg_x, G)
        out = x.clone(); out[:, :, q0:q1] = rec
        return out

    results = []
    for m in reg:
        ov_sum, nm_sum, en_sum = 0.0, 0.0, 0.0
        for li in range(nl):
            x = Ks[li]; q = Qs[li]; T = x.shape[-1]
            xr = quant_region(m, x)
            # overlap
            s_true = torch.einsum("hqd,hdt->hqt", q, x)
            s_rec = torch.einsum("hqd,hdt->hqt", q, xr)
            t1 = torch.zeros_like(s_true, dtype=torch.bool).scatter(
                -1, s_true.topk(topk, dim=-1).indices, True)
            t2 = torch.zeros_like(s_true, dtype=torch.bool).scatter(
                -1, s_rec.topk(topk, dim=-1).indices, True)
            ov_sum += ((t1 & t2).sum(-1).float() / topk).mean().item()
            # nmse over region
            q0, q1 = sink, T - recent
            e = (x[:, :, q0:q1] - xr[:, :, q0:q1]).pow(2).sum().item()
            en = x[:, :, q0:q1].pow(2).sum().item()
            nm_sum += e; en_sum += en
        results.append(dict(name=m["name"], axis=m["axis"], eff_bits=round(eff_bits(m), 4),
                            overlap=round(ov_sum / nl, 5), nmse=round(nm_sum / en_sum, 5),
                            note=m["note"]))

    results.sort(key=lambda r: -r["overlap"])
    # champion = best per_token within budget
    elig = [r for r in results if r["axis"] == "per_token" and r["eff_bits"] <= args.bit_ceiling + 1e-9]
    champ = max(elig, key=lambda r: r["overlap"]) if elig else None

    print(f"{'method':28s} {'axis':12s} {'bits':>6s} {'overlap':>8s} {'nmse':>8s}  note")
    print("-" * 86)
    for r in results:
        star = " *CHAMP*" if champ and r["name"] == champ["name"] else ""
        flag = "" if (r["axis"] != "per_token" or r["eff_bits"] <= args.bit_ceiling + 1e-9) else " (over-budget)"
        print(f"{r['name']:28s} {r['axis']:12s} {r['eff_bits']:6.3f} {r['overlap']:8.4f} {r['nmse']:8.4f}  {r['note']}{flag}{star}")

    out = dict(cache=os.path.basename(args.cache), src=meta["src"], bit_ceiling=args.bit_ceiling,
               topk=topk, n_layers=nl, head_dim=D, results=results,
               champion=champ)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("-" * 86)
    print(f"CHAMPION(per_token≤{args.bit_ceiling}b): {champ['name'] if champ else 'none'}  "
          f"overlap={champ['overlap'] if champ else 0}  bits={champ['eff_bits'] if champ else '-'}")
    print(f"METRIC={champ['overlap'] if champ else 0}")


if __name__ == "__main__":
    main()
