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

def cb_sign_sym(xg):                   # symmetric 1-bit around 0: NO mean stored (1 side = scale)
    s = xg.abs().mean(-1, keepdim=True)
    return torch.sign(xg) * s

# ----------------------------- axis drivers --------------------------------- #
def quant_per_token(xHDT, cb):
    """group along head_dim: per (head, token) one group of D channels."""
    x = xHDT.transpose(1, 2)           # [H,T,D]
    return cb(x).transpose(1, 2)       # back to [H,D,T]

def quant_per_token_sub(xHDT, cb, n_sub):
    """split head_dim into n_sub contiguous sub-groups, each its own per-token scale."""
    x = xHDT.transpose(1, 2)           # [H,T,D]
    H, T, D = x.shape
    rec = cb(x.reshape(H, T, n_sub, D // n_sub))
    return rec.reshape(H, T, D).transpose(1, 2)

def outlier_keep_pertoken(cb, k, outlier_bits=16, by="amax"):
    """keep top-k channels (fixed per head, selected by `by`) at `outlier_bits`
    (16=fp16, else per-channel uniform); per-token quantize the rest with the scale
    over ONLY the non-outlier channels (homogeneous group). `by`: amax (peak
    magnitude, what dominates the shared per-token scale) | var (residual sigma^2)."""
    def fn(xreg, G):
        H, D, T = xreg.shape
        score = xreg.abs().amax(dim=2) if by == "amax" else xreg.var(dim=2, unbiased=False)
        keep = torch.zeros(H, D, dtype=torch.bool, device=xreg.device)
        keep.scatter_(1, score.topk(k, dim=1).indices, True)
        out = xreg.clone()
        for h in range(H):
            nb = ~keep[h]
            out[h:h+1, nb, :] = quant_per_token(xreg[h:h+1, nb, :], cb)
            if outlier_bits < 16:                       # per-channel uniform on kept chans
                out[h:h+1, keep[h], :] = cb_uni(xreg[h:h+1, keep[h], :], 2 ** outlier_bits)
        return out
    return fn

def smooth_then(fn_region, alpha=0.5):
    """QServe channel smooth (K/lam) wrapped around any region method, fold lam back."""
    def fn(xreg, G):
        lam = smooth_lambda(xreg, alpha)
        return fn_region(xreg / lam, G) * lam
    return fn


_CW = {"sign_sym": 1.0, "sign": 1.0, "tern": float(np.log2(3)), "nf2": 2.0}
_CBFN = {"sign_sym": cb_sign_sym, "sign": cb_sign, "tern": cb_tern, "nf2": lambda g: cb_lloyd(g, 4)}

def mixed_base_pertoken(k, policy, outlier_bits=4):
    """Champion + sigma^2-MIXED base on the non-outlier channels: keep top-k amax
    channels per head @outlier_bits per-channel; bin the remaining channels by
    per-channel residual sigma^2 into len(policy) equal bins (low sigma^2 -> policy[0]);
    each bin's channels are per-token quantized with its codebook + own per-token scale.
    policy: list of codebook NAMES (low->high sigma^2). Mirrors qlutattn-k1v4 but per-token."""
    nbins = len(policy)
    fns = [_CBFN[p] for p in policy]
    def fn(xreg, G):
        H, D, T = xreg.shape
        keep_ids = xreg.abs().amax(2).topk(k, dim=1).indices
        out = xreg.clone()
        for h in range(H):
            keep = torch.zeros(D, dtype=torch.bool, device=xreg.device); keep[keep_ids[h]] = True
            nb_idx = (~keep).nonzero().squeeze(1)                 # non-outlier channel ids
            sig2 = xreg[h][nb_idx].var(dim=1)                     # per-channel residual variance
            order = sig2.argsort(); ranks = torch.empty_like(order); ranks[order] = torch.arange(len(order), device=xreg.device)
            binid = (ranks.float() * nbins / len(order)).floor().long().clamp(max=nbins - 1)
            for b in range(nbins):
                ch = nb_idx[(binid == b).nonzero().squeeze(1)]
                if ch.numel() == 0:
                    continue
                sub = xreg[h][ch].transpose(0, 1)[None]           # [1,T,nch] per-token group
                out[h][ch] = fns[b](sub)[0].transpose(0, 1)
            if outlier_bits < 16:                                 # outliers per-channel uniform @bits
                out[h][keep] = cb_uni(xreg[h][keep][None], 2 ** outlier_bits)[0]
        return out
    return fn

def mixed_base_bits(k, policy):
    """eff bits: each sigma^2 bin = equal share of (64-k) channels, 1 fp16 scale/bin/token; outliers k@4."""
    nb = 64 - k; nbins = len(policy)
    cw = sum((nb / nbins) * _CW[p] for p in policy)
    return (cw + 16 * nbins + k * 4) / 64

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

def make_method(name, axis, cw_bits, n_side, recon, note="", bits_override=None):
    return dict(name=name, axis=axis, cw_bits=cw_bits, n_side=n_side, recon=recon,
                note=note, bits_override=bits_override)


def had_pertoken(cb):
    """Hadamard rotate head_dim -> per-token codebook -> rotate back (R==R^-1)."""
    def fn(xreg, G):
        R = hadamard_R(xreg.shape[1], xreg.device)
        return rotate_D(quant_per_token(rotate_D(xreg, R), cb), R)
    return fn


def smooth_pertoken(cb, alpha=0.5, then_hadamard=False):
    """QServe smooth (K/lam, lam folds to Q) -> [Hadamard] -> per-token cb -> fold lam back."""
    def fn(xreg, G):
        lam = smooth_lambda(xreg, alpha)            # [H,D,1]
        xs = xreg / lam
        if then_hadamard:
            R = hadamard_R(xreg.shape[1], xreg.device)
            xs = rotate_D(quant_per_token(rotate_D(xs, R), cb), R)
        else:
            xs = quant_per_token(xs, cb)
        return xs * lam
    return fn


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
    # ---- iter 1: Hadamard rotation (incoherence) + per-token --------------- #
    reg.append(make_method("pt/had+sign", "per_token", 1.0, 2,
                           lambda x, G: had_pertoken(cb_sign)(x, G), "QuaRot rot + 1-bit"))
    reg.append(make_method("pt/had+uni2", "per_token", 2.0, 2,
                           lambda x, G: had_pertoken(lambda g: cb_uni(g, 4))(x, G), "QuaRot rot + uniform2"))
    reg.append(make_method("pt/had+nf2 fixed", "per_token", 2.0, 1,
                           lambda x, G: had_pertoken(lambda g: cb_nf_fixed(g, 4))(x, G), "rot + Gaussian NF, 1 side"))
    reg.append(make_method("pt/had+nf2 Lloyd", "per_token", 2.0, 2,
                           lambda x, G: had_pertoken(lambda g: cb_lloyd(g, 4))(x, G), "rot + per-group Lloyd"))
    # ---- iter 2: SmoothAttention channel equalization (+/- Hadamard) ------- #
    reg.append(make_method("pt/smooth+nf2 Lloyd", "per_token", 2.0, 2,
                           lambda x, G: smooth_pertoken(lambda g: cb_lloyd(g, 4))(x, G), "QServe smooth + Lloyd"))
    reg.append(make_method("pt/smooth+had+nf2 Lloyd", "per_token", 2.0, 2,
                           lambda x, G: smooth_pertoken(lambda g: cb_lloyd(g, 4), then_hadamard=True)(x, G), "smooth then rot + Lloyd"))
    reg.append(make_method("pt/smooth+had+nf2 fixed", "per_token", 2.0, 1,
                           lambda x, G: smooth_pertoken(lambda g: cb_nf_fixed(g, 4), then_hadamard=True)(x, G), "smooth+rot+NF 2.25b"))
    # ---- iter 3: finer scale (sub-group) + outlier-channel isolation ------- #
    reg.append(make_method("pt/2sub nf fixed", "per_token", 2.0, 2,
                           lambda x, G: quant_per_token_sub(x, lambda g: cb_nf_fixed(g, 4), 2), "2 head_dim sub-groups, 2.5b"))
    reg.append(make_method("pt/had+2sub nf fixed", "per_token", 2.0, 2,
                           lambda x, G: rotate_D(quant_per_token_sub(rotate_D(x, hadamard_R(x.shape[1], x.device)),
                                                                     lambda g: cb_nf_fixed(g, 4), 2),
                                                 hadamard_R(x.shape[1], x.device)), "rot + 2 sub-groups, 2.5b"))
    reg.append(make_method("pt/outlier1+nf2 fixed", "per_token", 2.0, 1,
                           lambda x, G: outlier_keep_pertoken(lambda g: cb_nf_fixed(g, 4), 1)(x, G),
                           "keep top-1 chan fp16 + rest per-tok NF", bits_override=(63 * 2 + 16 + 16) / 64))
    reg.append(make_method("pt/outlier1+nf2 Lloyd", "per_token", 2.0, 1,
                           lambda x, G: outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), 1)(x, G),
                           "keep top-1 chan fp16 + rest per-tok Lloyd", bits_override=(63 * 2 + 16 + 16) / 64))
    # ---- iter 4: stronger outlier isolation + smooth combos ---------------- #
    # keep top-2 outlier chans @ int8 (cheaper than fp16) + per-tok Lloyd on 62
    reg.append(make_method("pt/outlier2-int8+Lloyd", "per_token", 2.0, 1,
                           lambda x, G: outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), 2, outlier_bits=8)(x, G),
                           "keep top-2 chan int8 + rest per-tok Lloyd", bits_override=(62 * 2 + 2 * 8 + 16) / 64))
    reg.append(make_method("pt/smooth+outlier1+Lloyd", "per_token", 2.0, 1,
                           lambda x, G: smooth_then(outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), 1))(x, G),
                           "smooth then keep top-1 + Lloyd", bits_override=(63 * 2 + 16 + 16) / 64))
    reg.append(make_method("pt/outlier3-int8+Lloyd", "per_token", 2.0, 1,
                           lambda x, G: outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), 3, outlier_bits=8)(x, G),
                           "keep top-3 chan int8 + rest per-tok Lloyd", bits_override=(61 * 2 + 3 * 8 + 16) / 64))
    # ---- iter 5: sweep smooth + (k outliers @ b bits) within 2.5 budget ---- #
    reg.append(make_method("pt/smooth+outlier2-int8+Lloyd", "per_token", 2.0, 1,
                           lambda x, G: smooth_then(outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), 2, outlier_bits=8))(x, G),
                           "smooth + top-2 int8 + Lloyd", bits_override=(62 * 2 + 2 * 8 + 16) / 64))
    reg.append(make_method("pt/smooth+outlier3-6b+Lloyd", "per_token", 2.0, 1,
                           lambda x, G: smooth_then(outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), 3, outlier_bits=6))(x, G),
                           "smooth + top-3 @6bit + Lloyd", bits_override=(61 * 2 + 3 * 6 + 16) / 64))
    reg.append(make_method("pt/smooth+outlier4-4b+Lloyd", "per_token", 2.0, 1,
                           lambda x, G: smooth_then(outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), 4, outlier_bits=4))(x, G),
                           "smooth + top-4 @4bit + Lloyd", bits_override=(60 * 2 + 4 * 4 + 16) / 64))
    # ---- iter 6: push #isolated channels to saturation within 2.5b --------- #
    for k, ob in [(8, 3), (8, 4), (12, 3), (16, 2)]:
        bo = ((64 - k) * 2 + k * ob + 16) / 64
        reg.append(make_method(f"pt/smooth+outlier{k}-{ob}b+Lloyd", "per_token", 2.0, 1,
                               (lambda kk, obb: lambda x, G: smooth_then(
                                   outlier_keep_pertoken(lambda g: cb_lloyd(g, 4), kk, outlier_bits=obb))(x, G))(k, ob),
                               f"smooth + top-{k} @{ob}bit + Lloyd", bits_override=bo))
    # ---- iter 7: outlier-selection signal (amax vs sigma^2) at the k=8 spot - #
    reg.append(make_method("pt/smooth+outlier8-4b+Lloyd[var]", "per_token", 2.0, 1,
                           lambda x, G: smooth_then(outlier_keep_pertoken(
                               lambda g: cb_lloyd(g, 4), 8, outlier_bits=4, by="var"))(x, G),
                           "k=8 selected by sigma^2 (vs amax champion)", bits_override=(56 * 2 + 8 * 4 + 16) / 64))
    # ---- iter 8: LOWER-BIT BASE — replace nf2(2b) on the non-outlier channels --- #
    # with sign_sym(1b,1side) / sign(submean 1b,2side) / tern(1.58b,2side) / nf2(2b,2side).
    # honest accounting: per-token base group = (64-k) channels, +side fp16; outliers k@4bit.
    #   eff = ((64-k)*cw_base + 16*side_base + k*4) / 64
    BASES = {  # name -> (codebook_fn, codeword_bits, fp16_side_per_token_group)
        "sign_sym": (cb_sign_sym, 1.0, 1),
        "sign":     (cb_sign, 1.0, 2),
        "tern":     (cb_tern, math.log2(3), 2),
        "nf2":      (lambda g: cb_lloyd(g, 4), 2.0, 2),
    }
    for bname, (cbfn, cw, side) in BASES.items():
        for k in (8, 12, 16, 20):
            nb = 64 - k
            bo = (nb * cw + 16 * side + k * 4) / 64
            reg.append(make_method(
                f"pt/smooth+out{k}@4b+{bname}", "per_token", cw, side,
                (lambda fn, kk: lambda x, G: smooth_then(
                    outlier_keep_pertoken(fn, kk, outlier_bits=4))(x, G))(cbfn, k),
                f"{bname}-base on {nb}ch + top-{k}@4bit", bits_override=bo))
    # ---- iter 9: sigma^2-MIXED base on non-outlier channels (qlutattn-k1v4 per-token) -- #
    MIX = {
        "mix[s,nf2]":      ["sign", "nf2"],
        "mix[s,s,nf2]":    ["sign", "sign", "nf2"],
        "mix[s,t,nf2]":    ["sign", "tern", "nf2"],
        "mix[s,s,t,nf2]":  ["sign", "sign", "tern", "nf2"],
        "mix[ss,nf2]":     ["sign_sym", "nf2"],
        "mix[ss,t,nf2]":   ["sign_sym", "tern", "nf2"],
    }
    for mname, pol in MIX.items():
        for k in (8, 16):
            reg.append(make_method(
                f"pt/smooth+out{k}@4b+{mname}", "per_token", 1.5, 2,
                (lambda pp, kk: lambda x, G: smooth_then(mixed_base_pertoken(kk, pp))(x, G))(pol, k),
                f"sigma2-mix {pol} on {64-k}ch + top-{k}@4bit", bits_override=mixed_base_bits(k, pol)))
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
        if m["bits_override"] is not None:
            return m["bits_override"]
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
