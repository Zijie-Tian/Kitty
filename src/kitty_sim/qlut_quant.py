# -*- coding: utf-8 -*-
"""
QLUT-Attn k1v4 (per-channel) post-RoPE K-cache codebooks (formerly "typed").

Core idea under test: spend quant bits in proportion to a channel's residual
energy sigma^2. Channels are binned by per-layer sigma^2 quantile (bin 0 = lowest
sigma^2 ~ mu^2-dominated "red"; top bin = highest sigma^2 "blue"). A policy maps
each bin to a codebook. This subsumes the mu^2/sigma^2 split and lets the search
allocate cheap codebooks (meanonly/sign) to low-sigma^2 channels and expensive
ones (tern/nf2) to high-sigma^2 channels.

All codebooks stay in post-RoPE space (no de-RoPE) so the per-channel affine /
int-accumulator structure is preserved. Effective bits = codeword bits + fp16
side-info (16 bit each) / group. Pure fake-quant accuracy proxy.
"""
import numpy as np
import torch

_CODEWORD_BITS = {
    "fp16": 16.0, "meanonly": 0.0, "sign": 1.0, "tern": float(np.log2(3)),
    "uni2": 2.0, "nf2": 2.0, "uni3": 3.0,
}
_SIDE_FP16 = {
    "fp16": 0, "meanonly": 1, "sign": 2, "tern": 2, "uni2": 2, "nf2": 2, "uni3": 2,
}
CODEBOOKS = tuple(_CODEWORD_BITS.keys())

# Symmetric NF2 (IR-QLoRA appendix B.2, Table 11): fixed normalized levels
# {-1, -c, +c, +1}. The published +/- inner entries differ in the 8th decimal
# (float artifacts); we symmetrize to the fp32 magnitude of the negative entry.
# Nearest-neighbor boundary between inner and outer level is t = (1 + c) / 2;
# a strict `>` sends the exact boundary to the INNER level, matching argmin's
# first-hit tie rule over the sorted LUT. NOTE: this replaced the old adaptive
# per-group Lloyd-Max "nf2" -- the codebook is now a FIXED LUT, only the group
# mean mu and the absmax scale s are data-dependent (2 fp16 side values).
NF2_INNER = 0.25256848335266113
NF2_THRESH = (1.0 + NF2_INNER) / 2.0


def nf2_symmetric_lastdim(r):
    """Symmetric-NF2 snap of a ZERO-CENTERED group along the last axis:
    per-group absmax scale s, closed-form nearest level sign(r)*s*(c or 1)
    (no argmin tensor, no division -- s==0 all-zero groups reconstruct 0).
    sign(0)==0 maps an exact-zero value to 0 (measure-zero event; the strict
    LUT would give -c*s)."""
    s = r.abs().amax(-1, keepdim=True)
    level = torch.where(r.abs() > NF2_THRESH * s, s, NF2_INNER * s)
    return torch.sign(r) * level


def codebook_bits(name, group_size):
    return _CODEWORD_BITS[name] + 16.0 * _SIDE_FP16[name] / group_size


def _grouped(x, G):
    H, D, T = x.shape
    ng = T // G
    return x[:, :, : ng * G].reshape(H, D, ng, G), ng


def _ungroup(rec, x, ng, G):
    H, D, T = x.shape
    out = x.clone()
    out[:, :, : ng * G] = rec.reshape(H, D, ng * G).to(x.dtype)
    return out


def apply_codebook(x, G, name):
    """x:[H,D,T] post-RoPE -> reconstruction with one codebook on G-token groups."""
    if name == "fp16":
        return x
    xg, ng = _grouped(x.float(), G)
    if name == "meanonly":
        rec = xg.mean(-1, keepdim=True).expand_as(xg)
    elif name in ("sign", "tern"):
        mu = xg.mean(-1, keepdim=True)
        r = xg - mu
        thr = (0.5 if name == "tern" else 0.0) * r.abs().mean(-1, keepdim=True)
        mask = r.abs() > thr
        mag = (r.abs() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
        rec = mu + torch.sign(r) * mag * mask
    elif name in ("uni2", "uni3"):
        L = 4 if name == "uni2" else 8
        mn = xg.min(-1, keepdim=True).values
        mx = xg.max(-1, keepdim=True).values
        scale = (mx - mn).clamp(min=1e-6) / (L - 1)
        q = ((xg - mn) / scale).round().clamp(0, L - 1)
        rec = q * scale + mn
    elif name == "nf2":
        mu = xg.mean(-1, keepdim=True)
        rec = mu + nf2_symmetric_lastdim(xg - mu)
    else:
        raise ValueError(name)
    return _ungroup(rec, x, ng, G)


def channel_sigma2(x_quant, G):
    """x_quant:[H,D,Tq] -> per-channel residual sigma^2 [H,D] (submean over G-groups)."""
    xg, _ = _grouped(x_quant.float(), G)
    return (xg - xg.mean(-1, keepdim=True)).pow(2).mean(dim=(-1, -2))


def sigma2_bins(sig2, n_bins):
    """sig2:[H,D] -> bin id [H,D] in [0,n_bins), 0=lowest sigma^2, by equal-count quantiles."""
    flat = sig2.reshape(-1)
    order = flat.argsort()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(flat.numel(), device=flat.device)
    binid = (ranks.float() * n_bins / flat.numel()).floor().long().clamp(max=n_bins - 1)
    return binid.reshape(sig2.shape)


def apply_qlut(x_full, bin_codebooks, binid, sink, recent, G):
    """Reconstruct quant region: channels in bin b use codebook bin_codebooks[b];
    sink + recent kept fp16. bin_codebooks: list[str] len n_bins. Returns [H,D,T]."""
    H, D, T = x_full.shape
    q0, q1 = sink, T - recent
    xq = x_full[:, :, q0:q1]
    rec = xq.clone()
    for b, cb in enumerate(bin_codebooks):
        mask = (binid == b)
        if not mask.any():
            continue
        rq = apply_codebook(xq, G, cb)
        rec = torch.where(mask[:, :, None], rq, rec)
    out = x_full.clone()
    out[:, :, q0:q1] = rec
    return out


def effective_bits(bin_codebooks, binid, G):
    counts = torch.bincount(binid.reshape(-1), minlength=len(bin_codebooks)).float()
    frac = counts / counts.sum()
    return float(sum(frac[b].item() * codebook_bits(cb, G) for b, cb in enumerate(bin_codebooks)))


# --------------------------------------------------------------------------- #
# Integration helpers for the real KittyKVCache quantization path (LongBench).
# --------------------------------------------------------------------------- #
def compute_sigma_bins(key_region, group_size, n_bins):
    """key_region:[B,nh,D,Tq] post-RoPE quant region -> per-channel bin ids [nh,D]
    (B=0 used; bins = per-(layer) sigma^2 equal-count quantiles, 0=lowest sigma^2)."""
    sig2 = channel_sigma2(key_region[0], group_size)        # [nh,D]
    return sigma2_bins(sig2, n_bins)


def fake_quant_qlut_buffer(key_slice, bin_ids, bin_codebooks, group_size):
    """key_slice:[B,nh,D,T] post-RoPE -> per-channel qlut-codebook reconstruction.
    bin_ids:[nh,D] long (channel -> bin); bin_codebooks: list[str] (bin -> codebook)."""
    B = key_slice.shape[0]
    out = key_slice.clone()
    for b in range(B):
        x = key_slice[b].float()                            # [nh,D,T]
        rec = x.clone()
        for bi, cb in enumerate(bin_codebooks):
            mask = (bin_ids == bi)
            if not mask.any():
                continue
            rq = apply_codebook(x, group_size, cb)
            rec = torch.where(mask[:, :, None], rq, rec)
        out[b] = rec.to(key_slice.dtype)
    return out
