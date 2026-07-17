#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: V-cache quant-grouping study (per-token vs token-block / head_dim-block).

Question: instead of the current per-token grouping (one (scale, zp) per
(head, token) over the whole head_dim row), can V be quantized with
  * blocks of N consecutive tokens along the TOKEN axis (scale/zp on the
    channel axis -> constant along the P@V reduction, kernel-friendly), and/or
  * blocks of M channels along the HEAD_DIM axis (sub-row groups)?

Scheme grammar (all asymmetric min-max, round-to-nearest, --bits, fp16 side):
  per_token       group = 1 token x 64 ch   (current _quant_v)
  pt_hd{M}        group = 1 token x M ch    (head_dim split only)
  pc_blk{N}       group = N tokens x 1 ch   (token-axis blocking only)
  tile_t{N}c{M}   group = N tokens x M ch   (2D tile; scale/zp per channel-block,
                                             shared across the N-token block)
  ptensor_blk{N}  group = N tokens x 64 ch  (kept for reference, off by default)

Two studies on ONE real LongBench 32k prefill:
  1. reconstruction error of the V mid-region [sink, T-recent)
  2. attention-output error ||P@V_hat - P@V|| for the last --n-query REAL query
     rows (post-RoPE Q captured via hooks).  We report both the deployment-style
     per-scheme block frontier and a common frontier shared by every scheme;
     the latter removes the unfair advantage of a larger block leaving more
     pending tokens in fp16.

Usage (via the skill entry):
  bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh probe v-grouping-error \
    --model /path/to/Llama-3.2-1B-Instruct --longbench-dir /path/to/LongBench/data \
    --seq-len 32768 --tag llama32-1b-narrativeqa --outdir probe_out/v_grouping
"""
import argparse
import json
import math
import os

import torch

from lib.common import add_prefill_args, load_model, pick_doc

DEFAULT_SCHEMES = (
    "per_token,pt_hd32,pt_hd16,pt_hd8,"
    "pc_blk16,pc_blk64,pc_blk128,"
    "tile_t16c4,tile_t16c8,tile_t16c16,tile_t16c32,tile_t32c4,tile_t32c8"
)


# ---------------------------------------------------------------------------
# quantizers (fp32 in, fp16-round-tripped out, matching deployed storage)
# ---------------------------------------------------------------------------

def _minmax_fq(x: torch.Tensor, dim, bits: int) -> torch.Tensor:
    """Asymmetric min-max fake quant over reduction dims `dim` of x.

    Run the quantizer itself in fp16, matching ``fake_quant_groupwise_lastdim``
    (the deployed Kitty simulation path), then return fp32 for error reduction.
    Rounding only the final reconstruction is not equivalent at 2 bit: fp16
    min/scale rounding can move values across a code boundary.
    """
    xh = x.half()
    levels = 2 ** bits - 1
    mn = xh.amin(dim=dim, keepdim=True)
    mx = xh.amax(dim=dim, keepdim=True)
    scale = (mx - mn).clamp(min=1e-4) / levels
    max_val = torch.full_like(scale, levels)
    q = ((xh - mn) / scale).clamp(torch.zeros_like(max_val), max_val).round()
    return (q * scale + mn).float()


def _lloyd_fq_lastdim(x: torch.Tensor, levels: int = 4, iters: int = 10) -> torch.Tensor:
    """Per-group Lloyd-Max with four fp16-stored reconstruction levels."""
    x = x.half().float()
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    ar = torch.arange(levels, device=x.device, dtype=x.dtype)
    lev = lo + (hi - lo) * (ar + 0.5) / levels
    for _ in range(iters):
        assign = (x.unsqueeze(-1) - lev.unsqueeze(-2)).abs().argmin(-1)
        new = []
        for idx in range(levels):
            mask = assign == idx
            count = mask.sum(-1)
            total = (x * mask).sum(-1)
            new.append(torch.where(count > 0, total / count.clamp(min=1), lev[..., idx]))
        lev = torch.stack(new, dim=-1)
    # The four levels are side information in the intended representation.
    lev = lev.half().float()
    assign = (x.unsqueeze(-1) - lev.unsqueeze(-2)).abs().argmin(-1)
    return torch.gather(lev, -1, assign)


def _mse_uniform_fq_lastdim(x: torch.Tensor, bits: int, iters: int = 10) -> torch.Tensor:
    """MSE-fitted affine uniform quantizer with fp16 offset and scale.

    Alternates nearest-code assignment and the closed-form least-squares fit
    ``x ~= offset + scale*q``.  Unlike endpoint min-max it may clip outliers,
    but it keeps exactly the same two-fp16 side representation.
    """
    x = x.half().float()
    levels = 2 ** bits - 1
    offset = x.amin(dim=-1, keepdim=True)
    scale = (x.amax(dim=-1, keepdim=True) - offset).clamp(min=1e-4) / levels
    n = x.shape[-1]
    sum_x = x.sum(dim=-1, keepdim=True)
    for _ in range(iters):
        q = ((x - offset) / scale).round().clamp(0, levels)
        sum_q = q.sum(dim=-1, keepdim=True)
        sum_q2 = q.square().sum(dim=-1, keepdim=True)
        sum_qx = (q * x).sum(dim=-1, keepdim=True)
        denom = n * sum_q2 - sum_q.square()
        new_scale = (n * sum_qx - sum_q * sum_x) / denom.clamp(min=1e-12)
        valid = (denom > 0) & (new_scale > 1e-6)
        new_offset = (sum_x - new_scale * sum_q) / n
        scale = torch.where(valid, new_scale, scale)
        offset = torch.where(valid, new_offset, offset)
    offset = offset.half().float()
    scale = scale.clamp(min=1e-4).half().float()
    q = ((x - offset) / scale).round().clamp(0, levels)
    return q * scale + offset


_METHOD_PREFIXES = (
    "biascorr_", "pcaff_", "pcmean_", "pcrms_", "sortstd_",
    "rht_", "had_", "lloyd_", "mseuni_",
)


def parse_method(method: str):
    """Return (modifier set, base grouping scheme)."""
    modifiers = set()
    base = method
    while True:
        for prefix in _METHOD_PREFIXES:
            if base.startswith(prefix):
                modifiers.add(prefix[:-1])
                base = base[len(prefix):]
                break
        else:
            return modifiers, base


def parse_scheme(scheme: str, D: int):
    """Return (token_block N, chan_block M) of a scheme name."""
    _, scheme = parse_method(scheme)
    if scheme == "per_token":
        return 1, D
    if scheme.startswith("pt_hd"):
        return 1, int(scheme[len("pt_hd"):])
    if scheme.startswith("pc_blk"):
        return int(scheme[len("pc_blk"):]), 1
    if scheme.startswith("tile_t"):
        n, m = scheme[len("tile_t"):].split("c")
        return int(n), int(m)
    if scheme.startswith("ptensor_blk"):
        return int(scheme[len("ptensor_blk"):]), D
    raise ValueError(scheme)


def _fwht_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Normalized self-inverse Walsh-Hadamard transform."""
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Hadamard transform needs power-of-two D, got {n}")
    lead = x.shape[:-1]
    y = x
    h = 1
    while h < n:
        y = y.reshape(*lead, n // (2 * h), 2, h)
        a0, a1 = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a0 + a1, a0 - a1), dim=-2).reshape(*lead, n)
        h *= 2
    return y / math.sqrt(n)


def _rht_signs(D: int, device, dtype, seed: int) -> torch.Tensor:
    """Fixed zero-calibration random signs for a foldable RHT."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    signs = torch.randint(0, 2, (D,), generator=gen, dtype=torch.int8)
    return signs.to(device=device, dtype=dtype).mul_(2).sub_(1)


def _quant_grouped(
    x: torch.Tensor,
    base: str,
    bits: int,
    lloyd: bool,
    mse_uniform: bool,
    lloyd_iters: int,
    mse_iters: int,
) -> torch.Tensor:
    """Quantize [H,T,D] in N-token by M-channel groups."""
    H, T, D = x.shape
    N, M = parse_scheme(base, D)
    assert D % M == 0, f"{base}: D={D} not divisible by M={M}"

    def run_groups(g):
        if lloyd:
            if bits != 2:
                raise ValueError("Lloyd tile probe currently supports only 2-bit/4-level codes")
            candidate = _lloyd_fq_lastdim(g, iters=lloyd_iters)
            baseline = _minmax_fq(g, dim=-1, bits=bits)
            use_candidate = ((candidate - g).square().sum(-1, keepdim=True)
                             <= (baseline - g).square().sum(-1, keepdim=True))
            return torch.where(use_candidate, candidate, baseline)
        if mse_uniform:
            candidate = _mse_uniform_fq_lastdim(g, bits, iters=mse_iters)
            baseline = _minmax_fq(g, dim=-1, bits=bits)
            use_candidate = ((candidate - g).square().sum(-1, keepdim=True)
                             <= (baseline - g).square().sum(-1, keepdim=True))
            return torch.where(use_candidate, candidate, baseline)
        return _minmax_fq(g, dim=-1, bits=bits)

    out = torch.empty_like(x)
    nb, rem = divmod(T, N)
    if nb:
        groups = (x[:, :nb * N, :].reshape(H, nb, N, D // M, M)
                  .permute(0, 1, 3, 2, 4).reshape(H, nb, D // M, N * M))
        q = run_groups(groups).reshape(H, nb, D // M, N, M).permute(0, 1, 3, 2, 4)
        out[:, :nb * N, :] = q.reshape(H, nb * N, D)
    if rem:
        groups = (x[:, nb * N:, :].reshape(H, 1, rem, D // M, M)
                  .permute(0, 1, 3, 2, 4).reshape(H, 1, D // M, rem * M))
        q = run_groups(groups).reshape(H, 1, D // M, rem, M).permute(0, 1, 3, 2, 4)
        out[:, nb * N:, :] = q.reshape(H, rem, D)
    return out


def quant_v(
    x: torch.Tensor,
    scheme: str,
    bits: int,
    calib_fraction: float = 1.0,
    rht_seed: int = 20260711,
    lloyd_iters: int = 10,
    mse_iters: int = 10,
) -> torch.Tensor:
    """x: [H, T, D] fp32 (the region to quantize). Returns fake-quantized x.

    Group = N consecutive tokens x M consecutive channels; one (scale, zp) per
    group. Tail tokens (T % N != 0) form one smaller final block, matching the
    K cache's block-aligned tail schedule.

    Optional prefixes explore large-tile rescue methods:
      pcmean_ / pcrms_ / pcaff_: prompt-calibrated per-channel normalization;
      had_ / rht_: foldable Hadamard or randomized Hadamard rotation;
      sortstd_: oracle channel reorder by prompt sigma (diagnostic upper bound);
      lloyd_: four-level per-tile Lloyd codebook (four fp16 levels per group).
      mseuni_: MSE-fitted uniform offset/scale (same two-fp16 side as min-max).
      biascorr_: subtract one prompt-calibrated fp16 error bias per channel.
    """
    H, T, D = x.shape
    if not 0 < calib_fraction <= 1:
        raise ValueError(f"calib_fraction must be in (0,1], got {calib_fraction}")
    calib_end = max(1, int(T * calib_fraction))
    modifiers, base = parse_method(scheme)
    z = x
    mu = scale = perm = inv_perm = signs = None

    # Rotation/reorder come first so a fixed V transform can be folded into
    # W_v/W_o; prompt-adaptive normalization then lives in that stored basis.
    if "rht" in modifiers:
        signs = _rht_signs(D, z.device, z.dtype, rht_seed)
        z = _fwht_lastdim(z * signs).half().float()
    elif "had" in modifiers:
        z = _fwht_lastdim(z).half().float()

    if "sortstd" in modifiers:
        score = z[:, :calib_end].square().mean(dim=1).sqrt()
        perm = score.argsort(dim=-1)
        inv_perm = perm.argsort(dim=-1)
        z = torch.gather(z, -1, perm[:, None, :].expand(H, T, D))

    if "pcaff" in modifiers or "pcmean" in modifiers:
        mu = z[:, :calib_end].mean(dim=1, keepdim=True).half().float()
        z = z - mu
    if "pcaff" in modifiers:
        scale = (z[:, :calib_end].square().mean(dim=1, keepdim=True).sqrt()
                 .clamp(min=1e-4).half().float())
        z = z / scale
    elif "pcrms" in modifiers:
        scale = (z[:, :calib_end].square().mean(dim=1, keepdim=True).sqrt()
                 .clamp(min=1e-4).half().float())
        z = z / scale

    zq = _quant_grouped(
        z, base, bits,
        lloyd="lloyd" in modifiers,
        mse_uniform="mseuni" in modifiers,
        lloyd_iters=lloyd_iters,
        mse_iters=mse_iters,
    )

    if scale is not None:
        zq = zq * scale
    if mu is not None:
        zq = zq + mu
    if inv_perm is not None:
        zq = torch.gather(zq, -1, inv_perm[:, None, :].expand(H, T, D))
    if "rht" in modifiers:
        zq = _fwht_lastdim(zq) * signs
    elif "had" in modifiers:
        zq = _fwht_lastdim(zq)
    if "biascorr" in modifiers:
        error_bias = ((zq - x)[:, :calib_end].mean(dim=1, keepdim=True)
                      .half().float())
        zq = zq - error_bias
    return zq.half().float()


def side_bits_per_value(scheme: str, D: int, side_bits_per_group: int = 32) -> float:
    """Per-tile side information amortized per value."""
    modifiers, _ = parse_method(scheme)
    if "lloyd" in modifiers:
        side_bits_per_group = 64  # four fp16 reconstruction levels
    N, M = parse_scheme(scheme, D)
    return side_bits_per_group / (N * M)


def side_bits_per_group(scheme: str) -> int:
    modifiers, _ = parse_method(scheme)
    return 64 if "lloyd" in modifiers else 32


def global_side_bits_per_value(scheme: str, T: int) -> float:
    """Prompt-level per-channel normalization metadata amortized over T."""
    modifiers, _ = parse_method(scheme)
    side = 0
    if "pcaff" in modifiers:
        side += 32  # fp16 mean + fp16 rms per channel
    elif "pcmean" in modifiers or "pcrms" in modifiers:
        side += 16
    if "biascorr" in modifiers:
        side += 16  # one fp16 post-quant error correction per channel
    return side / T


# ---------------------------------------------------------------------------
# prefill with post-RoPE Q capture for the last n_query rows
# ---------------------------------------------------------------------------

def prefill_capture(model, ids, chunk, device, n_query):
    """Chunked prefill; returns (K, V, Q) per layer.

    K, V: [Hkv, T, D] fp16 cpu (full sequence, post-RoPE K / raw V)
    Q   : [Hq, n_query, D] fp32 cpu (post-RoPE, the last n_query positions)
    """
    from transformers import DynamicCache
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    T = ids.shape[1]
    nl = model.config.num_hidden_layers
    D = getattr(model.config, "head_dim",
                model.config.hidden_size // model.config.num_attention_heads)
    state = {"start": 0}
    q_parts = {li: [] for li in range(nl)}

    def make_hook(li):
        def hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings")
            if hs is None or pe is None:
                return
            cos, sin = pe
            B, L, _ = hs.shape
            pos = state["start"] + torch.arange(L, device=hs.device)
            keep = pos >= (T - n_query)
            if not bool(keep.any()):
                return
            q = module.q_proj(hs).view(B, L, -1, D).transpose(1, 2)  # [B,Hq,L,D]
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            q_parts[li].append(q[0, :, keep, :].float().cpu())
        return hook

    handles = [model.model.layers[li].self_attn.register_forward_pre_hook(
        make_hook(li), with_kwargs=True) for li in range(nl)]
    cache = DynamicCache()
    with torch.inference_mode():
        for s in range(0, T, chunk):
            state["start"] = s
            cache = model(input_ids=ids[:, s: s + chunk].to(device),
                          past_key_values=cache, use_cache=True).past_key_values
    for h in handles:
        h.remove()

    Ks, Vs, Qs = [], [], []
    for li in range(nl):
        if hasattr(cache, "layers"):
            K, V = cache.layers[li].keys, cache.layers[li].values
        else:
            K, V = cache.key_cache[li], cache.value_cache[li]
        Ks.append(K[0].half().cpu())
        Vs.append(V[0].half().cpu())
        Qs.append(torch.cat(q_parts[li], dim=1))
    return Ks, Vs, Qs


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def run(argv=None):
    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
    ap.add_argument("--n-query", type=int, default=256)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--calib-fraction", type=float, default=1.0,
                    help="fraction of the prompt used for prompt-level pc stats/bias correction")
    ap.add_argument("--rht-seed", type=int, default=20260711,
                    help="fixed randomized-Hadamard sign seed")
    ap.add_argument("--mse-iters", type=int, default=10,
                    help="alternating least-squares iterations for mseuni_ methods")
    ap.add_argument("--lloyd-iters", type=int, default=10,
                    help="Lloyd iterations for lloyd_ methods")
    ap.add_argument("--schemes", default=DEFAULT_SCHEMES,
                    help="comma-separated scheme names (see module docstring)")
    ap.add_argument("--lb-files", default=None,
                    help="comma-separated LongBench jsonl names (doc choice)")
    a = ap.parse_args(argv)
    dev = a.device
    os.makedirs(a.outdir, exist_ok=True)

    model, tok = load_model(a.model, dev)
    lb_files = a.lb_files.split(",") if a.lb_files else None
    ids = pick_doc(os.path.expanduser(a.longbench_dir), tok, a.seq_len, lb_files=lb_files)
    T = ids.shape[1]
    print(f"[v-grouping] tag={a.tag} T={T} bits={a.bits} sink={a.sink} recent={a.recent}",
          flush=True)

    Ks, Vs, Qs = prefill_capture(model, ids, a.chunk, dev, a.n_query)
    del model
    torch.cuda.empty_cache()

    nl = len(Ks)
    Hkv, _, D = Ks[0].shape
    Hq = Qs[0].shape[0]
    rep = Hq // Hkv
    schemes = [s.strip() for s in a.schemes.split(",") if s.strip()]

    q0, q1 = a.sink, T - a.recent          # one-shot mid region (recon study)
    rows = torch.arange(T - a.n_query, T)  # query positions (attn study)
    res = {s: {"recon_se": 0.0, "recon_ref": 0.0,
               "out_se": 0.0, "out_ref": 0.0,
               "out_common_se": 0.0, "out_common_ref": 0.0,
               "out_base_same_se": 0.0,
               "per_layer_recon": [], "per_layer_out": [],
               "per_layer_out_common": [],
               "per_layer_out_base_same": []} for s in schemes}
    pmass_quant = 0.0  # attention mass landing on the (largest) quantizable region
    pmass_common = 0.0
    token_blocks = sorted({parse_scheme(s, D)[0] for s in schemes})
    common_blk = max(token_blocks)

    for li in range(nl):
        V = Vs[li].float().to(dev)                       # [Hkv,T,D]
        K = Ks[li].float().to(dev)
        Q = Qs[li].float().to(dev)                       # [Hq,nq,D]
        # real attention rows: P[hq, r, t]
        scores = torch.einsum("hqd,htd->hqt", Q,
                              K.repeat_interleave(rep, 0)) / math.sqrt(D)
        col = torch.arange(T, device=dev)
        causal = col[None, :] > rows[:, None].to(dev)    # [nq,T]
        scores.masked_fill_(causal[None], float("-inf"))
        P = scores.softmax(-1)                           # fp32 [Hq,nq,T]
        del scores
        O_ref = torch.einsum("hqt,htd->hqd", P, V.repeat_interleave(rep, 0))
        res_ref_out = float((O_ref ** 2).sum())

        Vmid = V[:, q0:q1, :]
        ref_mid = float((Vmid ** 2).sum())
        # The existing per-token baseline, evaluated under every scheme's
        # frontier below.  Comparing against this same-frontier baseline
        # separates grouping error from the extra fp16 pending tail of N>1.
        dV_base = torch.zeros_like(V)
        dV_base[:, q0:q1, :] = quant_v(
            Vmid, "per_token", a.bits, calib_fraction=a.calib_fraction,
            rht_seed=a.rht_seed, lloyd_iters=a.lloyd_iters,
            mse_iters=a.mse_iters) - Vmid
        live_by_blk = {}
        base_same_se = {}
        for blk in token_blocks:
            frontier = ((rows.to(dev) - a.recent - q0).clamp(min=0) // blk) * blk + q0
            live = (col[None, :] >= q0) & (col[None, :] < frontier[:, None])
            live_by_blk[blk] = live
            dO_base = torch.einsum(
                "hqt,htd->hqd", P * live[None], dV_base.repeat_interleave(rep, 0))
            base_same_se[blk] = float((dO_base ** 2).sum())
        live_common = live_by_blk[common_blk]

        for s in schemes:
            Vq = quant_v(
                Vmid, s, a.bits, calib_fraction=a.calib_fraction,
                rht_seed=a.rht_seed, lloyd_iters=a.lloyd_iters,
                mse_iters=a.mse_iters)
            se = float(((Vq - Vmid) ** 2).sum())
            res[s]["recon_se"] += se
            res[s]["recon_ref"] += ref_mid
            res[s]["per_layer_recon"].append(se / max(ref_mid, 1e-30))

            # attention-output error with per-row fp16 windows + block-aligned
            # frontier: row p sees quantized cols [sink, align(p-recent)).
            dV = torch.zeros_like(V)
            dV[:, q0:q1, :] = Vq - Vmid
            blk = parse_scheme(s, D)[0]
            live = live_by_blk[blk]  # [nq,T]
            dO = torch.einsum("hqt,htd->hqd", P * live[None], dV.repeat_interleave(rep, 0))
            out_se = float((dO ** 2).sum())
            res[s]["out_se"] += out_se
            res[s]["out_ref"] += res_ref_out
            res[s]["per_layer_out"].append(out_se / max(res_ref_out, 1e-30))
            res[s]["out_base_same_se"] += base_same_se[blk]
            res[s]["per_layer_out_base_same"].append(
                base_same_se[blk] / max(res_ref_out, 1e-30))

            dO_common = torch.einsum(
                "hqt,htd->hqd", P * live_common[None], dV.repeat_interleave(rep, 0))
            out_common_se = float((dO_common ** 2).sum())
            res[s]["out_common_se"] += out_common_se
            res[s]["out_common_ref"] += res_ref_out
            res[s]["per_layer_out_common"].append(
                out_common_se / max(res_ref_out, 1e-30))

        live_any = (col[None, :] >= q0) & (col[None, :] < (rows.to(dev) - a.recent)[:, None])
        pmass_quant += float((P * live_any[None]).sum() / P.sum())
        pmass_common += float((P * live_common[None]).sum() / P.sum())
        del V, K, Q, P, O_ref, dV, dV_base
        torch.cuda.empty_cache()
        print(f"[v-grouping] layer {li} done", flush=True)

    print(f"\n== {a.tag}: V {a.bits}-bit grouping probe, T={T}, mid=[{q0},{q1}), "
          f"{a.n_query} real query rows, attn-mass on quantized region "
          f"{100 * pmass_quant / nl:.1f}% ==")
    hdr = (f"{'scheme':>14} {'grp NxM':>8} {'b/val':>6} | "
           f"{'recon relMSE':>12} {'SQNR dB':>8} | {'attn-out relMSE':>15} {'dB':>7}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for s in schemes:
        r = res[s]["recon_se"] / res[s]["recon_ref"]
        o = res[s]["out_se"] / res[s]["out_ref"]
        oc = res[s]["out_common_se"] / res[s]["out_common_ref"]
        same = res[s]["out_se"] / max(res[s]["out_base_same_se"], 1e-30)
        N, M = parse_scheme(s, D)
        global_side = global_side_bits_per_value(s, T)
        tot = a.bits + side_bits_per_value(s, D) + global_side
        mid = q1 - q0
        full_tot = (16 * (a.sink + a.recent) * D + a.bits * mid * D
                    + side_bits_per_group(s) * math.ceil(mid / N) * (D // M)) / (T * D)
        full_tot += global_side
        modifiers, base = parse_method(s)
        summary[s] = {
            "base_scheme": base, "modifiers": sorted(modifiers),
            "token_block": N, "chan_block": M, "bits_per_value": tot,
            "full_cache_bits_per_value": full_tot,
            "recon_relmse": r, "recon_sqnr_db": -10 * math.log10(max(r, 1e-30)),
            "out_relmse": o, "out_sqnr_db": -10 * math.log10(max(o, 1e-30)),
            "out_common_relmse": oc,
            "out_common_sqnr_db": -10 * math.log10(max(oc, 1e-30)),
            "out_vs_per_token_same_frontier": same,
            "out_per_token_same_frontier_relmse": (
                res[s]["out_base_same_se"] / res[s]["out_ref"]),
            "per_layer_recon": res[s]["per_layer_recon"],
            "per_layer_out": res[s]["per_layer_out"],
            "per_layer_out_common": res[s]["per_layer_out_common"],
            "per_layer_out_per_token_same_frontier": res[s]["per_layer_out_base_same"],
        }
        print(f"{s:>14} {f'{N}x{M}':>8} {tot:6.2f} | "
              f"{r:12.3e} {summary[s]['recon_sqnr_db']:8.2f} | "
              f"{o:15.3e} {summary[s]['out_sqnr_db']:7.2f}")

    out = {"tag": a.tag, "T": T, "bits": a.bits,
           "calib_fraction": a.calib_fraction,
           "rht_seed": a.rht_seed,
           "mse_iters": a.mse_iters,
           "lloyd_iters": a.lloyd_iters,
           "sink": a.sink, "recent": a.recent,
           "n_query": a.n_query, "layers": nl, "Hkv": Hkv, "Hq": Hq, "D": D,
           "attn_mass_on_quant_region": pmass_quant / nl, "schemes": summary}
    out["common_frontier_block"] = common_blk
    out["attn_mass_on_common_quant_region"] = pmass_common / nl
    path = os.path.join(a.outdir, f"v_grouping_{a.tag}_b{a.bits}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("[json]", path, flush=True)


if __name__ == "__main__":
    run()
