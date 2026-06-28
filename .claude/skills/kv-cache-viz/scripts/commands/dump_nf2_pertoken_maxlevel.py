#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump snf-pt's PER-TOKEN nf2 codebook max level and plot its distribution.

snf-pt's nf2 is per-token: for each (head, token) the nf2 channels (the high-σ²
half per head, sign-frac=0.5) get a FRESH 4-level Lloyd codebook computed from
that token's own nf2-channel residuals. This reproduces
kitty_simulate.KittyKVCache._masked_lloyd_lastdim (init levels = [min,max] of the
token's nf2 channels split in 4, then 10 Lloyd iterations: nearest-level assign +
centroid update along head_dim). We dump, for every (layer, head, token), the
codebook's max level l3 (= largest reconstruction level) and also max|level|, then
plot the distribution over all tokens. Residual = post-RoPE K minus per-channel
mean (snf-pt's submean). nf2 channels chosen by this doc's per-head σ² ranking.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from lib.common import (
    add_prefill_args,
    load_model,
    prefill_and_extract,
    save_fig,
    setup_matplotlib,
)


def pertoken_lloyd_levels(rr, L=4, iters=10):
    """rr: [T, C] (C = nf2 channels of this head) -> levels [T, L] (per token).
    Mirrors _masked_lloyd_lastdim's math, restricted to the extracted nf2 chans."""
    lo = rr.min(1, keepdim=True).values
    hi = rr.max(1, keepdim=True).values
    ar = torch.arange(L, device=rr.device, dtype=rr.dtype)
    lev = lo + (hi - lo) * (ar + 0.5) / L  # [T, L]
    for _ in range(iters):
        d = (rr.unsqueeze(-1) - lev.unsqueeze(1)).abs()  # [T, C, L]
        a = d.argmin(-1)  # [T, C]
        oh = F.one_hot(a, L).to(rr.dtype)  # [T, C, L]
        cnt = oh.sum(1)  # [T, L]
        summ = (oh * rr.unsqueeze(-1)).sum(1)
        lev = torch.where(cnt > 0, summ / cnt.clamp(min=1), lev)
    return lev  # [T, L] ascending-ish


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
    ap.add_argument("--block", type=int, default=16)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    model, tok = load_model(a.model, a.device)
    data = prefill_and_extract(
        model,
        tok,
        os.path.expanduser(a.longbench_dir),
        seq_len=a.seq_len,
        sink=a.sink,
        recent=a.recent,
        chunk=a.chunk,
        device=a.device,
    )
    T = data["T"]
    nl = data["layers"]
    D = data["D"]
    q0, q1 = data["region"]
    print(
        f"[prefill] {a.tag}: T={T} layers={nl} D={D} region=[{q0},{q1})",
        flush=True,
    )

    maxlev_layers, maxabs_layers, amax_layers = [], [], []
    for li in range(nl):
        K = data["K"][li].to(a.device)  # [nh, Tq, D] on GPU
        nh, Tq, _ = K.shape
        mu = K.mean(1, keepdim=True)
        r = K - mu
        var = r.var(1)  # [nh, D]
        ml = torch.empty(nh, Tq)
        mabs = torch.empty(nh, Tq)
        amx = torch.empty(nh, Tq)
        for h in range(nh):
            v = var[h]
            nf2 = (v > v.median()).nonzero().flatten()  # high-σ² half = nf2 channels
            rr = r[h][:, nf2].contiguous()  # [Tq, n_nf2]
            lev = pertoken_lloyd_levels(rr)  # [Tq, 4]
            ml[h] = lev.max(1).values.cpu()  # l3 = codebook max level
            mabs[h] = lev.abs().max(1).values.cpu()  # max|level|
            amx[h] = rr.abs().max(1).values.cpu()  # this token's nf2 amax (ref)
        maxlev_layers.append(ml)
        maxabs_layers.append(mabs)
        amax_layers.append(amx)
        del K, r
        torch.cuda.empty_cache()

    M = torch.stack(maxlev_layers)  # [nl, nh, Tq]  l3
    Mabs = torch.stack(maxabs_layers)
    A = torch.stack(amax_layers)
    torch.save(
        {
            "max_level_l3": M.half(),
            "max_abs_level": Mabs.half(),
            "nf2_amax": A.half(),
            "model": a.tag,
            "def": "per-token nf2 codebook max level (l3) over high-σ² half per head",
        },
        f"{a.outdir}/nf2_pertoken_maxlevel_{a.tag}.pt",
    )

    flat = M.flatten().numpy()
    blk = a.block
    cv_blk = []
    for li in range(nl):
        x = M[li]  # [nh, Tq]
        nb = x.shape[1] // blk
        xb = x[:, : nb * blk].reshape(x.shape[0], nb, blk)
        cv_blk.append(
            (xb.std(2) / xb.mean(2).clamp(min=1e-9)).mean().item()
        )
    stats = {
        "model": a.tag,
        "T": T,
        "Tq": q1 - q0,
        "layers": nl,
        "D": D,
        "n_nf2_per_head": int((M.shape) and D // 2),
        "metric": "per-token nf2 codebook max level l3",
        "global": {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "p50": float(np.median(flat)),
            "p99": float(np.percentile(flat, 99)),
            "max": float(flat.max()),
            "cv_global": float(flat.std() / (flat.mean() + 1e-9)),
        },
        "cv_within_16tok_block_per_layer_mean": float(np.mean(cv_blk)),
        "per_layer_mean_l3": [float(M[li].mean()) for li in range(nl)],
    }
    json.dump(
        stats,
        open(f"{a.outdir}/nf2_pertoken_maxlevel_{a.tag}.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(
        json.dumps(
            {
                k: stats[k]
                for k in ["global", "cv_within_16tok_block_per_layer_mean"]
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs[0, 0].hist(flat, bins=100, color="#d62728", alpha=0.8, density=True)
    axs[0, 0].axvline(
        np.median(flat),
        color="k",
        ls="--",
        lw=0.8,
        label=f"median={np.median(flat):.2f}",
    )
    axs[0, 0].set_title(
        f"per-token nf2 codebook max level (l3) — all {nl}×{M.shape[1]}×{M.shape[2]} tokens"
    )
    axs[0, 0].set_xlabel("codebook max level l3")
    axs[0, 0].set_ylabel("density")
    axs[0, 0].legend()
    parts = [M[li].flatten().numpy() for li in range(nl)]
    axs[0, 1].boxplot(parts, showfliers=False)
    axs[0, 1].set_title("max level l3 by layer")
    axs[0, 1].set_xlabel("layer")
    axs[0, 1].set_ylabel("l3")
    for li, col in [(0, "#1f77b4"), (nl // 2, "#2ca02c"), (nl - 1, "#d62728")]:
        axs[1, 0].plot(M[li, 0, :2000].numpy(), col, lw=0.5, label=f"L{li} h0")
    axs[1, 0].set_title("max level l3 along tokens [first 2000]")
    axs[1, 0].set_xlabel("token")
    axs[1, 0].set_ylabel("l3")
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    samp = np.random.default_rng(0).choice(
        flat.size, size=min(40000, flat.size), replace=False
    )
    axs[1, 1].scatter(
        A.flatten().numpy()[samp], flat[samp], s=3, alpha=0.3, color="#9467bd"
    )
    axs[1, 1].plot([0, flat.max()], [0, flat.max()], "k--", lw=0.6)
    axs[1, 1].set_title(
        "l3 vs token's nf2 amax (how close the top level sits to the peak)"
    )
    axs[1, 1].set_xlabel("nf2-channel amax (this token)")
    axs[1, 1].set_ylabel("codebook max level l3")
    fig.suptitle(
        f"snf-pt per-token nf2 codebook max level — {a.tag} "
        f"(D={D}, sign-frac=0.5, T={T})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"{a.outdir}/nf2_pertoken_maxlevel_{a.tag}.png", tight=False)


if __name__ == "__main__":
    run()
