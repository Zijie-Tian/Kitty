#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dequantize K with qlutattn-sign-pt (per-channel submean + per-token 1-bit sign)
and plot per-channel value distributions (original vs dequant) to verify the
quantization actually fired.

Uses the REAL kitty_simulate.KittyKVCache._quant_k_pertoken path via a duck-typed
self (k_codebook='qlut', pertoken_pc_submean=True, bin_codebooks=['sign']).
Per-token sign reconstruction = mu_channel + sign(K-mu_channel) * mag_token, so a
correctly-fired quant turns each channel into TWO clusters at mu +/- mag (vs the
original continuous distribution). Overall NMSE ~0.3-0.5 = fired; ~0 = identity.
"""
import argparse
import os
import sys

import numpy as np
import torch

from lib.common import (
    add_viz_args,
    default_layer_pt,
    save_fig,
    setup_matplotlib,
)


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_viz_args(ap)
    ap.add_argument("--base", type=float, default=500000.0)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    pt = a.pt or default_layer_pt(a.outdir, a.layer, "K")
    K = torch.load(pt, weights_only=False).float()  # [nh, T, D]
    nh, T, D = K.shape
    print(f"loaded K {tuple(K.shape)}  (layer{a.layer})")

    sys.path.insert(0, "src")
    from kitty_sim.kitty_simulate import KittyKVCache
    from types import SimpleNamespace

    # --- real qlutattn-sign-pt quant path ---
    mock = SimpleNamespace(
        k_codebook="qlut",
        bin_codebooks=["sign"],
        pertoken_offline=False,
        pertoken_mixed=False,
        pertoken_pc_submean=True,
        pertoken_rotate=False,
        k_pc_mean={},
    )
    ks = K[None].to(a.device)  # [1, nh, T, D]
    Khat = KittyKVCache._quant_k_pertoken(mock, ks, 0)[0].float().cpu()  # [nh, T, D]

    nmse = (((K - Khat) ** 2).sum() / (K ** 2).sum()).item()
    print(
        f"overall dequant NMSE = {nmse:.4f}  "
        f"({'FIRED (sign ~0.3-0.5)' if nmse > 0.05 else 'NOT fired / identity!'})"
    )

    head = a.head
    Ko, Kh = K[head], Khat[head]  # [T, D]
    # sanity: per-token mag should be constant across channels within a token
    r = K.float() - K.float().mean(2, keepdim=True)  # rough check only
    print(
        f"dequant value range: orig [{Ko.min():.2f},{Ko.max():.2f}]  "
        f"dequant [{Kh.min():.2f},{Kh.max():.2f}]"
    )

    # sample 8 channels by RoPE period (fast -> slow), same as the original dist plot
    j = np.arange(D) % (D // 2)
    period = 2 * np.pi * a.base ** (2.0 * j / D)
    order = np.argsort(period)
    chans = [int(order[i]) for i in np.linspace(0, D - 1, 8).round().astype(int)]

    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axs.flat, chans):
        o = Ko[:, c].numpy()
        h = Kh[:, c].numpy()
        lo = min(o.min(), h.min())
        hi = max(o.max(), h.max())
        bins = np.linspace(lo, hi, 70)
        ax.hist(o, bins=bins, color="#1f77b4", alpha=0.55, density=True, label="original")
        ax.hist(
            h, bins=bins, color="#d62728", alpha=0.60, density=True, label="dequant sign-pt"
        )
        mu_c = float(Ko[:, c].mean())
        ax.axvline(mu_c, color="k", ls=":", lw=0.8)
        nm = (
            ((Ko[:, c] - Kh[:, c]) ** 2).sum()
            / (Ko[:, c] ** 2).sum().clamp(min=1e-9)
        ).item()
        ax.set_title(f"K ch{c}  T={period[c]:.0f}\nnmse={nm:.2f}  μ={mu_c:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    fig.suptitle(
        f"qlutattn-sign-pt: per-channel value distribution, original vs dequant "
        f"(layer{a.layer} head{head}, overall NMSE={nmse:.3f})",
        fontsize=12,
    )
    fig.supxlabel("channel value")
    fig.supylabel("density")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_fig(
        fig,
        f"{a.outdir}/signpt_dequant_channel_dist_{a.tag}.png",
        tight=False,
    )


if __name__ == "__main__":
    run()
