#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot per-channel value distributions of the K-cache and V-cache.

For a sampled set of channels, show the histogram of that channel's values over
all tokens. K and V are plotted in SEPARATE figures.

K channels are sampled by RoPE period (fast-RoPE -> bimodal arcsine, slow-RoPE ->
peaked); V channels by per-channel amax (outlier -> normal)."""
import argparse
import os

import numpy as np
import torch

from lib.common import (
    add_viz_args,
    default_layer_pt,
    rope_period,
    save_fig,
    setup_matplotlib,
)


def panel(ax, vals, title, color):
    v = vals.numpy()
    ax.hist(v, bins=70, density=True, color=color, alpha=0.8)
    ax.axvline(v.mean(), color="k", ls="--", lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def make_fig(X, chans, titles, colors, suptitle, out):
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, c, t, col in zip(axs.flat, chans, titles, colors):
        panel(ax, X[:, c], t, col)
    for ax in axs.flat[len(chans):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.supxlabel("channel value", fontsize=10)
    fig.supylabel("density", fontsize=10)
    save_fig(fig, out, tight=False)


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_viz_args(ap)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    layer = a.layer
    head = a.head

    pt_k = a.pt if a.pt else default_layer_pt(a.outdir, layer, "K")
    pt_v = default_layer_pt(a.outdir, layer, "V")

    K = torch.load(pt_k, weights_only=False).float()[head]  # [T, D]
    V = torch.load(pt_v, weights_only=False).float()[head]  # [T, D]
    T, D = K.shape

    period = rope_period(D)

    # ---- K: sample 8 channels by RoPE period (fast -> slow) ----
    order = np.argsort(period)
    kch = [order[i] for i in np.linspace(0, D - 1, 8).round().astype(int)]
    kamax = K.abs().amax(0).numpy()
    ktitles = [
        f"K ch{c}  T={period[c]:.0f}{' (fast→bimodal)' if period[c] < 128 else ' (slow→peaked)'}\n"
        f"amax={kamax[c]:.1f}  std={K[:, c].std():.2f}"
        for c in kch
    ]
    kcolors = ["#1f77b4" if period[c] < 128 else "#9467bd" for c in kch]
    make_fig(
        K,
        kch,
        ktitles,
        kcolors,
        f"K-cache per-channel value distribution ({a.tag}, layer {layer}, head {head}, {K.shape[0]} tokens)\n"
        "sampled by RoPE period: fast-RoPE channels are bimodal (arcsine), slow-RoPE are peaked",
        os.path.join(a.outdir, "kv_channel_dist_K.png"),
    )

    # ---- V: sample 8 channels by amax (outlier -> normal) ----
    vamax = V.abs().amax(0).numpy()
    vorder = np.argsort(-vamax)
    vch = [vorder[i] for i in np.linspace(0, D - 1, 8).round().astype(int)]
    vtitles = [
        f"V ch{c}\namax={vamax[c]:.1f}  std={V[:, c].std():.2f}  "
        f"kurt={float(((V[:, c] - V[:, c].mean()) ** 4).mean() / V[:, c].var() ** 2):.1f}"
        for c in vch
    ]
    vcolors = ["#d62728" if vamax[c] > 2 * np.median(vamax) else "#2ca02c" for c in vch]
    make_fig(
        V,
        vch,
        vtitles,
        vcolors,
        f"V-cache per-channel value distribution ({a.tag}, layer {layer}, head {head}, {V.shape[0]} tokens)\n"
        "sampled by per-channel amax: red = outlier channels (amax > 2x median), green = normal",
        os.path.join(a.outdir, "kv_channel_dist_V.png"),
    )

    # quick stats to stdout
    print(f"\nK per-channel amax: min={kamax.min():.2f} median={np.median(kamax):.2f} max={kamax.max():.2f}")
    print(f"V per-channel amax: min={vamax.min():.2f} median={np.median(vamax):.2f} max={vamax.max():.2f}")
    print(f"K amax max/median ratio: {kamax.max()/np.median(kamax):.1f}x   V: {vamax.max()/np.median(vamax):.1f}x")


if __name__ == "__main__":
    run()
