#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D (head_dim x token_len) map of the K-cache AFTER the offline sigma^2 reorder.

One head of the post-RoPE K-cache (layer8). Rows = head_dim channels REORDERED by
their residual sigma^2 (the offline weight-reorder permutation); columns = a token
window. Color = K residual (K - per-channel mean; the mean is the DC the runtime
removes, free for attention). After reorder the low-sigma^2 channels (-> sign,
1.25 bit) are grouped on top and the high-sigma^2 channels (-> nf2, 2.25 bit) below,
two contiguous blocks; mixed = 1.88 bit/value. K-cache only (no V)."""
import argparse

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
    ap.add_argument("--sink", type=int, default=32)
    ap.add_argument("--recent", type=int, default=128)
    ap.add_argument("--win", type=int, default=256, help="Token window width shown on x-axis")
    ap.add_argument("--n-sign", type=int, default=24, help="Number of sign channels")
    a = ap.parse_args(argv)

    pt = a.pt or default_layer_pt(a.outdir, a.layer, "K")
    K = torch.load(pt, weights_only=False).float()  # [nh, T, D]
    nh, T, D = K.shape
    Kr = K[:, a.sink : T - a.recent, :]  # [nh, Treg, D]
    Treg = Kr.shape[1]
    mu = Kr.mean(1, keepdim=True)  # [nh, 1, D] per-channel mean
    R = Kr - mu  # residual [nh, Treg, D]
    sig2 = R.pow(2).mean(1)  # [nh, D] per-channel residual variance
    # pick the head with the widest sigma^2 dynamic range (clearest low->high gradient)
    dynrange = sig2.amax(1) / sig2.clamp(min=1e-9).amin(1)
    h0 = int(dynrange.argmax())
    order = torch.argsort(sig2[h0]).numpy()  # ascending sigma^2 -> reorder permutation
    n_nf2 = D - a.n_sign
    mixed = (a.n_sign * 1.25 + n_nf2 * 2.25) / D
    print(
        f"D={D} nh={nh} head={h0}  sig2 range {sig2[h0].min():.3e}..{sig2[h0].max():.3e}  "
        f"n_sign={a.n_sign} n_nf2={n_nf2} mixed={mixed:.4f} bit",
        flush=True,
    )

    w0 = Treg // 2 - a.win // 2  # a window from the middle of the prefill
    Rh = R[h0].numpy()[w0 : w0 + a.win, :]  # [WIN, D]
    after = Rh[:, order].T  # [D, WIN] rows reordered by sigma^2
    vlim = float(np.percentile(np.abs(after), 99))

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    im = ax.imshow(
        after,
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        aspect="auto",
        origin="upper",
        extent=[a.sink + w0, a.sink + w0 + a.win, D - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.axhline(a.n_sign - 0.5, color="k", lw=1.6, ls="--")
    x_lab = a.sink + w0 + a.win * 0.5
    GREEN, ORANGE = "#2ca02c", "#ff7f0e"
    ax.text(
        x_lab,
        (a.n_sign - 1) / 2,
        f"sign block  ·  {a.n_sign} ch  ·  1.25 bit",
        ha="center",
        va="center",
        color=GREEN,
        fontsize=12,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREEN, lw=1.6, alpha=0.92),
    )
    ax.text(
        x_lab,
        a.n_sign - 0.5 + n_nf2 / 2,
        f"nf2 block  ·  {n_nf2} ch  ·  2.25 bit",
        ha="center",
        va="center",
        color=ORANGE,
        fontsize=12,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ORANGE, lw=1.6, alpha=0.92),
    )
    # colored block markers on the left edge
    for y0, y1, c in [(-0.5, a.n_sign - 0.5, GREEN), (a.n_sign - 0.5, D - 0.5, ORANGE)]:
        ax.add_patch(
            plt.Rectangle(
                (a.sink + w0 - a.win * 0.022, y0),
                a.win * 0.012,
                y1 - y0,
                color=c,
                clip_on=False,
                transform=ax.transData,
            )
        )
    ax.set_xlabel("token position  (a 256-token window of the prefill)", fontsize=11)
    ax.set_ylabel("head_dim channel  (REORDERED by $\\sigma^2$, low → high)", fontsize=11)
    ax.set_title(
        f"K-cache spatial distribution AFTER offline $\\sigma^2$ reorder  "
        f"(layer{a.layer}, head {h0})\n"
        f"low-$\\sigma^2$ → sign (1.25b) on top · high-$\\sigma^2$ → nf2 (2.25b) below · "
        f"mixed = {mixed:.3f} ≈ 1.88 bit/value",
        fontsize=12,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("post-RoPE K residual  (K − per-channel mean $\\mu$)")
    out = f"{a.outdir}/kcache_reorder_2d_{a.tag}.png"
    save_fig(fig, out, dpi=130, tight=False, bbox_inches="tight")
    print("[done]", flush=True)


if __name__ == "__main__":
    run()
