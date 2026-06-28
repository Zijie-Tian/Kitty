#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample a few K-cache channels and plot EACH channel's value distribution as a
SEPARATE figure (one PNG per channel, not subplots in one image).

Channels are sampled evenly across the RoPE-period spectrum (fast-RoPE -> bimodal
arcsine, slow-RoPE -> peaked). Each figure overlays a Gaussian(mu, sigma)
reference so the departure from normal is visible.
"""
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


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_viz_args(ap)
    ap.add_argument("--num-channels", type=int, default=8)
    ap.add_argument("--bins", type=int, default=70)
    a = ap.parse_args(argv)

    outdir = os.path.join(a.outdir, "kv_channel_dist_K_per_channel")
    os.makedirs(outdir, exist_ok=True)

    pt = a.pt if a.pt else default_layer_pt(a.outdir, a.layer, "K")
    K = torch.load(pt, weights_only=False).float()
    if K.dim() == 3:
        K = K[a.head]  # [T, D]
    T, D = K.shape

    period = rope_period(D)

    # sample N channels evenly across the RoPE-period spectrum (fast -> slow)
    order = np.argsort(period)
    sel = [int(order[i]) for i in np.linspace(0, D - 1, a.num_channels).round().astype(int)]

    amax = K.abs().amax(0).numpy()
    saved = []
    for rank, c in enumerate(sel):
        v = K[:, c].numpy()
        mean, std = float(v.mean()), float(v.std())
        kurt = float(((v - mean) ** 4).mean() / (std ** 4 + 1e-12))
        fast = period[c] < 128
        kind = "fast-RoPE → bimodal (arcsine)" if fast else "slow-RoPE → peaked"
        color = "#1f77b4" if fast else "#9467bd"

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.hist(v, bins=a.bins, density=True, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        xs = np.linspace(float(v.min()), float(v.max()), 256)
        gauss = np.exp(-(xs - mean) ** 2 / (2 * std ** 2 + 1e-12)) / (std * np.sqrt(2 * np.pi) + 1e-12)
        ax.plot(xs, gauss, color="0.35", ls=":", lw=1.4, label="Gaussian(μ,σ) ref")
        ax.axvline(mean, color="k", ls="--", lw=1.0, label=f"mean={mean:.3f}")
        ax.set_title(f"K-cache channel {c}  (layer {a.layer}, head {a.head})\n"
                     f"RoPE period T={period[c]:.1f}  ·  {kind}", fontsize=10)
        ax.set_xlabel("channel value")
        ax.set_ylabel("density")
        txt = (f"amax = {amax[c]:.2f}\nstd  = {std:.3f}\nmean = {mean:.3f}\n"
               f"kurt = {kurt:.2f}  (Gauss=3)\nT    = {T} tokens")
        ax.text(0.975, 0.975, txt, transform=ax.transAxes, ha="right", va="top",
                family="monospace", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

        out = os.path.join(outdir, f"Kdist_r{rank}_ch{c:02d}_T{period[c]:.0f}.png")
        save_fig(fig, out, dpi=130)
        saved.append(out)
        print(f"[fig] {out}  T={period[c]:.0f} amax={amax[c]:.2f} std={std:.3f} kurt={kurt:.2f}")

    print(f"\n[done] {len(saved)} per-channel figures -> {outdir}")
    print(f"source: {pt}  head={a.head}  K[head]=[{T}, {D}]  sampled channels={sel}")


if __name__ == "__main__":
    run()
