#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-cache decomposition as 3D bars (Token x Channel x value, SIGNED):
   K  =  mu_K  +  (K - mu_K)
shown as three panels joined by big '=' and '+' signs. Vivid diverging colormap
(seismic), bars up=positive/down=negative, z centered at 0, shared symmetric
z-limit."""
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
    from matplotlib.colors import Normalize
    from matplotlib import cm

    ap = argparse.ArgumentParser()
    add_viz_args(ap)
    ap.add_argument("--ntok", type=int, default=64, help="Number of tokens to show")
    a = ap.parse_args(argv)

    pt = a.pt or default_layer_pt(a.outdir, a.layer, "K")
    K = torch.load(pt, weights_only=False).float()[a.head]  # [T, D]
    T, D = K.shape
    muK = K.mean(0)  # [D]
    seg = slice(0, a.ntok)
    panels = [
        ("K", K[seg].numpy()),
        ("mu_K  (per-channel mean)", muK[None, :].repeat(a.ntok, 1).numpy()),
        ("K - mu_K  (residual)", (K[seg] - muK).numpy()),
    ]
    zmax = float(max(abs(M).max() for _, M in panels))
    print(
        "K signed ranges: "
        + "  ".join(f"{n.split()[0]}=[{M.min():.2f},{M.max():.2f}]" for n, M in panels)
    )
    print(f"symmetric zmax={zmax:.2f}")

    xpos, ypos = np.meshgrid(np.arange(a.ntok), np.arange(D), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)
    norm = Normalize(-zmax, zmax)

    fig = plt.figure(figsize=(19, 6.5))
    for i, (title, M) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        dz = M.ravel()
        ax.bar3d(xpos, ypos, zpos, 0.92, 0.92, dz, color=cm.seismic(norm(dz)), shade=True)
        ax.set_zlim(-zmax, zmax)
        ax.set_xlabel("Token")
        ax.set_ylabel("Channel")
        ax.set_zlabel("Value")
        ax.set_title(title, fontsize=12, pad=0)
        ax.view_init(elev=20, azim=-60)

    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.08, top=0.9, bottom=0.04)
    fig.text(0.353, 0.5, "=", fontsize=48, ha="center", va="center", weight="bold")
    fig.text(0.673, 0.5, "+", fontsize=48, ha="center", va="center", weight="bold")
    fig.suptitle(
        f"K-cache decomposition:  K  =  mu_K  +  (K - mu_K)   "
        f"({a.tag}, layer{a.layer} head{a.head}, first {a.ntok} tokens, D={D})",
        fontsize=14,
    )
    out = f"{a.outdir}/kcache_decomp_3d_{a.tag}.png"
    save_fig(fig, out, tight=False)


if __name__ == "__main__":
    run()
