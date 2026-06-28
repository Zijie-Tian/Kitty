#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KIVI-style 3D bars of per-SEGMENT per-channel mu^2 and sigma^2.

Split the cache along tokens into 128-token segments. For each (segment, channel)
compute in-segment mu^2 (DC energy) and sigma^2 (residual variance). TWO figures:
one for K-cache, one for V-cache; each figure has two 3D subplots (mu^2 | sigma^2),
axes = Segment x Channel x value. coolwarm."""
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
    ap.add_argument("--G", type=int, default=128, help="Tokens per segment")
    ap.add_argument("--nseg", type=int, default=64, help="Max segments to show")
    a = ap.parse_args(argv)

    pt_k = a.pt or default_layer_pt(a.outdir, a.layer, "K")
    pt_v = a.pt.replace("_K_", "_V_") if a.pt else default_layer_pt(a.outdir, a.layer, "V")
    K = torch.load(pt_k, weights_only=False).float()[a.head]
    V = torch.load(pt_v, weights_only=False).float()[a.head]
    T, D = K.shape
    nseg = min(a.nseg, T // a.G)

    def seg_stats(X):
        Xs = X[: nseg * a.G].reshape(nseg, a.G, D)  # [nseg, G, D]
        return (Xs.mean(1) ** 2).numpy(), Xs.var(1).numpy()  # mu^2, sigma^2  [nseg, D]

    mu2K, s2K = seg_stats(K)
    mu2V, s2V = seg_stats(V)
    print(f"segments={nseg} (x{a.G} tok)  D={D}")
    print(
        f"K  mu2 max={mu2K.max():.2f} median={np.median(mu2K):.3f} | "
        f"sigma2 max={s2K.max():.2f} median={np.median(s2K):.2f}"
    )
    print(
        f"V  mu2 max={mu2V.max():.3f} median={np.median(mu2V):.4f} | "
        f"sigma2 max={s2V.max():.3f} median={np.median(s2V):.3f}"
    )

    xpos, ypos = np.meshgrid(np.arange(nseg), np.arange(D), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)

    def render_cache(mu2, s2, cache, fname):
        fig = plt.figure(figsize=(15, 6.5))
        for ci, (M, zlab, sub) in enumerate(
            [(mu2, "mu^2", "mu^2  (DC energy)"), (s2, "sigma^2", "sigma^2  (variance)")]
        ):
            ax = fig.add_subplot(1, 2, ci + 1, projection="3d")
            dz = M.ravel()
            vmax = dz.max() if dz.max() > 0 else 1.0
            ax.bar3d(
                xpos,
                ypos,
                zpos,
                0.9,
                0.9,
                dz,
                color=cm.coolwarm(Normalize(0, vmax)(dz)),
                shade=True,
            )
            ax.set_zlim(0, vmax)
            ax.set_xlabel("Segment (128-tok)")
            ax.set_ylabel("Channel")
            ax.set_zlabel(zlab)
            ax.set_title(f"{cache}-cache  per-seg per-channel  {sub}", fontsize=11)
            ax.view_init(elev=20, azim=-60)
        fig.suptitle(
            f"{cache}-cache: per-segment per-channel mu^2 and sigma^2 "
            f"({a.tag}, layer{a.layer} head{a.head}, {nseg} segs x {a.G} tok, D={D})",
            fontsize=12,
        )
        save_fig(fig, fname, tight=False)

    render_cache(mu2K, s2K, "K", f"{a.outdir}/kvseg_K_mu2sigma2_3d_{a.tag}.png")
    render_cache(mu2V, s2V, "V", f"{a.outdir}/kvseg_V_mu2sigma2_3d_{a.tag}.png")


if __name__ == "__main__":
    run()
