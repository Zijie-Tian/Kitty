#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-cache decomposition + sign quant, as FOUR separate 3D figures (signed value,
Token x Channel x value), coolwarm colormap:
  1) K (original)
  2) mu_K (per-channel mean)
  3) K - mu_K (residual)
  4) K_hat after qlutattn-sign-pt quant = mu_K + sign(K-mu_K)*mag
All share one symmetric z-limit."""
import argparse
import sys
from types import SimpleNamespace

import numpy as np
import torch

from lib.common import (
    add_viz_args,
    default_layer_pt,
    save_fig,
    setup_matplotlib,
)

sys.path.insert(0, "src")
from kitty_sim.kitty_simulate import KittyKVCache


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

    # real qlutattn-sign-pt quant (per-channel submean + per-token 1-bit sign)
    mock = SimpleNamespace(
        k_codebook="qlut",
        bin_codebooks=["sign"],
        pertoken_offline=False,
        pertoken_mixed=False,
        pertoken_pc_submean=True,
        pertoken_rotate=False,
        k_pc_mean={},
    )
    Khat = KittyKVCache._quant_k_pertoken(mock, K[None, None].to(a.device), 0)[0, 0].float().cpu()  # [T, D]
    nmse = (((K - Khat) ** 2).sum() / (K ** 2).sum()).item()
    print(f"sign-pt dequant NMSE={nmse:.4f}")

    seg = slice(0, a.ntok)
    panels = [
        ("1_value", "K  (original value)", K[seg].numpy()),
        ("2_mu", "mu_K  (per-channel mean)", muK[None, :].repeat(a.ntok, 1).numpy()),
        ("3_residual", "K - mu_K  (residual)", (K[seg] - muK).numpy()),
        ("4_signquant", "K_hat  (after sign-pt quant: mu + sign(r)*mag)", Khat[seg].numpy()),
    ]
    zmax = float(max(abs(M).max() for _, _, M in panels))
    norm = Normalize(-zmax, zmax)
    print(f"symmetric zmax={zmax:.2f}")

    xpos, ypos = np.meshgrid(np.arange(a.ntok), np.arange(D), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)

    cmap = cm.coolwarm
    for tag, title, M in panels:
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        dz = M.ravel()
        ax.bar3d(xpos, ypos, zpos, 0.92, 0.92, dz, color=cmap(norm(dz)), shade=True)
        ax.set_zlim(-zmax, zmax)
        ax.set_xlabel("Token")
        ax.set_ylabel("Channel")
        ax.set_zlabel("Value")
        ax.set_title(
            f"{title}\n({a.tag}, layer{a.layer} head{a.head}, first {a.ntok} tok, D={D})",
            fontsize=11,
        )
        ax.view_init(elev=20, azim=-60)
        out = f"{a.outdir}/kdecomp_{tag}_{a.tag}.png"
        save_fig(fig, out, tight=False)


if __name__ == "__main__":
    run()
