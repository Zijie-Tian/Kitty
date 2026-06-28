#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schematic: per-channel sign/nf2 MIXED K-cache codebook + OFFLINE weight reorder.

Each post-RoPE K head_dim channel is assigned a codebook by its residual sigma^2
(low sigma^2 -> sign 1.25 bit, high sigma^2 -> nf2 2.25 bit). An offline reorder of
the W_q / W_k OUTPUT channels by sigma^2 (RoPE-pair-preserving, folded into the
weights, math-identity, bit-neutral) turns the scattered sign/nf2 channels into two
CONTIGUOUS blocks. Mixed bit = (n_sign*1.25 + n_nf2*2.25)/D = 1.88 bit/value.
Real per-channel sigma^2 from layer8 K."""
import argparse
import sys

import numpy as np
import torch

from lib.common import (
    add_viz_args,
    default_layer_pt,
    save_fig,
    setup_matplotlib,
)

sys.path.insert(0, "src")
from kitty_sim.qlut_quant import channel_sigma2


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ap = argparse.ArgumentParser()
    add_viz_args(ap)
    ap.add_argument("--sink", type=int, default=32)
    ap.add_argument("--recent", type=int, default=128)
    ap.add_argument("--G", type=int, default=128, help="Group size for sigma^2")
    ap.add_argument("--n-sign", type=int, default=24, help="Number of sign channels")
    a = ap.parse_args(argv)

    pt = a.pt or default_layer_pt(a.outdir, a.layer, "K")
    K = torch.load(pt, weights_only=False).float()  # [nh, T, D]
    nh, T, D = K.shape
    x = K[:, a.sink : T - a.recent, :].permute(0, 2, 1).contiguous()  # [nh, D, Treg]
    sig2 = channel_sigma2(x, a.G).mean(0)  # [D] per-channel residual sigma^2 (mean over heads)
    sig = sig2.sqrt().numpy()
    order = np.argsort(sig)  # ascending sigma -> the reorder permutation
    n_nf2 = D - a.n_sign
    B_SIGN, B_NF2 = 1.25, 2.25
    mixed = (a.n_sign * B_SIGN + n_nf2 * B_NF2) / D
    print(f"D={D} nh={nh}  n_sign={a.n_sign} n_nf2={n_nf2}  mixed={mixed:.4f} bit/value", flush=True)

    rank = np.empty(D, dtype=int)
    rank[order] = np.arange(D)
    col = np.where(rank < a.n_sign, "#2ca02c", "#ff7f0e")
    ymax = float(sig.max())

    fig = plt.figure(figsize=(14.5, 9.4))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1, 1], hspace=0.7, left=0.075, right=0.97, top=0.9, bottom=0.07
    )

    # ---------------- BEFORE ----------------
    ax0 = fig.add_subplot(gs[0])
    ax0.bar(np.arange(D), sig, color=col, width=0.92)
    ax0.set_title(
        r"BEFORE reorder — each channel's codebook is fixed by its $\sigma^2$;  "
        r"sign & nf2 channels are INTERLEAVED along the raw head_dim",
        fontsize=11,
    )
    ax0.set_xlabel("head_dim channel index (original)")
    ax0.set_ylabel(r"per-channel $\sigma$ (K residual)")
    ax0.set_xlim(-0.7, D - 0.3)
    ax0.set_ylim(0, ymax * 1.18)
    leg = [
        Patch(fc="#2ca02c", label="sign — 2 levels (1-bit codeword)  ->  1.25 bit/value"),
        Patch(fc="#ff7f0e", label="nf2 — 4-level Lloyd (2-bit codeword)  ->  2.25 bit/value"),
    ]
    ax0.legend(handles=leg, fontsize=9, loc="upper center", ncol=2, framealpha=0.95)

    # ---------------- AFTER ----------------
    ax1 = fig.add_subplot(gs[1])
    sig_sorted = sig[order]
    col_sorted = np.where(np.arange(D) < a.n_sign, "#2ca02c", "#ff7f0e")
    ax1.bar(np.arange(D), sig_sorted, color=col_sorted, width=0.92)
    ax1.axvspan(-0.7, a.n_sign - 0.5, color="#2ca02c", alpha=0.10)
    ax1.axvspan(a.n_sign - 0.5, D - 0.3, color="#ff7f0e", alpha=0.10)
    ax1.axvline(a.n_sign - 0.5, color="k", lw=1.0, ls="--")
    ax1.set_title(
        r"AFTER offline weight reorder — channels sorted by $\sigma^2$  ->  two CONTIGUOUS codebook blocks",
        fontsize=11,
    )
    ax1.set_xlabel(r"head_dim channel index (reordered by $\sigma^2$)")
    ax1.set_ylabel(r"per-channel $\sigma$ (K residual)")
    ax1.set_xlim(-0.7, D - 0.3)
    ax1.set_ylim(0, ymax * 1.18)
    ax1.text(
        a.n_sign / 2 - 0.5,
        ymax * 1.10,
        f"sign block\n{a.n_sign} ch × 1.25 bit",
        ha="center",
        va="top",
        fontsize=11,
        color="#2ca02c",
        weight="bold",
    )
    ax1.text(
        a.n_sign + n_nf2 / 2 - 0.5,
        ymax * 1.10,
        f"nf2 block\n{n_nf2} ch × 2.25 bit",
        ha="center",
        va="top",
        fontsize=11,
        color="#ff7f0e",
        weight="bold",
    )

    # ---------------- reorder arrow + caption between panels ----------------
    fig.text(
        0.5,
        0.515,
        r"offline weight reorder:  permute  $W_q,\ W_k$  output channels by $\sigma^2$"
        "\n(RoPE-pair-preserving · folded into the weights · math-identity · bit-neutral)",
        ha="center",
        va="center",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="#eaf3ff", ec="#1f77b4", lw=1.4),
    )
    ax1.annotate(
        "",
        xy=(0.5, 0.45),
        xytext=(0.5, 0.485),
        xycoords="figure fraction",
        arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#1f77b4"),
    )

    fig.suptitle(
        r"$\sigma^2$-mixed K-cache codebook + offline channel reorder      "
        rf"mixed K = ({a.n_sign}×1.25 + {n_nf2}×2.25)/{D} = {mixed:.3f} ≈ 1.88 bit/value",
        fontsize=13,
    )
    out = f"{a.outdir}/reorder_mixed_codebook_{a.tag}.png"
    save_fig(fig, out, dpi=130, tight=False, bbox_inches="tight")
    print("[done]", flush=True)


if __name__ == "__main__":
    run()
