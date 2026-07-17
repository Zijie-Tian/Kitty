#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dequantize K with the canonical qlutattn K path (per-channel mean mu_d
removed, residual quantized per token with the offline sign/nf2 codebook mask)
and plot per-channel value distributions of the SIGN channels (original vs
dequant) to verify the quantization actually fired.

Uses the REAL kitty_simulate.KittyKVCache._quant_k_pertoken path via a
duck-typed self (k_codebook='qlut', bin_codebooks=['sign','nf2'],
pertoken_offline=True). Since this probe must stay self-contained, the offline
codebook mask is rebuilt on the fly from the dumped K tensor itself, exactly
like scripts/calibrate_qlutattn_mask.py: rank channels per head by residual
sigma^2 (G-token submean groups, sink skipped), the 50% lowest-sigma^2
channels -> sign (mask 0), the 50% highest -> nf2 (mask 1).

On a sign channel the per-token reconstruction is mu_d + sign(r) * mag_token
(mag = masked mean |r| over the head's sign channels), so a correctly-fired
quant turns the channel into TWO clusters at mu +/- mag (vs the original
continuous distribution). Overall NMSE clearly > 0 = fired; ~0 = identity.
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
    ap.add_argument("--group-size", type=int, default=128,
                    help="sigma^2 submean group along tokens (mask calibration)")
    ap.add_argument("--skip-first", type=int, default=32,
                    help="skip the sink window when measuring sigma^2")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    pt = a.pt or default_layer_pt(a.outdir, a.layer, "K")
    K = torch.load(pt, weights_only=False).float()  # [nh, T, D]
    nh, T, D = K.shape
    print(f"loaded K {tuple(K.shape)}  (layer{a.layer})")

    sys.path.insert(0, "src")
    from kitty_sim.kitty_simulate import KittyKVCache
    from kitty_sim.qlut_quant import channel_sigma2
    from types import SimpleNamespace

    # --- on-the-fly offline-style codebook mask (0=sign, 1=nf2), 50/50 ---
    skip = min(a.skip_first, max(T - a.group_size, 0))
    G = min(a.group_size, T - skip)
    sig2 = channel_sigma2(K[:, skip:, :].permute(0, 2, 1), G)  # [nh, D]
    n_sign = D // 2
    order = torch.argsort(sig2, dim=1)  # ascending sigma^2
    mask = torch.ones(nh, D, dtype=torch.long)  # 1 = nf2 (high sigma^2)
    mask.scatter_(1, order[:, :n_sign], 0)      # 0 = sign (low sigma^2)

    # --- real qlutattn quant path (new attribute surface) ---
    mock = SimpleNamespace(
        k_codebook="qlut",
        bin_codebooks=["sign", "nf2"],
        pertoken_offline=True,
        pertoken_cb_mask_path=None,   # mask injected directly below
        k_cb_mask={0: mask},
        k_pc_mean={},
        k_pt_quant_end={},
        group_size=128,
        kbits=2,
        # staticmethod the real path dispatches through via self
        _pt_codebook_masked=KittyKVCache._pt_codebook_masked,
    )
    ks = K[None].to(a.device)  # [1, nh, T, D]
    Khat = KittyKVCache._quant_k_pertoken(mock, ks, 0)[0].float().cpu()  # [nh, T, D]

    nmse = (((K - Khat) ** 2).sum() / (K ** 2).sum()).item()
    print(
        f"overall dequant NMSE = {nmse:.4f}  "
        f"({'FIRED' if nmse > 0.01 else 'NOT fired / identity!'})"
    )

    head = a.head
    Ko, Kh = K[head], Khat[head]  # [T, D]
    print(
        f"dequant value range: orig [{Ko.min():.2f},{Ko.max():.2f}]  "
        f"dequant [{Kh.min():.2f},{Kh.max():.2f}]"
    )

    # sample 8 SIGN channels of this head, spread by RoPE period (fast -> slow)
    j = np.arange(D) % (D // 2)
    period = 2 * np.pi * a.base ** (2.0 * j / D)
    sign_chans = np.array(
        [c for c in np.argsort(period) if mask[head, c].item() == 0], dtype=int
    )
    chans = [int(sign_chans[i]) for i in
             np.linspace(0, len(sign_chans) - 1, 8).round().astype(int)]

    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axs.flat, chans):
        o = Ko[:, c].numpy()
        h = Kh[:, c].numpy()
        lo = min(o.min(), h.min())
        hi = max(o.max(), h.max())
        bins = np.linspace(lo, hi, 70)
        ax.hist(o, bins=bins, color="#1f77b4", alpha=0.55, density=True, label="original")
        ax.hist(
            h, bins=bins, color="#d62728", alpha=0.60, density=True,
            label="qlutattn dequant",
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
        f"qlutattn (sign channels): per-channel value distribution, original vs dequant "
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
