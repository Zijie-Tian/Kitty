#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why 1-bit min-max (KIVI/Kitty) collapses but sign survives, on REAL K residuals
(per-channel + per-128-token-group submean).

Two SEPARATE figures:
  (a) residual histogram + the two 1-bit codebooks' levels (sign=centroid in the
      bulk, min-max=extreme in the empty tail);
  (b) distortion-vs-level parabola WITH a full derivation of D(L).
"""
import argparse
import os

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
    ap.add_argument("--group", type=int, default=128)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    layer = a.layer
    pt = a.pt if a.pt else default_layer_pt(a.outdir, layer, "K")
    K = torch.load(pt, weights_only=False).float()  # [nh, T, D]
    nh, T, D = K.shape
    G = a.group
    nb = T // G
    Kg = K[:, :nb * G, :].reshape(nh, nb, G, D)
    r = Kg - Kg.mean(2, keepdim=True)
    sig = r.std(2, keepdim=True).clamp(min=1e-6)
    rn = r / sig
    meanabs = rn.abs().mean(2, keepdim=True)
    maxabs = rn.abs().amax(2, keepdim=True)
    D_sign = ((rn - torch.sign(rn) * meanabs) ** 2).mean().item()
    D_mm = ((rn - torch.sign(rn) * maxabs) ** 2).mean().item()
    rho = rn.abs().mean().item()
    L_sign = meanabs.mean().item()
    L_mm = maxabs.mean().item()
    print(
        f"rho={rho:.3f} L_sign={L_sign:.2f} L_mm={L_mm:.2f} D_sign={D_sign:.3f} D_mm={D_mm:.3f} ratio={D_mm/D_sign:.1f}x"
    )

    rng = np.random.default_rng(0)
    flat = rn.flatten().numpy()
    rn_s = flat[rng.choice(flat.size, size=min(300000, flat.size), replace=False)]

    tag = a.tag
    out_a = os.path.join(a.outdir, f"why_minmax_a_residual_levels_{tag}.png")
    out_b = os.path.join(a.outdir, f"why_minmax_b_distortion_derivation_{tag}.png")

    # ---------- (a) residual + levels ----------
    figA, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.hist(rn_s, bins=130, density=True, color="#b0b0b0", alpha=0.9, label="K residual  $r/\\sigma$")
    for s in (+1, -1):
        ax.axvline(s * L_sign, color="#2ca02c", lw=2.4, label="sign level  $\\pm E|r|$" if s == 1 else None)
        ax.axvline(
            s * L_mm, color="#d62728", lw=2.4, ls="--", label="min-max level  $\\pm\\max|r|$" if s == 1 else None
        )
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.annotate(
        "sign: in the data bulk",
        xy=(L_sign, 0.30),
        xytext=(L_sign + 0.2, 0.34),
        color="#2ca02c",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#2ca02c"),
    )
    ax.annotate(
        "min-max: in the empty tail",
        xy=(L_mm, 0.02),
        xytext=(L_mm - 1.7, 0.12),
        color="#d62728",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#d62728"),
    )
    ax.set_xlim(-4, 4)
    ax.set_title(
        "(a) K residual distribution + the two 1-bit levels\n"
        "sign = centroid (data-dense)   vs   min-max = extreme (data-sparse)",
        fontsize=11,
    )
    ax.set_xlabel("$r/\\sigma$")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    save_fig(figA, out_a, dpi=130)

    # ---------- (b) parabola + derivation ----------
    figB = plt.figure(figsize=(15.5, 6.6))
    axp = figB.add_axes([0.06, 0.12, 0.45, 0.78])
    L = np.linspace(0, 3.0, 200)
    Dl = 1 - 2 * rho * L + L ** 2
    axp.plot(L, Dl, color="#1f77b4", lw=2.2, label="$D(L)=D^*+(L-E|r|)^2$")
    axp.axhline(1.0, color="gray", ls=":", lw=1.4, label="0-bit (drop $r$):  $D=\\sigma^2$")
    axp.scatter([L_sign], [D_sign], color="#2ca02c", s=110, zorder=5, label=f"sign  $L=E|r|$,  D={D_sign:.2f}")
    axp.scatter([L_mm], [D_mm], color="#d62728", s=110, zorder=5, label=f"min-max  $L=\\max|r|$,  D={D_mm:.2f}")
    axp.annotate(
        f"{D_mm/D_sign:.0f}x worse\n(negative optimization,\n D > $\\sigma^2$)",
        xy=(L_mm, D_mm),
        xytext=(1.15, D_mm * 0.78),
        fontsize=9.5,
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728"),
    )
    axp.scatter([rho], [1 - rho ** 2], facecolors="none", edgecolors="#2ca02c", s=180, lw=1.5, zorder=4)
    axp.set_title(f"(b) distortion vs level radius $L$   (real K, {tag} layer{layer})", fontsize=11)
    axp.set_xlabel("level radius  $L/\\sigma$")
    axp.set_ylabel("distortion  $D/\\sigma^2$")
    axp.legend(fontsize=9, loc="upper center")

    axt = figB.add_axes([0.55, 0.02, 0.43, 0.96])
    axt.axis("off")
    lines = [
        ("1-bit affine quant of the zero-mean residual $r$ (after submean):", 11, "bold"),
        (r"$\hat r = L\,\mathrm{sign}(r),\qquad L>0$  (1 bit/value; $L$=per-group side-info)", 12, None),
        ("", 6, None),
        (r"$D(L)=E[(r-L\,\mathrm{sign}\,r)^2]$", 13, None),
        (r"$\quad=E[r^2]-2L\,E[r\,\mathrm{sign}\,r]+L^2\,E[\mathrm{sign}^2 r]$", 12, None),
        (r"use  $r\,\mathrm{sign}\,r=|r|$  and  $\mathrm{sign}^2 r=1$ :", 10, "italic"),
        (r"$\quad=\sigma^2-2L\,E|r|+L^2$", 13, None),
        ("", 5, None),
        ("complete the square:", 10, "italic"),
        (r"$D(L)=[\sigma^2-(E|r|)^2]+(L-E|r|)^2 \equiv D^*+(L-E|r|)^2$", 12, None),
        ("", 5, None),
        (r"minimize:  $\frac{dD}{dL}=-2E|r|+2L=0 \;\Rightarrow\; L^*=E|r|$", 12, None),
        (r"(centroid level, distribution-free)   $D^*=\sigma^2(1-\rho^2),\ \rho=E|r|/\sigma$", 10, "italic"),
        ("", 7, None),
        (r"sign:     $L=E|r|\;\Rightarrow\;D=D^*$   (Lloyd-Max optimum)", 11, None),
        (r"min-max:  $L=\max|r|\approx2.5\sigma\;\Rightarrow\;D=D^*+(\max|r|-E|r|)^2$", 11, None),
        (r"0-bit:    $L=0\;\Rightarrow\;D(0)=\sigma^2$", 11, None),
        (r"$\Rightarrow$ min-max $D>\sigma^2$:  worse than dropping $r$ (negative opt.)", 10, "bold"),
        ("", 7, None),
        (rf"measured (K layer{layer}): $\rho={rho:.2f}$,  $D^*={D_sign:.2f}\sigma^2$,", 10, None),
        (rf"min-max $L={L_mm:.2f}\sigma,\ D={D_mm:.2f}\sigma^2\ ({D_mm/D_sign:.0f}\times)$", 10, None),
    ]
    y = 0.99
    for txt, fs, style in lines:
        if txt:
            kw = {}
            if style == "bold":
                kw["weight"] = "bold"
            if style == "italic":
                kw["style"] = "italic"
                kw["color"] = "#555555"
            axt.text(0.0, y, txt, fontsize=fs, va="top", ha="left", **kw)
            y -= 0.053
        else:
            y -= 0.022
    figB.savefig(out_b, dpi=110)
    plt.close(figB)
    print("[fig]", out_b)


if __name__ == "__main__":
    run()
