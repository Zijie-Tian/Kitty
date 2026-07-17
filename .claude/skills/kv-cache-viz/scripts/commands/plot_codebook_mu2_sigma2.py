#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two figures (sign / nf2): how each K-cache codebook's levels sit on a
mu^2-dominant (low sigma^2, high dc_share) channel vs a sigma^2-dominant
(high sigma^2, low dc_share) channel, on REAL post-RoPE K. This illustrates
exactly the canonical qlutattn K design (offline mask: low sigma^2 -> sign,
high sigma^2 -> nf2).
Reconstructions are inlined from the current qlutattn primitives:
  sign = mu + sign(r) * mean|r|            (r = x - group mean, 1-bit)
  nf2  = mu + nf2_symmetric_lastdim(x-mu)  (fixed symnf2-v1 LUT, absmax scale)
Style follows probe_out/sign_scale/why_minmax_a_residual_levels_llama32-1b.png."""
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
from kitty_sim.qlut_quant import nf2_symmetric_lastdim


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_viz_args(ap)
    ap.add_argument("--sink", type=int, default=32)
    ap.add_argument("--recent", type=int, default=128)
    a = ap.parse_args(argv)

    pt = a.pt or default_layer_pt(a.outdir, a.layer, "K")
    K = torch.load(pt, weights_only=False).float()  # [nh, T, D]
    nh, T, D = K.shape
    Kr = K[:, a.sink : T - a.recent, :]  # [nh, Treg, D]
    Treg = Kr.shape[1]
    mu = Kr.mean(1)
    var = Kr.var(1)
    dc = (mu ** 2) / (mu ** 2 + var + 1e-12)  # [nh, D] dc_share
    print(f"K {tuple(K.shape)}  region={Treg} tokens", flush=True)

    def info(h, c):
        vv = Kr[h, :, c].contiguous().reshape(-1)  # one group = all region tokens
        mu_g = vv.mean()                           # group mean (plays the role of mu_d)
        r = vv - mu_g                              # centered residual, qlutattn quantizes this
        rs = mu_g + torch.sign(r) * r.abs().mean()  # sign: 1-bit, mean-|r| scale
        rn = mu_g + nf2_symmetric_lastdim(r)        # nf2: fixed symnf2-v1 LUT, absmax scale
        s2 = vv.var().item()
        return dict(
            h=h,
            c=c,
            v=vv.numpy(),
            mu=vv.mean().item(),
            sig=vv.std().item(),
            s2=s2,
            dc=dc[h, c].item(),
            sign_lev=torch.unique(rs).numpy(),
            nf2_lev=torch.unique(rn).numpy(),
            mse_sign=((vv - rs) ** 2).mean().item(),
            mse_nf2=((vv - rn) ** 2).mean().item(),
        )

    # mu^2-dominant = highest dc_share ; sigma^2-dominant = highest residual variance (-> low dc_share)
    HI = info(*divmod(dc.reshape(-1).argmax().item(), D))
    LO = info(*divmod(var.reshape(-1).argmax().item(), D))
    for nm, I in [("mu2-dom ", HI), ("sig2-dom", LO)]:
        print(
            f"{nm}: head{I['h']} ch{I['c']:2d}  dc={I['dc']:.3f} mu={I['mu']:+.3f} sig={I['sig']:.3f}  "
            f"NMSE sign={I['mse_sign']/I['s2']:.3f} nf2={I['mse_nf2']/I['s2']:.3f}  "
            f"absMSE sign={I['mse_sign']:.2e} nf2={I['mse_nf2']:.2e}  "
            f"nlev sign={len(I['sign_lev'])} nf2={len(I['nf2_lev'])}",
            flush=True,
        )

    def plot(cb, lev_key, color, mse_key, other_key, other_name, fname, sub):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
        for ax, I, tag in [
            (axs[0], HI, r"$\mu^2$-dominant channel  (low $\sigma^2$, high dc_share)"),
            (axs[1], LO, r"$\sigma^2$-dominant channel  (high $\sigma^2$, low dc_share)"),
        ]:
            ax.hist(I["v"], bins=120, density=True, color="#b0b0b0", alpha=0.9, label="post-RoPE K value")
            for j, L in enumerate(I[lev_key]):
                ax.axvline(
                    L,
                    color=color,
                    lw=2.2,
                    label=f"{cb} levels ({len(I[lev_key])})" if j == 0 else None,
                )
            ax.axvline(I["mu"], color="k", lw=0.8, ls=":", label=r"per-channel mean $\mu$")
            ax.set_title(
                f"{tag}\nhead {I['h']}, ch {I['c']}   dc_share={I['dc']:.2f}   "
                f"$\\sigma$={I['sig']:.3f}",
                fontsize=10,
            )
            ax.set_xlabel("post-RoPE K value")
            ax.set_ylabel("density")
            ax.legend(fontsize=8, loc="upper right")
            ax.text(
                0.02,
                0.97,
                f"{cb}:  NMSE={I[mse_key]/I['s2']:.3f}\nabs MSE={I[mse_key]:.2e}\n"
                f"({other_name} abs MSE={I[other_key]:.2e})",
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9),
            )
        fig.suptitle(sub, fontsize=12)
        save_fig(fig, fname, tight=False, bbox_inches="tight")

    plot(
        "sign",
        "sign_lev",
        "#2ca02c",
        "mse_sign",
        "mse_nf2",
        "nf2",
        f"{a.outdir}/why_codebook_sign_mu2_vs_sigma2_{a.tag}.png",
        r"sign (1-bit, 2 levels $\mu\pm E|r|$): error $\approx0.36\,\sigma^2$ either way "
        r"— negligible on $\mu^2$-dominant (tiny $\sigma^2$), costly on $\sigma^2$-dominant (large $\sigma^2$)",
    )
    plot(
        "nf2",
        "nf2_lev",
        "#ff7f0e",
        "mse_nf2",
        "mse_sign",
        "sign",
        f"{a.outdir}/why_codebook_nf2_mu2_vs_sigma2_{a.tag}.png",
        r"nf2 (2-bit, fixed symnf2-v1 LUT $\{-1,-c,+c,+1\}$): lower error than sign "
        r"— overkill on $\mu^2$-dominant, necessary on $\sigma^2$-dominant to cut the large absolute error",
    )

    def plot_mix(fname):
        """The sigma^2-adaptive MIXED codebook: each channel shows BOTH candidate
codebooks, the one the mix actually selects (low sigma^2 -> sign, high -> nf2)
bold/solid, the other faint/dashed; a formula band documents both codebooks."""
        GREEN, ORANGE = "#2ca02c", "#ff7f0e"
        fig = plt.figure(figsize=(15.5, 9.0))
        gs = fig.add_gridspec(2, 2, height_ratios=[2.7, 1.3], hspace=0.6, wspace=0.16)
        specs = [
            (HI, "sign", GREEN, "nf2", ORANGE, r"$\mu^2$-dominant  (low $\sigma^2$, high dc_share)"),
            (LO, "nf2", ORANGE, "sign", GREEN, r"$\sigma^2$-dominant  (high $\sigma^2$, low dc_share)"),
        ]
        for col, (I, chosen, cc, alt, ac, tag) in enumerate(specs):
            ax = fig.add_subplot(gs[0, col])
            ax.hist(I["v"], bins=120, density=True, color="#c8c8c8", alpha=0.95, label="post-RoPE K value")
            lev = {"sign": I["sign_lev"], "nf2": I["nf2_lev"]}
            mse = {"sign": I["mse_sign"], "nf2": I["mse_nf2"]}
            for j, L in enumerate(lev[alt]):  # alternative: faint dashed
                ax.axvline(
                    L,
                    color=ac,
                    lw=1.3,
                    ls="--",
                    alpha=0.6,
                    label=f"{alt} ({len(lev[alt])} lev) — not used" if j == 0 else None,
                )
            for j, L in enumerate(lev[chosen]):  # chosen: bold solid
                ax.axvline(
                    L,
                    color=cc,
                    lw=2.8,
                    label=f"{chosen} ({len(lev[chosen])} lev) — CHOSEN" if j == 0 else None,
                )
            ax.axvline(I["mu"], color="k", lw=0.8, ls=":", label=r"mean $\mu$")
            ax.set_title(
                f"{tag}\nhead {I['h']}, ch {I['c']}   dc_share={I['dc']:.2f}   "
                f"$\\sigma$={I['sig']:.3f}",
                fontsize=10.5,
            )
            ax.set_xlabel("post-RoPE K value")
            ax.set_ylabel("density")
            ax.legend(fontsize=8, loc="upper right")
            bits = "1-bit" if chosen == "sign" else "2-bit"
            if chosen == "sign":
                verdict = (
                    f"abs MSE  sign={mse['sign']:.2f}  nf2={mse['nf2']:.2f}\n"
                    f"both tiny (σ²={I['s2']:.1f}) → keep cheap 1-bit"
                )
            else:
                verdict = (
                    f"abs MSE  sign={mse['sign']:.2f}  nf2={mse['nf2']:.2f}\n"
                    f"sign {mse['sign']/mse['nf2']:.1f}× worse (σ²={I['s2']:.1f}) → 2-bit nf2"
                )
            ax.text(
                0.02,
                0.97,
                f"✓ MIX picks {chosen} ({bits})\n{verdict}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec=cc, lw=2, alpha=0.95),
            )
        axt = fig.add_subplot(gs[1, :])
        axt.axis("off")
        axt.set_xlim(0, 1)
        axt.set_ylim(0, 1)
        axt.axvline(0.5, 0.04, 0.84, color="#cccccc", lw=1)
        axt.text(
            0.5,
            0.99,
            r"codebook formulas  (per $G$-token group $x_1,\dots,x_G$ of one channel; side-info fp16/group)",
            ha="center",
            va="top",
            fontsize=10.5,
            style="italic",
            color="#444",
        )
        axt.text(0.02, 0.78, "sign  (1-bit, 2 levels):", fontsize=11.5, weight="bold", color=GREEN, va="top")
        axt.text(0.02, 0.55, r"$\mu=\frac{1}{G}\sum_i x_i,\qquad m=\frac{1}{G}\sum_i|x_i-\mu|$", fontsize=13, va="top")
        axt.text(0.02, 0.27, r"$\hat x_i=\mu+m\,\mathrm{sign}(x_i-\mu)\in\{\mu-m,\ \mu+m\}$", fontsize=13, va="top")
        axt.text(0.02, 0.04, r"$\Rightarrow\ 1+32/G=1.25$ bit @ $G{=}128$", fontsize=9.5, color="#555", va="top")
        axt.text(0.53, 0.78, "nf2  (2-bit, fixed symnf2-v1 LUT):", fontsize=11.5, weight="bold", color=ORANGE, va="top")
        axt.text(
            0.53,
            0.57,
            r"fixed levels $s\cdot\{-1,-c,+c,+1\}$,  $c=0.2526$,  $s=\max_i|x_i-\mu|$ (absmax),",
            fontsize=11.5,
            va="top",
        )
        axt.text(
            0.53,
            0.37,
            r"$\hat x_i=\mu+s\,\mathrm{sign}(x_i{-}\mu)\cdot(1$ if $|x_i{-}\mu|>\frac{1+c}{2}s$ else $c)$",
            fontsize=11.5,
            va="top",
        )
        axt.text(
            0.53,
            0.10,
            r"LUT is fixed (no fitting); only $s$ is data-dependent;   $\approx 2+32/G=2.25$ bit",
            fontsize=9.5,
            color="#555",
            va="top",
        )
        fig.suptitle(
            r"$\sigma^2$-adaptive MIXED codebook:  low-$\sigma^2$ ($\mu^2$-dominant) channels $\to$ sign (1-bit),   "
            r"high-$\sigma^2$ ($\sigma^2$-dominant) channels $\to$ nf2 (2-bit)",
            fontsize=12.5,
        )
        save_fig(fig, fname, tight=False, bbox_inches="tight")

    plot_mix(f"{a.outdir}/why_codebook_mix_mu2_vs_sigma2_{a.tag}.png")
    print("[done]", flush=True)


if __name__ == "__main__":
    run()
