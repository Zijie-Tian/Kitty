#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-layer per-channel K/V distribution study. Prefill one 32k LongBench doc,
grab post-RoPE K and raw V for ALL layers, then plot:
  (1) per-layer heterogeneity summary (K vs V) over all layers
  (2) per-channel histograms at sampled layers (0/8/15), K and V separately.
Shows whether 'K heterogeneous / V homogeneous (Gaussian)' holds across depth.
"""
import argparse
import os

import numpy as np
import torch

from lib.common import (
    add_prefill_args,
    load_model,
    prefill_and_extract,
    rope_period,
    save_fig,
    setup_matplotlib,
)


def stats(X):
    """X: [Tq, D] -> per-channel arrays."""
    amax = X.abs().amax(0).numpy()
    mu = X.mean(0).numpy()
    sd = X.std(0).numpy()
    kurt = (((X - X.mean(0)) ** 4).mean(0) / (X.var(0) ** 2 + 1e-9)).numpy()
    return amax, mu, sd, kurt


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
    ap.add_argument("--head", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    model, tok = load_model(a.model, a.device)
    data = prefill_and_extract(
        model,
        tok,
        os.path.expanduser(a.longbench_dir),
        seq_len=a.seq_len,
        sink=a.sink,
        recent=a.recent,
        chunk=a.chunk,
        device=a.device,
    )
    T = data["T"]
    nl = data["layers"]
    D = data["D"]
    H = data["H"]
    q0, q1 = data["region"]
    head = a.head
    print(f"[prefill] {T} tok, {nl} layers, D={D}, H={H}", flush=True)

    def get_kv(li):
        K = data["K"][li][head]  # [Tq, D]
        V = data["V"][li][head]  # [Tq, D]
        return K, V

    KV = {li: get_kv(li) for li in range(nl)}

    # ---- (1) cross-layer summary ----
    het_K, het_V, dc_K, dc_V, ku_K, ku_V, sd_K, sd_V = [], [], [], [], [], [], [], []
    for li in range(nl):
        aK, mK, sK, kK = stats(KV[li][0])
        aV, mV, sV, kV = stats(KV[li][1])
        het_K.append(aK.max() / np.median(aK))
        het_V.append(aV.max() / np.median(aV))
        dc_K.append(np.mean(np.abs(mK) / (sK + 1e-9)))
        dc_V.append(np.mean(np.abs(mV) / (sV + 1e-9)))
        ku_K.append(np.median(kK))
        ku_V.append(np.median(kV))
        sd_K.append(sK.mean())
        sd_V.append(sV.mean())

    x = np.arange(nl)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    def line(ax, yk, yv, title, yl):
        ax.plot(x, yk, "o-", label="K-cache", color="#1f77b4")
        ax.plot(x, yv, "s-", label="V-cache", color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.set_ylabel(yl)
        ax.legend()
        ax.grid(alpha=0.3)

    line(axs[0, 0], het_K, het_V, "per-channel amax max/median (channel heterogeneity)", "ratio")
    line(axs[0, 1], dc_K, dc_V, "mean |per-channel DC| / std (DC offset)", "|mean|/std")
    line(axs[1, 0], ku_K, ku_V, "median per-channel kurtosis (3 = Gaussian)", "kurtosis")
    axs[1, 0].axhline(3, color="gray", ls=":", lw=1)
    line(axs[1, 1], sd_K, sd_V, "mean per-channel std (magnitude)", "std")
    fig.suptitle(f"K vs V per-channel statistics across all {nl} layers ({a.tag}, head {head})", fontsize=12)
    save_fig(fig, os.path.join(a.outdir, "kv_channel_dist_layers_summary.png"), tight=False)
    print("[fig] summary")

    # ---- (2) per-channel histograms at sampled layers ----
    SL = [0, nl // 2, nl - 1]
    period = rope_period(D)
    kchs = [
        int(c)
        for c in [
            np.argsort(period)[0],
            13 if D > 13 else 1,
            np.argsort(period)[-8],
            np.argsort(period)[-1],
        ]
    ]

    def grid(which, chans, fname):
        fig, axs = plt.subplots(len(SL), len(chans), figsize=(4 * len(chans), 3 * len(SL)))
        for r, li in enumerate(SL):
            X = KV[li][0 if which == "K" else 1]
            for cc, ch in enumerate(chans):
                ax = axs[r, cc]
                v = X[:, ch].numpy()
                ax.hist(v, bins=60, density=True, color="#1f77b4" if which == "K" else "#2ca02c", alpha=0.8)
                ax.axvline(v.mean(), color="k", ls="--", lw=0.7)
                extra = f" T={period[ch]:.0f}" if which == "K" else ""
                ax.set_title(f"L{li} {which} ch{ch}{extra} amax={np.abs(v).max():.1f}", fontsize=8)
                ax.tick_params(labelsize=6)
        fig.suptitle(f"{which}-cache per-channel distribution across layers {SL} (head {head})", fontsize=12)
        fig.supxlabel("value")
        fig.supylabel("density")
        save_fig(fig, os.path.join(a.outdir, fname), tight=False)
        print("[fig]", fname)

    grid("K", kchs, "kv_channel_dist_layers_K.png")

    # V channels: top-amax at mid layer (fixed indices across layers)
    aVmid = KV[nl // 2][1].abs().amax(0).numpy()
    vchs = [int(c) for c in np.argsort(-aVmid)[[0, 2, 5, 20]]]
    grid("V", vchs, "kv_channel_dist_layers_V.png")

    print(f"\nheterogeneity(amax max/med) K: L0={het_K[0]:.1f} Lmid={het_K[nl//2]:.1f} L{nl-1}={het_K[-1]:.1f}")
    print(f"                            V: L0={het_V[0]:.1f} Lmid={het_V[nl//2]:.1f} L{nl-1}={het_V[-1]:.1f}")
    print(f"median kurtosis K: {np.median(ku_K):.1f}  V: {np.median(ku_V):.1f}")


if __name__ == "__main__":
    run()
