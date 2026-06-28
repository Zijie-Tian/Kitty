#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-channel μ²/σ² distributions for the K-cache and V-cache.

Replaces scripts/dump_kv_mu2_sigma2_dist.py.
"""
import argparse
import json
import os

import numpy as np
import torch

from lib.common import (
    add_prefill_args,
    load_model,
    prefill_and_extract,
    save_fig,
    setup_matplotlib,
)


def summ(x):
    return {
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def panel(ax, data, title, color):
    pos = data[data > 0]
    lo = max(pos.min(), 1e-7) if pos.size else 1e-7
    hi = data.max() + 1e-9
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    ax.hist(np.clip(data, lo, None), bins=bins, color=color, alpha=0.85, density=True)
    ax.set_xscale("log")
    ax.axvline(
        np.median(data),
        color="k",
        ls="--",
        lw=0.9,
        label=f"median={np.median(data):.3g}",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("value (log scale)")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
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
    q0, q1 = data["region"]
    print(
        f"[prefill] {a.tag}: T={T} layers={nl} D={D} region=[{q0},{q1})",
        flush=True,
    )

    mu2K, s2K, mu2V, s2V = [], [], [], []
    for li in range(nl):
        K = data["K"][li]
        V = data["V"][li]
        mu2K.append((K.mean(1) ** 2).flatten())
        s2K.append(K.var(1).flatten())
        mu2V.append((V.mean(1) ** 2).flatten())
        s2V.append(V.var(1).flatten())
    mu2K = torch.cat(mu2K).numpy()
    s2K = torch.cat(s2K).numpy()
    mu2V = torch.cat(mu2V).numpy()
    s2V = torch.cat(s2V).numpy()

    torch.save(
        {
            "mu2_K": mu2K,
            "sigma2_K": s2K,
            "mu2_V": mu2V,
            "sigma2_V": s2V,
            "model": a.tag,
            "layers": nl,
            "D": D,
        },
        f"{a.outdir}/kv_mu2_sigma2_{a.tag}.pt",
    )

    stats = {
        "model": a.tag,
        "n_channels": int(mu2K.size),
        "K_mu2": summ(mu2K),
        "K_sigma2": summ(s2K),
        "V_mu2": summ(mu2V),
        "V_sigma2": summ(s2V),
    }
    json.dump(
        stats,
        open(f"{a.outdir}/kv_mu2_sigma2_{a.tag}.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    panel(
        axs[0, 0],
        mu2K,
        f"K-cache  per-channel μ²  (DC energy, n={mu2K.size})",
        "#1f77b4",
    )
    panel(
        axs[0, 1],
        s2K,
        f"K-cache  per-channel σ²  (AC energy, n={s2K.size})",
        "#1f77b4",
    )
    panel(
        axs[1, 0],
        mu2V,
        f"V-cache  per-channel μ²  (DC energy, n={mu2V.size})",
        "#2ca02c",
    )
    panel(
        axs[1, 1],
        s2V,
        f"V-cache  per-channel σ²  (AC energy, n={s2V.size})",
        "#2ca02c",
    )
    fig.suptitle(
        f"Per-channel μ² and σ² distributions — K-cache vs V-cache "
        f"({a.tag}, all {nl} layers, D={D}, T={T})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"{a.outdir}/kv_mu2_sigma2_{a.tag}.png", tight=False)


if __name__ == "__main__":
    run()
