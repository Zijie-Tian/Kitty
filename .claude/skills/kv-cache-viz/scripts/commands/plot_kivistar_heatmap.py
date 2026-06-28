#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heatmap of the kivi_star K x V bit sweep on LongBench (Llama-3.2-1B, 21 datasets).

Reads each combo's longbench_out/llama32-1b-instruct_kivi-star-k{K}v{V}/pred/result.json
(mean over the 21 datasets) for K, V in {1,2,3,4} and renders a 4x4 heatmap of
LongBench mean score, with the fp16 ceiling shown for reference and the best cell
outlined. Output: longbench_out/kivistar_kv_heatmap_llama32_1b.png
"""
import argparse
import json
import os

import numpy as np

from lib.common import save_fig, setup_matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "..", "..", "longbench_out")
KS = [1, 2, 3, 4]
VS = [1, 2, 3, 4]


def mean(d):
    with open(os.path.join(BASE, d, "pred", "result.json")) as fh:
        o = json.load(fh)
    vals = [x for x in o.values() if isinstance(x, (int, float))]
    return sum(vals) / len(vals)


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=BASE)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    M = np.full((len(KS), len(VS)), np.nan)
    for i, k in enumerate(KS):
        for j, v in enumerate(VS):
            d = f"llama32-1b-instruct_kivi-star-k{k}v{v}"
            try:
                M[i, j] = mean(d)
            except FileNotFoundError:
                print(f"[warn] missing {d}")

    fp16 = None
    try:
        fp16 = mean("llama32-1b-instruct_fp16")
    except FileNotFoundError:
        pass

    vmin, vmax = float(np.nanmin(M)), float(np.nanmax(M))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(len(KS)):
        for j in range(len(VS)):
            if np.isnan(M[i, j]):
                continue
            norm = (M[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm < 0.55 else "black",
                    fontsize=13, fontweight="bold")

    # outline the best cell
    bi, bj = np.unravel_index(np.nanargmax(M), M.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="#d61f26", lw=2.6, zorder=4))

    ax.set_xticks(range(len(VS)))
    ax.set_xticklabels([f"V{v}" for v in VS], fontsize=11)
    ax.set_yticks(range(len(KS)))
    ax.set_yticklabels([f"K{k}" for k in KS], fontsize=11)
    ax.set_xlabel("V-cache bits", fontsize=12)
    ax.set_ylabel("K-cache bits", fontsize=12)
    title = ("kivi_star K x V bit sweep - LongBench mean (21 datasets)\n"
             f"Llama-3.2-1B")
    if fp16 is not None:
        title += f" - fp16 ceiling = {fp16:.2f}"
    title += f" - best K{KS[bi]}V{VS[bj]} = {M[bi, bj]:.2f}"
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("LongBench mean score (21 datasets)", fontsize=11)

    out = os.path.join(a.outdir, "kivistar_kv_heatmap_llama32_1b.png")
    save_fig(fig, out, dpi=150)

    print(f"fp16 ceiling = {fp16}")
    print("matrix rows=K1..K4, cols=V1..V4:")
    print(np.array2string(M, formatter={"float_kind": lambda x: f"{x:6.2f}"}))


if __name__ == "__main__":
    run()
