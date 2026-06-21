#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heatmap of the kitty K-base x V bit sweep on LongBench (Llama-3.2-1B, 21 datasets).

Sweep config: boost (promote_bit) = 4, promote_ratio = 0.25 for K base in {1,2,3};
K base = 4 uses promote_ratio = 0 (base == boost, i.e. uniform 4-bit K). Reads each
combo's longbench_out/llama32-1b-instruct_kitty-k{K}b4v{V}-pr{...}/pred/result.json
(mean over 21 datasets) and renders a 4x4 heatmap with the fp16 ceiling and the
best cell outlined. Output: longbench_out/kitty_kv_heatmap_llama32_1b.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "longbench_out")
KS = [1, 2, 3, 4]
VS = [1, 2, 3, 4]


def slug(k, v):
    pr = "0p0" if k == 4 else "0p25"
    return f"llama32-1b-instruct_kitty-k{k}b4v{v}-pr{pr}"


def mean(d):
    with open(os.path.join(BASE, d, "pred", "result.json")) as fh:
        o = json.load(fh)
    vals = [x for x in o.values() if isinstance(x, (int, float))]
    return sum(vals) / len(vals)


def main():
    M = np.full((len(KS), len(VS)), np.nan)
    for i, k in enumerate(KS):
        for j, v in enumerate(VS):
            try:
                M[i, j] = mean(slug(k, v))
            except FileNotFoundError:
                print(f"[warn] missing {slug(k, v)}")

    fp16 = None
    try:
        fp16 = mean("llama32-1b-instruct_fp16")
    except FileNotFoundError:
        pass

    vmin, vmax = float(np.nanmin(M)), float(np.nanmax(M))
    fig, ax = plt.subplots(figsize=(7.6, 6.3))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(len(KS)):
        for j in range(len(VS)):
            if np.isnan(M[i, j]):
                continue
            norm = (M[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm < 0.55 else "black",
                    fontsize=13, fontweight="bold")

    bi, bj = np.unravel_index(np.nanargmax(M), M.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="#d61f26", lw=2.6, zorder=4))

    ax.set_xticks(range(len(VS)))
    ax.set_xticklabels([f"V{v}" for v in VS], fontsize=11)
    ax.set_yticks(range(len(KS)))
    ax.set_yticklabels([f"K{k}" for k in KS], fontsize=11)
    ax.set_xlabel("V-cache bits", fontsize=12)
    ax.set_ylabel("K base bits", fontsize=12)
    title = ("kitty K-base x V sweep (boost=4-bit, pr=0.25; K4 = uniform 4-bit)\n"
             "LongBench mean (21 datasets) - Llama-3.2-1B")
    if fp16 is not None:
        title += f" - fp16 ceiling = {fp16:.2f}"
    title += f" - best K{KS[bi]}V{VS[bj]} = {M[bi, bj]:.2f}"
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("LongBench mean score (21 datasets)", fontsize=11)
    fig.tight_layout()

    out = os.path.join(BASE, "kitty_kv_heatmap_llama32_1b.png")
    fig.savefig(out, dpi=150)
    print(f"fp16 ceiling = {fp16}")
    print("matrix rows=K1..K4, cols=V1..V4:")
    print(np.array2string(M, formatter={"float_kind": lambda x: f"{x:6.2f}"}))
    print("saved", out)


if __name__ == "__main__":
    main()
