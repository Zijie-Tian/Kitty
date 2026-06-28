#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped bar chart: Attention-operator latency vs KV-cache length, by KV format.
Measured data provided by the user. x = context length, one bar per KV format."""
import argparse
import os

import numpy as np

from lib.common import save_fig, setup_matplotlib

LENGTHS = ["4k", "16k", "32k", "64k", "128k"]
DATA = {                       # measured latency (lower = faster)
    "F16":         [275, 1328, 2638, 5571, 10633],
    "Q4_0":        [459, 1793, 3581, 7079, 14246],
    "Q8_0":        [435, 1635, 3245, 6716, 13341],
    "K2V2":  [212,  883, 1805, 3624,  7538],
    "K1V2":  [204,  814, 1666, 3324,  6624],
}
COLORS = {"F16": "#5b6770", "Q4_0": "#d62728", "Q8_0": "#ff7f0e",
          "K2V2": "#2ca02c", "K1V2": "#156915"}

methods = list(DATA)
ng, nm = len(LENGTHS), len(methods)
x = np.arange(ng)
bw = 0.8 / nm


def draw(ax, logy):
    for i, m in enumerate(methods):
        xs = x + (i - (nm - 1) / 2) * bw
        bars = ax.bar(xs, DATA[m], width=bw, color=COLORS[m], label=m,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, DATA[m]):
            ax.text(b.get_x() + b.get_width() / 2, v * (1.02 if logy else 1.0) + (0 if logy else max(map(max, DATA.values())) * 0.005),
                    f"{v}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("KV-cache context length")
    ax.set_ylabel("Attention-op latency  (measured · lower = faster)")
    if logy:
        ax.set_yscale("log"); ax.set_ylim(150, 22000)
        ax.set_title("log scale — every length readable", fontsize=10)
    else:
        ax.set_ylim(0, 15800)
        ax.set_title("linear scale — shows growth with length", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="probe_out/attn_bench")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    draw(axs[0], logy=False)
    draw(axs[1], logy=True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=nm, fontsize=10, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Attention operator latency vs KV-cache length, by KV format", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out = os.path.join(a.outdir, "attn_op_latency_bars.png")
    save_fig(fig, out, dpi=130, tight=False, bbox_inches="tight")


if __name__ == "__main__":
    run()
