#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped bar chart: KV-cache memory usage (MB) vs context length, by KV format.
Lower = better. Linear panel (absolute growth) + log panel (every length readable).
Measured data provided by the user."""
import argparse
import os

import numpy as np

from lib.common import save_fig, setup_matplotlib

LENGTHS = ["4k", "8k", "16k", "32k", "64k"]
DATA = {                       # KV-cache memory (MB), lower = better
    "K16V16 (F16)": [544, 1056, 2112, 4128, 8224],
    "K4V4 (Q4_0)":  [153, 297, 594, 1161, 2313],
    "K1V2":         [63.75, 123.75, 247.50, 483.75, 963.75],
    "K2V2":         [80.75, 156.75, 313.50, 612.75, 1220.75],
}
COLORS = {"K16V16 (F16)": "#5b6770", "K4V4 (Q4_0)": "#d62728",
          "K1V2": "#156915", "K2V2": "#2ca02c"}

methods = list(DATA)
nm = len(methods)
x = np.arange(len(LENGTHS))
bw = 0.8 / nm
vmax = max(max(v) for v in DATA.values())


def draw(ax, logy):
    for i, m in enumerate(methods):
        xs = x + (i - (nm - 1) / 2) * bw
        bars = ax.bar(xs, DATA[m], width=bw, color=COLORS[m], label=m,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, DATA[m]):
            ax.text(b.get_x() + b.get_width() / 2,
                    v * 1.03 if logy else v + vmax * 0.008,
                    f"{v:g}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("context length")
    ax.set_ylabel("KV-cache memory  (MB · lower = better)")
    ax.grid(axis="y", ls=":", alpha=0.4)
    if logy:
        ax.set_yscale("log"); ax.set_ylim(45, 13000)
        ax.set_title("log scale — every length readable", fontsize=10)
    else:
        ax.set_ylim(0, vmax * 1.12)
        ax.set_title("linear scale — absolute footprint", fontsize=10)


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="probe_out/attn_bench")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.6))
    draw(ax, logy=False)
    ax.set_title("KV-cache memory usage vs context length, by KV format", fontsize=13)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95, ncol=2)

    out = os.path.join(a.outdir, "kv_memory_bars.png")
    save_fig(fig, out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    run()
