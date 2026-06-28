#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped bar chart: end-to-end performance vs input length, by KV format.
Higher = better. Measured data provided by the user."""
import argparse
import os

import numpy as np

from lib.common import save_fig, setup_matplotlib

LENGTHS = ["4k", "8k", "16k", "32k", "64k"]
DATA = {                       # end-to-end performance (higher = better)
    "K16V16 (F16)": [4.44, 3.03, 1.95, 1.12, 0.59],
    "K4V4 (Q4_0)":  [4.67, 4.13, 3.45, 2.46, 1.63],
    "K1V2":         [5.50, 4.82, 4.79, 4.47, 3.10],
    "K2V2":         [5.35, 4.96, 4.72, 4.43, 3.26],
}
COLORS = {"K16V16 (F16)": "#5b6770", "K4V4 (Q4_0)": "#d62728",
          "K1V2": "#156915", "K2V2": "#2ca02c"}


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="probe_out/attn_bench")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    methods = list(DATA)
    nm = len(methods); x = np.arange(len(LENGTHS)); bw = 0.8 / nm
    fig, ax = plt.subplots(figsize=(12, 6.6))
    for i, m in enumerate(methods):
        xs = x + (i - (nm - 1) / 2) * bw
        bars = ax.bar(xs, DATA[m], width=bw, color=COLORS[m], label=m,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, DATA[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("input length")
    ax.set_ylabel("end-to-end performance  (higher = better)")
    ax.set_ylim(0, max(max(v) for v in DATA.values()) * 1.16)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95, ncol=2)
    ax.set_title("End-to-end performance vs input length, by KV format   (Quest budget = 2048)", fontsize=13)

    out = os.path.join(a.outdir, "e2e_perf_bars.png")
    save_fig(fig, out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    run()
