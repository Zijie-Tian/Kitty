#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped bar chart: end-to-end energy per token (J/tok), GPU vs QLUTATTN,
at Quest budget = 1024, across context lengths. Lower = better. The
QLUTATTN/GPU efficiency ratio is annotated above each group. Measured data."""
import argparse
import os

import numpy as np

from lib.common import save_fig, setup_matplotlib

LENGTHS = ["4k", "8k", "16k", "32k", "64k"]
GPU =     [4.36, 4.43, 4.75, 5.48, 6.63]      # J/tok
QLUT =    [3.14, 3.33, 3.56, 3.75, 5.27]      # J/tok
C_GPU, C_QLUT = "#5b6770", "#2ca02c"


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="probe_out/attn_bench")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    x = np.arange(len(LENGTHS)); bw = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.6))
    b1 = ax.bar(x - bw / 2, GPU, width=bw, color=C_GPU, label="GPU (baseline)",
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + bw / 2, QLUT, width=bw, color=C_QLUT, label="QLUTATTN",
                edgecolor="white", linewidth=0.5)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    for i in range(len(LENGTHS)):
        r = QLUT[i] / GPU[i]
        top = max(GPU[i], QLUT[i])
        ax.text(x[i], top + 0.62, f"{r:.2f}×\n(−{(1 - r) * 100:.0f}% energy)",
                ha="center", va="bottom", fontsize=9.5, color="#156915", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eaf7ea", ec=C_QLUT, lw=1.3))

    ax.set_xticks(x); ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("context length")
    ax.set_ylabel("energy per token  (J/tok · lower = better)")
    ax.set_ylim(0, max(GPU) * 1.34)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_title("End-to-end energy efficiency: QLUTATTN vs GPU baseline   (Quest budget = 1024)\n"
                 "green box = QLUTATTN/GPU energy ratio", fontsize=12)

    out = os.path.join(a.outdir, "qlutattn_energy_efficiency_quest1024.png")
    save_fig(fig, out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    run()
