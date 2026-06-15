#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot LongBench score vs K-cache bits/value for Llama-3.2-1B KV-quant variants.

Reads each variant's longbench_out/<dir>/pred/result.json (mean over 21 datasets)
and scatters it against the variant's K-cache bits/value. fp16 is drawn as a
ceiling line (16-bit, off the bit axis). K bit/value = codeword bits + fp16
side-info (scale+zero = 2*16/group, group=128 -> 0.25), sink/buffer ignored.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "longbench_out"
# (label, output-dir, K bits/value)
POINTS = [
    ("sign (submean 1-bit)", "llama32-1b-instruct-sign_qlutattn-k1v4", 1.25),
    ("k1v4 (KIVI 1-bit)",    "llama32-1b-instruct-k1v4pr0_kitty-k1v4", 1.25),
    ("k1v4 +25% promote",    "llama32-1b-instruct_kitty-k1v4",         1.50),
    ("qlutattn-k1v4",        "llama32-1b-instruct_qlutattn-k1v4",      1.68),
    ("tern",                 "llama32-1b-instruct_tern-uniform",       1.83),
    ("KIVI-2",               "llama32-1b-instruct_kivi-2",             2.25),
    ("KIVI*-2",              "llama32-1b-instruct_kivi-star-2",        2.25),
    ("kitty",                "llama32-1b-instruct_kitty",              2.50),
]
FP16_DIR = "llama32-1b-instruct_fp16"
# methods using the submean (qlut) codebook family — drawn green; rest are KIVI-style.
GREEN = {"sign (submean 1-bit)", "tern", "qlutattn-k1v4"}
# per-point label offset (x_off bits, y_off score) to avoid overlap
OFFS = {"sign (submean 1-bit)": (-0.05, 0.7), "k1v4 (KIVI 1-bit)": (0.05, -1.6),
        "k1v4 +25% promote": (0, -1.6), "qlutattn-k1v4": (0.06, 0.6),
        "tern": (0, -1.6), "KIVI-2": (0, -1.6), "KIVI*-2": (0, 0.6),
        "kitty": (0, 0.6)}


def mean(d):
    o = json.load(open(os.path.join(BASE, d, "pred", "result.json")))
    v = [x for x in o.values() if isinstance(x, (int, float))]
    return sum(v) / len(v)


def main():
    fp16 = mean(FP16_DIR)
    pts = [(lbl, bit, mean(d)) for lbl, d, bit in POINTS]
    print(f"{'variant':16s} {'Kbit':>5s} {'score':>7s}  retain%")
    for lbl, bit, sc in sorted(pts, key=lambda p: p[1]):
        print(f"{lbl:16s} {bit:5.2f} {sc:7.2f}  {100*sc/fp16:5.1f}%")
    print(f"{'fp16':16s} {16:5.0f} {fp16:7.2f}  100.0%")

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.axhline(fp16, ls="--", color="#999", lw=1.3)
    ax.text(1.02, fp16 + 0.12, f"fp16 ceiling = {fp16:.2f} (16-bit)",
            ha="left", va="bottom", color="#555", fontsize=10)

    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    colors = ["#2ca06c" if l in GREEN else "#d9663f" for l, _, _ in pts]
    ax.scatter(xs, ys, s=110, c=colors, zorder=3, edgecolors="white", linewidths=1.2)
    for lbl, bit, sc in pts:
        dx, dy = OFFS.get(lbl, (0, -1.1))
        ax.annotate(f"{lbl}\n{bit:g} bit · {sc:.2f}", (bit, sc),
                    xytext=(bit + dx, sc + dy), ha="center", fontsize=9.5,
                    color="#222")

    ax.set_xlabel("K-cache bits per value", fontsize=12)
    ax.set_ylabel("LongBench score (quality →)", fontsize=12)
    ax.set_title("Llama-3.2-1B · LongBench vs K-cache bit/value (21 datasets, 32k)",
                 fontsize=12)
    ax.set_xlim(1.0, 2.7)
    ax.set_ylim(min(ys) - 3, fp16 + 1.5)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(BASE, "kcache_pareto_llama32_1b.png")
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
