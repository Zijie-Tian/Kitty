#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combined LongBench-vs-K-cache-bit/value Pareto for Llama-3.2-1B.

Overlays the per-token offline sigma^2-mix sweeps (the NEW work) on top of the
discrete reference methods from scripts/plot_kcache_pareto.py:

  * REFERENCE discrete points (per-channel qlut family + KIVI/kitty), drawn as
    large markers, no connecting line. Source: the original
    longbench_out/kcache_pareto_llama32_1b.png (values stable / published).
  * per-token NON-ROTATED lines  (autoresearch/pareto-sign-ternnf2/pareto_data.tsv)
      - sign/nf2  = the DEFAULT recommended optimized algorithm (red, bold)
      - sign/tern
  * per-token ROTATED lines      (autoresearch/pareto-rotated/pareto_rotated_data.tsv)
      - rotated sign/nf2, rotated sign/tern (dashed, supplementary)

x = K-cache bit/value (effective, by mix ratio: bit = f*lo + (1-f)*hi).
y = LongBench 21-dataset average. fp16 = 16-bit ceiling line.

The two collapsed naive-KIVI-1-bit points (1.25b/9.20, 1.5b/10.76) from the old
figure are intentionally dropped so the y-axis can focus on the useful band.

Pure matplotlib, no GPU, no model load. Run:
  conda run -n kitty python scripts/plot_kcache_pareto_combined.py
"""
import argparse
import os

from lib.common import save_fig, setup_matplotlib

FP16 = 27.59          # fp16 ceiling (21-dataset mean)
K1V4_PC = 24.88       # per-channel qlutattn-k1v4 (reference horizontal guide)

# --- reference discrete methods (label, bit/value, score, family) -----------
# family: 'qlut_pc' = per-channel submean qlut codebooks (green);
#         'kivi'    = KIVI / kitty dense-uniform style (orange).
REF = [
    ("sign (per-ch)",         1.25, 21.23, "qlut_pc"),
    ("qlutattn-k1v4 (per-ch)", 1.68, 24.88, "qlut_pc"),
    ("tern (per-ch)",         1.83, 22.96, "qlut_pc"),
    ("KIVI-2",                2.25, 24.24, "kivi"),
    ("KIVI*-2",               2.25, 25.48, "kivi"),
    ("kitty",                 2.50, 26.25, "kivi"),
]
REF_OFF = {  # label -> (dx, dy) text offset in data units
    "sign (per-ch)": (-0.02, -0.95), "qlutattn-k1v4 (per-ch)": (-0.34, 0.30),
    "tern (per-ch)": (0.0, -1.05), "KIVI-2": (0.05, -1.0),
    "KIVI*-2": (0.06, 0.30), "kitty": (0.0, 0.34),
}

# --- per-token sweeps: (bit, score) sorted by bit -------------------------- #
# NON-ROTATED (autoresearch/pareto-sign-ternnf2/pareto_data.tsv)
NR_SNF = [(1.2500, 21.39), (1.5625, 24.07), (1.8750, 25.03), (2.1875, 25.42), (2.5000, 25.65)]
NR_ST  = [(1.2500, 21.39), (1.4000, 22.73), (1.5500, 22.89), (1.6800, 22.75), (1.7000, 22.75), (1.8500, 22.54)]
# ROTATED (autoresearch/pareto-rotated/pareto_rotated_data.tsv)
ROT_SNF = [(1.2500, 22.14), (1.5625, 23.44), (1.8750, 24.19), (2.1875, 25.16), (2.5000, 26.04)]
ROT_ST  = [(1.2500, 22.14), (1.4000, 22.72), (1.5500, 23.15), (1.7000, 23.67), (1.8500, 24.21)]


def xs(pts):
    return [p[0] for p in pts]


def ys(pts):
    return [p[1] for p in pts]


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.0, 7.0))

    # ceiling + per-channel k1v4 guide lines
    ax.axhline(FP16, ls="--", color="#999", lw=1.3)
    ax.text(1.12, FP16 + 0.08, f"fp16 ceiling = {FP16} (16-bit)",
            ha="left", va="bottom", color="#555", fontsize=10)
    ax.axhline(K1V4_PC, ls=":", color="#444", lw=1.0)
    ax.text(2.58, K1V4_PC + 0.06, f"per-channel k1v4 {K1V4_PC}",
            ha="right", va="bottom", color="#444", fontsize=8.5)

    # reference discrete points
    for fam, color, marker in [("qlut_pc", "#2ca06c", "P"), ("kivi", "#d9663f", "X")]:
        sub = [r for r in REF if r[3] == fam]
        ax.scatter(xs([(b, s) for _, b, s, _ in sub]),
                   ys([(b, s) for _, b, s, _ in sub]),
                   s=150, c=color, marker=marker, zorder=4,
                   edgecolors="white", linewidths=1.3,
                   label=("per-channel qlut (ref)" if fam == "qlut_pc"
                          else "KIVI / kitty (ref)"))
    for lbl, b, s, _ in REF:
        dx, dy = REF_OFF.get(lbl, (0.0, -1.0))
        ax.annotate(f"{lbl}\n{b:g}b·{s:.2f}", (b, s), xytext=(b + dx, s + dy),
                    ha="center", fontsize=8.3, color="#333")

    # per-token lines
    ax.plot(xs(NR_ST), ys(NR_ST), color="#1f77b4", lw=2.0, marker="o", ms=7,
            zorder=5, label="per-token sign/tern (no rot)")
    ax.plot(xs(ROT_ST), ys(ROT_ST), color="#2ca02c", lw=1.7, ls="--", marker="^",
            ms=7, alpha=0.85, zorder=5, label="per-token rotated sign/tern")
    ax.plot(xs(ROT_SNF), ys(ROT_SNF), color="#9467bd", lw=1.7, ls="--", marker="D",
            ms=6.5, alpha=0.85, zorder=5, label="per-token rotated sign/nf2")
    # DEFAULT method last so it sits on top
    ax.plot(xs(NR_SNF), ys(NR_SNF), color="#d62728", lw=3.0, marker="s", ms=8.5,
            zorder=6, label="per-token sign/nf2 (no rot) ★ DEFAULT")

    # highlight the default sweet spot f=0.5 (1.875b, 25.03)
    ax.annotate("★ default sweet spot\n1.875b · 25.03  (> per-ch k1v4)",
                (1.8750, 25.03), xytext=(1.62, 26.7), fontsize=9.5,
                color="#d62728", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4))

    ax.set_xlabel("K-cache bit / value  (effective, by mix ratio)", fontsize=12)
    ax.set_ylabel("LongBench average (21 datasets, 32k)  quality →", fontsize=12)
    ax.set_title("Llama-3.2-1B · K-cache Pareto — per-token σ²-mix lines vs reference methods",
                 fontsize=12.5)
    ax.set_xlim(1.12, 2.62)
    ax.set_ylim(20.5, 28.0)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    out = os.path.join(a.outdir, "kcache_pareto_combined_llama32_1b.png")
    save_fig(fig, out, dpi=140)

    # console summary
    print(f"\n{'line':28s} {'bit':>6s} {'score':>7s}")
    for name, pts in [("per-token sign/nf2 (DEFAULT)", NR_SNF),
                      ("per-token sign/tern", NR_ST),
                      ("per-token rotated sign/nf2", ROT_SNF),
                      ("per-token rotated sign/tern", ROT_ST)]:
        for b, s in pts:
            print(f"{name:28s} {b:6.4f} {s:7.2f}")


if __name__ == "__main__":
    run()
