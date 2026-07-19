# -*- coding: utf-8 -*-
"""Pareto figure for the research_mixed loop: subtask screen vs full LongBench."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1a1a19", "#52514e", "#e8e7e3"
C_UNI, C_Q, C_TIER = "#2a78d6", "#008300", "#e87ba4"   # validated categorical order

sub = {
    "uniform": [(1.25, 17.15), (1.50, 22.45), (1.60, 24.72), (1.75, 25.06), (2.00, 26.61), (2.25, 25.49)],
    "q":       [(1.35, 21.35), (1.40, 23.63), (1.45, 26.10), (1.50, 26.53), (1.60, 27.80), (1.75, 26.57)],
    "tier":    [(1.40, 25.23), (1.50, 27.22)],
    "fp16": 33.45,
}
full = {
    "uniform": [(1.50, 23.78), (1.60, 24.28), (1.75, 24.23)],
    "q":       [(1.50, 24.27), (1.60, 24.53)],
    "tier":    [(1.50, 24.52)],
    "fp16": 27.59,
}
SERIES = [("uniform σ²", C_UNI, "o", "uniform"),
          ("+ query-aware (σ²×E|q|)", C_Q, "s", "q"),
          ("+ token tiering (2D)", C_TIER, "D", "tier")]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)
fig.patch.set_facecolor("white")

for ax, data, title in [
    (axes[0], sub, "Screening metric (qasper + multifieldqa_en, full samples)"),
    (axes[1], full, "Full LongBench (21 datasets)"),
]:
    ax.set_facecolor("white")
    ax.axhline(data["fp16"], color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(f"FP16 {data['fp16']:.2f}", xy=(0.99, data["fp16"]), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8.5, color=MUTED)
    for label, color, mk, key in SERIES:
        pts = data[key]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, marker=mk, ms=8, mfc=color, mec="white", mew=1.2,
                label=label, zorder=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("nominal K-cache bits / value", color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=10.5, pad=10, loc="left")

axes[0].set_ylabel("score", color=INK, fontsize=10)
# selective direct labels: the story points only
a = axes[0]
a.annotate("interior optimum\n~35% nf2", xy=(1.60, 27.80), xytext=(1.72, 29.3), fontsize=8.5,
           color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.annotate("2D 27.22", xy=(1.50, 27.22), xytext=(1.33, 28.6), fontsize=8.5, color=INK,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.annotate("canonical f50", xy=(1.75, 25.06), xytext=(1.86, 23.6), fontsize=8.5, color=INK,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.annotate("cliff < 1.45b", xy=(1.40, 23.63), xytext=(1.26, 25.6), fontsize=8.5, color=MUTED)
b = axes[1]
b.annotate("canonical 24.23 @1.75b", xy=(1.75, 24.23), xytext=(1.60, 23.35), fontsize=8.5, color=INK,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
b.annotate("2D 24.52 @1.50b\n(−0.25 bit, +0.29)", xy=(1.50, 24.52), xytext=(1.53, 25.2),
           fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
b.set_ylim(23.0, 28.4)
axes[0].legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK)
fig.suptitle("Llama-3.2-1B — sign/nf2 K-cache mixed-precision Pareto (V fixed 2-bit tile16c64)",
             color=INK, fontsize=11.5, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = "autoresearch/loop-260717-1200/pareto.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
