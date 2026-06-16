#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""per-token K (uniform vs qlut-nf2, +/-SmoothAttention) vs per-channel, Llama-3.2-1B.

Shows that the per-token accuracy loss comes from the *codebook*, not the
quantization axis: uniform per-token needs SmoothAttention (+3.66) and still
lags; qlut-nf2 per-token (Lloyd self-adaptive) reaches ~per-channel WITHOUT
smooth (+0.58 from smooth). Reads each variant's result.json. K bit/value for
per-token = 2.5 (2-bit + per-token side 0.5 over head_dim=64)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "longbench_out"
def mean(d):
    o = json.load(open(os.path.join(BASE, d, "pred", "result.json")))
    v = [x for x in o.values() if isinstance(x, (int, float))]
    return sum(v) / len(v)

fp16 = mean("llama32-1b-instruct_fp16")
# per-token pairs at 2.5 bit: (orig, smooth)
PT = {
    "uniform":  (mean("llama32-1b-instruct_kitty-pertoken"),
                 mean("llama32-1b-instruct-smooth_kitty-pertoken")),
    "qlut-nf2": (mean("llama32-1b-instruct_qlutattn-pertoken"),
                 mean("llama32-1b-instruct-smooth_qlutattn-pertoken")),
}
# per-channel references: (label, K bit, score)
PC = [("qlut σ²-mix", 1.68, mean("llama32-1b-instruct_qlutattn-k1v4")),
      ("KIVI-2",      2.25, mean("llama32-1b-instruct_kivi-2")),
      ("kitty",       2.50, mean("llama32-1b-instruct_kitty"))]

fig, ax = plt.subplots(figsize=(10, 6.4))
ax.axhline(fp16, ls="--", color="#999", lw=1.3)
ax.text(1.42, fp16 + 0.15, f"fp16 ceiling = {fp16:.2f} (16-bit)", color="#555", fontsize=10)

# per-channel: gray squares
for l, b, s in PC:
    ax.scatter(b, s, marker="s", s=95, c="#8a8a8a", zorder=3, edgecolors="white", linewidths=1)
    ax.annotate(f"{l}\n{b:g}b · {s:.2f}", (b, s), xytext=(b, s + 0.55),
                ha="center", fontsize=9, color="#555")

# per-token: orig -> smooth arrows at 2.5 bit, uniform/nf2 offset to avoid overlap
colors = {"uniform": "#d9663f", "qlut-nf2": "#2ca06c"}
xoff = {"uniform": 2.40, "qlut-nf2": 2.60}
for name, (o, sm) in PT.items():
    x = xoff[name]; c = colors[name]
    ax.annotate("", xy=(x, sm), xytext=(x, o),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
    ax.scatter([x, x], [o, sm], s=95, c=c, zorder=4, edgecolors="white", linewidths=1)
    ax.annotate(f"per-token {name}\norig {o:.2f}", (x, o), xytext=(x, o - 1.4),
                ha="center", fontsize=9, color=c)
    ax.annotate(f"+smooth {sm:.2f}  (Δ{sm-o:+.2f})", (x, sm), xytext=(x, sm + 0.35),
                ha="center", fontsize=9, color=c, fontweight="bold")

ax.set_xlabel("K-cache bits per value", fontsize=12)
ax.set_ylabel("LongBench score (21 datasets, 32k)", fontsize=12)
ax.set_title("Llama-3.2-1B · per-token K: uniform vs qlut-nf2 (±SmoothAttention) vs per-channel",
             fontsize=11.5)
ax.set_xlim(1.4, 2.8); ax.set_ylim(10, 28.5); ax.grid(True, ls=":", alpha=0.5)
fig.tight_layout()
out = os.path.join(BASE, "pertoken_smooth_llama32_1b.png")
fig.savefig(out, dpi=140)
print(f"uniform : orig {PT['uniform'][0]:.2f} -> smooth {PT['uniform'][1]:.2f} (Δ{PT['uniform'][1]-PT['uniform'][0]:+.2f})")
print(f"qlut-nf2: orig {PT['qlut-nf2'][0]:.2f} -> smooth {PT['qlut-nf2'][1]:.2f} (Δ{PT['qlut-nf2'][1]-PT['qlut-nf2'][0]:+.2f})")
print(f"per-channel: " + ", ".join(f"{l} {s:.2f}@{b}b" for l, b, s in PC))
print("saved", out)
