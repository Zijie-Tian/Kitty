# -*- coding: utf-8 -*-
"""Render the M2 per-layer NF2 quota stability figure (paper-figure candidate).

Ported from worktree eval_design_fig_6/overlap_pipeline.

Reads the GPU-computed TSVs (nf2_counts / layer_ratio_summary) and the
per-task masks artifact; plotting is presentation only — all statistics were
computed on GPU by overlap_metrics.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mdir = Path(args.metrics_dir)

    masks = torch.load(mdir / f"masks_{args.tag}.pt", map_location="cpu",
                       weights_only=True)
    tasks = masks["tasks"]
    counts = masks["nf2_counts_per_layer"].double()          # [T, nl]
    N = 512 if counts.shape[1] == 16 else 1024
    ratios = counts / N
    T, nl = ratios.shape

    rows = list(csv.reader(open(mdir / f"layer_ratio_summary_{args.tag}.tsv"),
                           delimiter="\t"))
    hdr = rows[0]
    data = {h: [float(r[i + 1]) for r in rows[1:]] for i, h in enumerate(hdr[1:])}
    layers = list(range(nl))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.fill_between(layers, data["min"], data["max"], alpha=0.25, color="tab:blue",
                    label=f"cross-task min–max ({T} tasks)")
    ax.plot(layers, data["mean"], color="tab:blue", lw=1.8,
            label="cross-task mean")
    for i in range(T):
        ax.plot(layers, ratios[i].tolist(), color="tab:gray", alpha=0.28, lw=0.7)
    if "wikitext_ref" in data:
        ax.plot(layers, data["wikitext_ref"], color="tab:red", lw=1.6, ls="--",
                marker="o", ms=3.5, label="WikiText-2 calibration (deployed)")
    ax.set_xlabel("layer")
    ax.set_ylabel("NF2 channel ratio @ ρ=0.62")
    ax.set_ylim(0, max(data["max"]) * 1.25)
    ax.set_xlim(0, nl - 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"{args.model_label}: per-layer NF2 quota structure across "
                 f"{T} LongBench tasks", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
