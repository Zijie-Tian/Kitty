#!/usr/bin/env python
"""Combine several NIAH arms' depth x length heatmaps into one comparison figure.

Reads each arm's ``pred/result.json`` (written by kitty_sim.cli.score_niah) and
renders a single row/grid of heatmaps with a shared color scale, plus the
per-length task-mean accuracy under each panel title.

Usage:
  PYTHONPATH=src python scripts/plot_niah_montage.py \
      --arms niah_out/llama32-1b-instruct_fp16 \
             niah_out/llama32-1b-instruct_qlutattn \
      --labels "fp16" "qlutattn" \
      --task pooled \
      --output niah_out/niah_heatmap_montage.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_result(arm_dir: Path) -> dict:
    path = arm_dir / "pred" / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run kitty_sim.cli.score_niah first")
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_from_result(result: dict, task: str) -> np.ndarray:
    source = (
        result["depth_matrix_pooled"]
        if task == "pooled"
        else result["depth_matrix_by_task"][task]
    )
    return np.array(
        [[np.nan if v is None else v for v in row] for row in source["acc"]],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True, help="arm dirs (contain pred/result.json)")
    parser.add_argument("--labels", nargs="+", default=None, help="panel titles (default: dir names)")
    parser.add_argument("--task", default="pooled", help="'pooled' or a task name, e.g. niah_single_2")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arm_dirs = [Path(a) for a in args.arms]
    labels = args.labels or [d.name for d in arm_dirs]
    if len(labels) != len(arm_dirs):
        raise ValueError("--labels must match --arms in length")

    results = [load_result(d) for d in arm_dirs]
    seq_lens = results[0]["seq_lens"]
    n_bins = results[0]["n_depth_bins"]
    for r, d in zip(results, arm_dirs):
        if r["seq_lens"] != seq_lens or r["n_depth_bins"] != n_bins:
            raise ValueError(f"{d} has mismatched seq_lens/depth bins vs the first arm")

    n = len(arm_dirs)
    fig, axes = plt.subplots(
        1, n,
        figsize=(0.62 * len(seq_lens) * n + 2.4, 4.6),
        constrained_layout=True,
        sharey=True,
    )
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#dddddd")
    im = None
    for ax, result, label in zip(axes, results, labels):
        acc = matrix_from_result(result, args.task)
        im = ax.imshow(
            np.ma.masked_invalid(acc),
            aspect="auto", cmap=cmap, vmin=0.0, vmax=100.0, origin="upper",
        )
        for i in range(acc.shape[0]):
            for j in range(acc.shape[1]):
                if not np.isnan(acc[i, j]):
                    ax.text(j, i, f"{acc[i, j]:.0f}", ha="center", va="center",
                            fontsize=7, color="black")
        means = result["per_len_mean"]
        mean_line = " ".join(f"{s // 1024}k:{means[str(s)]:.0f}" for s in seq_lens)
        ax.set_title(f"{label}\nmean {mean_line}", fontsize=9)
        ax.set_xticks(range(len(seq_lens)))
        ax.set_xticklabels([f"{s // 1024}k" for s in seq_lens], fontsize=8)
        ax.set_xlabel("context length")
    axes[0].set_yticks(range(n_bins))
    axes[0].set_yticklabels(
        [f"{int(100 * i / n_bins)}-{int(100 * (i + 1) / n_bins)}%" for i in range(n_bins)],
        fontsize=8,
    )
    axes[0].set_ylabel("needle depth")
    fig.colorbar(im, ax=axes, label="string-match accuracy (%)", shrink=0.9)
    suptitle = args.title or f"RULER-NIAH depth x length ({args.task})"
    fig.suptitle(suptitle, fontsize=12)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"[montage] wrote {out}")


if __name__ == "__main__":
    main()
