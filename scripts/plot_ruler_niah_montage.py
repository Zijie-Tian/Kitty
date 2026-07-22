#!/usr/bin/env python
"""Combine several RULER arms' NIAH depth heatmaps into one figure.

Each arm must contain ``pred/result.json`` written by
``kitty_sim.cli.score_ruler``. For example::

    PYTHONPATH=src python scripts/plot_ruler_niah_montage.py \
        --arms ruler_out/llama32-1b-instruct_fp16 \
               ruler_out/llama32-1b-instruct_qlutattn \
        --labels fp16 qlutattn \
        --task pooled \
        --output ruler_out/ruler_niah_heatmap_montage.png

Smoke arms can be passed explicitly from ``ruler_out/smoke/`` in the same way.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def load_result(arm_dir: Path) -> dict[str, Any]:
    path = Path(arm_dir) / "pred" / "result.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run kitty_sim.cli.score_ruler first"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("benchmark") != "ruler":
        raise ValueError(f"{path} is not a RULER result")
    return result


def matrix_from_result(result: dict[str, Any], task: str):
    """Return a NumPy depth matrix, importing NumPy only when plotting."""

    import numpy as np

    if task == "pooled":
        source = result["depth_matrix_pooled"]
    else:
        by_task = result["depth_matrix_by_task"]
        if task not in by_task:
            available = ", ".join(by_task) or "none"
            raise ValueError(
                f"No depth matrix for {task!r}; available per-task matrices: {available}"
            )
        source = by_task[task]
    return np.array(
        [
            [np.nan if value is None else value for value in row]
            for row in source["acc"]
        ],
        dtype=float,
    )


def _format_length(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


def _format_mean(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a comparison montage from ruler_out arm result files."
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        required=True,
        help="RULER arm directories containing pred/result.json",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Panel titles (default: arm directory names)",
    )
    parser.add_argument(
        "--task",
        default="pooled",
        help="'pooled' or a depth-eligible NIAH task name",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arm_dirs = [Path(value).expanduser() for value in args.arms]
    labels = args.labels or [arm_dir.name for arm_dir in arm_dirs]
    if len(labels) != len(arm_dirs):
        raise ValueError("--labels must match --arms in length")

    results = [load_result(arm_dir) for arm_dir in arm_dirs]
    seq_lens = results[0]["seq_lens"]
    n_depth_bins = results[0]["n_depth_bins"]
    for result, arm_dir in zip(results, arm_dirs):
        if (
            result["seq_lens"] != seq_lens
            or result["n_depth_bins"] != n_depth_bins
        ):
            raise ValueError(
                f"{arm_dir} has mismatched seq_lens/depth bins vs the first arm"
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrices = [matrix_from_result(result, args.task) for result in results]
    for matrix, arm_dir in zip(matrices, arm_dirs):
        if matrix.shape != (n_depth_bins, len(seq_lens)):
            raise ValueError(
                f"{arm_dir} has depth matrix shape {matrix.shape}, expected "
                f"({n_depth_bins}, {len(seq_lens)})"
            )
        if np.isnan(matrix).all():
            raise ValueError(
                f"{arm_dir} has no NIAH depth observations for {args.task!r}"
            )

    panel_count = len(arm_dirs)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(max(1.1, 0.62 * len(seq_lens)) * panel_count + 2.4, 4.6),
        constrained_layout=True,
        sharey=True,
    )
    if panel_count == 1:
        axes = [axes]
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#dddddd")
    image = None
    for ax, result, label, matrix in zip(axes, results, labels, matrices):
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=100.0,
            origin="upper",
        )
        for row_index in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if not np.isnan(matrix[row_index, column]):
                    ax.text(
                        column,
                        row_index,
                        f"{matrix[row_index, column]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
        means = result["per_len_mean"]
        mean_line = " ".join(
            f"{_format_length(seq_len)}:{_format_mean(means.get(str(seq_len)))}"
            for seq_len in seq_lens
        )
        status_suffix = "" if result["status"] == "ok" else f" [{result['status']}]"
        ax.set_title(
            f"{label}{status_suffix}\nmean {mean_line}",
            fontsize=9,
        )
        ax.set_xticks(range(len(seq_lens)))
        ax.set_xticklabels(
            [_format_length(seq_len) for seq_len in seq_lens], fontsize=8
        )

    axes[0].set_yticks(range(n_depth_bins))
    axes[0].set_yticklabels(
        [
            f"{int(100 * index / n_depth_bins)}-"
            f"{int(100 * (index + 1) / n_depth_bins)}%"
            for index in range(n_depth_bins)
        ],
        fontsize=8,
    )
    axes[0].set_ylabel("needle depth")
    fig.supxlabel("context length")
    if image is None:
        raise RuntimeError("No montage panels were created")
    fig.colorbar(
        image,
        ax=axes,
        label="string-match accuracy (%)",
        shrink=0.9,
    )
    fig.suptitle(
        args.title or f"RULER NIAH depth x length ({args.task})",
        fontsize=12,
    )
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[montage] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
