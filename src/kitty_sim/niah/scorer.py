"""Scoring + depth/length aggregation for RULER-NIAH predictions.

Metric is RULER's ``string_match_all``: for each sample, the fraction of gold
needle values contained (case-insensitive) in the prediction; task score is
the mean over samples x 100.

Depth heatmap: the RULER fork records ``token_position_answer`` (needle token
offset). ``depth = token_position_answer / length`` is binned into
``n_depth_bins`` uniform bins, producing an accuracy matrix depth-bin x
seq_len per task (and pooled over tasks) -- the classic NIAH heatmap.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PAIR_FILE_RE = re.compile(r"^(?P<task>[a-z0-9_]+)__(?P<len>\d+)\.jsonl$")


def string_match_all_score(pred: str, refs: list[str]) -> float:
    """Per-sample RULER string_match_all: fraction of refs present in pred."""
    if not refs:
        raise ValueError("string_match_all needs at least one reference")
    pred_lower = pred.lower()
    return sum(1.0 for r in refs if str(r).lower() in pred_lower) / len(refs)


def load_pred_rows(pred_dir: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(pred_dir.glob("*.jsonl")):
        match = PAIR_FILE_RE.match(path.name)
        if not match:
            continue
        task, seq_len = match.group("task"), int(match.group("len"))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows[(task, seq_len)].append(json.loads(line))
    return dict(rows)


def score_pred_dir(pred_dir: Path, *, n_depth_bins: int = 10) -> dict[str, Any]:
    rows_by_pair = load_pred_rows(pred_dir)
    if not rows_by_pair:
        raise FileNotFoundError(f"No <task>__<len>.jsonl prediction files in {pred_dir}")

    tasks = sorted({task for task, _ in rows_by_pair})
    seq_lens = sorted({seq_len for _, seq_len in rows_by_pair})

    scores: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    # depth_hits[task][(bin, len)] = [hit_sum, n]
    depth_acc: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0])
    )

    for (task, seq_len), rows in sorted(rows_by_pair.items()):
        sample_scores = []
        for row in rows:
            s = string_match_all_score(row["pred"], row["outputs"])
            sample_scores.append(s)
            tpa = row.get("token_position_answer")
            length = row.get("length")
            if tpa is not None and length:
                depth = min(max(float(tpa) / float(length), 0.0), 1.0)
                bin_idx = min(int(depth * n_depth_bins), n_depth_bins - 1)
                cell = depth_acc[task][(bin_idx, seq_len)]
                cell[0] += s
                cell[1] += 1
        scores.setdefault(task, {})[str(seq_len)] = round(
            100.0 * sum(sample_scores) / len(sample_scores), 2
        )
        counts.setdefault(task, {})[str(seq_len)] = len(sample_scores)

    per_len_mean = {
        str(seq_len): round(
            sum(scores[task][str(seq_len)] for task in tasks if str(seq_len) in scores[task])
            / sum(1 for task in tasks if str(seq_len) in scores[task]),
            2,
        )
        for seq_len in seq_lens
    }

    def _depth_matrix(cells: dict[tuple[int, int], list[float]]) -> dict[str, Any]:
        matrix = [[None] * len(seq_lens) for _ in range(n_depth_bins)]
        n_matrix = [[0] * len(seq_lens) for _ in range(n_depth_bins)]
        for (bin_idx, seq_len), (hit_sum, n) in cells.items():
            col = seq_lens.index(seq_len)
            matrix[bin_idx][col] = round(100.0 * hit_sum / n, 2) if n else None
            n_matrix[bin_idx][col] = n
        return {"acc": matrix, "n": n_matrix}

    pooled_cells: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0])
    for task_cells in depth_acc.values():
        for key, (hit_sum, n) in task_cells.items():
            pooled_cells[key][0] += hit_sum
            pooled_cells[key][1] += n

    result = {
        "benchmark": "ruler_niah",
        "metric": "string_match_all",
        "tasks": tasks,
        "seq_lens": seq_lens,
        "scores": scores,
        "counts": counts,
        "per_len_mean": per_len_mean,
        "overall_mean": round(
            sum(per_len_mean.values()) / len(per_len_mean), 2
        ) if per_len_mean else None,
        "n_depth_bins": n_depth_bins,
        "depth_matrix_pooled": _depth_matrix(pooled_cells),
        "depth_matrix_by_task": {
            task: _depth_matrix(cells) for task, cells in depth_acc.items()
        },
    }
    return result


def plot_depth_heatmap(
    result: dict[str, Any],
    out_path: Path,
    *,
    title: str,
    task: str | None = None,
) -> Path:
    """Render the classic NIAH depth x context-length heatmap (green=found)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    source = (
        result["depth_matrix_by_task"][task]
        if task is not None
        else result["depth_matrix_pooled"]
    )
    acc = np.array(
        [[np.nan if v is None else v for v in row] for row in source["acc"]],
        dtype=float,
    )
    seq_lens = result["seq_lens"]
    n_bins = result["n_depth_bins"]

    fig, ax = plt.subplots(
        figsize=(1.6 + 1.15 * len(seq_lens), 4.8), constrained_layout=True
    )
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(
        np.ma.masked_invalid(acc),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=100.0,
        origin="upper",
    )
    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels([f"{s // 1024}k" for s in seq_lens])
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(
        [f"{int(100 * i / n_bins)}-{int(100 * (i + 1) / n_bins)}%" for i in range(n_bins)]
    )
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("needle depth")
    ax.set_title(title)
    ns = source["n"]
    for i in range(n_bins):
        for j in range(len(seq_lens)):
            if not np.isnan(acc[i, j]):
                ax.text(
                    j,
                    i,
                    f"{acc[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    fig.colorbar(im, ax=ax, label="string-match accuracy (%)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
