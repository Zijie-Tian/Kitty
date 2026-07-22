"""Score RULER predictions and aggregate NIAH depth diagnostics.

Prediction files are named ``<task>__<length>.jsonl``.  Task metrics and depth
eligibility come exclusively from :mod:`kitty_sim.ruler.tasks`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable

from kitty_sim.ruler.tasks import DEFAULT_SEQ_LENS, DEFAULT_TASKS, TASK_SPECS

PAIR_FILE_RE = re.compile(r"^(?P<task>[a-z0-9_]+)__(?P<len>\d+)\.jsonl$")

SampleMetric = Callable[[str, Sequence[str]], float]


def string_match_all_score(pred: str, refs: Sequence[str]) -> float:
    """Return the fraction of references contained in ``pred``.

    This is the per-sample form of NVIDIA RULER/COMPASS
    ``string_match_all``. Matching is case-insensitive substring matching.
    """

    if not refs:
        raise ValueError("string_match_all needs at least one reference")
    pred_lower = pred.lower()
    return sum(1.0 for ref in refs if str(ref).lower() in pred_lower) / len(refs)


def string_match_part_score(pred: str, refs: Sequence[str]) -> float:
    """Return one when any reference is contained in ``pred``, else zero."""

    if not refs:
        raise ValueError("string_match_part needs at least one reference")
    pred_lower = pred.lower()
    return max(1.0 if str(ref).lower() in pred_lower else 0.0 for ref in refs)


def _score_batch(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    sample_metric: SampleMetric,
) -> float:
    if not predictions:
        raise ValueError("RULER metrics need at least one prediction")
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have the same length "
            f"({len(predictions)} != {len(references)})"
        )
    score = sum(
        sample_metric(pred, refs) for pred, refs in zip(predictions, references)
    )
    return round(100.0 * score / len(predictions), 2)


def string_match_all(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """Return COMPASS ``string_match_all`` on the 0--100 scale."""

    return _score_batch(predictions, references, string_match_all_score)


def string_match_part(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """Return COMPASS ``string_match_part`` on the 0--100 scale."""

    return _score_batch(predictions, references, string_match_part_score)


_SAMPLE_METRICS: dict[str, SampleMetric] = {
    "string_match_all": string_match_all_score,
    "string_match_part": string_match_part_score,
}


def _pair_name(task: str, seq_len: int) -> str:
    return f"{task}__{seq_len}"


def _normalize_tasks(tasks: Iterable[str] | str | None) -> list[str]:
    values: Iterable[str]
    if tasks is None:
        values = DEFAULT_TASKS
    elif isinstance(tasks, str):
        values = tasks.split(",")
    else:
        values = tasks
    normalized = [str(task).strip() for task in values if str(task).strip()]
    if not normalized:
        raise ValueError("tasks must be non-empty")
    if "all" in normalized:
        if normalized != ["all"]:
            raise ValueError("'all' cannot be combined with explicit task names")
        return list(DEFAULT_TASKS)
    if len(set(normalized)) != len(normalized):
        raise ValueError("tasks must not contain duplicates")
    unknown = [task for task in normalized if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unknown RULER task(s): {', '.join(unknown)}")
    return normalized


def _normalize_seq_lens(seq_lens: Iterable[int] | str | None) -> list[int]:
    values: Iterable[int | str]
    if seq_lens is None:
        values = DEFAULT_SEQ_LENS
    elif isinstance(seq_lens, str):
        values = seq_lens.split(",")
    else:
        values = seq_lens
    try:
        normalized = [int(value) for value in values if str(value).strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError("seq_lens must be a comma-separated list of integers") from exc
    if not normalized:
        raise ValueError("seq_lens must be non-empty")
    if any(seq_len <= 0 for seq_len in normalized):
        raise ValueError("seq_lens must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seq_lens must not contain duplicates")
    return normalized


def load_pred_rows(pred_dir: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Load all pair JSONLs in ``pred_dir``, retaining empty pair files."""

    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")

    rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in sorted(pred_dir.glob("*.jsonl")):
        match = PAIR_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        pair = (match.group("task"), int(match.group("len")))
        if pair in rows:
            raise ValueError(
                f"Multiple prediction files resolve to {_pair_name(*pair)} in {pred_dir}"
            )
        pair_rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object in {path}:{line_number}")
                pair_rows.append(row)
        rows[pair] = pair_rows
    return rows


def _partial_pair_detail(
    pred_dir: Path,
    task: str,
    seq_len: int,
    count: int,
) -> tuple[dict[str, Any] | None, bool]:
    """Validate a prediction sidecar and return diagnostics for any defect."""

    pair = _pair_name(task, seq_len)
    manifest_path = pred_dir / f"{pair}.manifest.json"
    prediction_path = pred_dir / f"{pair}.jsonl"
    if not manifest_path.is_file():
        return None, True

    reasons: list[str] = []
    manifest: dict[str, Any]
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("manifest root is not an object")
        manifest = loaded
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return (
            {
                "pair": pair,
                "count": count,
                "expected_count": None,
                "manifest_status": None,
                "reasons": ["invalid_manifest"],
                "detail": str(exc),
            },
            False,
        )

    expected = manifest.get("expected_samples")
    expected_count: int | None = None
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        reasons.append("invalid_expected_samples")
    else:
        expected_count = expected
        if count != expected_count:
            reasons.append("row_count_mismatch")

    written = manifest.get("written_samples")
    if isinstance(written, bool) or not isinstance(written, int) or written < 0:
        reasons.append("invalid_written_samples")
    elif written != count:
        reasons.append("written_samples_mismatch")

    manifest_status = manifest.get("status")
    if manifest_status != "ok":
        reasons.append("manifest_not_ok")
    if manifest.get("benchmark") != "ruler":
        reasons.append("benchmark_mismatch")
    if manifest.get("task") != task or manifest.get("seq_len") != seq_len:
        reasons.append("pair_identity_mismatch")

    run_config = manifest.get("run_config")
    run_config_hash = manifest.get("run_config_hash")
    if not isinstance(run_config, dict):
        reasons.append("invalid_run_config")
    else:
        encoded = json.dumps(
            run_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        actual_run_config_hash = hashlib.sha256(encoded).hexdigest()
        if (
            not isinstance(run_config_hash, str)
            or run_config_hash != actual_run_config_hash
        ):
            reasons.append("run_config_hash_mismatch")
        if (
            run_config.get("benchmark") != "ruler"
            or run_config.get("task") != task
            or run_config.get("seq_len") != seq_len
            or run_config.get("expected_samples") != expected_count
        ):
            reasons.append("run_config_pair_mismatch")

    output_sha = manifest.get("jsonl_sha256")
    if not isinstance(output_sha, str) or len(output_sha) != 64:
        reasons.append("invalid_jsonl_sha256")
    else:
        actual_output_sha = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if output_sha != actual_output_sha:
            reasons.append("jsonl_sha256_mismatch")

    if not reasons:
        return None, False
    return (
        {
            "pair": pair,
            "count": count,
            "expected_count": expected_count,
            "manifest_status": manifest_status,
            "reasons": reasons,
        },
        False,
    )


def _depth_matrix(
    cells: dict[tuple[int, int], list[float]],
    seq_lens: Sequence[int],
    n_depth_bins: int,
) -> dict[str, list[list[float | int | None]]]:
    acc: list[list[float | None]] = [
        [None] * len(seq_lens) for _ in range(n_depth_bins)
    ]
    counts: list[list[int]] = [[0] * len(seq_lens) for _ in range(n_depth_bins)]
    columns = {seq_len: index for index, seq_len in enumerate(seq_lens)}
    for (bin_index, seq_len), (score_sum, count) in cells.items():
        column = columns.get(seq_len)
        if column is None:
            continue
        acc[bin_index][column] = (
            round(100.0 * score_sum / count, 2) if count else None
        )
        counts[bin_index][column] = int(count)
    return {"acc": acc, "n": counts}


def score_pred_dir(
    pred_dir: Path,
    *,
    tasks: Iterable[str] | str | None = None,
    seq_lens: Iterable[int] | str | None = None,
    n_depth_bins: int = 10,
) -> dict[str, Any]:
    """Score a requested RULER task-by-length profile.

    Missing requested files or absent/invalid/incomplete sidecars remain scorable,
    but produce ``status == "incomplete"`` and machine-readable diagnostics.
    Empty requested files, or a profile with no scored samples, fail instead of
    producing a vacuous score.
    """

    if isinstance(n_depth_bins, bool) or not isinstance(n_depth_bins, int):
        raise ValueError("n_depth_bins must be an integer")
    if n_depth_bins <= 0:
        raise ValueError("n_depth_bins must be positive")

    pred_dir = Path(pred_dir)
    requested_tasks = _normalize_tasks(tasks)
    requested_seq_lens = _normalize_seq_lens(seq_lens)
    rows_by_pair = load_pred_rows(pred_dir)
    if not rows_by_pair:
        raise FileNotFoundError(
            f"No <task>__<length>.jsonl prediction files in {pred_dir}"
        )

    requested_pairs = {
        (task, seq_len)
        for task in requested_tasks
        for seq_len in requested_seq_lens
    }
    empty_pairs = [
        _pair_name(task, seq_len)
        for task in requested_tasks
        for seq_len in requested_seq_lens
        if (task, seq_len) in rows_by_pair and not rows_by_pair[(task, seq_len)]
    ]
    if empty_pairs:
        raise ValueError(
            "Requested prediction pair(s) contain zero samples: "
            + ", ".join(empty_pairs)
        )

    length_keys = [str(seq_len) for seq_len in requested_seq_lens]
    scores: dict[str, dict[str, float | None]] = {
        task: {key: None for key in length_keys} for task in requested_tasks
    }
    counts: dict[str, dict[str, int]] = {
        task: {key: 0 for key in length_keys} for task in requested_tasks
    }
    depth_cells: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )

    scored_pairs: list[str] = []
    partial_pairs: list[dict[str, Any]] = []
    unverified_pairs: list[str] = []
    for task in requested_tasks:
        spec = TASK_SPECS[task]
        sample_metric = _SAMPLE_METRICS[spec.metric]
        for seq_len in requested_seq_lens:
            pair_rows = rows_by_pair.get((task, seq_len))
            if pair_rows is None:
                continue

            sample_scores: list[float] = []
            for row_index, row in enumerate(pair_rows):
                pred = row.get("pred")
                refs = row.get("outputs")
                if not isinstance(pred, str):
                    raise ValueError(
                        f"{_pair_name(task, seq_len)} row {row_index} has non-string pred"
                    )
                if not isinstance(refs, list) or not refs:
                    raise ValueError(
                        f"{_pair_name(task, seq_len)} row {row_index} needs non-empty outputs"
                    )
                if any(not isinstance(ref, str) for ref in refs):
                    raise ValueError(
                        f"{_pair_name(task, seq_len)} row {row_index} outputs must be strings"
                    )

                sample_score = sample_metric(pred, refs)
                sample_scores.append(sample_score)

                if not spec.depth_eligible:
                    continue
                token_position = row.get("token_position_answer")
                actual_length = row.get("length")
                if token_position is None or actual_length is None:
                    continue
                try:
                    actual_length_float = float(actual_length)
                    depth = float(token_position) / actual_length_float
                except (TypeError, ValueError, ZeroDivisionError) as exc:
                    raise ValueError(
                        f"{_pair_name(task, seq_len)} row {row_index} has invalid "
                        "token_position_answer/length"
                    ) from exc
                depth = min(max(depth, 0.0), 1.0)
                bin_index = min(int(depth * n_depth_bins), n_depth_bins - 1)
                cell = depth_cells[task][(bin_index, seq_len)]
                cell[0] += sample_score
                cell[1] += 1.0

            key = str(seq_len)
            scores[task][key] = round(
                100.0 * sum(sample_scores) / len(sample_scores), 2
            )
            counts[task][key] = len(sample_scores)
            scored_pairs.append(_pair_name(task, seq_len))

            partial, manifest_absent = _partial_pair_detail(
                pred_dir, task, seq_len, len(sample_scores)
            )
            if partial is not None:
                partial_pairs.append(partial)
            elif manifest_absent:
                unverified_pairs.append(_pair_name(task, seq_len))

    if not scored_pairs:
        raise ValueError("The requested RULER profile contains zero prediction samples")

    missing_pairs = [
        _pair_name(task, seq_len)
        for task in requested_tasks
        for seq_len in requested_seq_lens
        if (task, seq_len) not in rows_by_pair
    ]
    per_len_mean: dict[str, float | None] = {}
    for seq_len in requested_seq_lens:
        key = str(seq_len)
        present_scores = [
            score
            for task in requested_tasks
            if (score := scores[task][key]) is not None
        ]
        per_len_mean[key] = (
            round(sum(present_scores) / len(present_scores), 2)
            if present_scores
            else None
        )
    present_length_means = [
        score for score in per_len_mean.values() if score is not None
    ]
    overall_mean = (
        round(sum(present_length_means) / len(present_length_means), 2)
        if present_length_means
        else None
    )

    pooled_cells: dict[tuple[int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    for task_cells in depth_cells.values():
        for cell_key, (score_sum, count) in task_cells.items():
            pooled_cells[cell_key][0] += score_sum
            pooled_cells[cell_key][1] += count

    unexpected_pairs = sorted(
        _pair_name(task, seq_len)
        for task, seq_len in rows_by_pair
        if (task, seq_len) not in requested_pairs
    )
    manifest_sidecars_present = len(unverified_pairs) < len(scored_pairs)
    profile_unverified = bool(unverified_pairs)
    status = (
        "incomplete"
        if missing_pairs or partial_pairs or profile_unverified
        else "ok"
    )
    return {
        "benchmark": "ruler",
        "status": status,
        "tasks": requested_tasks,
        "seq_lens": requested_seq_lens,
        "metrics": {
            task: TASK_SPECS[task].metric for task in requested_tasks
        },
        "scores": scores,
        "counts": counts,
        "per_len_mean": per_len_mean,
        "overall_mean": overall_mean,
        "n_depth_bins": n_depth_bins,
        "depth_matrix_pooled": _depth_matrix(
            pooled_cells, requested_seq_lens, n_depth_bins
        ),
        "depth_matrix_by_task": {
            task: _depth_matrix(
                depth_cells[task], requested_seq_lens, n_depth_bins
            )
            for task in requested_tasks
            if task in depth_cells
        },
        "diagnostics": {
            "requested_pair_count": len(requested_pairs),
            "scored_pair_count": len(scored_pairs),
            "scored_pairs": scored_pairs,
            "missing_pairs": missing_pairs,
            "partial_pairs": partial_pairs,
            "unverified_pairs": unverified_pairs,
            "manifest_sidecars_present": manifest_sidecars_present,
            "unexpected_pairs": unexpected_pairs,
            "zero_sample_pairs": [],
        },
    }


def plot_depth_heatmap(
    result: dict[str, Any],
    out_path: Path,
    *,
    title: str,
    task: str | None = None,
) -> Path:
    """Render a pooled or per-task NIAH depth-by-length heatmap."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if task is None:
        source = result["depth_matrix_pooled"]
    else:
        by_task = result["depth_matrix_by_task"]
        if task not in by_task:
            raise ValueError(f"No depth observations for RULER task {task!r}")
        source = by_task[task]

    acc = np.array(
        [[np.nan if value is None else value for value in row] for row in source["acc"]],
        dtype=float,
    )
    seq_lens = result["seq_lens"]
    n_depth_bins = result["n_depth_bins"]

    fig, ax = plt.subplots(
        figsize=(1.6 + 1.15 * len(seq_lens), 4.8), constrained_layout=True
    )
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#dddddd")
    image = ax.imshow(
        np.ma.masked_invalid(acc),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=100.0,
        origin="upper",
    )
    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels([f"{seq_len // 1024}k" for seq_len in seq_lens])
    ax.set_yticks(range(n_depth_bins))
    ax.set_yticklabels(
        [
            f"{int(100 * index / n_depth_bins)}-"
            f"{int(100 * (index + 1) / n_depth_bins)}%"
            for index in range(n_depth_bins)
        ]
    )
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("needle depth")
    ax.set_title(title)
    for row_index in range(n_depth_bins):
        for column in range(len(seq_lens)):
            if not np.isnan(acc[row_index, column]):
                ax.text(
                    column,
                    row_index,
                    f"{acc[row_index, column]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    fig.colorbar(image, ax=ax, label="string-match accuracy (%)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
