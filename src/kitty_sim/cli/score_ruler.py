"""Score one Kitty RULER prediction arm."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kitty_sim.ruler.scorer import plot_depth_heatmap, score_pred_dir
from kitty_sim.ruler.tasks import DEFAULT_SEQ_LENS, DEFAULT_TASKS


def _parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("task list must be non-empty")
    if "all" in tasks:
        if tasks != ["all"]:
            raise argparse.ArgumentTypeError(
                "'all' cannot be combined with explicit task names"
            )
        return list(DEFAULT_TASKS)
    return tasks


def _parse_seq_lens(value: str) -> list[int]:
    try:
        seq_lens = [
            int(item.strip()) for item in value.split(",") if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sequence lengths must be comma-separated integers"
        ) from exc
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise argparse.ArgumentTypeError(
            "sequence lengths must be non-empty positive integers"
        )
    return seq_lens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score <task>__<length>.jsonl files for one RULER arm. "
            "The requested profile is complete only when every pair is present."
        )
    )
    parser.add_argument("pred_dir", help="Arm pred directory containing pair JSONLs")
    parser.add_argument(
        "--tasks",
        type=_parse_tasks,
        default=list(DEFAULT_TASKS),
        help="Comma-separated task names (default: all 13 RULER tasks)",
    )
    parser.add_argument(
        "--seq-lens",
        type=_parse_seq_lens,
        default=list(DEFAULT_SEQ_LENS),
        help="Comma-separated nominal lengths (default: 4096,8192,16384,32768)",
    )
    parser.add_argument("--n-depth-bins", type=int, default=10)
    parser.add_argument(
        "--heatmap-dir",
        default=None,
        help="Optional logs directory for pooled and per-task NIAH heatmaps",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Heatmap title prefix (default: arm directory name)",
    )
    return parser


def _format_score(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def write_summary_csv(
    result: dict[str, Any],
    out_path: Path,
    *,
    arm: str,
) -> Path:
    """Write the one-row-per-arm RULER summary table."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq_lens = result["seq_lens"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", *[str(seq_len) for seq_len in seq_lens], "Avg"])
        writer.writerow(
            [
                arm,
                *[
                    _format_score(result["per_len_mean"].get(str(seq_len)))
                    for seq_len in seq_lens
                ],
                _format_score(result["overall_mean"]),
            ]
        )
    return out_path


def _has_depth_observations(result: dict[str, Any]) -> bool:
    return any(
        count
        for row in result["depth_matrix_pooled"]["n"]
        for count in row
    )


def _write_heatmaps(
    result: dict[str, Any],
    heatmap_dir: Path,
    *,
    title_prefix: str,
) -> list[Path]:
    if not _has_depth_observations(result):
        return []

    heatmap_dir = Path(heatmap_dir)
    written = [
        plot_depth_heatmap(
            result,
            heatmap_dir / "ruler_niah_depth_heatmap_pooled.png",
            title=f"{title_prefix} (NIAH pooled)",
        )
    ]
    for task in result["depth_matrix_by_task"]:
        written.append(
            plot_depth_heatmap(
                result,
                heatmap_dir / f"ruler_niah_depth_heatmap_{task}.png",
                title=f"{title_prefix} ({task})",
                task=task,
            )
        )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pred_dir = Path(args.pred_dir).expanduser()
    result = score_pred_dir(
        pred_dir,
        tasks=args.tasks,
        seq_lens=args.seq_lens,
        n_depth_bins=args.n_depth_bins,
    )

    result_path = pred_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    arm = pred_dir.parent.name if pred_dir.name == "pred" else pred_dir.name
    summary_path = write_summary_csv(
        result,
        pred_dir / "summary.csv",
        arm=arm,
    )
    print(f"[score-ruler] wrote {result_path}")
    print(f"[score-ruler] wrote {summary_path}")
    print(
        json.dumps(
            {
                "status": result["status"],
                "per_len_mean": result["per_len_mean"],
                "overall_mean": result["overall_mean"],
                "diagnostics": result["diagnostics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.heatmap_dir:
        title_prefix = args.title or arm
        for path in _write_heatmaps(
            result,
            Path(args.heatmap_dir).expanduser(),
            title_prefix=title_prefix,
        ):
            print(f"[score-ruler] heatmap: {path}")

    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
