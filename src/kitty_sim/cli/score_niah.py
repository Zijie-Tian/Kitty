"""Score RULER-NIAH prediction dirs and render depth x length heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kitty_sim.niah.scorer import plot_depth_heatmap, score_pred_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a NIAH pred dir (string_match_all) and plot heatmaps."
    )
    parser.add_argument("pred_dir", help="Directory with <task>__<len>.jsonl files")
    parser.add_argument("--n-depth-bins", type=int, default=10)
    parser.add_argument(
        "--heatmap-dir",
        default=None,
        help="Write depth x length heatmap PNGs here (pooled + per single-needle task)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Heatmap title prefix (default: pred dir's parent name)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pred_dir = Path(args.pred_dir)
    result = score_pred_dir(pred_dir, n_depth_bins=args.n_depth_bins)
    out_path = pred_dir / "result.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[score-niah] wrote {out_path}")
    print(json.dumps(
        {
            "scores": result["scores"],
            "per_len_mean": result["per_len_mean"],
            "overall_mean": result["overall_mean"],
        },
        ensure_ascii=False, indent=2,
    ))

    if args.heatmap_dir:
        heatmap_dir = Path(args.heatmap_dir)
        title_prefix = args.title or pred_dir.resolve().parent.name
        written = [
            plot_depth_heatmap(
                result,
                heatmap_dir / "niah_depth_heatmap_pooled.png",
                title=f"{title_prefix} (all tasks)",
            )
        ]
        for task in result["tasks"]:
            if task.startswith("niah_single"):
                written.append(
                    plot_depth_heatmap(
                        result,
                        heatmap_dir / f"niah_depth_heatmap_{task}.png",
                        title=f"{title_prefix} ({task})",
                        task=task,
                    )
                )
        for path in written:
            print(f"[score-niah] heatmap: {path}")


if __name__ == "__main__":
    main()
