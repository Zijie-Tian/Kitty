"""CLI wrapper for Kitty LongBench scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from kitty_sim.longbench.scorer import score_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Kitty LongBench predictions.")
    parser.add_argument("--model", required=True, help="Model tag or prediction directory")
    parser.add_argument("--output-dir", default="longbench_out/pred", help="Prediction root directory")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--no-strict-complete", action="store_true", help="Allow scoring partial outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model)
    if not model_dir.exists():
        root = Path("pred_e" if args.e else args.output_dir)
        model_dir = root / args.model
    scores = score_directory(model_dir, is_longbench_e=args.e, strict_complete=not args.no_strict_complete)
    print(f"[score] wrote {model_dir / 'result.json'}")
    print(scores)


if __name__ == "__main__":
    main()
