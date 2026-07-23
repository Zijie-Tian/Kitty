"""Score and optionally compare cache-aware PPL arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kitty_sim.ppl.scorer import (
    compare_results,
    score_pred_dir,
    write_comparison_csv,
    write_comparison_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score one or more Kitty PPL arms.")
    parser.add_argument("pred_dirs", nargs="+", help="PPL arm prediction directories")
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--comparison-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = [score_pred_dir(Path(path)) for path in args.pred_dirs]
    payload: dict[str, object] = {"results": results}
    if len(results) > 1:
        comparison = compare_results(results)
        payload["comparison"] = comparison
        if args.comparison_json:
            write_comparison_json(comparison, args.comparison_json)
        if args.comparison_csv:
            write_comparison_csv(comparison, args.comparison_csv)
    elif args.comparison_json or args.comparison_csv:
        raise SystemExit("comparison outputs require at least two pred directories")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
