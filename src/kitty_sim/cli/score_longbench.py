"""CLI wrapper for Kitty LongBench scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from kitty_sim.longbench.scorer import score_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Kitty LongBench predictions.")
    parser.add_argument("--model", required=True, help="Model tag or prediction directory")
    parser.add_argument("--output-dir", default=None, help="Optional prediction root for legacy model-tag lookup")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--no-strict-complete", action="store_true", help="Allow scoring partial outputs")
    return parser


def resolve_model_dir(model: str, output_dir: str, *, is_longbench_e: bool = False) -> Path:
    """Resolve a model tag or explicit prediction directory for scoring.

    Strings containing a path separator, or absolute paths, are treated as
    explicit directories. Only bare model tags are resolved relative to the
    prediction root. This avoids turning ``longbench_out/pref/foo`` into
    ``longbench_out/pred/longbench_out/pref/foo`` when callers use the newer
    flat output convention.
    """

    model_dir = Path(model).expanduser()
    if model_dir.is_absolute() or len(model_dir.parts) > 1:
        return model_dir
    if model_dir.exists():
        return model_dir
    root = Path("pred_e" if is_longbench_e else output_dir)
    return root / model_dir


def main() -> None:
    args = build_parser().parse_args()
    model_dir = resolve_model_dir(args.model, args.output_dir, is_longbench_e=args.e)
    scores = score_directory(model_dir, is_longbench_e=args.e, strict_complete=not args.no_strict_complete)
    print(f"[score] wrote {model_dir / 'result.json'}")
    print(scores)


if __name__ == "__main__":
    main()
