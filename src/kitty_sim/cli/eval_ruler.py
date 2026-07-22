"""CLI for Kitty NVIDIA RULER generation."""

from __future__ import annotations

import argparse
import json
import os

from kitty_sim.cli.eval_longbench import _VARIANT_CHOICES
from kitty_sim.cli.utils_cli import update_parser
from kitty_sim.ruler.runner import resolve_ruler_preflight, run_ruler
from kitty_sim.ruler.tasks import DEFAULT_SEQ_LENS

RULER_VARIANT_CHOICES = tuple(
    choice
    for choice in _VARIANT_CHOICES
    if "-" not in choice or choice.replace("-", "_") not in _VARIANT_CHOICES
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Kitty KV-cache variants on NVIDIA RULER."
    )
    parser.add_argument(
        "model", nargs="?", default=None, help="Model alias, HF model id, or local path"
    )
    parser.add_argument("--model-path", default=None, help="Override model path for loading")
    parser.add_argument("--model-tag", default=None, help="Output model tag/slug override")
    parser.add_argument("--model-family", default=None, help="Prompt family override")
    parser.add_argument("--variant", default="fp16", choices=RULER_VARIANT_CHOICES)
    parser.add_argument(
        "--resolve-config-only",
        action="store_true",
        help="CPU-only preflight; resolve variant, pair hashes, and output slugs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON")
    parser.add_argument(
        "--expected-preflight-hash",
        default=None,
        help="Shell preflight fingerprint; the worker must recompute it before model loading.",
    )
    parser.add_argument("--shadowkv-budget", type=int, default=None)
    parser.add_argument("--shadowkv-rank", type=int, default=None)
    parser.add_argument("--shadowkv-chunk-size", type=int, default=None)
    parser.add_argument("--quest-kernel", action="store_true")
    parser.add_argument("--quest-token-budget", type=int, default=None)
    parser.add_argument("--quest-skip-layers", type=int, default=0)
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated canonical RULER task names, or 'all' (default).",
    )
    parser.add_argument(
        "--seq-lens",
        default=",".join(str(length) for length in DEFAULT_SEQ_LENS),
        help="Comma-separated nominal context lengths matching the data directories.",
    )
    parser.add_argument(
        "--data-root", default=None, help="RULER data root (default: $RULER_DATA_ROOT)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction directory; defaults to ruler_out/[smoke/]<model>_<method>/pred.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=-1, help="Samples per pair; -1 means all"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Hard prompt-length cap; RULER prompts are never truncated.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Debug-only global generation-cap override; defaults to each task's registry cap.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-cuda-visible-devices",
        default=None,
        help="Fail unless CUDA_VISIBLE_DEVICES equals this exact value.",
    )
    parser.add_argument("--report-json", default=None, help="Write run metadata JSON")
    parser = update_parser(parser)
    parser.set_defaults(vbits=None)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.vbits is None:
        raw = os.environ.get("VBITS", "").strip()
        args.vbits = int(raw) if raw else 2
    return args


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    if args.resolve_config_only:
        if not args.model:
            args.model = "preflight-placeholder"
        payload = resolve_ruler_preflight(args)
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return
    if not args.model:
        raise SystemExit("model is required unless --resolve-config-only is set")
    report = run_ruler(args)
    print(
        json.dumps(
            {"prediction_dir": report["prediction_dir"], "status": report["status"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
