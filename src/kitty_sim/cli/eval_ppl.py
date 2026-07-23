"""CLI for cache-aware streaming perplexity evaluation."""

from __future__ import annotations

import argparse
import json
import os

from kitty_sim.cli.eval_longbench import _VARIANT_CHOICES
from kitty_sim.cli.utils_cli import update_parser
from kitty_sim.ppl.data import DEFAULT_TEXT_FIELD
from kitty_sim.ppl.runner import resolve_ppl_preflight, run_ppl

PPL_VARIANT_CHOICES = tuple(
    choice
    for choice in _VARIANT_CHOICES
    if "-" not in choice or choice.replace("-", "_") not in _VARIANT_CHOICES
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Kitty KV-cache variants with cache-aware streaming PPL."
    )
    parser.add_argument(
        "model", nargs="?", default=None, help="Model id, alias, or local path"
    )
    parser.add_argument("--model-path", default=None, help="Local model path")
    parser.add_argument("--model-tag", default=None, help="Output model slug override")
    parser.add_argument("--model-family", default=None, help="Model family override")
    parser.add_argument("--variant", default="fp16", choices=PPL_VARIANT_CHOICES)
    parser.add_argument("--resolve-config-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit compact preflight JSON")
    parser.add_argument("--expected-preflight-hash", default=None)
    parser.add_argument("--shadowkv-budget", type=int, default=None)
    parser.add_argument("--shadowkv-rank", type=int, default=None)
    parser.add_argument("--shadowkv-chunk-size", type=int, default=None)
    parser.add_argument("--quest-kernel", action="store_true")
    parser.add_argument("--quest-token-budget", type=int, default=None)
    parser.add_argument("--quest-skip-layers", type=int, default=0)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Local EleutherAI document-level WikiText-2 test parquet",
    )
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--score-tokens", type=int, default=256)
    parser.add_argument(
        "--max-samples", type=int, default=-1, help="Window limit; <=0 means all"
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--output-dir", default=None, help="Arm prediction directory")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-cuda-visible-devices", default=None)
    parser.add_argument("--report-json", default=None)
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
    if not args.model:
        raise SystemExit("model is required")
    if args.resolve_config_only:
        payload = resolve_ppl_preflight(args)
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return
    report = run_ppl(args)
    print(
        json.dumps(
            {"prediction_dir": report["prediction_dir"], "status": report["status"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
