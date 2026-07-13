"""CLI for Kitty RULER-NIAH generation."""

from __future__ import annotations

import argparse
import json
import os

from kitty_sim.cli.eval_longbench import _VARIANT_CHOICES
from kitty_sim.cli.utils_cli import update_parser
from kitty_sim.niah.runner import DEFAULT_MAX_NEW_TOKENS, run_niah


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Kitty/FP16 models on RULER-NIAH (needle-in-a-haystack)."
    )
    parser.add_argument("model", help="Model alias, HF model id, or local path")
    parser.add_argument("--model-path", default=None, help="Override model path for loading")
    parser.add_argument("--model-tag", default=None, help="Output model tag/slug override")
    parser.add_argument("--model-family", default=None, help="Prompt family override, e.g. llama3.2")
    parser.add_argument("--variant", default="fp16", choices=_VARIANT_CHOICES)
    parser.add_argument(
        "--v-tile-channels",
        type=int,
        default=None,
        help="Channel block C for rescued V tile16cC variants (required for *_vtile16).",
    )
    parser.add_argument(
        "--tasks",
        default="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1",
        help="CSV of RULER niah task names",
    )
    parser.add_argument(
        "--seq-lens",
        default="4096,8192,16384,32768",
        help="CSV of nominal context lengths (must match data dirs)",
    )
    parser.add_argument("--data-root", default=None, help="NIAH data root (default $NIAH_DATA_ROOT)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction dir; omitted uses niah_out/[smoke/]<model>_<method>/pred",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Samples per (task,len); -1 = all")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Hard prompt-length cap (never truncates)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="Generation budget per sample (RULER niah default 128)",
    )
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing pair output before running")
    parser.add_argument(
        "--require-cuda-visible-devices",
        default=None,
        help="Fail unless CUDA_VISIBLE_DEVICES equals this string (e.g. '0')",
    )
    parser.add_argument("--report-json", default=None, help="Write run metadata JSON")
    parser = update_parser(parser)
    # Keep --vbits resolution identical to eval_longbench: CLI > VBITS env > 2.
    parser.set_defaults(vbits=None)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.vbits is None:
        raw = os.environ.get("VBITS", "").strip()
        args.vbits = int(raw) if raw else 2
    # Fields build_variant() looks up but which have no NIAH CLI flags.
    args.quest_kernel = False
    args.quest_token_budget = None
    args.quest_skip_layers = 0
    args.shadowkv_budget = None
    args.shadowkv_rank = None
    args.shadowkv_chunk_size = None
    return args


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    report = run_niah(args)
    print(json.dumps(
        {"prediction_dir": report["prediction_dir"], "status": report["status"]},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
