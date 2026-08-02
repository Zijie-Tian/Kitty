"""CLI for Kitty LongBench generation."""

from __future__ import annotations

import argparse
import json
import os

from kitty_sim.cli.utils_cli import update_parser
from kitty_sim.longbench.runner import resolve_longbench_preflight, run_longbench

_VARIANT_CHOICES = [
    "fp16",
    "kitty",
    "shadowkv",
    "qlutattn",
    "kivi",
    "kivi_star",
    "llamacpp_q40",
    "llamacpp-q40",
    "llamacpp_q40_star",
    "llamacpp-q40-star",
    "custom",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Kitty/FP16 models on LongBench.")
    parser.add_argument("model", nargs="?", default=None, help="Model alias, HF model id, or local path")
    parser.add_argument("--model-path", default=None, help="Override model path for loading")
    parser.add_argument("--model-tag", default=None, help="Output model tag (variant suffix is added automatically)")
    parser.add_argument("--model-family", default=None, help="Prompt family override, e.g. qwen, llama3, llama2")
    parser.add_argument(
        "--variant",
        default="kitty",
        choices=_VARIANT_CHOICES,
    )
    parser.add_argument(
        "--resolve-config-only",
        action="store_true",
        help="No-GPU preflight: resolve variant/slug/hash and exit (use with --json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit preflight / report as JSON.",
    )
    parser.add_argument(
        "--expected-run-config-hash",
        default=None,
        help="Shell-preflight fingerprint; worker recomputes and must match before model loading.",
    )
    # ShadowKV sim controls (variant=shadowkv only).
    parser.add_argument(
        "--shadowkv-budget",
        type=int,
        default=None,
        help="ShadowKV sparse token budget (chunks selected per decode step). Defaults to 2048.",
    )
    parser.add_argument(
        "--shadowkv-rank",
        type=int,
        default=None,
        help="ShadowKV SVD rank for low-rank key compression. Defaults to 160.",
    )
    parser.add_argument(
        "--shadowkv-chunk-size",
        type=int,
        default=None,
        help="ShadowKV landmark chunk size (tokens per chunk). Defaults to 8.",
    )
    parser.add_argument(
        "--quest-kernel",
        action="store_true",
        help="Overlay Triton QUEST sparse decode on a Kitty/KIVI fake-quant variant (not qlutattn).",
    )
    parser.add_argument(
        "--quest-token-budget",
        type=int,
        default=None,
        help="QUEST sparse token budget for --quest-kernel. Defaults to 2048.",
    )
    parser.add_argument(
        "--quest-skip-layers",
        type=int,
        default=0,
        help="Disable QUEST sparse selection for the first N layers when --quest-kernel is used.",
    )
    parser.add_argument("--dataset", default=None, help="Run one LongBench dataset only")
    parser.add_argument(
        "--datasets-csv",
        default=None,
        help=argparse.SUPPRESS,  # scheduler preflight: resolve several dataset hashes once
    )
    parser.add_argument("--e", action="store_true", help="Evaluate LongBench-E")
    parser.add_argument("--data-root", default=None, help="LongBench root containing data/*.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction directory/root; omitted uses the normalized longbench_out/<model>-<method>/pred layout",
    )
    parser.add_argument(
        "--flat-output-dir",
        action="store_true",
        help="Write dataset JSONL/result files directly under --output-dir instead of appending model-tag/variant",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Samples per dataset; -1 means all")
    parser.add_argument("--max-model-len", type=int, default=None, help="Prompt truncation length")
    parser.add_argument("--default-max-model-len", type=int, default=3500, help="Fallback prompt length for unknown model aliases")
    parser.add_argument("--max-gen", type=int, default=None, help="Override dataset generation length")
    parser.add_argument(
        "--qwen-thinking",
        action="store_true",
        help=(
            "Use the fixed deterministic Qwen3 thinking sampling policy and "
            "score only the final-answer suffix."
        ),
    )
    parser.add_argument("--prompt-token-reserve", type=int, default=0)
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing dataset output before running")
    parser.add_argument("--no-strict-complete", dest="strict_complete", action="store_false", help="Do not fail on partial output")
    parser.set_defaults(strict_complete=True)
    parser.add_argument("--require-gpu1", action="store_true", help="Require CUDA_VISIBLE_DEVICES=1")
    parser.add_argument("--report-json", default=None, help="Write run metadata JSON")
    parser = update_parser(parser)
    # Preserve whether --vbits was explicitly supplied so the final resolver can
    # implement CLI > VBITS env > default.  update_parser's global default is 2,
    # which otherwise makes an explicit stale VBITS=4 impossible to detect.
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
        payload = resolve_longbench_preflight(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2 if not args.json else None))
        return
    if args.datasets_csv:
        raise SystemExit("--datasets-csv is reserved for --resolve-config-only/preflight")
    if not args.model:
        raise SystemExit("model is required unless --resolve-config-only is set")
    report = run_longbench(args)
    print(json.dumps({"prediction_dir": report["prediction_dir"], "status": report["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
