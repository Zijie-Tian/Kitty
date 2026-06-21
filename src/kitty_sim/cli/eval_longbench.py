"""CLI for Kitty LongBench generation."""

from __future__ import annotations

import argparse
import json

from kitty_sim.cli.utils_cli import update_parser
from kitty_sim.longbench.runner import run_longbench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Kitty/FP16 models on LongBench.")
    parser.add_argument("model", help="Model alias, HF model id, or local path")
    parser.add_argument("--model-path", default=None, help="Override model path for loading")
    parser.add_argument("--model-tag", default=None, help="Output model tag (variant suffix is added automatically)")
    parser.add_argument("--model-family", default=None, help="Prompt family override, e.g. qwen, llama3, llama2")
    parser.add_argument(
        "--variant",
        default="kitty",
        choices=[
            "fp16",
            "kitty",
            "shadowkv",
            "qlutattn_pertoken",
            "kivi",
            "kivi_star",
            "custom",
            "qlutattn_k1v4",
            "qlutattn-k1v4",
            "qlutattn_k184v4",
            "qlutattn-k184v4",
            "qlutattn_k125v4",
            "qlutattn-k125v4",
            "qlutattn_k125v4_pt",
            "qlutattn-k125v4-pt",
            "qlutattn_k185v4_pt",
            "qlutattn-k185v4-pt",
            "qlutattn_k168v4_pt",
            "qlutattn-k168v4-pt",
            "qlutattn-k1.68v4-pt",
            "qlutattn_k188v4_pt",
            "qlutattn-k188v4-pt",
        ],
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
    parser.add_argument("--dataset", default=None, help="Run one LongBench dataset only")
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
    parser.add_argument("--prompt-token-reserve", type=int, default=0)
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing dataset output before running")
    parser.add_argument("--no-strict-complete", dest="strict_complete", action="store_false", help="Do not fail on partial output")
    parser.set_defaults(strict_complete=True)
    parser.add_argument("--require-gpu1", action="store_true", help="Require CUDA_VISIBLE_DEVICES=1")
    parser.add_argument("--report-json", default=None, help="Write run metadata JSON")
    parser = update_parser(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_longbench(args)
    print(json.dumps({"prediction_dir": report["prediction_dir"], "status": report["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
