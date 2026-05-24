"""Shared constants and config loading for Kitty LongBench evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config"

LONG_BENCH_DATASETS: tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_retrieval_en",
    "passage_count",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
)

LONG_BENCH_E_DATASETS: tuple[str, ...] = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

NO_CHAT_DATASETS: frozenset[str] = frozenset(
    {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
)


def load_json_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
