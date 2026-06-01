"""LongBench scoring with optional strict completeness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import DATASET_TO_METRIC


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _manifest_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".manifest.json")


def scorer(dataset: str, predictions: list[str], answers: list[Any], all_classes: Any) -> float:
    metric = DATASET_TO_METRIC[dataset]
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, metric(prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2) if predictions else 0.0


def scorer_e(
    dataset: str,
    predictions: list[str],
    answers: list[Any],
    lengths: list[int],
    all_classes: Any,
) -> dict[str, float]:
    metric = DATASET_TO_METRIC[dataset]
    scores: dict[str, list[float]] = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, metric(prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    return {key: round(100 * float(np.mean(value)), 2) if value else 0.0 for key, value in scores.items()}


def score_directory(
    model_dir: str | Path,
    *,
    is_longbench_e: bool = False,
    strict_complete: bool = True,
    output_name: str = "result.json",
) -> dict[str, Any]:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {path}")

    scores: dict[str, Any] = {}
    incomplete: dict[str, dict[str, Any]] = {}
    jsonl_files = sorted(path.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl prediction files found in {path}")

    for jsonl_path in jsonl_files:
        dataset = jsonl_path.stem
        metric_dataset = dataset.removesuffix("_e") if dataset.endswith("_e") else dataset
        if metric_dataset not in DATASET_TO_METRIC:
            continue
        rows = _read_jsonl(jsonl_path)
        manifest_file = _manifest_path(jsonl_path)
        manifest: dict[str, Any] = {}
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            expected = int(manifest.get("expected_samples", len(rows)))
            failed = manifest.get("failed_sample_ids", [])
            if len(rows) != expected or failed:
                incomplete[dataset] = {
                    "expected": expected,
                    "actual": len(rows),
                    "failed_sample_ids": failed,
                    "manifest": str(manifest_file),
                }
        elif strict_complete:
            incomplete[dataset] = {
                "expected": None,
                "actual": len(rows),
                "failed_sample_ids": [],
                "manifest": "missing",
            }

        predictions = [row["pred"] for row in rows]
        answers = [row["answers"] for row in rows]
        all_classes = rows[-1].get("all_classes", []) if rows else []
        if is_longbench_e:
            lengths = [int(row.get("length", 0)) for row in rows]
            scores[metric_dataset] = scorer_e(metric_dataset, predictions, answers, lengths, all_classes)
        else:
            scores[metric_dataset] = scorer(metric_dataset, predictions, answers, all_classes)

    result_path = path / output_name
    if incomplete and strict_complete:
        partial_path = path / "result.partial.json"
        partial_path.write_text(
            json.dumps({"scores": scores, "incomplete": incomplete}, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        raise RuntimeError(f"Incomplete LongBench predictions under {path}; wrote {partial_path}")

    result_path.write_text(json.dumps(scores, ensure_ascii=False, indent=4), encoding="utf-8")
    stale_partial_path = path / "result.partial.json"
    if output_name == "result.json" and stale_partial_path.exists():
        stale_partial_path.unlink()
    return scores
