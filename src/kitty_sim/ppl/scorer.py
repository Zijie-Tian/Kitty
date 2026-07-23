"""Strict aggregation and cross-method comparison for cache-aware PPL."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from kitty_sim.longbench.runner import _sha256_file, _stable_json_hash

from .runner import CORPUS_SLUG, PPL_PROTOCOL


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def _validate_engagement(manifest: dict[str, Any]) -> None:
    variant = manifest.get("variant")
    engagement = manifest.get("engagement")
    if not isinstance(variant, dict) or not isinstance(engagement, dict):
        raise RuntimeError("PPL manifest lacks variant/engagement metadata")
    if not bool(variant.get("use_kitty")):
        return
    if bool(variant.get("shadowkv")):
        if (
            int(engagement.get("shadowkv_prefill_calls", 0)) <= 0
            or int(engagement.get("shadowkv_decode_calls", 0)) <= 0
        ):
            raise RuntimeError("PPL ShadowKV manifest lacks positive hook engagement")
        return
    if (
        int(engagement.get("k_quantized_tokens", 0)) <= 0
        or int(engagement.get("v_quantized_tokens", 0)) <= 0
        or int(engagement.get("decode_steps", 0)) <= 0
    ):
        raise RuntimeError("PPL quantized manifest lacks K/V/decode engagement")
    if variant.get("name") == "qlutattn":
        if (
            int(engagement.get("k_prompt_mean_layers", 0)) <= 0
            or int(engagement.get("v_tile_blocks", 0)) <= 0
            or engagement.get("last_v_quant_mode") != "tile16_rescued"
            or int(engagement.get("last_v_tile_channels") or 0)
            != int(variant.get("v_tile_channels") or 0)
        ):
            raise RuntimeError(
                "PPL QLUTATTN manifest lacks prompt-mean/rescued V-tile engagement"
            )
    if bool(variant.get("quest_kernel")) and int(
        engagement.get("quest_decode_calls", 0)
    ) <= 0:
        raise RuntimeError("PPL QUEST manifest lacks sparse decode engagement")


def score_pred_dir(pred_dir: str | Path) -> dict[str, Any]:
    """Validate and score one PPL arm, writing ``result.json``."""

    pred_dir = Path(pred_dir)
    jsonl_path = pred_dir / f"{CORPUS_SLUG}.jsonl"
    manifest_path = jsonl_path.with_suffix(".manifest.json")
    if not jsonl_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"PPL arm requires {jsonl_path.name} and {manifest_path.name} under {pred_dir}"
        )
    manifest = _load_json_object(manifest_path)
    if manifest.get("benchmark") != "ppl" or manifest.get("protocol") != PPL_PROTOCOL:
        raise RuntimeError("PPL manifest benchmark/protocol mismatch")
    if manifest.get("status") != "ok":
        raise RuntimeError(f"PPL manifest is not complete: status={manifest.get('status')!r}")

    run_config = manifest.get("run_config")
    run_config_hash = manifest.get("run_config_hash")
    if (
        not isinstance(run_config, dict)
        or not isinstance(run_config_hash, str)
        or _stable_json_hash(run_config) != run_config_hash
    ):
        raise RuntimeError("PPL manifest run_config hash is invalid")
    if manifest.get("comparison_config_hash") != run_config.get(
        "comparison_config_hash"
    ):
        raise RuntimeError("PPL manifest comparison_config_hash mismatch")
    if manifest.get("jsonl_sha256") != _sha256_file(jsonl_path):
        raise RuntimeError("PPL JSONL checksum does not match manifest")

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid PPL JSONL {jsonl_path}:{line_no}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"PPL row {line_no} is not an object")
            rows.append(row)

    expected_samples = int(run_config.get("expected_samples", -1))
    if (
        len(rows) != expected_samples
        or int(manifest.get("written_samples", -1)) != expected_samples
    ):
        raise RuntimeError(
            f"PPL row count mismatch: rows={len(rows)}, expected={expected_samples}"
        )

    total_nll = 0.0
    total_tokens = 0
    expected_input_tokens = int(run_config["window_tokens"])
    expected_score_tokens = int(run_config["score_tokens"])
    for sample_idx, row in enumerate(rows):
        if int(row.get("sample_idx", -1)) != sample_idx:
            raise RuntimeError(f"PPL row {sample_idx} has a non-sequential sample_idx")
        if int(row.get("input_tokens", -1)) != expected_input_tokens:
            raise RuntimeError(f"PPL row {sample_idx} input_tokens mismatch")
        scored_tokens = int(row.get("scored_tokens", 0))
        nll_sum = float(row.get("nll_sum", float("nan")))
        if scored_tokens != expected_score_tokens:
            raise RuntimeError(f"PPL row {sample_idx} scored_tokens mismatch")
        if not math.isfinite(nll_sum) or nll_sum < 0:
            raise RuntimeError(f"PPL row {sample_idx} has invalid nll_sum={nll_sum}")
        if not bool(row.get("cache_length_monotonic", False)):
            raise RuntimeError(f"PPL row {sample_idx} lacks monotonic cache evidence")
        prefill_length = int(row.get("prefill_cache_length", -1))
        final_length = int(row.get("final_cache_length", -1))
        if prefill_length != int(run_config["prefill_tokens"]):
            raise RuntimeError(f"PPL row {sample_idx} prefill cache length mismatch")
        if final_length != prefill_length + scored_tokens:
            raise RuntimeError(f"PPL row {sample_idx} final cache length mismatch")
        total_nll += nll_sum
        total_tokens += scored_tokens

    if total_tokens <= 0 or total_tokens != int(run_config["expected_scored_tokens"]):
        raise RuntimeError(
            f"PPL scored-token total mismatch: {total_tokens} vs "
            f"{run_config['expected_scored_tokens']}"
        )
    avg_nll = total_nll / total_tokens
    token_ppl = math.exp(avg_nll)
    if not math.isfinite(token_ppl):
        raise RuntimeError(f"PPL aggregation produced non-finite value {token_ppl}")
    _validate_engagement(manifest)

    result = {
        "status": "ok",
        "benchmark": "ppl",
        "protocol": PPL_PROTOCOL,
        "corpus": run_config["corpus"],
        "split": run_config["split"],
        "model_slug": run_config["model_slug"],
        "method_slug": manifest["method_slug"],
        "variant": manifest["variant"],
        "run_config_hash": run_config_hash,
        "comparison_config_hash": run_config["comparison_config_hash"],
        "windows": len(rows),
        "scored_tokens": total_tokens,
        "total_nll": total_nll,
        "avg_nll": avg_nll,
        "token_ppl": token_ppl,
        "engagement": manifest["engagement"],
        "jsonl_sha256": manifest["jsonl_sha256"],
    }
    _write_json_atomic(pred_dir / "result.json", result)
    return result


def compare_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    if len(rows) < 2:
        raise ValueError("PPL comparison requires at least two method results")
    hashes = {row.get("comparison_config_hash") for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(
            "PPL methods do not share one comparison_config_hash; targets/config differ"
        )
    fp16_rows = [row for row in rows if row.get("method_slug") == "fp16"]
    if len(fp16_rows) != 1:
        raise RuntimeError("PPL comparison requires exactly one fp16 baseline")
    baseline = fp16_rows[0]
    baseline_nll = float(baseline["avg_nll"])
    baseline_ppl = float(baseline["token_ppl"])

    comparison_rows = []
    for row in sorted(rows, key=lambda item: (item.get("method_slug") != "fp16", item.get("method_slug", ""))):
        avg_nll = float(row["avg_nll"])
        token_ppl = float(row["token_ppl"])
        delta_nll = avg_nll - baseline_nll
        comparison_rows.append(
            {
                "method": row["method_slug"],
                "avg_nll": avg_nll,
                "token_ppl": token_ppl,
                "delta_nll_vs_fp16": delta_nll,
                "ppl_delta_vs_fp16": token_ppl - baseline_ppl,
                "ppl_ratio_vs_fp16": math.exp(delta_nll),
                "windows": int(row["windows"]),
                "scored_tokens": int(row["scored_tokens"]),
                "run_config_hash": row["run_config_hash"],
            }
        )
    return {
        "status": "ok",
        "benchmark": "ppl-comparison",
        "protocol": PPL_PROTOCOL,
        "comparison_config_hash": hashes.pop(),
        "model_slug": baseline["model_slug"],
        "baseline": "fp16",
        "rows": comparison_rows,
    }


def write_comparison_csv(comparison: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "avg_nll",
        "token_ppl",
        "delta_nll_vs_fp16",
        "ppl_delta_vs_fp16",
        "ppl_ratio_vs_fp16",
        "windows",
        "scored_tokens",
        "run_config_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison["rows"])
    return path


def write_comparison_json(comparison: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, comparison)
    return path
