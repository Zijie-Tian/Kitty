"""Cache-aware streaming perplexity runner for Kitty KV-cache variants."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from kitty_sim.longbench.runner import (
    QLUTATTN_VARIANT,
    VariantConfig,
    _cache_factory,
    _maybe_enable_quest_kernel,
    _model_config_digest,
    _model_source_identity,
    _sha256_file,
    _shadowkv_cache,
    _stable_json_hash,
    build_variant,
    load_model_and_tokenizer,
    method_layout_slug,
    model_layout_slug,
    validate_qlutattn_model_config,
    validate_qlutattn_model_family,
    validate_qlutattn_preload,
    variant_semantic_hash,
    variant_semantic_payload,
)
from kitty_sim.longbench.templates import infer_model_family, use_fast_tokenizer
from kitty_sim.ruler.runner import TOKENIZER_CONFIG_FILES

from .data import (
    CORPUS_NAME,
    CORPUS_SPLIT,
    DEFAULT_TEXT_FIELD,
    DETOKENIZER_VERSION,
    TOKENIZATION_POLICY,
    WINDOW_POLICY,
    PPLWindow,
    PPLWindowPlan,
    build_window_plan,
    load_wikitext_documents,
)

PPL_PROTOCOL = "cache-streaming-suffix-v1"
PPL_SCHEMA_VERSION = 1
TOKENIZER_IDENTITY_ALGORITHM = "ppl-tokenizer-config-v1"
DECODE_CHUNK_SIZE = 1
BATCH_SIZE = 1
CORPUS_SLUG = "wikitext2"


@dataclass(frozen=True)
class WindowEvaluation:
    nll_sum: float
    scored_tokens: int
    prefill_cache_length: int
    final_cache_length: int
    decode_steps: int
    cache_length_monotonic: bool
    engagement: dict[str, Any]


def _model_family(args: Any) -> str:
    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None) or model
    return getattr(args, "model_family", None) or infer_model_family(
        getattr(args, "model_tag", None) or model, model_path
    )


def _model_slug(args: Any) -> str:
    model = str(getattr(args, "model", None) or "model")
    model_path = getattr(args, "model_path", None) or model
    return getattr(args, "model_tag", None) or model_layout_slug(model, model_path)


def resolve_data_path(args: Any) -> Path:
    value = (
        getattr(args, "data_path", None)
        or os.environ.get("PPL_DATA_PATH")
        or os.environ.get("KITTY_WIKITEXT2_TEST_PATH")
    )
    if not value:
        raise ValueError(
            "PPL requires a local document-level WikiText-2 test parquet: pass "
            "--data-path or set PPL_DATA_PATH/KITTY_WIKITEXT2_TEST_PATH"
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PPL data parquet does not exist: {path}")
    return path


def _tokenizer_file_hashes(model_path: str | os.PathLike[str]) -> dict[str, str]:
    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise ValueError(
            f"PPL requires a local model directory to hash tokenizer artifacts: {root}"
        )
    hashes = {
        name: _sha256_file(root / name)
        for name in TOKENIZER_CONFIG_FILES
        if (root / name).is_file()
    }
    if not hashes:
        raise ValueError(f"No canonical tokenizer files found under {root}")
    return dict(sorted(hashes.items()))


def _tokenizer_identity(
    config_hashes: dict[str, str],
    *,
    requested_use_fast: bool,
    tokenizer_class: str,
    is_fast: bool,
) -> str:
    return _stable_json_hash(
        {
            "algorithm": TOKENIZER_IDENTITY_ALGORITHM,
            "config_hashes": config_hashes,
            "requested_use_fast": requested_use_fast,
            "class": tokenizer_class,
            "is_fast": is_fast,
        }
    )


def _load_preflight_tokenizer(args: Any, model_family: str) -> tuple[Any, dict[str, Any]]:
    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None) or model
    if not model_path:
        raise ValueError("PPL preflight requires --model or --model-path")
    requested_use_fast = use_fast_tokenizer(model_family)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=requested_use_fast,
        local_files_only=bool(getattr(args, "local_files_only", False)),
    )
    config_hashes = _tokenizer_file_hashes(model_path)
    tokenizer_class = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    is_fast = bool(getattr(tokenizer, "is_fast", False))
    identity = _tokenizer_identity(
        config_hashes,
        requested_use_fast=requested_use_fast,
        tokenizer_class=tokenizer_class,
        is_fast=is_fast,
    )
    return tokenizer, {
        "algorithm": TOKENIZER_IDENTITY_ALGORITHM,
        "requested_use_fast": requested_use_fast,
        "class": tokenizer_class,
        "is_fast": is_fast,
        "config_hashes": config_hashes,
        "identity_sha256": identity,
    }


def _validate_runtime_tokenizer(tokenizer: Any, run_config: dict[str, Any]) -> None:
    expected = run_config["tokenizer"]
    actual_class = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    actual_is_fast = bool(getattr(tokenizer, "is_fast", False))
    if actual_class != expected["class"] or actual_is_fast != expected["is_fast"]:
        raise RuntimeError(
            "PPL runtime tokenizer differs from preflight: "
            f"expected class={expected['class']!r}, is_fast={expected['is_fast']!r}; "
            f"got class={actual_class!r}, is_fast={actual_is_fast!r}"
        )
    model_path = run_config["model_source_identity"]
    actual_hashes = _tokenizer_file_hashes(model_path)
    actual_identity = _tokenizer_identity(
        actual_hashes,
        requested_use_fast=expected["requested_use_fast"],
        tokenizer_class=actual_class,
        is_fast=actual_is_fast,
    )
    if (
        actual_hashes != expected["config_hashes"]
        or actual_identity != expected["identity_sha256"]
    ):
        raise RuntimeError("PPL runtime tokenizer artifacts differ from preflight")


def _load_window_plan(args: Any, tokenizer: Any) -> PPLWindowPlan:
    documents = load_wikitext_documents(
        resolve_data_path(args),
        text_field=str(getattr(args, "text_field", DEFAULT_TEXT_FIELD)),
    )
    return build_window_plan(
        documents,
        tokenizer,
        prefill_tokens=int(getattr(args, "prefill_tokens", 4096)),
        score_tokens=int(getattr(args, "score_tokens", 256)),
        max_samples=int(getattr(args, "max_samples", -1)),
    )


def _head_dim(config: Any) -> int:
    value = getattr(config, "head_dim", None)
    if value is not None:
        return int(value)
    return int(config.hidden_size) // int(config.num_attention_heads)


def _reject_unsupported(variant: VariantConfig, model_family: str) -> None:
    from kitty_sim.glm_kitty_patch import is_glm_family

    if is_glm_family(model_family):
        raise ValueError(
            "PPL currently requires an HF Cache model family; GLM legacy tuple-cache "
            "models are not supported"
        )


def _common_run_config(
    args: Any,
    *,
    model_family: str,
    tokenizer_metadata: dict[str, Any],
    plan: PPLWindowPlan,
    model_config_sha256: str,
) -> dict[str, Any]:
    data_path = resolve_data_path(args)
    prefill_tokens = int(getattr(args, "prefill_tokens", 4096))
    score_tokens = int(getattr(args, "score_tokens", 256))
    return {
        "schema_version": PPL_SCHEMA_VERSION,
        "benchmark": "ppl",
        "protocol": PPL_PROTOCOL,
        "model_id": getattr(args, "model", None),
        "model_source_identity": _model_source_identity(
            getattr(args, "model_path", None) or getattr(args, "model", None)
        ),
        "model_config_sha256": model_config_sha256,
        "model_slug": _model_slug(args),
        "model_family": model_family,
        "tokenizer": tokenizer_metadata,
        "corpus": CORPUS_NAME,
        "split": CORPUS_SPLIT,
        "text_field": str(getattr(args, "text_field", DEFAULT_TEXT_FIELD)),
        "data_file": data_path.name,
        "data_sha256": _sha256_file(data_path),
        "detokenizer_version": DETOKENIZER_VERSION,
        "tokenization_policy": TOKENIZATION_POLICY,
        "window_policy": WINDOW_POLICY,
        "selected_token_sha256": plan.selected_token_sha256,
        "prefill_tokens": prefill_tokens,
        "bridge_tokens": 1,
        "score_tokens": score_tokens,
        "window_tokens": plan.window_tokens,
        "decode_chunk_size": DECODE_CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "max_model_len": int(getattr(args, "max_model_len", 32768)),
        "max_samples": int(getattr(args, "max_samples", -1)),
        "total_documents": plan.total_documents,
        "eligible_documents": plan.eligible_documents,
        "skipped_short_documents": plan.skipped_short_documents,
        "available_windows": plan.available_windows,
        "expected_samples": plan.selected_windows,
        "expected_scored_tokens": plan.selected_windows * score_tokens,
        "torch_dtype": str(getattr(args, "torch_dtype", "float16")),
    }


def ppl_run_config_payload(args: Any, variant: VariantConfig) -> dict[str, Any]:
    """Build the stable PPL fingerprint without loading model weights."""

    model_family = _model_family(args)
    _reject_unsupported(variant, model_family)
    validate_qlutattn_model_family(variant, model_family)
    validate_qlutattn_preload(args, variant, model_family)

    prefill_tokens = int(getattr(args, "prefill_tokens", 4096))
    score_tokens = int(getattr(args, "score_tokens", 256))
    max_model_len = int(getattr(args, "max_model_len", 32768))
    if prefill_tokens <= 1 or score_tokens <= 0:
        raise ValueError("PPL requires prefill_tokens > 1 and score_tokens > 0")
    if prefill_tokens + score_tokens + 1 > max_model_len:
        raise ValueError(
            "PPL window exceeds max_model_len: "
            f"{prefill_tokens}+1+{score_tokens}>{max_model_len}"
        )

    model_path = getattr(args, "model_path", None) or getattr(args, "model", None)
    model_config_sha256 = _model_config_digest(model_path)
    if model_config_sha256 is None:
        raise ValueError(f"PPL requires local model config.json under {model_path!r}")
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=bool(getattr(args, "local_files_only", False)),
    )
    declared_max = int(getattr(config, "max_position_embeddings", max_model_len))
    if prefill_tokens + score_tokens + 1 > declared_max:
        raise ValueError(
            f"PPL window exceeds model max_position_embeddings={declared_max}"
        )
    if variant.name == QLUTATTN_VARIANT:
        settled = prefill_tokens - variant.sink_length - variant.buffer_length
        required = _head_dim(config)
        if settled < required:
            raise ValueError(
                "QLUTATTN PPL prefill is too short for canonical prompt-mean "
                f"calibration: settled={settled}, head_dim={required}"
            )
    if variant.shadowkv and prefill_tokens <= int(variant.sparse_budget):
        raise ValueError(
            "ShadowKV PPL prefill must exceed its sparse budget to exercise "
            f"retrieval: prefill={prefill_tokens}, budget={variant.sparse_budget}"
        )

    tokenizer, tokenizer_metadata = _load_preflight_tokenizer(args, model_family)
    plan = _load_window_plan(args, tokenizer)
    common = _common_run_config(
        args,
        model_family=model_family,
        tokenizer_metadata=tokenizer_metadata,
        plan=plan,
        model_config_sha256=model_config_sha256,
    )
    comparison_config_hash = _stable_json_hash(common)
    return {
        **common,
        "comparison_config_hash": comparison_config_hash,
        "canonical_variant": variant.name,
        "method_slug": method_layout_slug(variant),
        "variant_semantic_hash": variant_semantic_hash(variant),
        "mask_sha256": variant_semantic_payload(variant)["mask_sha256"],
    }


def ppl_run_config_hash(args: Any, variant: VariantConfig) -> str:
    return _stable_json_hash(ppl_run_config_payload(args, variant))


def resolve_ppl_preflight(args: Any) -> dict[str, Any]:
    variant = _maybe_enable_quest_kernel(build_variant(args), args)
    run_config = ppl_run_config_payload(args, variant)
    run_config_hash = _stable_json_hash(run_config)
    result = {
        "canonical_variant": variant.name,
        "method_slug": method_layout_slug(variant),
        "model_slug": run_config["model_slug"],
        "resolved_variant": asdict(variant),
        "variant_semantic_hash": run_config["variant_semantic_hash"],
        "mask_sha256": run_config["mask_sha256"],
        "comparison_config_hash": run_config["comparison_config_hash"],
        "run_config_hash": run_config_hash,
        "run_config": run_config,
    }
    result["preflight_hash"] = _stable_json_hash(
        {
            "schema_version": PPL_SCHEMA_VERSION,
            "canonical_variant": result["canonical_variant"],
            "method_slug": result["method_slug"],
            "model_slug": result["model_slug"],
            "run_config_hash": run_config_hash,
        }
    )
    return result


def _cache_seq_length(past_key_values: Any) -> int:
    if past_key_values is None:
        raise RuntimeError("Model did not return past_key_values with use_cache=True")
    getter = getattr(past_key_values, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        layer = past_key_values[0]
        key = layer[0] if isinstance(layer, (tuple, list)) else layer
        return int(key.shape[-2])
    raise RuntimeError(
        f"Cannot determine cache length for {type(past_key_values).__name__}"
    )


def _logit_slice_kwargs(model: Any) -> dict[str, int]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    if "logits_to_keep" in parameters:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {"num_logits_to_keep": 1}
    return {}


def _cache_engagement(cache: Any) -> dict[str, Any]:
    return {
        "prefill_updates": int(getattr(cache, "prefill_updates", 0)),
        "decode_updates": int(getattr(cache, "decode_updates", 0)),
        "k_quant_calls": int(getattr(cache, "k_quant_calls", 0)),
        "k_quantized_tokens": int(getattr(cache, "k_quantized_tokens", 0)),
        "last_k_quant_mode": getattr(cache, "last_k_quant_mode", None),
        "v_quant_calls": int(getattr(cache, "v_quant_calls", 0)),
        "v_quantized_tokens": int(getattr(cache, "v_quantized_tokens", 0)),
        "v_tile_blocks": int(getattr(cache, "v_tile_blocks", 0)),
        "last_v_quant_mode": getattr(cache, "last_v_quant_mode", None),
        "last_v_tile_channels": getattr(cache, "last_v_tile_channels", None),
        "k_prompt_mean_layers": len(getattr(cache, "k_pc_mean", {})),
    }


def _validate_engagement(
    variant: VariantConfig,
    engagement: dict[str, Any],
) -> None:
    if not variant.use_kitty:
        return
    if variant.shadowkv:
        if (
            engagement["shadowkv_prefill_calls_delta"] <= 0
            or engagement["shadowkv_decode_calls_delta"] <= 0
        ):
            raise RuntimeError(
                f"ShadowKV did not engage prefill+decode hooks: {engagement}"
            )
        return
    if engagement["decode_updates"] <= 0:
        raise RuntimeError(f"Kitty cache recorded no decode updates: {engagement}")
    if engagement["k_quantized_tokens"] <= 0 or engagement["v_quantized_tokens"] <= 0:
        raise RuntimeError(f"Kitty cache recorded no K/V quantized tokens: {engagement}")
    if variant.name == QLUTATTN_VARIANT:
        if (
            engagement["k_prompt_mean_layers"] <= 0
            or engagement["v_tile_blocks"] <= 0
            or engagement["last_v_quant_mode"] != "tile16_rescued"
            or engagement["last_v_tile_channels"] != variant.v_tile_channels
        ):
            raise RuntimeError(
                f"QLUTATTN prompt calibration/V tile path did not engage: {engagement}"
            )
    if variant.quest_kernel and engagement.get("quest_decode_calls_delta", 0) <= 0:
        raise RuntimeError(f"QUEST sparse decode did not engage: {engagement}")


def evaluate_window_streaming(
    *,
    model: Any,
    window: PPLWindow,
    variant: VariantConfig,
    prefill_tokens: int,
    score_tokens: int,
    runtime_stats: dict[str, Any] | None = None,
) -> WindowEvaluation:
    """Score one suffix after a dense prefill using one-token cache handoffs."""

    expected_tokens = prefill_tokens + score_tokens + 1
    if window.input_tokens != expected_tokens:
        raise ValueError(
            f"PPL window has {window.input_tokens} tokens, expected {expected_tokens}"
        )
    device = getattr(
        model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    all_ids = torch.tensor(window.token_ids, dtype=torch.long, device=device).unsqueeze(0)
    if variant.shadowkv:
        initial_cache = _shadowkv_cache(variant, model, prefill_tokens, score_tokens)
    else:
        initial_cache = _cache_factory(variant)

    shadow_prefill_before = int(runtime_stats.get("prefill_calls", 0)) if runtime_stats else 0
    shadow_decode_before = int(runtime_stats.get("decode_calls", 0)) if runtime_stats else 0
    quest_decode_before = int(runtime_stats.get("decode_calls", 0)) if runtime_stats else 0
    attention_mask = torch.ones((1, prefill_tokens), dtype=torch.long, device=device)
    forward_extras = _logit_slice_kwargs(model)

    with torch.inference_mode():
        outputs = model(
            input_ids=all_ids[:, :prefill_tokens],
            attention_mask=attention_mask,
            past_key_values=initial_cache,
            use_cache=True,
            return_dict=True,
            **forward_extras,
        )
        # Always use the returned cache, including custom mutable Cache objects.
        past_key_values = outputs.past_key_values
        previous_length = _cache_seq_length(past_key_values)
        prefill_cache_length = previous_length
        if prefill_cache_length != prefill_tokens:
            raise RuntimeError(
                "PPL prefill cache length mismatch: "
                f"expected={prefill_tokens}, actual={prefill_cache_length}"
            )

        nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        cache_length_monotonic = True
        for offset in range(score_tokens):
            token_index = prefill_tokens + offset
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((1, 1))), dim=-1
            )
            outputs = model(
                input_ids=all_ids[:, token_index : token_index + 1],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                **forward_extras,
            )
            logits = outputs.logits[:, -1, :].float()
            target = all_ids[:, token_index + 1]
            nll_sum = nll_sum + F.cross_entropy(
                logits, target, reduction="sum"
            ).to(torch.float64)

            next_past_key_values = outputs.past_key_values
            current_length = _cache_seq_length(next_past_key_values)
            if current_length != previous_length + DECODE_CHUNK_SIZE:
                cache_length_monotonic = False
                raise RuntimeError(
                    "PPL cache handoff did not grow by one token: "
                    f"previous={previous_length}, current={current_length}, offset={offset}"
                )
            past_key_values = next_past_key_values
            previous_length = current_length

    engagement: dict[str, Any]
    if variant.shadowkv:
        engagement = {
            "shadowkv_prefill_calls_delta": int(runtime_stats.get("prefill_calls", 0))
            - shadow_prefill_before,
            "shadowkv_decode_calls_delta": int(runtime_stats.get("decode_calls", 0))
            - shadow_decode_before,
            "shadowkv_last_selected_chunks": runtime_stats.get("last_selected_chunks"),
            "shadowkv_last_seq_length": runtime_stats.get("last_seq_length"),
        }
    else:
        engagement = _cache_engagement(past_key_values)
        if variant.quest_kernel:
            engagement["quest_decode_calls_delta"] = (
                int(runtime_stats.get("decode_calls", 0)) - quest_decode_before
            )
            engagement["quest_last_path"] = runtime_stats.get("last_path")
    _validate_engagement(variant, engagement)

    return WindowEvaluation(
        nll_sum=float(nll_sum.item()),
        scored_tokens=score_tokens,
        prefill_cache_length=prefill_cache_length,
        final_cache_length=previous_length,
        decode_steps=score_tokens,
        cache_length_monotonic=cache_length_monotonic,
        engagement=engagement,
    )


def _completed_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid PPL manifest {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"PPL manifest must contain a JSON object: {path}")
    return value


def _validate_existing_output(
    *,
    out_path: Path,
    manifest_path: Path,
    windows: tuple[PPLWindow, ...],
    run_config: dict[str, Any],
    run_config_hash: str,
) -> dict[str, Any] | None:
    completed = _completed_rows(out_path)
    expected = len(windows)
    if completed > expected:
        raise RuntimeError(
            f"Refusing PPL resume: {completed} rows exceed expected {expected}"
        )
    if not manifest_path.is_file():
        if completed:
            raise RuntimeError("Refusing PPL resume without a manifest")
        return None
    manifest = _load_manifest(manifest_path)
    embedded = manifest.get("run_config")
    if (
        manifest.get("run_config_hash") != run_config_hash
        or not isinstance(embedded, dict)
        or embedded != run_config
        or _stable_json_hash(embedded) != run_config_hash
        or int(manifest.get("expected_samples", -1)) != expected
    ):
        raise RuntimeError("Refusing PPL resume: manifest/run-config mismatch")

    if completed:
        with out_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for sample_idx, row in enumerate(rows):
            window = windows[sample_idx]
            if (
                int(row.get("sample_idx", -1)) != sample_idx
                or row.get("document_id") != window.document_id
                or int(row.get("start_token", -1)) != window.start_token
            ):
                raise RuntimeError(
                    f"Refusing PPL resume: row {sample_idx} does not match window plan"
                )
    if completed == expected:
        if (
            manifest.get("status") != "ok"
            or int(manifest.get("written_samples", -1)) != completed
            or manifest.get("jsonl_sha256") != _sha256_file(out_path)
        ):
            raise RuntimeError("Refusing completed PPL output with stale manifest/checksum")
    elif manifest.get("status") not in {"running", "partial"}:
        raise RuntimeError(
            f"Refusing PPL resume from status {manifest.get('status')!r}"
        )
    return manifest


def _empty_engagement() -> dict[str, Any]:
    return {
        "windows_observed": 0,
        "decode_steps": 0,
        "k_quant_calls": 0,
        "k_quantized_tokens": 0,
        "v_quant_calls": 0,
        "v_quantized_tokens": 0,
        "v_tile_blocks": 0,
        "k_prompt_mean_layers": 0,
        "shadowkv_prefill_calls": 0,
        "shadowkv_decode_calls": 0,
        "quest_decode_calls": 0,
        "last_k_quant_mode": None,
        "last_v_quant_mode": None,
        "last_v_tile_channels": None,
    }


def _merge_engagement(total: dict[str, Any], row: dict[str, Any], decode_steps: int) -> None:
    total["windows_observed"] += 1
    total["decode_steps"] += decode_steps
    for source, target in (
        ("k_quant_calls", "k_quant_calls"),
        ("k_quantized_tokens", "k_quantized_tokens"),
        ("v_quant_calls", "v_quant_calls"),
        ("v_quantized_tokens", "v_quantized_tokens"),
        ("v_tile_blocks", "v_tile_blocks"),
        ("k_prompt_mean_layers", "k_prompt_mean_layers"),
        ("shadowkv_prefill_calls_delta", "shadowkv_prefill_calls"),
        ("shadowkv_decode_calls_delta", "shadowkv_decode_calls"),
        ("quest_decode_calls_delta", "quest_decode_calls"),
    ):
        total[target] += int(row.get(source, 0))
    for key in ("last_k_quant_mode", "last_v_quant_mode", "last_v_tile_channels"):
        if row.get(key) is not None:
            total[key] = row[key]


def evaluate_ppl_windows(
    *,
    model: Any,
    windows: tuple[PPLWindow, ...],
    variant: VariantConfig,
    prefill_tokens: int,
    score_tokens: int,
    out_path: Path,
    model_name: str,
    model_family: str,
    run_config: dict[str, Any],
    expected_run_config_hash: str,
    runtime_stats: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate deterministic PPL windows with manifest-backed resume."""

    if _stable_json_hash(run_config) != expected_run_config_hash:
        raise ValueError("expected_run_config_hash does not match PPL run_config")
    if int(run_config["expected_samples"]) != len(windows):
        raise ValueError("PPL run_config expected_samples does not match window plan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    if overwrite:
        out_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    previous = _validate_existing_output(
        out_path=out_path,
        manifest_path=manifest_path,
        windows=windows,
        run_config=run_config,
        run_config_hash=expected_run_config_hash,
    )
    completed = _completed_rows(out_path)
    if completed == len(windows):
        assert previous is not None
        print(f"[skip] {out_path.name}: all {completed}/{len(windows)} windows present")
        return previous

    engagement = _empty_engagement()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if previous is not None:
        prior_engagement = previous.get("engagement")
        if isinstance(prior_engagement, dict):
            engagement.update(prior_engagement)
        created_at = str(previous.get("created_at") or created_at)

    manifest: dict[str, Any] = {
        "status": "running",
        "benchmark": "ppl",
        "protocol": PPL_PROTOCOL,
        "corpus": CORPUS_NAME,
        "expected_samples": len(windows),
        "written_samples": completed,
        "written_this_run": 0,
        "expected_scored_tokens": int(run_config["expected_scored_tokens"]),
        "output_path": str(out_path),
        "variant": asdict(variant),
        "method_slug": method_layout_slug(variant),
        "variant_semantic_hash": variant_semantic_hash(variant),
        "mask_sha256": variant_semantic_payload(variant)["mask_sha256"],
        "model_name": model_name,
        "model_family": model_family,
        "comparison_config_hash": run_config["comparison_config_hash"],
        "run_config_hash": expected_run_config_hash,
        "run_config": run_config,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engagement": engagement,
        "created_at": created_at,
        "updated_at": created_at,
    }
    _write_json_atomic(manifest_path, manifest)

    written_now = 0
    try:
        for sample_idx in tqdm(range(completed, len(windows)), desc="ppl"):
            window = windows[sample_idx]
            started = time.perf_counter()
            result = evaluate_window_streaming(
                model=model,
                window=window,
                variant=variant,
                prefill_tokens=prefill_tokens,
                score_tokens=score_tokens,
                runtime_stats=runtime_stats,
            )
            elapsed = time.perf_counter() - started
            if not math.isfinite(result.nll_sum) or result.nll_sum < 0:
                raise RuntimeError(
                    f"PPL window {sample_idx} produced invalid NLL {result.nll_sum}"
                )
            row = {
                "sample_idx": sample_idx,
                "document_id": window.document_id,
                "source_index": window.source_index,
                "window_index": window.window_index,
                "start_token": window.start_token,
                "input_tokens": window.input_tokens,
                "prefill_tokens": prefill_tokens,
                "scored_tokens": result.scored_tokens,
                "nll_sum": result.nll_sum,
                "avg_nll": result.nll_sum / result.scored_tokens,
                "decode_steps": result.decode_steps,
                "prefill_cache_length": result.prefill_cache_length,
                "final_cache_length": result.final_cache_length,
                "cache_length_monotonic": result.cache_length_monotonic,
                "elapsed_seconds": round(elapsed, 6),
                "engagement": result.engagement,
            }
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_now += 1
            _merge_engagement(engagement, result.engagement, result.decode_steps)
            manifest["written_samples"] = _completed_rows(out_path)
            manifest["written_this_run"] = written_now
            manifest["engagement"] = engagement
            manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _write_json_atomic(manifest_path, manifest)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        manifest["status"] = "partial"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["written_samples"] = _completed_rows(out_path)
        manifest["written_this_run"] = written_now
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if out_path.is_file():
            manifest["jsonl_sha256"] = _sha256_file(out_path)
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest["written_samples"] = _completed_rows(out_path)
    manifest["written_this_run"] = written_now
    manifest["status"] = (
        "ok" if manifest["written_samples"] == len(windows) else "partial"
    )
    manifest["jsonl_sha256"] = _sha256_file(out_path)
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json_atomic(manifest_path, manifest)
    if manifest["status"] != "ok":
        raise RuntimeError(
            f"Incomplete PPL output: {manifest['written_samples']}/{len(windows)}"
        )
    return manifest


def _install_runtime_hooks(
    model: Any,
    variant: VariantConfig,
) -> dict[str, Any] | None:
    if variant.shadowkv:
        from kitty_sim.shadowkv_sim import ShadowKVSimConfig, install_shadowkv_sim

        return install_shadowkv_sim(
            model,
            ShadowKVSimConfig(
                sparse_budget=variant.sparse_budget,
                rank=variant.rank,
                chunk_size=variant.chunk_size,
            ),
        )
    if variant.quest_kernel:
        from kitty_sim.quest_kernel import QuestConfig, install_quest_kernel

        return install_quest_kernel(
            model,
            QuestConfig(
                page_size=16,
                token_budget=variant.quest_token_budget,
                skip_layers=variant.quest_skip_layers,
                sink_length=variant.sink_length,
                recent_length=variant.buffer_length,
            ),
        )
    return None


def run_ppl(args: Any) -> dict[str, Any]:
    required_cvd = getattr(args, "require_cuda_visible_devices", None)
    if required_cvd is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != required_cvd:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={required_cvd!r} required, got "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    preflight = resolve_ppl_preflight(args)
    expected_preflight_hash = getattr(args, "expected_preflight_hash", None)
    if expected_preflight_hash and expected_preflight_hash != preflight["preflight_hash"]:
        raise RuntimeError(
            "PPL worker preflight hash mismatch before model loading: "
            f"expected={expected_preflight_hash}, actual={preflight['preflight_hash']}"
        )

    variant = _maybe_enable_quest_kernel(build_variant(args), args)
    model_family = _model_family(args)
    run_config = preflight["run_config"]
    output_root = getattr(args, "output_dir", None)
    if output_root:
        pred_dir = Path(output_root)
    else:
        base = Path("ppl_out")
        if int(getattr(args, "max_samples", -1)) > 0:
            base = base / "smoke"
        pred_dir = base / f"{preflight['model_slug']}_{preflight['method_slug']}" / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model_path = getattr(args, "model_path", None) or getattr(args, "model", None)
    print("=" * 80)
    print("Kitty cache-aware streaming PPL")
    print(f"model={getattr(args, 'model', None)} path={model_path} family={model_family}")
    print(f"variant={variant.tag} protocol={PPL_PROTOCOL}")
    print(
        f"prefill={run_config['prefill_tokens']} score={run_config['score_tokens']} "
        f"windows={run_config['expected_samples']}"
    )
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"output={pred_dir}")
    print("=" * 80)

    model_obj, tokenizer, resolved_model_path = load_model_and_tokenizer(
        getattr(args, "model"),
        model_path=model_path,
        model_family=model_family,
        dtype=getattr(args, "torch_dtype", "float16"),
        local_files_only=bool(getattr(args, "local_files_only", False)),
    )
    _validate_runtime_tokenizer(tokenizer, run_config)
    validate_qlutattn_model_config(variant, model_obj.config, model_obj.dtype)
    if variant.promote_ratio_per_layer:
        n_layers = int(getattr(model_obj.config, "num_hidden_layers", 0))
        invalid = [
            layer_idx
            for layer_idx, _ in variant.promote_ratio_per_layer
            if not 0 <= layer_idx < n_layers
        ]
        if invalid:
            raise ValueError(
                f"promote-ratio config references invalid layers {invalid} for {n_layers} layers"
            )

    runtime_plan = _load_window_plan(args, tokenizer)
    if (
        runtime_plan.selected_token_sha256 != run_config["selected_token_sha256"]
        or runtime_plan.selected_windows != run_config["expected_samples"]
    ):
        raise RuntimeError("PPL runtime token windows differ from CPU preflight")
    runtime_stats = _install_runtime_hooks(model_obj, variant)

    try:
        manifest = evaluate_ppl_windows(
            model=model_obj,
            windows=runtime_plan.windows,
            variant=variant,
            prefill_tokens=int(run_config["prefill_tokens"]),
            score_tokens=int(run_config["score_tokens"]),
            out_path=pred_dir / f"{CORPUS_SLUG}.jsonl",
            model_name=resolved_model_path,
            model_family=model_family,
            run_config=run_config,
            expected_run_config_hash=preflight["run_config_hash"],
            runtime_stats=runtime_stats,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    finally:
        model_obj.to("cpu")
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "status": manifest["status"],
        "benchmark": "ppl",
        "prediction_dir": str(pred_dir),
        "model": getattr(args, "model", None),
        "model_path": resolved_model_path,
        "model_slug": preflight["model_slug"],
        "model_family": model_family,
        "method_slug": preflight["method_slug"],
        "variant": asdict(variant),
        "variant_semantic_hash": preflight["variant_semantic_hash"],
        "comparison_config_hash": preflight["comparison_config_hash"],
        "preflight_hash": preflight["preflight_hash"],
        "run_config_hash": preflight["run_config_hash"],
        "manifest": manifest,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report_json = getattr(args, "report_json", None)
    if report_json:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(report_path, report)
    return report
