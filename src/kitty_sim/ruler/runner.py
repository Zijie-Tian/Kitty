"""NVIDIA RULER generation runner for Kitty KV-cache variants."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from kitty_sim.longbench.runner import (
    QLUTATTN_VARIANT,
    VariantConfig,
    _cache_factory,
    _shadowkv_cache,
    _maybe_enable_quest_kernel,
    _model_config_digest,
    _sha256_file,
    _stable_json_hash,
    build_variant,
    load_model_and_tokenizer,
    method_layout_slug,
    model_layout_slug,
    validate_qlutattn_model_config,
    validate_qlutattn_preload,
    variant_semantic_hash,
    variant_semantic_payload,
)
from kitty_sim.longbench.templates import (
    build_chat,
    infer_model_family,
    use_fast_tokenizer,
)

from .data import (
    load_ruler_data_manifest,
    load_ruler_records,
    resolve_ruler_file,
    resolve_ruler_manifest,
)
from .tasks import TaskSpec, get_task_spec, resolve_task_names

PROMPT_CONTRACT = (
    'build_chat(tokenizer, record["input"], model_family, '
    'template_date="26 Jul 2024") + record["gen_prefix"]'
)
TOKENIZER_IDENTITY_ALGORITHM = "ruler-tokenizer-config-v2"
TOKENIZER_CONFIG_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
)


def _model_source_identity(model_path: str | None) -> str | None:
    if not model_path:
        return None
    path = Path(model_path).expanduser()
    return str(path.resolve()) if path.exists() else str(model_path)


def _model_slug(args: Any) -> str:
    model = str(getattr(args, "model", None) or "model")
    model_path = getattr(args, "model_path", None) or model
    return getattr(args, "model_tag", None) or model_layout_slug(model, model_path)


def _model_family(args: Any) -> str:
    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None) or model
    return getattr(args, "model_family", None) or infer_model_family(
        getattr(args, "model_tag", None) or model, model_path
    )


def _parse_seq_lens(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            seq_lens = tuple(
                int(part.strip()) for part in value.split(",") if part.strip()
            )
        except ValueError as exc:
            raise ValueError(f"Invalid --seq-lens CSV: {value!r}") from exc
    else:
        seq_lens = tuple(int(length) for length in value)
    if not seq_lens or any(length <= 0 for length in seq_lens):
        raise ValueError("--seq-lens must contain positive integers")
    if len(set(seq_lens)) != len(seq_lens):
        raise ValueError(f"--seq-lens contains duplicates: {seq_lens}")
    return seq_lens


def _effective_max_new_tokens(args: Any, spec: TaskSpec) -> int:
    override = getattr(args, "max_new_tokens", None)
    value = spec.max_new_tokens if override is None else int(override)
    if value <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {value}")
    return value


def build_ruler_prompt(
    tokenizer: Any, record: dict[str, Any], model_family: str
) -> str:
    """Chat-wrap the input reproducibly and append the generation prefix once."""

    return (
        build_chat(
            tokenizer,
            record["input"],
            model_family,
            template_date="26 Jul 2024",
        )
        + record["gen_prefix"]
    )


def pair_name(task: str, seq_len: int) -> str:
    return f"{task}__{int(seq_len)}"


def select_task_shard(
    tasks: tuple[str, ...],
    shard_index: int | None,
    shard_count: int | None,
) -> tuple[str, ...]:
    """Select one deterministic round-robin task shard."""

    if shard_index is None and shard_count is None:
        return tasks
    if shard_index is None or shard_count is None:
        raise ValueError("--task-shard-index and --task-shard-count must be set together")
    if shard_count <= 0:
        raise ValueError("--task-shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"--task-shard-index must be in [0, {shard_count}), got {shard_index}"
        )
    shard = tasks[shard_index::shard_count]
    if not shard:
        raise ValueError(
            f"task shard {shard_index}/{shard_count} is empty for {len(tasks)} tasks"
        )
    return shard


def _prompt_source_sha256() -> str:
    source = inspect.getsource(build_ruler_prompt).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _templates_source_sha256() -> str:
    source_path = inspect.getsourcefile(build_chat)
    if not source_path:
        raise RuntimeError("Cannot resolve LongBench template source for RULER preflight")
    return _sha256_file(source_path)


def tokenizer_config_hashes(model_path: str | os.PathLike[str]) -> dict[str, str]:
    """Hash the canonical local tokenizer-file set in deterministic order."""

    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise ValueError(
            f"Cannot hash tokenizer files for remote/non-local model {str(model_path)!r}; "
            "provide a local --model-path"
        )
    hashes = {
        name: _sha256_file(root / name)
        for name in TOKENIZER_CONFIG_FILES
        if (root / name).is_file()
    }
    if not hashes:
        raise ValueError(f"No canonical tokenizer files found under target model {root}")
    return hashes


def tokenizer_identity_sha256(
    config_hashes: dict[str, str],
    *,
    requested_use_fast: bool,
    tokenizer_class: str,
    is_fast: bool,
) -> str:
    payload = {
        "algorithm": TOKENIZER_IDENTITY_ALGORITHM,
        "config_hashes": dict(sorted(config_hashes.items())),
        "requested_use_fast": requested_use_fast,
        "class": tokenizer_class,
        "is_fast": is_fast,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_tokenizer_provenance(
    manifest: dict[str, Any],
    model_path: str | None,
    model_family: str,
    manifest_path: Path,
) -> dict[str, Any]:
    """Match tokenizer artifacts and loader policy before loading model weights."""

    tokenizer_metadata = manifest.get("tokenizer")
    identity_sha = manifest.get("tokenizer_identity_sha256")
    declared_hashes = manifest.get("tokenizer_config_hashes")
    if not isinstance(tokenizer_metadata, dict):
        raise ValueError(f"RULER data manifest lacks tokenizer metadata: {manifest_path}")
    if tokenizer_metadata.get("algorithm") != TOKENIZER_IDENTITY_ALGORITHM:
        raise ValueError(
            f"RULER data manifest {manifest_path} uses tokenizer algorithm "
            f"{tokenizer_metadata.get('algorithm')!r}; expected "
            f"{TOKENIZER_IDENTITY_ALGORITHM!r}"
        )
    requested_use_fast = tokenizer_metadata.get("requested_use_fast")
    tokenizer_class = tokenizer_metadata.get("class")
    is_fast = tokenizer_metadata.get("is_fast")
    if type(requested_use_fast) is not bool:
        raise ValueError(
            f"RULER data manifest has invalid requested_use_fast: {manifest_path}"
        )
    if not isinstance(tokenizer_class, str) or not tokenizer_class:
        raise ValueError(f"RULER data manifest has invalid tokenizer class: {manifest_path}")
    if type(is_fast) is not bool:
        raise ValueError(f"RULER data manifest has invalid is_fast: {manifest_path}")
    expected_use_fast = use_fast_tokenizer(model_family)
    if requested_use_fast != expected_use_fast:
        raise ValueError(
            f"RULER tokenizer policy mismatch in {manifest_path}: data requested "
            f"use_fast={requested_use_fast}, model family {model_family!r} requires "
            f"use_fast={expected_use_fast}"
        )
    if not isinstance(identity_sha, str) or len(identity_sha) != 64:
        raise ValueError(
            f"RULER data manifest lacks a valid tokenizer_identity_sha256: {manifest_path}"
        )
    if not isinstance(declared_hashes, dict) or not declared_hashes:
        raise ValueError(
            f"RULER data manifest lacks tokenizer_config_hashes: {manifest_path}"
        )
    normalized_hashes: dict[str, str] = {}
    for name, digest in declared_hashes.items():
        if name not in TOKENIZER_CONFIG_FILES:
            raise ValueError(
                f"Unexpected tokenizer file {name!r} in RULER data manifest {manifest_path}"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest.lower() != digest
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(
                f"Invalid tokenizer hash for {name!r} in RULER data manifest "
                f"{manifest_path}"
            )
        normalized_hashes[name] = digest
    normalized_hashes = dict(sorted(normalized_hashes.items()))
    identity_fields = {
        "requested_use_fast": requested_use_fast,
        "tokenizer_class": tokenizer_class,
        "is_fast": is_fast,
    }
    declared_identity = tokenizer_identity_sha256(
        normalized_hashes, **identity_fields
    )
    if identity_sha != declared_identity:
        raise ValueError(
            f"RULER data manifest {manifest_path} tokenizer identity does not match "
            "its artifacts/class/fast-mode metadata"
        )

    if not model_path:
        raise ValueError(
            f"Cannot verify RULER tokenizer provenance from {manifest_path}: no target "
            "model_path was provided"
        )
    target_hashes = tokenizer_config_hashes(model_path)
    target_identity = tokenizer_identity_sha256(target_hashes, **identity_fields)
    if target_hashes != normalized_hashes or target_identity != identity_sha:
        raise ValueError(
            f"RULER tokenizer mismatch: data identity={identity_sha}, target "
            f"identity={target_identity}. Data generated with tokenizer metadata "
            f"{tokenizer_metadata!r} cannot be reused with model {model_path!r}."
        )
    return {
        "data_tokenizer": tokenizer_metadata,
        "tokenizer_identity_sha256": target_identity,
        "tokenizer_config_hashes": target_hashes,
    }


def _validate_runtime_tokenizer(
    tokenizer: Any, preflight: dict[str, Any], model_family: str
) -> None:
    """Require the loaded tokenizer implementation to match every data manifest."""

    actual_class = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    actual_is_fast = bool(getattr(tokenizer, "is_fast", False))
    expected_use_fast = use_fast_tokenizer(model_family)
    for pair_name, pair in preflight["pairs"].items():
        metadata = pair["run_config"].get("data_tokenizer")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"RULER preflight pair {pair_name} lacks tokenizer metadata")
        if metadata.get("requested_use_fast") != expected_use_fast:
            raise RuntimeError(
                f"RULER preflight pair {pair_name} requested use_fast="
                f"{metadata.get('requested_use_fast')!r}, expected {expected_use_fast}"
            )
        if metadata.get("class") != actual_class or metadata.get("is_fast") != actual_is_fast:
            raise RuntimeError(
                f"RULER runtime tokenizer mismatch for {pair_name}: data used "
                f"class={metadata.get('class')!r}, is_fast={metadata.get('is_fast')!r}; "
                f"runtime loaded class={actual_class!r}, is_fast={actual_is_fast!r}"
            )


def ruler_run_config_payload(
    args: Any,
    variant: VariantConfig,
    task: str,
    seq_len: int,
    *,
    task_spec: TaskSpec | None = None,
) -> dict[str, Any]:
    """Build a stable per-pair fingerprint without loading model weights."""

    spec = task_spec or get_task_spec(task)
    if spec.name != task:
        raise ValueError(f"TaskSpec name {spec.name!r} does not match task {task!r}")
    nominal_length = int(seq_len)
    max_model_len = int(getattr(args, "max_model_len", 32768))
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {max_model_len}")
    if nominal_length > max_model_len:
        raise ValueError(
            f"RULER length {nominal_length} exceeds max_model_len={max_model_len}"
        )

    data_root = getattr(args, "data_root", None)
    data_path = resolve_ruler_file(task, nominal_length, data_root)
    records = load_ruler_records(task, nominal_length, data_root)
    data_manifest = load_ruler_data_manifest(task, nominal_length, data_root)
    data_manifest_path = resolve_ruler_manifest(task, nominal_length, data_root)
    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None) or model
    local_model_path = str(Path(model_path).expanduser()) if model_path else None
    model_family = _model_family(args)
    model_config_sha256 = _model_config_digest(local_model_path)
    if model_config_sha256 is None:
        raise ValueError(
            f"Target model config.json is missing under local model path {model_path!r}"
        )
    tokenizer_provenance = _validate_tokenizer_provenance(
        data_manifest, model_path, model_family, data_manifest_path
    )

    max_samples = int(getattr(args, "max_samples", -1))
    total_samples = len(records)
    expected_samples = min(total_samples, max_samples) if max_samples > 0 else total_samples
    max_new_tokens = _effective_max_new_tokens(args, spec)
    return {
        "schema_version": 2,
        "benchmark": "ruler",
        "variant_semantic_hash": variant_semantic_hash(variant),
        "model_id": model,
        "model_source_identity": _model_source_identity(model_path),
        "model_config_sha256": model_config_sha256,
        "model_slug": _model_slug(args),
        "model_family": model_family,
        "task": task,
        "task_spec": asdict(spec),
        "seq_len": nominal_length,
        "data_sha256": _sha256_file(data_path),
        "data_manifest_sha256": _sha256_file(data_manifest_path),
        "data_tokenizer": tokenizer_provenance["data_tokenizer"],
        "tokenizer_identity_sha256": tokenizer_provenance[
            "tokenizer_identity_sha256"
        ],
        "tokenizer_config_hashes": tokenizer_provenance["tokenizer_config_hashes"],
        "dataset_total_samples": total_samples,
        "max_samples": max_samples,
        "expected_samples": expected_samples,
        "max_model_len": max_model_len,
        "max_new_tokens": max_new_tokens,
        "max_new_tokens_override": getattr(args, "max_new_tokens", None),
        "torch_dtype": str(getattr(args, "torch_dtype", "float16")),
        "prompt_contract": PROMPT_CONTRACT,
        "prompt_source_sha256": _prompt_source_sha256(),
        "templates_source_sha256": _templates_source_sha256(),
    }


def ruler_run_config_hash(
    args: Any,
    variant: VariantConfig,
    task: str,
    seq_len: int,
    *,
    task_spec: TaskSpec | None = None,
) -> str:
    return _stable_json_hash(
        ruler_run_config_payload(
            args, variant, task, seq_len, task_spec=task_spec
        )
    )


def resolve_ruler_preflight(args: Any) -> dict[str, Any]:
    """Resolve canonical variant, output slugs, and every pair hash on CPU."""

    variant = _maybe_enable_quest_kernel(build_variant(args), args)
    tasks = resolve_task_names(getattr(args, "tasks", None))
    seq_lens = _parse_seq_lens(getattr(args, "seq_lens", ""))
    model_family = _model_family(args)
    validate_qlutattn_preload(args, variant, model_family)

    resolved_pairs: dict[str, dict[str, Any]] = {}
    for task in tasks:
        spec = get_task_spec(task)
        for seq_len in seq_lens:
            run_config = ruler_run_config_payload(
                args, variant, task, seq_len, task_spec=spec
            )
            name = pair_name(task, seq_len)
            resolved_pairs[name] = {
                "task": task,
                "seq_len": seq_len,
                "run_config_hash": _stable_json_hash(run_config),
                "expected_rows": run_config["expected_samples"],
                "data_sha256": run_config["data_sha256"],
                "run_config": run_config,
            }

    semantic_payload = variant_semantic_payload(variant)
    result: dict[str, Any] = {
        "canonical_variant": variant.name,
        "method_slug": method_layout_slug(variant),
        "model_slug": _model_slug(args),
        "resolved_variant": asdict(variant),
        "variant_semantic_hash": _stable_json_hash(semantic_payload),
        "mask_sha256": semantic_payload["mask_sha256"],
        "tasks": list(tasks),
        "seq_lens": list(seq_lens),
        "pairs": resolved_pairs,
    }
    result["preflight_hash"] = _stable_json_hash(
        {
            "schema_version": 1,
            "canonical_variant": result["canonical_variant"],
            "method_slug": result["method_slug"],
            "model_slug": result["model_slug"],
            "variant_semantic_hash": result["variant_semantic_hash"],
            "pairs": {
                name: pair["run_config_hash"] for name, pair in resolved_pairs.items()
            },
        }
    )
    return result


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


def _load_output_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid output manifest {path}: {exc.msg}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Output manifest must be a JSON object: {path}")
    return manifest


def _validate_existing_output(
    *,
    out_path: Path,
    manifest_path: Path,
    completed: int,
    expected_samples: int,
    run_config: dict[str, Any],
    run_config_hash: str,
) -> dict[str, Any] | None:
    if completed > expected_samples:
        raise RuntimeError(
            f"Refusing to reuse {out_path.name}: {completed} rows exceed expected "
            f"{expected_samples}"
        )
    if not manifest_path.is_file():
        if completed:
            raise RuntimeError(
                f"Refusing to reuse {out_path.name} without a matching manifest"
            )
        return None

    manifest = _load_output_manifest(manifest_path)
    embedded_config = manifest.get("run_config")
    if (
        manifest.get("run_config_hash") != run_config_hash
        or not isinstance(embedded_config, dict)
        or _stable_json_hash(embedded_config) != run_config_hash
        or embedded_config != run_config
        or int(manifest.get("expected_samples", -1)) != expected_samples
    ):
        raise RuntimeError(
            f"Refusing to reuse {out_path.name}: manifest/run-config mismatch"
        )

    manifest_written = int(manifest.get("written_samples", -1))
    if manifest_written < 0 or manifest_written > completed:
        raise RuntimeError(
            f"Refusing to reuse {out_path.name}: manifest row count is stale"
        )
    if completed == expected_samples:
        if manifest.get("status") != "ok" or manifest_written != completed:
            raise RuntimeError(
                f"Refusing to reuse completed {out_path.name}: manifest is not complete"
            )
        output_sha = manifest.get("jsonl_sha256")
        if (
            not isinstance(output_sha, str)
            or len(output_sha) != 64
            or output_sha != _sha256_file(out_path)
        ):
            raise RuntimeError(
                f"Refusing to reuse completed {out_path.name}: output checksum mismatch"
            )
    elif manifest.get("status") not in {"running", "partial"}:
        raise RuntimeError(
            f"Refusing to resume {out_path.name}: manifest status is "
            f"{manifest.get('status')!r}"
        )
    return manifest


def _reject_unsupported(variant: VariantConfig, model_family: str) -> None:
    if getattr(variant, "quest_kernel", False):
        raise ValueError(
            "The RULER runner does not support QUEST overlays; use a canonical "
            "standalone LongBench variant."
        )
    from kitty_sim.glm_kitty_patch import is_glm_family

    if is_glm_family(model_family):
        raise ValueError(
            "GLM legacy-cache models are not wired into the RULER runner; use an "
            "HF-Cache model family."
        )


def _new_engagement() -> dict[str, Any]:
    return {
        "kitty_cache_checked": False,
        "kitty_cache_engaged": False,
        "kitty_cache_seq_length": None,
        "samples_observed": 0,
        "v_quant_calls": 0,
        "v_quantized_tokens": 0,
        "v_tile_blocks": 0,
        "last_v_quant_mode": None,
        "last_v_tile_channels": None,
        "shadowkv_installed": None,
        "shadowkv_prefill_calls": 0,
        "shadowkv_decode_calls": 0,
        "shadowkv_decode_calls_at_pair_start": 0,
        "shadowkv_decode_calls_delta": 0,
        "shadowkv_last_selected_chunks": None,
        "shadowkv_last_seq_length": None,
    }


def generate_ruler_pair(
    *,
    model: Any,
    tokenizer: Any,
    task: str,
    seq_len: int,
    records: list[dict[str, Any]],
    task_spec: TaskSpec,
    max_model_len: int,
    max_new_tokens: int,
    out_path: Path,
    variant: VariantConfig,
    model_family: str,
    model_name: str,
    run_config: dict[str, Any],
    expected_run_config_hash: str,
    kitty_stats: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one RULER pair with strict manifest-backed resume semantics."""

    if task_spec.name != task:
        raise ValueError(f"TaskSpec name {task_spec.name!r} does not match task {task!r}")
    if _stable_json_hash(run_config) != expected_run_config_hash:
        raise ValueError("expected_run_config_hash does not match run_config")
    if run_config.get("task_spec") != asdict(task_spec):
        raise ValueError("run_config task_spec does not match generation task_spec")
    if int(run_config.get("max_new_tokens", -1)) != int(max_new_tokens):
        raise ValueError("run_config max_new_tokens does not match generation cap")
    if int(run_config.get("expected_samples", -1)) != len(records):
        raise ValueError("run_config expected_samples does not match loaded records")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    if overwrite:
        out_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    expected_samples = len(records)
    completed = _completed_rows(out_path)
    previous = _validate_existing_output(
        out_path=out_path,
        manifest_path=manifest_path,
        completed=completed,
        expected_samples=expected_samples,
        run_config=run_config,
        run_config_hash=expected_run_config_hash,
    )
    if completed == expected_samples:
        assert previous is not None
        print(f"[skip] {out_path.name}: all {completed}/{expected_samples} samples present")
        return previous
    if completed:
        print(
            f"[resume] {out_path.name}: skip {completed}, remaining "
            f"{expected_samples - completed}"
        )
    if variant.shadowkv and kitty_stats is None:
        raise RuntimeError(
            "ShadowKV generation requires the run-level install_shadowkv_sim hook"
        )

    engagement = _new_engagement()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if previous is not None:
        prior_engagement = previous.get("engagement")
        if isinstance(prior_engagement, dict):
            engagement.update(prior_engagement)
        created_at = str(previous.get("created_at") or created_at)
    shadow_pair_decode_start = (
        int(kitty_stats.get("decode_calls", 0))
        if variant.shadowkv and kitty_stats is not None
        else 0
    )
    if variant.shadowkv and kitty_stats is not None:
        engagement["shadowkv_installed"] = int(kitty_stats.get("installed", 0))
        engagement["shadowkv_prefill_calls"] = int(
            kitty_stats.get("prefill_calls", 0)
        )
        engagement["shadowkv_decode_calls"] = int(
            kitty_stats.get("decode_calls", 0)
        )
        engagement["shadowkv_decode_calls_at_pair_start"] = shadow_pair_decode_start
        engagement["shadowkv_decode_calls_delta"] = 0
        engagement["shadowkv_last_selected_chunks"] = kitty_stats.get(
            "last_selected_chunks"
        )
        engagement["shadowkv_last_seq_length"] = kitty_stats.get("last_seq_length")

    manifest: dict[str, Any] = {
        "status": "running",
        "benchmark": "ruler",
        "task": task,
        "task_spec": asdict(task_spec),
        "seq_len": int(seq_len),
        "expected_samples": expected_samples,
        "written_samples": completed,
        "written_this_run": 0,
        "output_path": str(out_path),
        "variant": asdict(variant),
        "variant_semantic_hash": variant_semantic_hash(variant),
        "mask_sha256": variant_semantic_payload(variant)["mask_sha256"],
        "model_name": model_name,
        "model_family": model_family,
        "model_config_sha256": run_config["model_config_sha256"],
        "data_sha256": run_config["data_sha256"],
        "data_manifest_sha256": run_config["data_manifest_sha256"],
        "tokenizer_identity_sha256": run_config["tokenizer_identity_sha256"],
        "torch_dtype": run_config["torch_dtype"],
        "max_model_len": int(max_model_len),
        "max_new_tokens": int(max_new_tokens),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engagement": engagement,
        "run_config_hash": expected_run_config_hash,
        "run_config": run_config,
        "created_at": created_at,
        "updated_at": created_at,
    }
    _write_json_atomic(manifest_path, manifest)

    device = getattr(
        model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    kitty_checked = False
    written_now = 0
    pair = pair_name(task, seq_len)
    for sample_idx in tqdm(range(completed, expected_samples), desc=pair):
        record = records[sample_idx]
        prompt = build_ruler_prompt(tokenizer, record, model_family)
        bos = getattr(tokenizer, "bos_token", None)
        add_special_tokens = not (bos and prompt.startswith(bos))
        inputs = tokenizer(
            prompt,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        ).to(device)
        inputs.pop("token_type_ids", None)
        context_length = int(inputs.input_ids.shape[-1])
        if context_length > max_model_len:
            raise RuntimeError(
                f"{pair} sample {sample_idx}: prompt has {context_length} tokens > "
                f"max_model_len={max_model_len}. RULER prompts are never truncated; "
                "regenerate the data with a larger margin."
            )

        shadow_decode_before = (
            int(kitty_stats.get("decode_calls", 0))
            if variant.shadowkv and kitty_stats is not None
            else 0
        )
        if variant.shadowkv:
            kv_cache = _shadowkv_cache(
                variant, model, context_length, max_new_tokens
            )
        else:
            kv_cache = _cache_factory(variant)
        generation_kwargs: dict[str, Any] = {
            "past_key_values": kv_cache,
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 1.0,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "use_cache": True,
        }
        if variant.shadowkv:
            generation_kwargs["disable_compile"] = True
            generation_kwargs["temperature"] = None
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)[0]
        elapsed = time.perf_counter() - started
        prediction = tokenizer.decode(
            output[context_length:], skip_special_tokens=True
        )

        if variant.name == QLUTATTN_VARIANT and kv_cache is not None:
            engagement["samples_observed"] += 1
            engagement["v_quant_calls"] += int(
                getattr(kv_cache, "v_quant_calls", 0)
            )
            engagement["v_quantized_tokens"] += int(
                getattr(kv_cache, "v_quantized_tokens", 0)
            )
            engagement["v_tile_blocks"] += int(
                getattr(kv_cache, "v_tile_blocks", 0)
            )
            mode_seen = getattr(kv_cache, "last_v_quant_mode", None)
            channels_seen = getattr(kv_cache, "last_v_tile_channels", None)
            if mode_seen is not None:
                engagement["last_v_quant_mode"] = mode_seen
            if channels_seen is not None:
                engagement["last_v_tile_channels"] = channels_seen

        if variant.shadowkv and kitty_stats is not None:
            engagement["shadowkv_installed"] = int(kitty_stats.get("installed", 0))
            engagement["shadowkv_prefill_calls"] = int(
                kitty_stats.get("prefill_calls", 0)
            )
            engagement["shadowkv_decode_calls"] = int(
                kitty_stats.get("decode_calls", 0)
            )
            engagement["shadowkv_decode_calls_at_pair_start"] = (
                shadow_pair_decode_start
            )
            engagement["shadowkv_decode_calls_delta"] = (
                engagement["shadowkv_decode_calls"] - shadow_pair_decode_start
            )
            engagement["shadowkv_last_selected_chunks"] = kitty_stats.get(
                "last_selected_chunks"
            )
            engagement["shadowkv_last_seq_length"] = kitty_stats.get(
                "last_seq_length"
            )

        if variant.use_kitty and not kitty_checked:
            seqlen = kv_cache.get_seq_length() if kv_cache is not None else 0
            engaged = kv_cache is not None and seqlen > 0
            detail = f"KittyKVCache.get_seq_length()={seqlen}"
            if variant.shadowkv:
                installed = (
                    int(kitty_stats.get("installed", 0)) if kitty_stats else 0
                )
                decode_after = (
                    int(kitty_stats.get("decode_calls", 0)) if kitty_stats else 0
                )
                selected = (
                    kitty_stats.get("last_selected_chunks") if kitty_stats else None
                )
                engaged = (
                    engaged
                    and installed > 0
                    and decode_after > shadow_decode_before
                )
                detail = (
                    f"installed={installed}, decode_calls_before="
                    f"{shadow_decode_before}, decode_calls_after={decode_after}, "
                    f"seq_length={seqlen}, last_selected_chunks={selected}"
                )
            elif variant.name == QLUTATTN_VARIANT and kv_cache is not None:
                blocks = int(getattr(kv_cache, "v_tile_blocks", 0))
                mode = getattr(kv_cache, "last_v_quant_mode", None)
                channels_seen = getattr(kv_cache, "last_v_tile_channels", None)
                ready = max(
                    0, context_length - variant.sink_length - variant.buffer_length
                )
                detail += (
                    f", v_tile_blocks={blocks}, last_v_quant_mode={mode}, "
                    f"last_v_tile_channels={channels_seen}"
                )
                if ready >= 16:
                    engaged = engaged and blocks > 0 and mode == "tile16_rescued"
                    if variant.v_tile_channels is not None:
                        engaged = engaged and channels_seen == variant.v_tile_channels
            engagement["kitty_cache_checked"] = True
            engagement["kitty_cache_engaged"] = bool(engaged)
            engagement["kitty_cache_seq_length"] = int(seqlen)
            if not engaged:
                raise RuntimeError(
                    f"Kitty variant {variant.name!r} did not engage ({detail}); refusing "
                    "to emit dense fp16 results under a quantized method label."
                )
            kitty_checked = True

        row = {
            "sample_idx": sample_idx,
            "index": record["index"],
            "task": task,
            "seq_len": int(seq_len),
            "pred": prediction,
            "outputs": record["outputs"],
            "prompt_tokens": context_length,
            "generation_seconds": round(elapsed, 3),
            "token_position_answer": record.get("token_position_answer"),
            "length": record["length"],
            "max_length": record["max_length"],
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written_now += 1
        manifest["written_samples"] = _completed_rows(out_path)
        manifest["written_this_run"] = written_now
        manifest["engagement"] = engagement
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json_atomic(manifest_path, manifest)
        del kv_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["written_samples"] = _completed_rows(out_path)
    manifest["written_this_run"] = written_now
    manifest["status"] = (
        "ok" if manifest["written_samples"] == expected_samples else "partial"
    )
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["jsonl_sha256"] = _sha256_file(out_path)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def run_ruler(args: Any) -> dict[str, Any]:
    """Load one model and serially evaluate this worker's RULER task shard."""

    required_cvd = getattr(args, "require_cuda_visible_devices", None)
    if required_cvd is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != required_cvd:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={required_cvd!r} required, got "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    preflight = resolve_ruler_preflight(args)
    expected_preflight_hash = getattr(args, "expected_preflight_hash", None)
    if (
        expected_preflight_hash is not None
        and expected_preflight_hash != preflight["preflight_hash"]
    ):
        raise RuntimeError(
            "RULER worker preflight hash mismatch before model loading: "
            f"expected={expected_preflight_hash}, actual={preflight['preflight_hash']}"
        )

    variant = _maybe_enable_quest_kernel(build_variant(args), args)
    model_family = _model_family(args)
    _reject_unsupported(variant, model_family)
    requested_tasks = tuple(preflight["tasks"])
    worker_tasks = select_task_shard(
        requested_tasks,
        getattr(args, "task_shard_index", None),
        getattr(args, "task_shard_count", None),
    )
    seq_lens = tuple(int(length) for length in preflight["seq_lens"])
    max_model_len = int(args.max_model_len)
    model_path = args.model_path or args.model

    output_root = getattr(args, "output_dir", None)
    if output_root in (None, ""):
        base = Path("ruler_out")
        if int(args.max_samples) > 0:
            base = base / "smoke"
        pred_dir = base / f"{preflight['model_slug']}_{preflight['method_slug']}" / "pred"
    else:
        pred_dir = Path(output_root)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Kitty NVIDIA RULER evaluation")
    print(f"model={args.model} path={model_path} family={model_family}")
    print(f"variant={variant.tag}")
    print(
        f"tasks={list(requested_tasks)} worker_tasks={list(worker_tasks)} "
        f"seq_lens={list(seq_lens)} max_samples={args.max_samples}"
    )
    print(f"max_model_len={max_model_len} max_new_tokens={args.max_new_tokens}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"output={pred_dir}")
    print("=" * 80)

    model_obj, tokenizer, resolved_model_path = load_model_and_tokenizer(
        args.model,
        model_path=model_path,
        model_family=model_family,
        dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    _validate_runtime_tokenizer(tokenizer, preflight, model_family)
    validate_qlutattn_model_config(variant, model_obj.config, model_obj.dtype)
    kitty_stats: dict[str, Any] | None = None
    if variant.shadowkv:
        from kitty_sim.shadowkv_sim import ShadowKVSimConfig, install_shadowkv_sim

        kitty_stats = install_shadowkv_sim(
            model_obj,
            ShadowKVSimConfig(
                sparse_budget=variant.sparse_budget,
                rank=variant.rank,
                chunk_size=variant.chunk_size,
            ),
        )
        print(
            f"[shadowkv] installed pure-torch hook on {kitty_stats['installed']} "
            f"layers (budget={variant.sparse_budget}, rank={variant.rank}, "
            f"chunk={variant.chunk_size})"
        )

    manifests: list[dict[str, Any]] = []
    try:
        for task in worker_tasks:
            spec = get_task_spec(task)
            max_new_tokens = _effective_max_new_tokens(args, spec)
            for seq_len in seq_lens:
                name = pair_name(task, seq_len)
                pair_preflight = preflight["pairs"][name]
                records = load_ruler_records(task, seq_len, data_root=args.data_root)
                max_samples = int(args.max_samples)
                if max_samples > 0:
                    records = records[:max_samples]
                manifest = generate_ruler_pair(
                    model=model_obj,
                    tokenizer=tokenizer,
                    task=task,
                    seq_len=seq_len,
                    records=records,
                    task_spec=spec,
                    max_model_len=max_model_len,
                    max_new_tokens=max_new_tokens,
                    out_path=pred_dir / f"{name}.jsonl",
                    variant=variant,
                    model_family=model_family,
                    model_name=resolved_model_path,
                    run_config=pair_preflight["run_config"],
                    expected_run_config_hash=pair_preflight["run_config_hash"],
                    overwrite=bool(getattr(args, "overwrite", False)),
                    kitty_stats=kitty_stats,
                )
                manifests.append(manifest)
    finally:
        model_obj.to("cpu")
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "status": "ok" if all(item.get("status") == "ok" for item in manifests) else "partial",
        "benchmark": "ruler",
        "prediction_dir": str(pred_dir),
        "model": args.model,
        "model_path": resolved_model_path,
        "model_slug": preflight["model_slug"],
        "model_family": model_family,
        "method_slug": preflight["method_slug"],
        "variant": asdict(variant),
        "variant_semantic_hash": preflight["variant_semantic_hash"],
        "preflight_hash": preflight["preflight_hash"],
        "tasks": list(worker_tasks),
        "requested_tasks": list(requested_tasks),
        "task_shard": {
            "index": getattr(args, "task_shard_index", None),
            "count": getattr(args, "task_shard_count", None),
        },
        "seq_lens": list(seq_lens),
        "pairs": preflight["pairs"],
        "manifests": manifests,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hook_engagement": dict(kitty_stats) if kitty_stats is not None else None,
    }
    if getattr(args, "report_json", None):
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(report_path, report)
    return report
