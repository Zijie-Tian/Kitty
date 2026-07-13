"""RULER-NIAH generation runner for Kitty KV-cache variants.

Reuses the LongBench variant machinery (``build_variant`` / per-sample
``KittyKVCache`` / engagement evidence / manifest+resume) but swaps the data
source for RULER-NIAH jsonl and the prompt assembly for the RULER protocol:

    prompt = chat_wrap(record["input"]) + record["answer_prefix"]

The answer prefix sits AFTER the assistant header (RULER's model-template
semantics). Prompts are never truncated: data is generated with a token margin
and an over-long prompt is a hard error, because middle-truncation could
silently delete the needle.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from kitty_sim.longbench.runner import (
    NEW_V2_VARIANTS,
    VariantConfig,
    _cache_factory,
    _sha256_file,
    _stable_json_hash,
    build_variant,
    load_model_and_tokenizer,
    method_layout_slug,
    model_layout_slug,
    validate_new_v2_model_config,
    validate_new_v2_model_family,
    validate_new_v2_preload,
    variant_semantic_hash,
    variant_semantic_payload,
)
from kitty_sim.longbench.templates import build_chat, infer_model_family

from .data import load_niah_records, resolve_niah_file

DEFAULT_MAX_NEW_TOKENS = 128  # RULER niah tokens_to_generate


def _model_source_identity(model_path: str | None) -> str | None:
    if not model_path:
        return None
    path = Path(model_path).expanduser()
    return str(path.resolve()) if path.exists() else str(model_path)


def niah_run_config_payload(
    args: Any, variant: VariantConfig, task: str, seq_len: int
) -> dict[str, Any]:
    """Stable per-(task, seq_len) generation fingerprint (no model weights)."""
    data_path = resolve_niah_file(task, seq_len, getattr(args, "data_root", None))
    if not data_path.is_file():
        raise FileNotFoundError(f"NIAH data file not found for preflight: {data_path}")
    with data_path.open("r", encoding="utf-8") as handle:
        total_samples = sum(1 for line in handle if line.strip())
    max_samples = int(getattr(args, "max_samples", -1))
    expected = min(total_samples, max_samples) if max_samples > 0 else total_samples
    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None) or model
    model_family = getattr(args, "model_family", None) or infer_model_family(
        getattr(args, "model_tag", None) or model, model_path or model
    )
    return {
        "schema_version": 1,
        "benchmark": "ruler_niah",
        "variant_semantic_hash": variant_semantic_hash(variant),
        "model_id": model,
        "model_source_identity": _model_source_identity(model_path),
        "model_family": model_family,
        "task": task,
        "seq_len": int(seq_len),
        "dataset_sha256": _sha256_file(data_path),
        "dataset_total_samples": total_samples,
        "max_samples": max_samples,
        "expected_samples": expected,
        "max_model_len": int(getattr(args, "max_model_len", 32768)),
        "max_new_tokens": int(getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        "torch_dtype": str(getattr(args, "torch_dtype", "float16")),
    }


def niah_run_config_hash(args: Any, variant: VariantConfig, task: str, seq_len: int) -> str:
    return _stable_json_hash(niah_run_config_payload(args, variant, task, seq_len))


def build_niah_prompt(
    tokenizer: Any, record: dict[str, Any], model_family: str
) -> str:
    """Chat-wrap the task input, then append the assistant-side answer prefix."""
    prompt = build_chat(tokenizer, record["input"], model_family)
    return prompt + record["answer_prefix"]


def _pair_name(task: str, seq_len: int) -> str:
    return f"{task}__{int(seq_len)}"


def _completed_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _reject_unsupported(variant: VariantConfig, model_family: str) -> None:
    if variant.shadowkv or getattr(variant, "quest_kernel", False):
        raise ValueError(
            "The NIAH runner supports fp16 and KittyKVCache fake-quant variants "
            "only (no shadowkv / quest kernel overlays)."
        )
    from kitty_sim.glm_kitty_patch import is_glm_family

    if is_glm_family(model_family):
        raise ValueError(
            "GLM legacy-cache models are not wired into the NIAH runner; "
            "use an HF-Cache model family."
        )


def generate_niah_pair(
    *,
    model: Any,
    tokenizer: Any,
    task: str,
    seq_len: int,
    records: list[dict[str, Any]],
    max_model_len: int,
    max_new_tokens: int,
    out_path: Path,
    variant: VariantConfig,
    model_family: str,
    model_name: str,
    overwrite: bool = False,
    expected_run_config_hash: str | None = None,
) -> dict[str, Any]:
    """Generate one (task, seq_len) jsonl with LongBench-style manifest/resume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    if overwrite:
        out_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    expected_samples = len(records)
    completed = _completed_rows(out_path)
    if completed >= expected_samples:
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Refusing to reuse completed {out_path.name} without a manifest"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not expected_run_config_hash
            or manifest.get("run_config_hash") != expected_run_config_hash
            or manifest.get("status") != "ok"
            or int(manifest.get("written_samples", -1)) != completed
            or completed != expected_samples
        ):
            raise RuntimeError(
                f"Refusing to reuse completed {out_path.name}: manifest/hash mismatch "
                f"(expected hash {expected_run_config_hash!r}, "
                f"manifest {manifest.get('run_config_hash')!r})"
            )
        print(f"[skip] {out_path.name}: all {completed}/{expected_samples} samples present")
        return manifest

    if completed:
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Refusing to resume partial {out_path.name} without a manifest"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_config_hash") != expected_run_config_hash:
            raise RuntimeError(
                f"Refusing to resume partial {out_path.name}: run_config_hash mismatch"
            )
        print(f"[resume] {out_path.name}: skip {completed}, remaining "
              f"{expected_samples - completed}")

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    engagement = {
        "samples_observed": 0,
        "v_quant_calls": 0,
        "v_quantized_tokens": 0,
        "v_tile_blocks": 0,
        "last_v_quant_mode": None,
        "last_v_tile_channels": None,
    }
    kitty_checked = False
    written_now = 0

    pair = _pair_name(task, seq_len)
    for sample_idx in tqdm(range(completed, expected_samples), desc=pair):
        record = records[sample_idx]
        prompt = build_niah_prompt(tokenizer, record, model_family)
        # apply_chat_template renders the BOS token as text (llama3.x); avoid a
        # duplicated BOS by disabling add_special_tokens in that case.
        bos = getattr(tokenizer, "bos_token", None)
        add_special = not (bos and prompt.startswith(bos))
        inputs = tokenizer(
            prompt,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=add_special,
        ).to(device)
        inputs.pop("token_type_ids", None)
        context_length = int(inputs.input_ids.shape[-1])
        if context_length > max_model_len:
            raise RuntimeError(
                f"{pair} sample {sample_idx}: prompt has {context_length} tokens "
                f"> max_model_len={max_model_len}. NIAH prompts are never "
                "truncated (a truncation could delete the needle); regenerate "
                "the data with a larger margin."
            )
        kv_cache = _cache_factory(variant)
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                past_key_values=kv_cache,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )[0]
        elapsed = time.perf_counter() - t0
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        if variant.use_kitty and not kitty_checked:
            seqlen = kv_cache.get_seq_length() if kv_cache is not None else 0
            if kv_cache is None or seqlen <= 0:
                raise RuntimeError(
                    f"Kitty variant '{variant.name}' requested but the KittyKVCache "
                    f"never engaged (get_seq_length()={seqlen}); results would be "
                    "dense fp16 mislabelled as quantized. Refusing to proceed."
                )
            kitty_checked = True
        if variant.name in NEW_V2_VARIANTS and kv_cache is not None:
            engagement["samples_observed"] += 1
            engagement["v_quant_calls"] += int(getattr(kv_cache, "v_quant_calls", 0))
            engagement["v_quantized_tokens"] += int(
                getattr(kv_cache, "v_quantized_tokens", 0)
            )
            engagement["v_tile_blocks"] += int(getattr(kv_cache, "v_tile_blocks", 0))
            mode_seen = getattr(kv_cache, "last_v_quant_mode", None)
            channels_seen = getattr(kv_cache, "last_v_tile_channels", None)
            if mode_seen is not None:
                engagement["last_v_quant_mode"] = mode_seen
            if channels_seen is not None:
                engagement["last_v_tile_channels"] = channels_seen

        row = {
            "sample_idx": sample_idx,
            "task": task,
            "seq_len": int(seq_len),
            "pred": pred,
            "outputs": record["outputs"],
            "prompt_tokens": context_length,
            "generation_seconds": round(elapsed, 3),
            "token_position_answer": record.get("token_position_answer"),
            "length": record.get("length"),
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written_now += 1
        del kv_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "status": "ok",
        "benchmark": "ruler_niah",
        "task": task,
        "seq_len": int(seq_len),
        "expected_samples": expected_samples,
        "written_samples": _completed_rows(out_path),
        "written_this_run": written_now,
        "output_path": str(out_path),
        "variant": asdict(variant),
        "variant_semantic_hash": variant_semantic_hash(variant),
        "mask_sha256": variant_semantic_payload(variant)["mask_sha256"],
        "model_name": model_name,
        "model_family": model_family,
        "max_model_len": int(max_model_len),
        "max_new_tokens": int(max_new_tokens),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engagement": engagement,
        "run_config_hash": expected_run_config_hash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if manifest["written_samples"] != expected_samples:
        manifest["status"] = "partial"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run_niah(args: Any) -> dict[str, Any]:
    required_cvd = getattr(args, "require_cuda_visible_devices", None)
    if required_cvd is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != required_cvd:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={required_cvd!r} required, "
            f"got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    variant = build_variant(args)
    model_family = args.model_family or infer_model_family(
        args.model_tag or args.model, args.model_path or args.model
    )
    _reject_unsupported(variant, model_family)
    validate_new_v2_model_family(variant, model_family)
    validate_new_v2_preload(args, variant, model_family)

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    seq_lens = [int(s) for s in str(args.seq_lens).split(",") if str(s).strip()]
    if not tasks or not seq_lens:
        raise ValueError("--tasks and --seq-lens must be non-empty CSV lists")

    max_model_len = int(args.max_model_len)
    max_new_tokens = int(getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    model_path = args.model_path or args.model

    output_root = getattr(args, "output_dir", None)
    if output_root in (None, ""):
        model_slug = args.model_tag or model_layout_slug(args.model, model_path)
        method_slug = method_layout_slug(variant)
        base = Path("niah_out")
        if int(args.max_samples) > 0:
            base = base / "smoke"
        pred_dir = base / f"{model_slug}_{method_slug}" / "pred"
    else:
        pred_dir = Path(output_root)
    pred_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(task, seq_len) for task in tasks for seq_len in seq_lens]
    run_hashes = {
        _pair_name(task, seq_len): niah_run_config_hash(args, variant, task, seq_len)
        for task, seq_len in pairs
    }

    print("=" * 80)
    print("Kitty RULER-NIAH evaluation")
    print(f"model={args.model} path={model_path} family={model_family}")
    print(f"variant={variant.tag}")
    print(f"tasks={tasks} seq_lens={seq_lens} max_samples={args.max_samples}")
    print(f"max_model_len={max_model_len} max_new_tokens={max_new_tokens}")
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
    validate_new_v2_model_config(variant, model_obj.config, model_obj.dtype)

    manifests: list[dict[str, Any]] = []
    try:
        for task, seq_len in pairs:
            records = load_niah_records(task, seq_len, data_root=args.data_root)
            max_samples = int(args.max_samples)
            if max_samples > 0:
                records = records[:max_samples]
            out_path = pred_dir / f"{_pair_name(task, seq_len)}.jsonl"
            manifest = generate_niah_pair(
                model=model_obj,
                tokenizer=tokenizer,
                task=task,
                seq_len=seq_len,
                records=records,
                max_model_len=max_model_len,
                max_new_tokens=max_new_tokens,
                out_path=out_path,
                variant=variant,
                model_family=model_family,
                model_name=resolved_model_path,
                overwrite=bool(getattr(args, "overwrite", False)),
                expected_run_config_hash=run_hashes[_pair_name(task, seq_len)],
            )
            manifests.append(manifest)
    finally:
        model_obj.to("cpu")
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "status": "ok",
        "benchmark": "ruler_niah",
        "prediction_dir": str(pred_dir),
        "model": args.model,
        "model_path": resolved_model_path,
        "model_family": model_family,
        "variant": asdict(variant),
        "variant_semantic_hash": variant_semantic_hash(variant),
        "tasks": tasks,
        "seq_lens": seq_lens,
        "run_config_hashes": run_hashes,
        "manifests": manifests,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if getattr(args, "report_json", None):
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report
