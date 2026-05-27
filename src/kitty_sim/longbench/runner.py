"""LongBench generation runner for Kitty KV-cache variants."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kitty_sim import get_kvcache_kitty

from .config import LONG_BENCH_DATASETS, LONG_BENCH_E_DATASETS, load_json_config
from .data import load_longbench_dataset
from .templates import format_longbench_prompt, infer_model_family, post_process


@dataclass(frozen=True)
class VariantConfig:
    name: str
    use_kitty: bool
    sink_length: int = 32
    buffer_length: int = 128
    group_size: int = 128
    kbits: int = 2
    vbits: int = 2
    promote_ratio: float = 0.125
    promote_bit: int = 4
    channel_selection: int = 1

    @property
    def tag(self) -> str:
        if not self.use_kitty:
            return "fp16"
        ratio = str(self.promote_ratio).replace(".", "p")
        return (
            f"{self.name}_g{self.group_size}_b{self.buffer_length}_s{self.sink_length}"
            f"_sel{self.channel_selection}_k{self.kbits}_v{self.vbits}"
            f"_pb{self.promote_bit}_pr{ratio}"
        )


def build_variant(args: Any) -> VariantConfig:
    variant = args.variant.lower()
    if variant == "fp16":
        return VariantConfig(name="fp16", use_kitty=False, promote_ratio=0.0)
    if variant == "kitty":
        return VariantConfig(name="kitty", use_kitty=True, promote_ratio=0.125)
    if variant == "kitty_page16":
        return VariantConfig(
            name="kitty_page16",
            use_kitty=True,
            sink_length=32,
            buffer_length=16,
            group_size=16,
            promote_ratio=0.125,
        )
    if variant == "kitty_pro":
        return VariantConfig(name="kitty_pro", use_kitty=True, promote_ratio=0.25)
    if variant == "kivi_2":
        return VariantConfig(name="kivi_2", use_kitty=True, sink_length=0, promote_ratio=0.0, channel_selection=0)
    if variant == "kivi_star_2":
        return VariantConfig(name="kivi_star_2", use_kitty=True, sink_length=32, promote_ratio=0.0, channel_selection=0)
    if variant == "custom":
        return VariantConfig(
            name="custom",
            use_kitty=True,
            sink_length=args.sink_length,
            buffer_length=args.buffer_length,
            group_size=args.group_size,
            kbits=args.kbits,
            vbits=args.vbits,
            promote_ratio=args.promote_ratio,
            promote_bit=args.promote_bit,
            channel_selection=args.channel_selection,
        )
    raise ValueError(f"Unknown variant: {args.variant}")


def _cache_factory(config: VariantConfig):
    if not config.use_kitty:
        return None
    ns = SimpleNamespace(
        sink_length=config.sink_length,
        buffer_length=config.buffer_length,
        group_size=config.group_size,
        kbits=config.kbits,
        vbits=config.vbits,
        promote_ratio=config.promote_ratio,
        promote_bit=config.promote_bit,
        channel_selection=config.channel_selection,
    )
    return get_kvcache_kitty(ns)


def _safe_tag(value: str) -> str:
    return value.strip().replace("/", "_").replace(" ", "_")


def model_basename(model: str, model_path: str | None = None) -> str:
    source = model_path or model
    return _safe_tag(Path(source).name if source else "model")


def output_model_dir(output_dir: str | os.PathLike[str], model_tag: str, variant: VariantConfig) -> Path:
    return Path(output_dir) / f"{_safe_tag(model_tag)}-{variant.tag}"


def config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_model_and_tokenizer(
    model: str,
    *,
    model_path: str | None = None,
    model_family: str,
    dtype: str = "float16",
    local_files_only: bool = False,
):
    resolved = model_path or model
    torch_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        trust_remote_code=True,
        use_fast=("llama3" in model_family or "qwen" in model_family),
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_obj = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model_obj.eval()
    return model_obj, tokenizer, resolved


def _select_data(data: Any, max_samples: int) -> Any:
    if max_samples > 0:
        return data.select(range(min(max_samples, len(data))))
    return data


def _completed_rows(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    with out_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate_dataset(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    dataset: str,
    records: Any,
    prompt_format: str,
    max_model_len: int,
    max_gen: int,
    out_path: Path,
    variant: VariantConfig,
    model_family: str,
    prompt_token_reserve: int = 0,
    overwrite: bool = False,
    strict_complete: bool = True,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        out_path.unlink(missing_ok=True)
        out_path.with_suffix(".manifest.json").unlink(missing_ok=True)

    expected_samples = len(records)
    completed = _completed_rows(out_path)
    if completed >= expected_samples:
        manifest = {
            "status": "ok",
            "dataset": dataset,
            "expected_samples": expected_samples,
            "written_samples": completed,
            "failed_sample_ids": [],
            "output_path": str(out_path),
            "resumed": True,
        }
        _write_manifest(out_path, manifest)
        print(f"[skip] {dataset}: all {completed}/{expected_samples} samples already present")
        return manifest

    if completed:
        print(f"[resume] {dataset}: skip {completed}, remaining {expected_samples - completed}")
        records = records.skip(completed)

    failed: list[dict[str, Any]] = []
    written_now = 0
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for local_idx, json_obj in enumerate(tqdm(records, desc=dataset)):
        sample_idx = completed + local_idx
        prompt = prompt_format.format(**json_obj)
        prompt = format_longbench_prompt(
            tokenizer,
            prompt,
            dataset=dataset,
            model_family=model_family,
            max_model_len=max_model_len,
            prompt_token_reserve=prompt_token_reserve,
        )
        try:
            inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = inputs.input_ids.shape[-1]
            kv_cache = _cache_factory(variant)
            eos_token_id: int | list[int] | None = tokenizer.eos_token_id
            if dataset == "samsum" and tokenizer.eos_token_id is not None:
                newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                eos_token_id = [tokenizer.eos_token_id]
                if newline_ids:
                    eos_token_id.append(newline_ids[-1])
            with torch.inference_mode():
                gen_kwargs = {
                    "past_key_values": kv_cache,
                    "max_new_tokens": max_gen,
                    "num_beams": 1,
                    "do_sample": False,
                    "temperature": 1.0,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "use_cache": True,
                }
                if dataset == "samsum":
                    gen_kwargs["min_length"] = context_length + 1
                output = model.generate(
                    **inputs,
                    **gen_kwargs,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_family)
            row = {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
            with out_path.open("a", encoding="utf-8") as handle:
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
            written_now += 1
        except Exception as exc:  # keep manifest evidence for strict failure
            failed.append({"sample_id": sample_idx, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {dataset} sample {sample_idx}: {type(exc).__name__}: {exc}")
        finally:
            try:
                del inputs  # type: ignore[name-defined]
            except UnboundLocalError:
                pass
            try:
                del output  # type: ignore[name-defined]
            except UnboundLocalError:
                pass
            try:
                del kv_cache  # type: ignore[name-defined]
            except UnboundLocalError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    total_written = _completed_rows(out_path)
    status = "ok" if total_written == expected_samples and not failed else "partial"
    manifest = {
        "status": status,
        "dataset": dataset,
        "expected_samples": expected_samples,
        "written_samples": total_written,
        "written_this_run": written_now,
        "failed_sample_ids": failed,
        "output_path": str(out_path),
        "variant": asdict(variant),
        "model_name": model_name,
        "model_family": model_family,
        "max_model_len": max_model_len,
        "max_gen": max_gen,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["config_hash"] = config_hash(manifest)
    _write_manifest(out_path, manifest)
    if strict_complete and status != "ok":
        raise RuntimeError(f"Incomplete dataset {dataset}: wrote {total_written}/{expected_samples}; failures={failed}")
    return manifest


def run_longbench(args: Any) -> dict[str, Any]:
    if args.require_gpu1 and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError(
            "GPU1-only run required: set CUDA_VISIBLE_DEVICES=1 before launching "
            f"(current={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )

    variant = build_variant(args)
    model_family = args.model_family or infer_model_family(args.model_tag or args.model, args.model_path or args.model)
    model_tag = args.model_tag or model_basename(args.model, args.model_path)
    output_root = "pred_e" if args.e and args.output_dir == "longbench_out/pred" else args.output_dir
    pred_dir = output_model_dir(output_root, model_tag, variant)
    dataset2prompt = load_json_config("dataset2prompt.json")
    dataset2maxlen = load_json_config("dataset2maxlen.json")
    model2maxlen = load_json_config("model2maxlen.json")
    # Do not resolve models through a tracked path map. Local filesystem
    # locations are intentionally provided only by --model-path / MODEL_PATH /
    # the ignored repo-root .env; otherwise args.model is treated as a
    # Hugging Face id or other transformers-compatible model identifier.
    model_path = args.model_path or args.model
    max_model_len = args.max_model_len or model2maxlen.get(args.model, args.default_max_model_len)

    if args.dataset:
        datasets = [args.dataset]
    elif args.e:
        datasets = list(LONG_BENCH_E_DATASETS)
    else:
        datasets = list(LONG_BENCH_DATASETS)

    print("=" * 80)
    print("Kitty LongBench evaluation")
    print(f"model={args.model} path={model_path}")
    print(f"model_tag={model_tag} model_family={model_family}")
    print(f"variant={variant.tag}")
    print(f"datasets={datasets}")
    print(f"max_samples={args.max_samples} max_model_len={max_model_len}")
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

    manifests: list[dict[str, Any]] = []
    try:
        for dataset in datasets:
            data_name = f"{dataset}_e" if args.e else dataset
            data = load_longbench_dataset(data_name, data_root=args.data_root)
            data = _select_data(data, args.max_samples)
            max_gen = args.max_gen or dataset2maxlen[dataset]
            out_path = pred_dir / f"{dataset}.jsonl"
            manifest = generate_dataset(
                model_name=resolved_model_path,
                model=model_obj,
                tokenizer=tokenizer,
                dataset=dataset,
                records=data,
                prompt_format=dataset2prompt[dataset],
                max_model_len=max_model_len,
                max_gen=max_gen,
                out_path=out_path,
                variant=variant,
                model_family=model_family,
                prompt_token_reserve=args.prompt_token_reserve,
                overwrite=args.overwrite,
                strict_complete=args.strict_complete,
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
        "prediction_dir": str(pred_dir),
        "model": args.model,
        "model_path": resolved_model_path,
        "model_tag": model_tag,
        "model_family": model_family,
        "variant": asdict(variant),
        "datasets": datasets,
        "manifests": manifests,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
