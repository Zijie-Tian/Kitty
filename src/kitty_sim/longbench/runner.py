"""LongBench generation runner for Kitty KV-cache variants."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kitty_sim import get_kvcache_kitty
from kitty_sim.glm_kitty_patch import (
    cache_config_from_variant,
    install_glm_kitty_fakequant,
    is_glm_family,
)

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
    # K-cache quant orientation: "per_channel" (KIVI-style token-axis groups +
    # promote/qlut codebook) or "per_token" (K quantized like V along head_dim,
    # uniform, no promote) -- the QServe SmoothAttention study mode, pair with a
    # smooth-calibrated checkpoint (scripts/calibrate_smooth_qk.py).
    k_quant_mode: str = "per_channel"
    # ShadowKV sim: pure-torch faithful port of ShadowKV's accuracy cache (SVD
    # low-rank pre-RoPE keys + landmark chunk selection + outlier/local chunks).
    # Accuracy + relative-timing proxy; NOT a memory/speed proof.
    shadowkv: bool = False
    sparse_budget: int = 2048
    rank: int = 160
    chunk_size: int = 8
    # Per-layer promote_ratio override (kitty_k1v4 only): a tuple of
    # (layer_idx, ratio) pairs -- hashable (frozen dataclass safe) and
    # asdict-friendly. None => scalar promote_ratio for every layer.
    # promote_ratio_config_path keeps the source JSON path for provenance.
    promote_ratio_per_layer: tuple[tuple[int, float], ...] | None = None
    promote_ratio_config_path: str | None = None
    # QLUT (sigma^2-binned) K codebook = qlutattn-k1v4. k_codebook='qlut'
    # uses bin_codebooks (a per-sigma^2-bin codebook list); 'kivi' = default path.
    k_codebook: str = "kivi"
    bin_codebooks: tuple[str, ...] | None = None
    n_bins: int = 6

    @property
    def tag(self) -> str:
        if not self.use_kitty:
            return "fp16"
        ratio = str(self.promote_ratio).replace(".", "p")
        if self.k_codebook == "qlut":
            h = hashlib.sha256(repr(self.bin_codebooks).encode()).hexdigest()[:6]
            return f"{self.name}_nb{self.n_bins}_v{self.vbits}_cb{h}"
        if self.shadowkv:
            return f"{self.name}_sb{self.sparse_budget}_r{self.rank}_c{self.chunk_size}"
        suffix = ""
        if self.promote_ratio_per_layer:
            h = hashlib.sha256(repr(self.promote_ratio_per_layer).encode()).hexdigest()[:6]
            suffix = f"-prcfg{h}"
        if self.k_quant_mode == "per_token":
            suffix += "_kpt"
        return (
            f"{self.name}_g{self.group_size}_b{self.buffer_length}_s{self.sink_length}"
            f"_sel{self.channel_selection}_k{self.kbits}_v{self.vbits}"
            f"_pb{self.promote_bit}_pr{ratio}{suffix}"
        )


def _load_promote_ratio_config(
    path: str, default_ratio: float
) -> tuple[float, tuple[tuple[int, float], ...] | None]:
    """Load a per-layer promote_ratio schedule from a JSON file.

    Accepted forms:
      * {"default": 0.25, "layers": {"0": 0.75, "1": 0.5}}
            -- any layer absent from "layers" falls back to "default".
      * [0.75, 0.5, ...]  (a bare list: one ratio per layer index, in order).
    Returns (default_ratio, per_layer), where per_layer is a tuple of
    (layer_idx, ratio) pairs (None when there are no explicit per-layer
    overrides). Ratios are validated to lie in [0, 1]; layer-index range vs the
    model's actual layer count is checked later in run_longbench (after load).
    """
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    def _check(value: Any, where: str) -> float:
        r = float(value)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"promote-ratio-config {where} must be in [0, 1]; got {r}")
        return r

    if isinstance(cfg, list):
        pairs = tuple((i, _check(v, f"layers[{i}]")) for i, v in enumerate(cfg))
        return default_ratio, (pairs or None)
    if isinstance(cfg, dict):
        resolved_default = _check(cfg.get("default", default_ratio), "default")
        layers = cfg.get("layers", {}) or {}
        if not isinstance(layers, dict):
            raise ValueError("promote-ratio-config 'layers' must be an object {layer_idx: ratio}")
        pairs = tuple(
            (int(k), _check(v, f"layers[{k}]"))
            for k, v in sorted(layers.items(), key=lambda kv: int(kv[0]))
        )
        return resolved_default, (pairs or None)
    raise ValueError(
        'promote-ratio-config must be a JSON object {"default": r, "layers": {...}} '
        "or a list [r0, r1, ...]"
    )


def build_variant(args: Any) -> VariantConfig:
    variant = args.variant.lower()
    config_path = getattr(args, "promote_ratio_config", None)
    if config_path and variant != "kitty_k1v4":
        raise ValueError(
            "--promote-ratio-config is only supported for --variant kitty_k1v4; "
            f"got '{variant}'."
        )
    if variant == "fp16":
        return VariantConfig(name="fp16", use_kitty=False, promote_ratio=0.0)
    if variant == "kitty":
        return VariantConfig(name="kitty", use_kitty=True, promote_ratio=0.125)
    if variant == "shadowkv":
        # Faithful pure-torch port of ShadowKV's accuracy cache (SVD low-rank
        # pre-RoPE keys + landmark chunk selection). Accuracy proxy, not a
        # speed/memory proof. Budget/rank/chunk default to the paper values.
        budget = getattr(args, "shadowkv_budget", None)
        budget = 2048 if budget in (None, 0) else int(budget)
        rank = int(getattr(args, "shadowkv_rank", None) or 160)
        chunk = int(getattr(args, "shadowkv_chunk_size", None) or 8)
        return VariantConfig(
            name="shadowkv",
            use_kitty=True,
            shadowkv=True,
            sparse_budget=budget,
            rank=rank,
            chunk_size=chunk,
        )
    if variant == "kitty_pro":
        return VariantConfig(name="kitty_pro", use_kitty=True, promote_ratio=0.25)
    if variant == "kitty_k1v4":
        # Low-bit K (1-bit base + 2-bit magnitude channel boost), V relaxed to
        # 4-bit. The K boost fraction (promote_ratio) is PER-LAYER when
        # --promote-ratio-config is supplied: a JSON object
        # {"default": r, "layers": {idx: r}} or a bare list [r0, r1, ...].
        # Without a config, the scalar default below applies to every layer.
        default_ratio = 0.25
        per_layer = None
        if config_path:
            default_ratio, per_layer = _load_promote_ratio_config(config_path, default_ratio)
        return VariantConfig(
            name="kitty_k1v4", use_kitty=True,
            kbits=1, vbits=4, promote_bit=2, promote_ratio=default_ratio,
            sink_length=32, buffer_length=128, group_size=128,
            channel_selection=1,
            promote_ratio_per_layer=per_layer,
            promote_ratio_config_path=config_path,
        )
    if variant in ("qlutattn_k1v4", "qlutattn-k1v4"):
        # QLUT-Attn k1v4 winner: per-layer sigma^2-binned K codebooks (6 bins,
        # low sigma^2 -> sign, high sigma^2 -> nf2). V per-token 4-bit. Effective
        # K ~1.68 bit (vs uniform tern 1.83); beats iso-tern on LongBench.
        cb = os.environ.get("QLUT_BIN_CODEBOOKS")
        bins = tuple(cb.split(",")) if cb else ("sign", "sign", "sign", "tern", "nf2", "nf2")
        return VariantConfig(
            name="qlutattn_k1v4", use_kitty=True, k_codebook="qlut", bin_codebooks=bins,
            n_bins=len(bins), vbits=4, promote_ratio=0.0, channel_selection=0,
            sink_length=32, buffer_length=128, group_size=128)
    if variant in ("qlutattn_k184v4", "qlutattn-k184v4"):
        # Uniform-tern K (all channels tern) + V 4-bit, ~1.84 bit K: the iso-tern
        # reference the qlutattn-k1v4 winner is compared against. (Formerly
        # tern_uniform; renamed into the qlutattn-k<bits>v4 family.)
        return VariantConfig(
            name="qlutattn_k184v4", use_kitty=True, k_codebook="qlut",
            bin_codebooks=("tern",) * 6, n_bins=6, vbits=4, promote_ratio=0.0,
            channel_selection=0, sink_length=32, buffer_length=128, group_size=128)
    if variant in ("qlutattn_pertoken", "qlut_pertoken"):
        # qlutattn-k1v4 turned per-token: a SINGLE submean codebook applied along
        # head_dim per token (sigma^2 binning has no per-channel axis in per-token
        # mode, so QLUT_BIN_CODEBOOKS gives one codebook, default nf2). V per-token
        # 4-bit, no promote. Pair with a SmoothAttention checkpoint
        # (scripts/calibrate_smooth_qk.py) to flatten per-channel K outliers that
        # the shared per-token scale would otherwise smear.
        cb = os.environ.get("QLUT_BIN_CODEBOOKS", "nf2")
        return VariantConfig(
            name="qlutattn_pertoken", use_kitty=True, k_codebook="qlut",
            bin_codebooks=(cb,), n_bins=1, vbits=4, promote_ratio=0.0,
            channel_selection=0, k_quant_mode="per_token")
    if variant in ("kivi", "kivi_star"):
        # KIVI-style uniform quant (NO promote, NO channel-select): K per-channel
        # + V per-token. kbits/vbits are free via --kbits/--vbits (default 2/2 =
        # the old kivi_2/kivi_star_2). kivi has no sink; kivi_star keeps sink=32.
        kb = int(getattr(args, "kbits", 2))
        vb = int(getattr(args, "vbits", 2))
        if not (1 <= kb <= 16 and 1 <= vb <= 16):
            raise ValueError(f"kivi kbits/vbits must be in [1, 16]; got kbits={kb}, vbits={vb}.")
        return VariantConfig(
            name=variant, use_kitty=True,
            kbits=kb, vbits=vb,
            sink_length=(32 if variant == "kivi_star" else 0),
            promote_ratio=0.0, channel_selection=0,
            buffer_length=128, group_size=128)
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
            k_quant_mode=getattr(args, "k_quant_mode", "per_channel"),
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
        promote_ratio_per_layer=(
            dict(config.promote_ratio_per_layer) if config.promote_ratio_per_layer else None
        ),
        channel_selection=config.channel_selection,
        k_quant_mode=config.k_quant_mode,
        k_codebook=config.k_codebook,
        bin_codebooks=(list(config.bin_codebooks) if config.bin_codebooks else None),
        n_bins=config.n_bins,
    )
    return get_kvcache_kitty(ns)


def _shadowkv_cache(variant: VariantConfig, model: Any, context_length: int, max_gen: int):
    """Build the per-sample ShadowKV sim cache (pure-torch accuracy port).

    Sized to context_length + max_gen (batch size 1, one prompt at a time). Uses
    the model's own rotary embedding so reconstructed keys are RoPE'd with the
    exact (rope-scaled) frequencies, matching the query path.
    """
    from kitty_sim.shadowkv_sim import ShadowKVSimCache

    config = model.config
    if getattr(config, "head_dim", None) is None:
        config.head_dim = config.hidden_size // config.num_attention_heads
    max_length = int(context_length) + int(max_gen)
    return ShadowKVSimCache(
        config,
        sparse_budget=variant.sparse_budget,
        rank=variant.rank,
        chunk_size=variant.chunk_size,
        max_length=max_length,
        max_gen=int(max_gen),
        device=getattr(model, "device", "cuda:0"),
        dtype=model.dtype,
        rotary_emb=getattr(model.model, "rotary_emb", None),
    )


def _safe_tag(value: str) -> str:
    return value.strip().replace("/", "_").replace(" ", "_")


def _layout_slug(value: str) -> str:
    slug = value.strip().lower().replace("/", "-").replace("_", "-")
    slug = re.sub(r"[^a-z0-9.+-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "model"


def model_layout_slug(model: str, model_path: str | None = None) -> str:
    value = model.lower()
    source = model_path or model
    if any(token in value for token in ("llama-3.1-8b", "llama3.1-8b", "llama31-8b", "llama31_8b")):
        return "llama31-8b-instruct"
    if any(token in value for token in ("llama-3.2-1b", "llama3.2-1b", "llama32-1b", "llama32_1b")):
        return "llama32-1b-instruct"
    if "qwen3" in value and "8b" in value:
        return "qwen3-8b"
    if ("glm-4" in value or "glm4" in value) and "9b" in value:
        return "glm4-9b-chat-1m"
    if "deepseek" in value and "r1" in value and "distill" in value and "llama" in value and "8b" in value:
        return "deepseek-r1-distill-llama-8b"
    return _layout_slug(Path(source).name if source else "model")


def method_layout_slug(variant: VariantConfig | str) -> str:
    name = (variant.name if isinstance(variant, VariantConfig) else str(variant)).lower()
    # kivi / kivi_star encode their K/V bit-width into the slug so different bit
    # combinations land in distinct output dirs (kivi-k2v4, kivi-star-k4v4, ...).
    if name in ("kivi", "kivi_star"):
        base = "kivi-star" if name == "kivi_star" else "kivi"
        if isinstance(variant, VariantConfig):
            return f"{base}-k{variant.kbits}v{variant.vbits}"
        return base  # str fallback: no bit info available
    return {
        "kitty": "kitty",
        "kitty_pro": "kitty-pro",
        "kitty_k1v4": "kitty-k1v4",
        "qlutattn_pertoken": "qlutattn-pertoken",
        "fp16": "fp16",
        "custom": "custom-kitty",
        "shadowkv": "shadowkv",
        "qlutattn_k1v4": "qlutattn-k1v4",
        "qlutattn_k184v4": "qlutattn-k184v4",
    }.get(name, _layout_slug(name))


def default_prediction_dir(model: str, model_path: str | None, variant: VariantConfig, max_samples: int) -> Path:
    model_slug = model_layout_slug(model, model_path)
    method_slug = method_layout_slug(variant)
    if max_samples > 0:
        return Path("longbench_out") / "smoke" / f"{model_slug}-{method_slug}" / "pred"
    return Path("longbench_out") / f"{model_slug}-{method_slug}" / "pred"


def model_basename(model: str, model_path: str | None = None) -> str:
    source = model_path or model
    return _safe_tag(Path(source).name if source else "model")


def output_model_dir(output_dir: str | os.PathLike[str], model_tag: str, variant: VariantConfig) -> Path:
    return Path(output_dir) / f"{_safe_tag(model_tag)}-{variant.tag}"


def resolve_prediction_dir(
    output_dir: str | os.PathLike[str],
    model_tag: str,
    variant: VariantConfig,
    *,
    flat_output_dir: bool = False,
) -> Path:
    if flat_output_dir:
        return Path(output_dir)
    return output_model_dir(output_dir, model_tag, variant)


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
        use_fast=("llama3" in model_family or "qwen" in model_family or "phi" in model_family),
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # All sim/fake-quant paths load the stock (remote-code) model.
    model_obj = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model_obj.eval()
    # If this is a QK-channel-reordered checkpoint (scripts/preprocess_qlutattn_model.py),
    # re-apply RoPE inv_freq[pair_perm] (persistent=False, not saved). No-op otherwise.
    from ..qk_reorder import apply_qk_reorder
    if apply_qk_reorder(model_obj, resolved):
        print(f"[qk-reorder] applied RoPE inv_freq permutation from {resolved}")
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
    legacy_cache_model: bool = False,
    kitty_stats: dict[str, Any] | None = None,
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
    kitty_engagement_checked = False
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
            # GLM-family models ignore an HF Cache object (legacy tuple cache); for
            # them Kitty fake-quant is applied by the SelfAttention patch installed
            # in run_longbench, so we pass past_key_values=None and let GLM build
            # its own (now-quantized) cache. All other models use the KittyKVCache.
            if legacy_cache_model:
                kv_cache = None
            elif variant.shadowkv:
                # ShadowKV sim: per-sample pure-torch cache sized to this prompt.
                kv_cache = _shadowkv_cache(variant, model, context_length, max_gen)
            else:
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
                if variant.shadowkv:
                    # ShadowKV's gather hook has data-dependent shapes, so
                    # torch.compile must stay off.
                    gen_kwargs["disable_compile"] = True
                    gen_kwargs["temperature"] = None
                output = model.generate(
                    **inputs,
                    **gen_kwargs,
                )[0]
            # Guardrail: refuse to silently report dense fp16 as Kitty. Verify the
            # KV quantization path actually engaged on the first generated sample.
            if variant.use_kitty and not kitty_engagement_checked:
                if legacy_cache_model:
                    engaged = bool(kitty_stats and kitty_stats.get("calls", 0) > 0)
                    detail = f"glm fake-quant patch calls={kitty_stats.get('calls') if kitty_stats else None}"
                    raise_msg = (
                        f"Kitty variant '{variant.name}' was requested but KV quantization "
                        f"never engaged for model_family='{model_family}' ({detail}). The model "
                        f"bypassed the KittyKVCache (e.g. a legacy tuple-cache remote modeling), "
                        f"so results would be plain dense fp16 mislabelled as Kitty. Refusing to "
                        f"proceed. See kitty_sim/glm_kitty_patch.py."
                    )
                elif variant.shadowkv:
                    # Pure-torch ShadowKV: prove the decode hook ran (vs the
                    # attention never being patched / the cache bypassed -> silent
                    # dense fp16). last_selected_chunks is informational: short
                    # contexts legitimately select few/zero chunks.
                    decode_calls = int(kitty_stats.get("decode_calls", 0)) if kitty_stats else 0
                    seqlen = kv_cache.get_seq_length() if kv_cache is not None else 0
                    engaged = (
                        kv_cache is not None and seqlen > 0
                        and kitty_stats is not None
                        and int(kitty_stats.get("installed", 0)) > 0
                        and decode_calls > 0
                    )
                    detail = (
                        f"installed={kitty_stats.get('installed') if kitty_stats else None} "
                        f"decode_calls={decode_calls} seq_length={seqlen} "
                        f"last_selected_chunks={kitty_stats.get('last_selected_chunks') if kitty_stats else None}"
                    )
                    print(f"[shadowkv] {dataset} first-sample evidence: {detail}")
                    raise_msg = (
                        f"ShadowKV variant '{variant.name}' did not run the pure-torch ShadowKV "
                        f"decode hook for model_family='{model_family}' ({detail}). Either the "
                        f"attention forward was not patched or the ShadowKVSimCache was bypassed, "
                        f"so results would be plain dense fp16 mislabelled as ShadowKV. "
                        f"Refusing to proceed. See kitty_sim/shadowkv_sim.py."
                    )
                else:
                    engaged = kv_cache is not None and kv_cache.get_seq_length() > 0
                    detail = (
                        f"KittyKVCache.get_seq_length()="
                        f"{kv_cache.get_seq_length() if kv_cache is not None else None}"
                    )
                    raise_msg = (
                        f"Kitty variant '{variant.name}' was requested but KV quantization "
                        f"never engaged for model_family='{model_family}' ({detail}). The model "
                        f"bypassed the KittyKVCache (e.g. a legacy tuple-cache remote modeling), "
                        f"so results would be plain dense fp16 mislabelled as Kitty. Refusing to "
                        f"proceed. See kitty_sim/glm_kitty_patch.py."
                    )
                if not engaged:
                    raise RuntimeError(raise_msg)
                kitty_engagement_checked = True
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
    output_root = args.output_dir
    flat_output_dir = getattr(args, "flat_output_dir", False)
    if output_root in (None, ""):
        output_root = default_prediction_dir(args.model, args.model_path, variant, args.max_samples)
        flat_output_dir = True
    elif Path(output_root) == Path("longbench_out/pred"):
        raise ValueError("longbench_out/pred is retired; use the normalized longbench_out/<model>-<method>/pred layout")
    pred_dir = resolve_prediction_dir(
        output_root,
        model_tag,
        variant,
        flat_output_dir=flat_output_dir,
    )
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

    # Validate a per-layer promote_ratio schedule against the model's real layer
    # count now that the model is loaded (build_variant only checked ratios).
    if variant.promote_ratio_per_layer:
        n_layers = int(getattr(model_obj.config, "num_hidden_layers", 0))
        bad = [idx for idx, _ in variant.promote_ratio_per_layer if not (0 <= idx < n_layers)]
        if bad:
            raise ValueError(
                f"--promote-ratio-config references layer indices {bad} outside the model's "
                f"[0, {n_layers}) range (num_hidden_layers={n_layers})."
            )
        print(
            f"[per-layer-pr] {variant.name}: default={variant.promote_ratio} "
            f"overrides={dict(variant.promote_ratio_per_layer)} "
            f"(config={variant.promote_ratio_config_path})"
        )

    # GLM-family remote modeling uses a legacy tuple KV cache and never calls
    # Cache.update(), so a KittyKVCache passed via past_key_values is a no-op
    # (it silently runs dense fp16). For Kitty variants on GLM, install a
    # SelfAttention forward patch that fake-quantizes the cached KV with the same
    # KittyKVCache logic used for HF-Cache models.
    legacy_cache_model = is_glm_family(model_family)
    kitty_stats: dict[str, Any] | None = None
    if variant.shadowkv:
        # Pure-torch ShadowKV hook on the stock model; the ShadowKVSimCache
        # (passed per sample via past_key_values) holds the SVD/landmark/buffer
        # state, since the HF Cache.update interface never sees the query.
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
            f"[shadowkv] installed pure-torch ShadowKV sim hook on {kitty_stats['installed']} layers "
            f"(variant={variant.tag}, budget={variant.sparse_budget}, rank={variant.rank}, "
            f"chunk={variant.chunk_size})"
        )
    elif variant.use_kitty and legacy_cache_model:
        kitty_stats = install_glm_kitty_fakequant(
            model_obj, cache_config_from_variant(variant)
        )
        print(
            f"[kitty] GLM legacy-cache model: installed SelfAttention fake-quant on "
            f"{kitty_stats['installed']} layers (variant={variant.tag})"
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
                legacy_cache_model=legacy_cache_model,
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
