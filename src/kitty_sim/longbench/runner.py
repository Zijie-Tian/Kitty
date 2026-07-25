"""LongBench generation runner for Kitty KV-cache variants."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from kitty_sim import get_kvcache_kitty
from kitty_sim.glm_kitty_patch import (
    cache_config_from_variant,
    install_glm_kitty_fakequant,
    is_glm_family,
)

from .config import LONG_BENCH_DATASETS, LONG_BENCH_E_DATASETS, load_json_config
from .data import default_data_root, load_longbench_dataset
from .templates import (
    format_longbench_prompt,
    infer_model_family,
    post_process,
    use_fast_tokenizer,
)

# nf2 algorithm generation stamped into run_config_hash via VariantConfig.nf2_impl.
# "symnf2-v1" = fixed symmetric NF2 LUT (IR-QLoRA), replacing the unversioned
# Lloyd-Max era. Bump on any future nf2 semantics change.
NF2_IMPL_VERSION = "symnf2-v1"

# The single public QLUTATTN variant. K: post-RoPE per-token quant on the
# per-channel-mean-centered residual, per-channel codebook fixed by an OFFLINE
# sign/nf2 mask ranked by sigma^2 x E|q| (difficulty x query usage), 65% sign /
# 35% nf2 -> nominal 0.65*1.25 + 0.35*2.25 = 1.60 bit/value; Q stays FP16.
# V: rescued 2-bit tile16c64 (rht-pcaff-mse1-bias-v1).
QLUTATTN_VARIANT = "qlutattn"
QLUTATTN_BIN_CODEBOOKS = ("sign", "nf2")
QLUTATTN_SIGN_FRACTION = 0.65
QLUTATTN_TOP_P_SELECTION_METHOD = "layer_channel_top_p"
QLUTATTN_TOP_P_FORMAT_VERSION = 3
QLUTATTN_TOP_P_SELECTION_AXIS = "per_layer_flattened_kv_head_channel"
QLUTATTN_TOP_P_THRESHOLD_RULE = "minimal_desc_prefix_cumsum_ge_p"
QLUTATTN_TOP_P_TIE_RULE = "score_desc_flat_index_asc"
QLUTATTN_TOP_P_SCORE_DTYPE = "float64"
QLUTATTN_V_TILE_CHANNELS = 64


def _top_p_threshold_slug(value: float) -> str:
    text = repr(float(value))
    if "e" not in text.lower():
        whole, separator, fraction = text.partition(".")
        fraction = fraction if separator else ""
        text = f"{whole}.{fraction.ljust(2, '0')}"
    return text.replace(".", "p").replace("+", "")


def _qlutattn_top_p_slug(config: Any) -> str:
    if not config.pertoken_cb_mask:
        raise ValueError("top-p qlutattn requires a codebook-mask path")
    threshold = _top_p_threshold_slug(config.top_p_threshold)
    mask_digest = _sha256_file(config.pertoken_cb_mask)
    return f"qlutattn-topp-p{threshold}-m{mask_digest}"




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
    # promote) or "per_token" (K quantized like V along head_dim; qlutattn and
    # the q4_0/custom per-token modes).
    k_quant_mode: str = "per_channel"
    # ShadowKV sim: pure-torch faithful port of ShadowKV's accuracy cache (SVD
    # low-rank pre-RoPE keys + landmark chunk selection + outlier/local chunks).
    # Accuracy + relative-timing proxy; NOT a memory/speed proof.
    shadowkv: bool = False
    sparse_budget: int = 2048
    rank: int = 160
    chunk_size: int = 8
    # Per-layer promote_ratio override (kitty only): a tuple of
    # (layer_idx, ratio) pairs -- hashable (frozen dataclass safe) and
    # asdict-friendly. None => scalar promote_ratio for every layer.
    # promote_ratio_config_path keeps the source JSON path for provenance.
    promote_ratio_per_layer: tuple[tuple[int, float], ...] | None = None
    promote_ratio_config_path: str | None = None
    # K-cache codebook: 'kivi' = min-max groupwise (default path); 'qlut' = the
    # canonical qlutattn offline per-channel sign/nf2 mask (per-token);
    # 'q4_0' = llama.cpp Q4_0 (per-token 32-channel symmetric blocks).
    k_codebook: str = "kivi"
    # V-cache codebook: 'kivi' (configured-group min-max), 'q4_0', or the
    # rescued 2-bit 'tile16_rescued' (qlutattn).
    v_codebook: str = "kivi"
    # Rescued V tile16cC provenance (None on non-tile variants).
    v_tile_tokens: int | None = None
    v_tile_channels: int | None = None
    v_tile_algo_version: str | None = None
    v_rht_seed: int | None = None
    v_mse_iters: int | None = None
    # qlut only: per-channel codebook names, mask value -> codebook. Fixed to
    # ("sign", "nf2") for the canonical qlutattn variant.
    bin_codebooks: tuple[str, ...] | None = None
    # qlutattn: path to the OFFLINE per-(layer, kv-head, channel) codebook mask
    # (scripts/calibrate_qlutattn_mask.py). Required for the qlut K codebook.
    pertoken_cb_mask: Optional[str] = None
    # Optional top-p research metadata. Canonical qlutattn leaves these unset.
    selection_method: Optional[str] = None
    top_p_threshold: Optional[float] = None
    actual_nf2_frac: Optional[float] = None
    # Exact theoretical packed-K accounting for top-p artifacts.
    actual_nf2_count: Optional[int] = None
    packed_bits_total: Optional[int] = None
    packed_k_values_total: Optional[int] = None
    # nf2 implementation marker: "symnf2-v1" whenever the effective codebooks include
    # "nf2" (static bin_codebooks or the offline mask blob), None otherwise. Purely a
    # run_config_hash version gate -- Lloyd-era nf2 manifests must NOT be resumed or
    # reused after the symmetric-NF2 (IR-QLoRA LUT) switch. variant_semantic_payload
    # drops the key when None so non-nf2 variants keep their historical hashes.
    nf2_impl: Optional[str] = None
    # Triton QUEST page-selection overlay for sim/fake-quant variants.
    # Query-aware sparse decode lives in an attention-forward hook because the HF
    # Cache.update() interface does not receive Q. The decode attention itself is
    # a Triton kernel over the dense fake-quant K/V tensors.
    quest_kernel: bool = False
    quest_token_budget: int | None = None
    quest_skip_layers: int = 0

    @property
    def tag(self) -> str:
        if not self.use_kitty:
            return "fp16"
        ratio = str(self.promote_ratio).replace(".", "p")
        if self.k_codebook == "qlut":
            if self.selection_method == QLUTATTN_TOP_P_SELECTION_METHOD:
                return _qlutattn_top_p_slug(self)
            if self.selection_method is not None:
                raise ValueError(
                    f"unsupported qlutattn selection_method="
                    f"{self.selection_method!r}"
                )
            return self.name
        if self.shadowkv:
            return f"{self.name}_sb{self.sparse_budget}_r{self.rank}_c{self.chunk_size}"
        suffix = ""
        if self.promote_ratio_per_layer:
            h = hashlib.sha256(repr(self.promote_ratio_per_layer).encode()).hexdigest()[:6]
            suffix = f"-prcfg{h}"
        if self.k_quant_mode == "per_token":
            suffix += "_kpt"
        base = (
            f"{self.name}_g{self.group_size}_b{self.buffer_length}_s{self.sink_length}"
            f"_sel{self.channel_selection}_k{self.kbits}_v{self.vbits}"
            f"_pb{self.promote_bit}_pr{ratio}{suffix}"
        )
        if self.quest_kernel:
            base += f"_qb{self.quest_token_budget}_qsl{self.quest_skip_layers}_questkernel"
        return base


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_enable_quest_kernel(variant: VariantConfig, args: Any) -> VariantConfig:
    requested = (
        bool(getattr(args, "quest_kernel", False))
        or _env_flag("QUEST_KERNEL")
        or _env_flag("QUEST_TRITON")
        or _env_flag("SIM_QUEST")
        or _env_flag("QUEST_SIM")
    )
    if not requested:
        return variant
    if not variant.use_kitty:
        raise ValueError("QUEST_KERNEL=1 requires a Kitty-style KV-cache variant, not fp16.")
    if variant.shadowkv:
        raise ValueError("QUEST_KERNEL=1 cannot be combined with shadowkv.")
    if variant.name == QLUTATTN_VARIANT:
        raise ValueError(
            "QUEST_KERNEL/QUEST_TRITON/SIM_QUEST/QUEST_SIM cannot be combined with "
            "--variant qlutattn: the canonical qlutattn algorithm is fixed and a "
            "QUEST overlay would silently turn it into a different method. "
            "Use QUEST with the kitty/kivi variants instead."
        )
    budget = (
        getattr(args, "quest_token_budget", None)
        or os.environ.get("QUEST_TOKEN_BUDGET")
        or os.environ.get("QUEST_BUDGET")
        or 2048
    )
    skip_layers = (
        getattr(args, "quest_skip_layers", None)
        or os.environ.get("QUEST_SKIP_LAYERS")
        or 0
    )
    return replace(
        variant,
        quest_kernel=True,
        quest_token_budget=int(budget),
        quest_skip_layers=int(skip_layers),
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


def _reject_conflicting_vbits(args: Any, variant_name: str) -> None:
    """qlutattn hard-locks the final parsed vbits value to two.

    ``scripts/run_exp.sh`` translates ``VBITS`` into ``--vbits``, so checking the
    parsed argument here preserves the repository-wide CLI-over-environment rule.
    """
    vb = getattr(args, "vbits", None)
    if vb is None:
        raw = os.environ.get("VBITS", "").strip()
        vb = int(raw) if raw else 2
    if int(vb) != 2:
        raise ValueError(
            f"{variant_name} requires vbits=2; got --vbits/{vb}"
        )


def _reject_retired_env(variant_name: str, env_names: tuple[str, ...]) -> None:
    """Retired knobs must never (silently or otherwise) reconfigure a variant.

    The canonical qlutattn algorithm is fixed; a non-empty value in any of
    these environment variables signals a stale launcher and is a hard error.
    Empty string is the explicit unset sentinel.
    """
    for env_name in env_names:
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip() != "":
            raise ValueError(
                f"variant {variant_name}: the retired knob "
                f"{env_name}={raw!r} must be unset (or empty)."
            )


def _reject_pertoken_block_env() -> None:
    """PERTOKEN_BLOCK is retired: only unset, empty, or '1' are accepted."""
    raw = os.environ.get("PERTOKEN_BLOCK")
    if raw is None or raw.strip() in ("", "1"):
        return
    raise ValueError(
        f"PERTOKEN_BLOCK is retired (block-shared per-token codebooks were "
        f"removed); got PERTOKEN_BLOCK={raw!r}. Unset it or set it to 1."
    )


def _validate_qlutattn_top_p_blob(
    blob: dict[str, Any], mask_path: str, mask: torch.Tensor
) -> None:
    """Validate the layer-global top-p selection and head-local layout."""
    shape = tuple(mask.shape)
    n_layers, n_heads, head_dim = shape
    if not all(dim > 0 for dim in shape):
        raise ValueError(
            f"qlutattn top-p codebook_mask dimensions must be positive; got {shape}"
        )

    def require_tensor(
        key: str, expected_shape: tuple[int, ...], dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        value = blob.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"qlutattn top-p mask {mask_path} requires tensor field {key!r}"
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"qlutattn top-p field {key!r} must have shape {expected_shape}; "
                f"got {tuple(value.shape)}"
            )
        if dtype is not None and value.dtype != dtype:
            raise ValueError(
                f"qlutattn top-p field {key!r} must have dtype {dtype}; "
                f"got {value.dtype}"
            )
        return value

    identity_fields = {
        "format_version": QLUTATTN_TOP_P_FORMAT_VERSION,
        "selection_axis": QLUTATTN_TOP_P_SELECTION_AXIS,
        "threshold_rule": QLUTATTN_TOP_P_THRESHOLD_RULE,
        "tie_rule": QLUTATTN_TOP_P_TIE_RULE,
        "score_dtype_for_selection": QLUTATTN_TOP_P_SCORE_DTYPE,
        "nf2_impl": NF2_IMPL_VERSION,
        "ranking_signal": "sigma2_x_q",
    }
    for key, expected in identity_fields.items():
        actual = blob.get(key)
        if (key == "format_version" and isinstance(actual, bool)) or actual != expected:
            raise ValueError(
                f"qlutattn top-p field {key!r} must equal {expected!r}; "
                f"got {actual!r}"
            )
        if key == "format_version" and (
            isinstance(actual, bool) or not isinstance(actual, int)
        ):
            raise ValueError(
                "qlutattn top-p field 'format_version' must be an integer"
            )
    if blob.get("codebooks") != list(QLUTATTN_BIN_CODEBOOKS):
        raise ValueError(
            "qlutattn top-p codebooks must be exactly ['sign', 'nf2']"
        )
    for key in ("model", "calib_data"):
        value = blob.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"qlutattn top-p field {key!r} must be a nonempty string; "
                f"got {value!r}"
            )
    for key in ("group_size", "num_samples", "sample_len"):
        value = blob.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"qlutattn top-p field {key!r} must be a positive integer; "
                f"got {value!r}"
            )
    skip_first = blob.get("skip_first")
    if (
        isinstance(skip_first, bool)
        or not isinstance(skip_first, int)
        or skip_first < 0
    ):
        raise ValueError(
            "qlutattn top-p field 'skip_first' must be a nonnegative integer; "
            f"got {skip_first!r}"
        )
    seed = blob.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"qlutattn top-p field 'seed' must be an integer; got {seed!r}"
        )
    for key, expected in (
        ("n_layers", n_layers),
        ("n_kv", n_heads),
        ("head_dim", head_dim),
    ):
        value = blob.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"qlutattn top-p field {key!r} must match codebook_mask "
                f"geometry ({expected}); got {value!r}"
            )

    threshold_raw = blob.get("top_p_threshold")
    if not isinstance(threshold_raw, float) or not math.isfinite(threshold_raw):
        raise ValueError(
            f"qlutattn top-p mask {mask_path} requires a finite float "
            f"top_p_threshold; got {threshold_raw!r}"
        )
    threshold = float(threshold_raw)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"qlutattn top_p_threshold must be in (0, 1]; got {threshold}"
        )

    values = set(torch.unique(mask).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(
            "qlutattn top-p codebook_mask values must be in {0, 1} "
            f"(0=sign, 1=nf2); got {sorted(values)}"
        )

    expected_int_fields = {
        "actual_nf2_count": int(mask.sum()),
        "packed_bits_total": _qlutattn_packed_k_bits_total(mask),
        "packed_k_values_total": mask.numel(),
    }
    for key, expected in expected_int_fields.items():
        value = blob.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"qlutattn top-p field {key!r} must equal recomputed "
                f"value {expected}; got {value!r}"
            )

    reorder = require_tensor("reorder_index", shape, torch.int64)
    inverse = require_tensor("inverse_reorder_index", shape, torch.int64)
    count_per_head = require_tensor(
        "nf2_count_per_head", (n_layers, n_heads), torch.int32
    )
    count_per_layer = require_tensor(
        "nf2_count_per_layer", (n_layers,), torch.int32
    )
    actual_count_per_head = (mask == 1).sum(dim=-1).to(torch.int32)
    actual_count_per_layer = actual_count_per_head.sum(dim=-1).to(torch.int32)
    if not torch.equal(count_per_head, actual_count_per_head):
        raise ValueError(
            "qlutattn top-p nf2_count_per_head does not match codebook_mask"
        )
    if not torch.equal(count_per_layer, actual_count_per_layer):
        raise ValueError(
            "qlutattn top-p nf2_count_per_layer does not match codebook_mask"
        )

    identity = torch.arange(head_dim, dtype=torch.int64).view(1, 1, head_dim)
    identity = identity.expand(n_layers, n_heads, head_dim)
    if not torch.equal(torch.sort(reorder, dim=-1).values, identity):
        raise ValueError(
            "qlutattn top-p reorder_index must be a permutation of every head"
        )
    if not torch.equal(torch.sort(inverse, dim=-1).values, identity):
        raise ValueError(
            "qlutattn top-p inverse_reorder_index must be a permutation of every head"
        )
    if not torch.equal(torch.gather(reorder, -1, inverse), identity):
        raise ValueError(
            "qlutattn top-p inverse_reorder_index is not the inverse of reorder_index"
        )

    expected_reorder = torch.empty_like(reorder)
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            head_mask = mask[layer_idx, head_idx]
            expected_reorder[layer_idx, head_idx] = torch.cat(
                (
                    torch.nonzero(head_mask == 1, as_tuple=False).flatten(),
                    torch.nonzero(head_mask == 0, as_tuple=False).flatten(),
                )
            )
    if not torch.equal(reorder, expected_reorder):
        raise ValueError(
            "qlutattn top-p reorder_index must be the original-index-stable "
            "head-local [NF2 prefix | sign suffix] permutation"
        )
    reordered_mask = torch.gather(mask, -1, reorder)
    expected_prefix = (
        torch.arange(head_dim).view(1, 1, head_dim)
        < count_per_head.unsqueeze(-1)
    ).to(torch.uint8)
    if not torch.equal(reordered_mask, expected_prefix):
        raise ValueError(
            "qlutattn top-p reorder/count metadata does not form an NF2 prefix"
        )

    sigma2 = require_tensor("sigma2", shape)
    q_absmean = require_tensor("q_absmean", shape)
    ranking = require_tensor("ranking_score", shape, torch.float64)
    for key, statistic in (("sigma2", sigma2), ("q_absmean", q_absmean)):
        if not torch.is_floating_point(statistic):
            raise ValueError(
                f"qlutattn top-p field {key!r} must be floating point; "
                f"got {statistic.dtype}"
            )
        if not bool(torch.isfinite(statistic).all()) or bool((statistic < 0).any()):
            raise ValueError(
                f"qlutattn top-p field {key!r} must contain finite nonnegative values"
            )
    if not bool(torch.isfinite(ranking).all()) or bool((ranking < 0).any()):
        raise ValueError(
            "qlutattn top-p ranking_score must contain finite nonnegative values"
        )
    if not torch.equal(ranking, sigma2.double() * q_absmean.double()):
        raise ValueError(
            "qlutattn top-p ranking_score must equal sigma2.double() * "
            "q_absmean.double() exactly"
        )

    ratio_per_layer = require_tensor(
        "nf2_ratio_per_layer", (n_layers,), torch.float64
    )
    captured = require_tensor(
        "captured_mass_per_layer", (n_layers,), torch.float64
    )
    previous = require_tensor(
        "previous_prefix_mass_per_layer", (n_layers,), torch.float64
    )
    boundary = require_tensor(
        "boundary_score_per_layer", (n_layers,), torch.float64
    )
    for key, statistic in (
        ("nf2_ratio_per_layer", ratio_per_layer),
        ("captured_mass_per_layer", captured),
        ("previous_prefix_mass_per_layer", previous),
        ("boundary_score_per_layer", boundary),
    ):
        if not bool(torch.isfinite(statistic).all()):
            raise ValueError(
                f"qlutattn top-p field {key!r} must contain only finite values"
            )

    expected_mask = torch.zeros_like(mask).reshape(n_layers, -1)
    expected_counts = torch.empty(n_layers, dtype=torch.int32)
    expected_captured = torch.empty(n_layers, dtype=torch.float64)
    expected_previous = torch.empty(n_layers, dtype=torch.float64)
    expected_boundary = torch.empty(n_layers, dtype=torch.float64)
    flat_ranking = ranking.reshape(n_layers, -1)
    for layer_idx in range(n_layers):
        order = torch.argsort(
            flat_ranking[layer_idx], descending=True, stable=True
        )
        ordered_scores = flat_ranking[layer_idx, order]
        cumulative = ordered_scores.cumsum(dim=0)
        total = cumulative[-1]
        if not bool(total > 0):
            raise ValueError(
                f"qlutattn top-p ranking_score layer {layer_idx} must have "
                "positive total mass"
            )
        target = total * threshold
        boundary_idx = int(torch.searchsorted(cumulative, target, right=False))
        if boundary_idx >= order.numel():
            raise ValueError(
                f"qlutattn top-p threshold {threshold} is unreachable at "
                f"layer {layer_idx}"
            )
        count = boundary_idx + 1
        expected_mask[layer_idx, order[:count]] = 1
        expected_counts[layer_idx] = count
        expected_captured[layer_idx] = cumulative[count - 1] / total
        expected_previous[layer_idx] = (
            cumulative[count - 2] / total if count > 1 else 0.0
        )
        expected_boundary[layer_idx] = ordered_scores[count - 1]

    if not torch.equal(mask.reshape(n_layers, -1), expected_mask):
        raise ValueError(
            "qlutattn top-p codebook_mask is not the deterministic "
            "score-desc/flat-index prefix selected by top_p_threshold"
        )
    if not torch.equal(count_per_layer, expected_counts):
        raise ValueError(
            "qlutattn top-p nf2_count_per_layer does not match the selected prefix"
        )
    expected_ratio = expected_counts.double() / float(n_heads * head_dim)
    if not torch.equal(ratio_per_layer, expected_ratio):
        raise ValueError(
            "qlutattn top-p nf2_ratio_per_layer does not match selected counts"
        )
    for key, actual, expected in (
        ("captured_mass_per_layer", captured, expected_captured),
        ("previous_prefix_mass_per_layer", previous, expected_previous),
        ("boundary_score_per_layer", boundary, expected_boundary),
    ):
        if not torch.equal(actual, expected):
            raise ValueError(
                f"qlutattn top-p {key} does not match ranking_score selection"
            )

    actual_nf2 = blob.get("actual_nf2_frac")
    low_frac = blob.get("low_frac")
    if not isinstance(actual_nf2, float) or not math.isfinite(actual_nf2):
        raise ValueError(
            f"qlutattn top-p actual_nf2_frac must be a finite float; "
            f"got {actual_nf2!r}"
        )
    if not isinstance(low_frac, float) or not math.isfinite(low_frac):
        raise ValueError(
            f"qlutattn top-p low_frac must be a finite float; got {low_frac!r}"
        )
    expected_actual_nf2 = int(actual_count_per_layer.sum()) / mask.numel()
    expected_low_frac = (mask.numel() - int(actual_count_per_layer.sum())) / mask.numel()
    if actual_nf2 != expected_actual_nf2:
        raise ValueError(
            "qlutattn top-p actual_nf2_frac does not match codebook_mask"
        )
    if low_frac != expected_low_frac:
        raise ValueError(
            "qlutattn top-p low_frac does not match codebook_mask"
        )


def _qlutattn_packed_k_bits_total(codebook_mask: torch.Tensor) -> int:
    """Exact packed K bits for one token over a full QLUTATTN mask."""
    if (
        codebook_mask.ndim != 3
        or codebook_mask.dtype != torch.uint8
        or not bool(((codebook_mask == 0) | (codebook_mask == 1)).all())
    ):
        raise ValueError(
            "qlutattn packed-bit accounting requires a uint8 "
            "[layer, kv_head, channel] sign/NF2 mask"
        )
    head_dim = codebook_mask.shape[-1]
    nf2_counts = codebook_mask.to(torch.int64).sum(dim=-1)
    code_bits = int(((head_dim - nf2_counts) + 2 * nf2_counts).sum())
    scale_bits = 16 * int((nf2_counts > 0).sum())
    scale_bits += 16 * int((nf2_counts < head_dim).sum())
    return code_bits + scale_bits


def load_qlutattn_mask_blob(mask_path: str) -> dict[str, Any]:
    """Load and strictly validate a canonical or top-p QLUTATTN mask."""
    blob = torch.load(mask_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"qlutattn mask {mask_path} must be a dict payload")
    codebooks = tuple(blob.get("codebooks") or ())
    if codebooks != QLUTATTN_BIN_CODEBOOKS:
        raise ValueError(
            f"qlutattn requires codebooks={list(QLUTATTN_BIN_CODEBOOKS)} in the "
            f"mask payload; got {list(codebooks)} in {mask_path}"
        )
    mask = blob.get("codebook_mask")
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"qlutattn mask {mask_path} has no codebook_mask tensor")
    if mask.ndim != 3:
        raise ValueError(
            f"qlutattn codebook_mask must be [n_layers, n_kv_heads, head_dim] "
            f"(ndim=3); got shape {tuple(mask.shape)}"
        )
    if mask.dtype != torch.uint8:
        raise ValueError(
            f"qlutattn codebook_mask must be torch.uint8; got {mask.dtype}"
        )

    selection_method = blob.get("selection_method")
    if selection_method == QLUTATTN_TOP_P_SELECTION_METHOD:
        _validate_qlutattn_top_p_blob(blob, mask_path, mask)
        return blob
    if selection_method is not None:
        raise ValueError(
            f"unsupported qlutattn selection_method={selection_method!r} "
            f"in {mask_path}"
        )
    removed_control_fields = (
        "control_kind",
        "reference_mask_sha256",
        "reference_semantic_sha256",
        "reference_selection_method",
        "reference_top_p_threshold",
        "reference_codebook_mask",
        "reference_nf2_count",
        "reference_packed_bits_total",
    )
    present_control_fields = sorted(
        key for key in removed_control_fields if key in blob
    )
    if present_control_fields:
        raise ValueError(
            "uniform top-k control artifacts are no longer supported; "
            "use a canonical fixed-65/35 or layer-channel top-p artifact. "
            f"Found removed fields: {present_control_fields}"
        )

    top_p_only_fields = (
        "format_version",
        "selection_axis",
        "threshold_rule",
        "tie_rule",
        "score_dtype_for_selection",
        "top_p_threshold",
        "reorder_index",
        "inverse_reorder_index",
        "nf2_count_per_head",
        "nf2_count_per_layer",
        "nf2_ratio_per_layer",
        "captured_mass_per_layer",
        "previous_prefix_mass_per_layer",
        "boundary_score_per_layer",
        "actual_nf2_frac",
        "ranking_score",
    )
    orphaned_top_p_fields = sorted(
        key for key in top_p_only_fields if key in blob
    )
    if orphaned_top_p_fields:
        raise ValueError(
            "qlutattn top-p metadata requires "
            f"selection_method={QLUTATTN_TOP_P_SELECTION_METHOD!r}; "
            f"found top-p-only fields without it: {orphaned_top_p_fields}"
        )

    values = set(torch.unique(mask).tolist())
    if values != {0, 1}:
        raise ValueError(
            f"qlutattn codebook_mask values must be exactly {{0, 1}} "
            f"(0=sign, 1=nf2); got {sorted(values)}"
        )
    n_per_layer = mask.shape[1] * mask.shape[2]
    expected_sign = int(round(QLUTATTN_SIGN_FRACTION * n_per_layer))
    per_layer_sign = (mask == 0).reshape(mask.shape[0], -1).sum(dim=1)
    if not bool((per_layer_sign == expected_sign).all()):
        bad = {
            int(i): int(c)
            for i, c in enumerate(per_layer_sign.tolist())
            if c != expected_sign
        }
        raise ValueError(
            f"qlutattn requires exactly {expected_sign}/{n_per_layer} sign channels "
            f"per layer (= round({QLUTATTN_SIGN_FRACTION} * n_kv * head_dim)); "
            f"mask in {mask_path} deviates at layers {bad}"
        )
    sign_frac = float((mask == 0).float().mean())
    low_frac = blob.get("low_frac")
    if low_frac is None or abs(float(low_frac) - sign_frac) > 1e-6:
        raise ValueError(
            f"qlutattn requires low_frac metadata equal to the actual sign "
            f"fraction {sign_frac:.6f}; got {low_frac!r} in {mask_path}"
        )
    return blob


def validate_qlutattn_model_family(variant: VariantConfig, model_family: str | None) -> None:
    """Fail before model loading when a legacy-cache GLM would bypass this path."""
    if variant.name == QLUTATTN_VARIANT and is_glm_family(model_family):
        raise ValueError(
            f"GLM parity for {variant.name} is not implemented yet; "
            "refusing to silently fall back to KIVI V. Use a non-GLM model family."
        )


def validate_qlutattn_model_config(
    variant: VariantConfig, model_config: Any, model_dtype: torch.dtype
) -> None:
    """Validate model-dependent qlutattn constraints before any sample loop."""
    if variant.name != QLUTATTN_VARIANT:
        return
    if model_dtype != torch.float16:
        raise ValueError(
            f"{variant.name} requires an FP16 model for the fake-quant path; "
            f"got model dtype={model_dtype}"
        )
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(model_config, "hidden_size", None)
        n_heads = getattr(model_config, "num_attention_heads", None)
        if hidden_size is None or not n_heads:
            raise ValueError("Cannot infer head_dim from model config")
        if int(hidden_size) % int(n_heads) != 0:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by "
                f"num_attention_heads={n_heads}; cannot infer head_dim"
            )
        head_dim = int(hidden_size) // int(n_heads)
    head_dim = int(head_dim)
    C = int(variant.v_tile_channels)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got head_dim={head_dim}")
    if head_dim % C != 0:
        raise ValueError(
            f"head_dim={head_dim} must be divisible by v_tile_channels={C}"
        )
    if head_dim & (head_dim - 1):
        raise ValueError(
            f"Hadamard/RHT requires power-of-two head_dim, got head_dim={head_dim}"
        )
    # The offline codebook mask must match this exact model geometry.
    n_layers = getattr(model_config, "num_hidden_layers", None)
    n_kv = getattr(model_config, "num_key_value_heads", None) or getattr(
        model_config, "num_attention_heads", None
    )
    if n_layers is None or n_kv is None:
        raise ValueError(
            "Cannot resolve num_hidden_layers/num_key_value_heads from the "
            "model config for qlutattn mask shape validation"
        )
    blob = load_qlutattn_mask_blob(variant.pertoken_cb_mask)
    got = tuple(blob["codebook_mask"].shape)
    expected = (int(n_layers), int(n_kv), head_dim)
    if got != expected:
        raise ValueError(
            f"qlutattn codebook_mask shape {got} does not match the model's "
            f"[num_layers, num_key_value_heads, head_dim] = {expected}; "
            f"recalibrate with scripts/calibrate_qlutattn_mask.py"
        )


def _torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[str(name).lower()]


def validate_qlutattn_preload(args: Any, variant: VariantConfig, model_family: str) -> None:
    """Config-only validation before scheduling or loading model weights.

    AutoConfig may read a local config or the normal HF metadata cache/network,
    but it never creates a model or CUDA context.  The same checks run again on
    the loaded model config to guard custom/remote modeling discrepancies.
    """
    if variant.name != QLUTATTN_VARIANT:
        return
    validate_qlutattn_model_family(variant, model_family)
    source = getattr(args, "model_path", None) or getattr(args, "model", None)
    if not source:
        raise ValueError(f"{variant.name} preflight requires model metadata")
    try:
        config = AutoConfig.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=bool(getattr(args, "local_files_only", False)),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load model metadata for {variant.name} from {source!r}; "
            "refusing to schedule before head_dim/dtype/mask validation"
        ) from exc
    validate_qlutattn_model_config(
        variant,
        config,
        _torch_dtype_from_name(getattr(args, "torch_dtype", "float16")),
    )


def _apply_nf2_impl_marker(config: VariantConfig) -> VariantConfig:
    """Stamp nf2_impl on variants whose codebooks include "nf2" (qlutattn).

    This is a pure run_config_hash version gate for the symmetric-NF2 switch
    (Lloyd-era nf2 results must not be resumed/reused). Variants without nf2
    keep nf2_impl=None, and variant_semantic_payload drops the None key, so
    their historical hashes stay byte-identical. For qlutattn the offline mask
    blob's codebooks are validated to equal the static ("sign", "nf2") list in
    load_qlutattn_mask_blob, so the static list is authoritative here."""
    if "nf2" in (config.bin_codebooks or ()):
        return replace(config, nf2_impl=NF2_IMPL_VERSION)
    return config


def build_variant(args: Any) -> VariantConfig:
    return _apply_nf2_impl_marker(_build_variant_impl(args))


def _build_variant_impl(args: Any) -> VariantConfig:
    variant = args.variant.lower()
    # Retired knobs are rejected for every variant: no launcher may carry them.
    _reject_pertoken_block_env()
    _reject_retired_env(
        variant, ("QLUT_BIN_CODEBOOKS", "V_TILE_CHANNELS", "PERTOKEN_OUTLIER_K")
    )
    config_path = getattr(args, "promote_ratio_config", None)
    if config_path and variant != "kitty":
        raise ValueError(
            "--promote-ratio-config is only supported for --variant kitty; "
            f"got '{variant}'."
        )
    if variant == "fp16":
        return VariantConfig(name="fp16", use_kitty=False, promote_ratio=0.0)
    if variant == "kitty":
        # Paper-style Kitty machinery (magnitude channel-select + sink=32),
        # generalized so K base (kbits), boost (promote_bit), boost fraction
        # (promote_ratio) and V (vbits) are all tunable. Defaults reproduce the
        # paper Kitty (k2 / boost4 / pr0.125 / v2). promote_ratio is per-layer
        # when --promote-ratio-config is given (JSON {"default": r, "layers": {..}}
        # or a bare [r0, r1, ...]); else the scalar --promote_ratio (default
        # 0.125) applies to every layer. Subsumes the old kitty_pro (pr=0.25) and
        # kitty_k1v4 (k1 / boost2 / v4 / pr0.25).
        kb = int(getattr(args, "kbits", 2))
        vb = int(getattr(args, "vbits", 2))
        pbit = int(getattr(args, "promote_bit", 4))
        if not (1 <= kb <= 16 and 1 <= vb <= 16 and 1 <= pbit <= 16):
            raise ValueError(
                f"kitty kbits/vbits/promote_bit must be in [1, 16]; "
                f"got kbits={kb}, vbits={vb}, promote_bit={pbit}.")
        default_ratio = 0.125
        per_layer = None
        if config_path:
            default_ratio, per_layer = _load_promote_ratio_config(config_path, default_ratio)
        else:
            pr = getattr(args, "promote_ratio", None)
            if pr is not None:
                default_ratio = float(pr)
        return VariantConfig(
            name="kitty", use_kitty=True,
            kbits=kb, vbits=vb, promote_bit=pbit, promote_ratio=default_ratio,
            channel_selection=1, sink_length=32, buffer_length=128, group_size=128,
            promote_ratio_per_layer=per_layer,
            promote_ratio_config_path=config_path,
        )
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
    if variant == "qlutattn":
        # The single canonical QLUTATTN variant.
        #   Q: FP16, untouched.
        #   K: post-RoPE per-token quant along head_dim. A per-channel mean
        #      mu_d is self-calibrated from each prompt at prefill (free for
        #      attention: q.mu is a per-query constant that cancels in softmax)
        #      and subtracted; the residual is quantized per token with a fixed
        #      per-channel codebook assignment loaded from an OFFLINE mask
        #      (scripts/calibrate_qlutattn_mask.py, ranking = sigma^2 x E|q|):
        #      65% lowest-ranked channels -> sign (1-bit + per-token |r| mean
        #      scale), 35% highest -> fixed symmetric NF2 (symnf2-v1 LUT,
        #      per-token absmax scale). Nominal 0.65*1.25 + 0.35*2.25 = 1.60
        #      bit/value. No rotation, no online binning, no block sharing.
        #   V: rescued 2-bit tile16c64 (rht-pcaff-mse1-bias-v1): 16 consecutive
        #      tokens x 64 channels per tile, RHT + frozen per-channel affine +
        #      one MSE refit + bias correction.
        #   Protection: sink=32 + recent-128 FP16 window (group_size=128).
        from kitty_sim.v_tile_quant import (
            V_MSE_ITERS,
            V_RHT_SEED,
            V_TILE_ALGO_VERSION,
            V_TILE_TOKENS,
        )
        _reject_conflicting_vbits(args, variant)
        _reject_retired_env(variant, ("PROMOTE_RATIO_CONFIG",))
        mask = os.environ.get("QLUT_CB_MASK", "")
        if not mask or not os.path.exists(mask):
            raise FileNotFoundError(
                "qlutattn requires an OFFLINE codebook mask: set "
                "QLUT_CB_MASK=/path/to/mask.pt (generate via "
                "scripts/calibrate_qlutattn_mask.py). "
                f"Got QLUT_CB_MASK='{mask}'")
        mask_blob = load_qlutattn_mask_blob(mask)
        selection_method = mask_blob.get("selection_method")
        return VariantConfig(
            name=QLUTATTN_VARIANT, use_kitty=True, k_codebook="qlut",
            bin_codebooks=QLUTATTN_BIN_CODEBOOKS, vbits=2,
            v_codebook="tile16_rescued", v_tile_tokens=V_TILE_TOKENS,
            v_tile_channels=QLUTATTN_V_TILE_CHANNELS,
            v_tile_algo_version=V_TILE_ALGO_VERSION, v_rht_seed=V_RHT_SEED,
            v_mse_iters=V_MSE_ITERS, promote_ratio=0.0, channel_selection=0,
            k_quant_mode="per_token", pertoken_cb_mask=mask,
            selection_method=selection_method,
            top_p_threshold=(
                float(mask_blob["top_p_threshold"])
                if selection_method == QLUTATTN_TOP_P_SELECTION_METHOD else None
            ),
            actual_nf2_frac=(
                float(mask_blob["actual_nf2_frac"])
                if selection_method == QLUTATTN_TOP_P_SELECTION_METHOD else None
            ),
            actual_nf2_count=(
                int(mask_blob["actual_nf2_count"])
                if selection_method == QLUTATTN_TOP_P_SELECTION_METHOD else None
            ),
            packed_bits_total=(
                int(mask_blob["packed_bits_total"])
                if selection_method == QLUTATTN_TOP_P_SELECTION_METHOD else None
            ),
            packed_k_values_total=(
                int(mask_blob["packed_k_values_total"])
                if selection_method == QLUTATTN_TOP_P_SELECTION_METHOD else None
            ))
    if variant in ("llamacpp_q40", "llamacpp-q40"):
        # llama.cpp Q4_0 KV cache, faithful port (sim fake-quant): K and V both
        # per-token with 32-channel symmetric absmax blocks (d = signed_max/-8,
        # fp16 scale -> 4.5 bit/value), quantize-on-write. Unlike every other
        # variant here there is NO sink and NO fp16 recent window (buffer=0).
        # kbits/vbits=4 are bookkeeping only (the q4_0 codebook fixes the width);
        # group_size=32 documents the block size (unused by the q4_0 kernel).
        return VariantConfig(
            name="llamacpp_q40", use_kitty=True,
            k_quant_mode="per_token", k_codebook="q4_0", v_codebook="q4_0",
            kbits=4, vbits=4, promote_ratio=0.0, channel_selection=0,
            sink_length=0, buffer_length=0, group_size=32)
    if variant in ("llamacpp_q40_star", "llamacpp-q40-star"):
        # Q4_0 codebook under the Kitty protection policy (sink=32 + recent-128
        # fp16 window): isolates how much of the q4_0 gap comes from the codebook
        # itself vs from having no sink/recent protection.
        return VariantConfig(
            name="llamacpp_q40_star", use_kitty=True,
            k_quant_mode="per_token", k_codebook="q4_0", v_codebook="q4_0",
            kbits=4, vbits=4, promote_ratio=0.0, channel_selection=0,
            sink_length=32, buffer_length=128, group_size=32)
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
            promote_ratio=(args.promote_ratio if getattr(args, "promote_ratio", None) is not None else 0.0),
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
        v_codebook=config.v_codebook,
        bin_codebooks=(list(config.bin_codebooks) if config.bin_codebooks else None),
        pertoken_cb_mask=config.pertoken_cb_mask,
        v_tile_tokens=config.v_tile_tokens,
        v_tile_channels=config.v_tile_channels,
        v_tile_algo_version=config.v_tile_algo_version,
        v_rht_seed=config.v_rht_seed,
        v_mse_iters=config.v_mse_iters,
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
            slug = f"{base}-k{variant.kbits}v{variant.vbits}"
        else:
            slug = base  # str fallback: no bit info available
    # kitty encodes K base / boost / V / ratio into the slug (subsumes the old
    # kitty / kitty_pro / kitty_k1v4), e.g. kitty-k2b4v2-pr0p125.
    elif name == "kitty":
        if isinstance(variant, VariantConfig):
            pr = str(variant.promote_ratio).replace(".", "p")
            slug = f"kitty-k{variant.kbits}b{variant.promote_bit}v{variant.vbits}-pr{pr}"
        else:
            slug = "kitty"  # str fallback: no bit info available
    elif (
        name == QLUTATTN_VARIANT
        and isinstance(variant, VariantConfig)
        and variant.selection_method == QLUTATTN_TOP_P_SELECTION_METHOD
    ):
        slug = _qlutattn_top_p_slug(variant)
    elif (
        name == QLUTATTN_VARIANT
        and isinstance(variant, VariantConfig)
        and variant.selection_method is not None
    ):
        raise ValueError(
            f"unsupported qlutattn selection_method="
            f"{variant.selection_method!r}"
        )
    else:
        slug = {
            "fp16": "fp16",
            "custom": "custom-kitty",
            "shadowkv": "shadowkv",
            "qlutattn": "qlutattn",
        }.get(name, _layout_slug(name))
    # A QUEST overlay must never share an output dir with the plain variant.
    if isinstance(variant, VariantConfig) and getattr(variant, "quest_kernel", False):
        slug = f"{slug}-quest-kernel"
    return slug


def _sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def variant_semantic_payload(variant: VariantConfig) -> dict[str, Any]:
    """Portable algorithm payload: content digests replace host-local paths."""
    payload = asdict(variant)
    mask_path = payload.pop("pertoken_cb_mask", None)
    payload.pop("promote_ratio_config_path", None)
    # Drop the nf2 marker when unset so variants without nf2 keep the exact
    # pre-symnf2 payload (their historical run_config_hashes stay valid).
    if payload.get("nf2_impl") is None:
        payload.pop("nf2_impl", None)
    optional_research_fields = (
        "selection_method",
        "top_p_threshold",
        "actual_nf2_frac",
        "actual_nf2_count",
        "packed_bits_total",
        "packed_k_values_total",
    )
    for key in optional_research_fields:
        if payload.get(key) is None:
            payload.pop(key, None)
    payload["mask_sha256"] = _sha256_file(mask_path) if mask_path else None
    return payload


def variant_semantic_hash(variant: VariantConfig) -> str:
    return _stable_json_hash(variant_semantic_payload(variant))


def _longbench_data_file(data_root: str | os.PathLike[str], data_name: str) -> Path:
    return Path(data_root) / "data" / f"{data_name}.jsonl"


def _nonblank_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _model_config_digest(model_path: str | None) -> str | None:
    if not model_path:
        return None
    config_path = Path(model_path) / "config.json"
    return _sha256_file(config_path) if config_path.is_file() else None


def _model_source_identity(model_path: str | None) -> str | None:
    """Distinguish local checkpoints that share an HF model id/config."""
    if not model_path:
        return None
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path.resolve())
    return str(model_path)


def longbench_run_config_payload(
    args: Any,
    variant: VariantConfig,
    dataset: str,
) -> dict[str, Any]:
    """Build the stable per-dataset generation fingerprint without model weights."""
    dataset2prompt = load_json_config("dataset2prompt.json")
    dataset2maxlen = load_json_config("dataset2maxlen.json")
    model2maxlen = load_json_config("model2maxlen.json")
    data_name = f"{dataset}_e" if bool(getattr(args, "e", False)) else dataset
    data_root = getattr(args, "data_root", None) or default_data_root()
    data_path = _longbench_data_file(data_root, data_name)
    if not data_path.is_file():
        raise FileNotFoundError(f"LongBench data file not found for preflight: {data_path}")
    total_samples = _nonblank_line_count(data_path)
    max_samples = int(getattr(args, "max_samples", -1))
    expected_samples = min(total_samples, max_samples) if max_samples > 0 else total_samples
    model = getattr(args, "model", None)
    model_path = getattr(args, "model_path", None) or model
    default_max = int(getattr(args, "default_max_model_len", 3500))
    max_model_len = getattr(args, "max_model_len", None) or model2maxlen.get(model, default_max)
    max_gen = getattr(args, "max_gen", None) or dataset2maxlen[dataset]
    model_family = getattr(args, "model_family", None) or infer_model_family(
        getattr(args, "model_tag", None) or model, model_path or model
    )
    templates_source = Path(__file__).with_name("templates.py")
    return {
        "schema_version": 1,
        "variant_semantic_hash": variant_semantic_hash(variant),
        "model_id": model,
        "model_source_identity": _model_source_identity(model_path),
        "model_config_sha256": _model_config_digest(model_path),
        "model_family": model_family,
        "dataset": dataset,
        "dataset_sha256": _sha256_file(data_path),
        "dataset_total_samples": total_samples,
        "max_samples": max_samples,
        "expected_samples": expected_samples,
        "prompt_sha256": hashlib.sha256(dataset2prompt[dataset].encode("utf-8")).hexdigest(),
        "templates_source_sha256": _sha256_file(templates_source),
        "max_model_len": int(max_model_len),
        "max_gen": int(max_gen),
        "prompt_token_reserve": int(getattr(args, "prompt_token_reserve", 0)),
        "torch_dtype": str(getattr(args, "torch_dtype", "float16")),
    }


def longbench_run_config_hash(args: Any, variant: VariantConfig, dataset: str) -> str:
    return _stable_json_hash(longbench_run_config_payload(args, variant, dataset))


def resolve_longbench_preflight(args: Any) -> dict[str, Any]:
    """No-GPU resolver: canonical variant, method slug, semantic hash.

    Shared by shell preflight and direct runner. Does not load model weights.
    """
    variant = _maybe_enable_quest_kernel(build_variant(args), args)
    semantic_payload = variant_semantic_payload(variant)
    result = {
        "canonical_variant": variant.name,
        "method_slug": method_layout_slug(variant),
        "resolved_variant": asdict(variant),
        "variant_semantic_hash": _stable_json_hash(semantic_payload),
        "mask_sha256": semantic_payload["mask_sha256"],
    }
    dataset = getattr(args, "dataset", None)
    datasets_csv = getattr(args, "datasets_csv", None)
    if dataset and datasets_csv:
        raise ValueError("Use either --dataset or --datasets-csv, not both")
    datasets = ([dataset] if dataset else [
        item.strip() for item in str(datasets_csv or "").split(",") if item.strip()
    ])
    if datasets:
        model = getattr(args, "model", None)
        model_path = getattr(args, "model_path", None) or model
        model_family = getattr(args, "model_family", None) or infer_model_family(
            getattr(args, "model_tag", None) or model, model_path or model
        )
        validate_qlutattn_preload(args, variant, model_family)
        resolved_datasets: dict[str, dict[str, Any]] = {}
        for dataset_name in datasets:
            run_payload = longbench_run_config_payload(args, variant, dataset_name)
            resolved_datasets[dataset_name] = {
                "run_config_hash": _stable_json_hash(run_payload),
                "expected_rows": run_payload["expected_samples"],
                "run_config": run_payload,
            }
        result["datasets"] = resolved_datasets
        if len(datasets) == 1:
            only = resolved_datasets[datasets[0]]
            result["run_config_hash"] = only["run_config_hash"]
            result["run_config"] = only["run_config"]
        else:
            result["run_config_hash"] = None
    else:
        result["run_config_hash"] = None
    return result


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
        use_fast=use_fast_tokenizer(model_family),
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
    expected_run_config_hash: str | None = None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        out_path.unlink(missing_ok=True)
        out_path.with_suffix(".manifest.json").unlink(missing_ok=True)

    expected_samples = len(records)
    completed = _completed_rows(out_path)
    if completed >= expected_samples:
        manifest_path = out_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Refusing to reuse completed {dataset} output without a manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = manifest.get("run_config_hash")
        if not expected_run_config_hash or actual_hash != expected_run_config_hash:
            raise RuntimeError(
                f"Refusing to reuse completed {dataset}: run_config_hash mismatch "
                f"(expected={expected_run_config_hash!r}, manifest={actual_hash!r})."
            )
        if completed != expected_samples:
            raise RuntimeError(
                f"Refusing to reuse completed {dataset}: output has {completed} rows "
                f"but this run expects exactly {expected_samples}"
            )
        if (
            manifest.get("status") != "ok"
            or int(manifest.get("written_samples", -1)) != completed
            or int(manifest.get("expected_samples", -1)) != expected_samples
        ):
            raise RuntimeError(
                f"Refusing to reuse completed {dataset}: manifest status/count is inconsistent"
            )
        print(f"[skip] {dataset}: all {completed}/{expected_samples} samples already present")
        return manifest

    if completed:
        manifest_path = out_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Refusing to resume partial {dataset} output without a manifest"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_config_hash") != expected_run_config_hash:
            raise RuntimeError(
                f"Refusing to resume partial {dataset}: run_config_hash mismatch"
            )
        print(f"[resume] {dataset}: skip {completed}, remaining {expected_samples - completed}")
        records = records.skip(completed)

    failed: list[dict[str, Any]] = []
    written_now = 0
    kitty_engagement_checked = False
    engagement_evidence = {
        "samples_observed": 0,
        "k_quant_calls": 0,
        "k_quantized_tokens": 0,
        "k_prompt_mean_layers": 0,
        "last_k_quant_mode": None,
        "v_quant_calls": 0,
        "v_quantized_tokens": 0,
        "v_tile_blocks": 0,
        "last_v_quant_mode": None,
        "last_v_tile_channels": None,
    }
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
            # Some fast tokenizers (e.g. MiniCPM5) emit token_type_ids, which a
            # Llama-style generate() rejects as an unused model_kwarg. Drop it —
            # the attention / KV path never consumes it.
            inputs.pop("token_type_ids", None)
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
                if variant.shadowkv or variant.quest_kernel:
                    # ShadowKV / QUEST kernel hooks have data-dependent shapes, so
                    # torch.compile must stay off.
                    gen_kwargs["disable_compile"] = True
                    if variant.shadowkv:
                        gen_kwargs["temperature"] = None
                output = model.generate(
                    **inputs,
                    **gen_kwargs,
                )[0]
            if variant.use_kitty and kv_cache is not None:
                engagement_evidence["samples_observed"] += 1
                engagement_evidence["k_quant_calls"] += int(
                    getattr(kv_cache, "k_quant_calls", 0)
                )
                engagement_evidence["k_quantized_tokens"] += int(
                    getattr(kv_cache, "k_quantized_tokens", 0)
                )
                prompt_means = getattr(kv_cache, "k_pc_mean", None)
                engagement_evidence["k_prompt_mean_layers"] += (
                    len(prompt_means) if isinstance(prompt_means, dict) else 0
                )
                k_mode_seen = getattr(kv_cache, "last_k_quant_mode", None)
                if k_mode_seen is not None:
                    engagement_evidence["last_k_quant_mode"] = k_mode_seen
                engagement_evidence["v_quant_calls"] += int(
                    getattr(kv_cache, "v_quant_calls", 0)
                )
                engagement_evidence["v_quantized_tokens"] += int(
                    getattr(kv_cache, "v_quantized_tokens", 0)
                )
                engagement_evidence["v_tile_blocks"] += int(
                    getattr(kv_cache, "v_tile_blocks", 0)
                )
                mode_seen = getattr(kv_cache, "last_v_quant_mode", None)
                channels_seen = getattr(kv_cache, "last_v_tile_channels", None)
                if mode_seen is not None:
                    engagement_evidence["last_v_quant_mode"] = mode_seen
                if channels_seen is not None:
                    engagement_evidence["last_v_tile_channels"] = channels_seen
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
                elif variant.quest_kernel:
                    decode_calls = int(kitty_stats.get("decode_calls", 0)) if kitty_stats else 0
                    prefill_calls = int(kitty_stats.get("prefill_calls", 0)) if kitty_stats else 0
                    last_path = kitty_stats.get("last_path") if kitty_stats else None
                    seqlen = kv_cache.get_seq_length() if kv_cache is not None else 0
                    engaged = (
                        kv_cache is not None and seqlen > 0
                        and kitty_stats is not None
                        and int(kitty_stats.get("installed", 0)) > 0
                        and prefill_calls > 0
                        and decode_calls > 0
                        and last_path in {"triton_sparse_reduced_budget", "triton_sparse_forced_all_pages"}
                    )
                    detail = (
                        f"installed={kitty_stats.get('installed') if kitty_stats else None} "
                        f"prefill_calls={prefill_calls} decode_calls={decode_calls} "
                        f"seq_length={seqlen} "
                        f"last_path={last_path} "
                        f"last_selected_pages={kitty_stats.get('last_selected_pages') if kitty_stats else None} "
                        f"last_page_count={kitty_stats.get('last_page_count') if kitty_stats else None}"
                    )
                    print(f"[quest-kernel] {dataset} first-sample evidence: {detail}")
                    raise_msg = (
                        f"QUEST_KERNEL was requested for variant '{variant.name}' but the Triton "
                        f"QUEST sparse decode kernel did not engage ({detail}). Results would be "
                        f"dense attention over the QLUTATTN fake-quant KV cache, not kernelized "
                        f"QUEST+QLUTATTN. Refusing to proceed. See kitty_sim/quest_kernel.py."
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
                    seqlen = kv_cache.get_seq_length() if kv_cache is not None else 0
                    engaged = kv_cache is not None and seqlen > 0
                    detail = (
                        f"KittyKVCache.get_seq_length()="
                        f"{seqlen}"
                    )
                    if variant.name == QLUTATTN_VARIANT and kv_cache is not None:
                        k_mode = getattr(kv_cache, "last_k_quant_mode", None)
                        k_calls = int(getattr(kv_cache, "k_quant_calls", 0))
                        k_tokens = int(getattr(kv_cache, "k_quantized_tokens", 0))
                        prompt_means = getattr(kv_cache, "k_pc_mean", None)
                        k_prompt_mean_layers = (
                            len(prompt_means)
                            if isinstance(prompt_means, dict)
                            else 0
                        )
                        mode = getattr(kv_cache, "last_v_quant_mode", None)
                        calls = int(getattr(kv_cache, "v_quant_calls", 0))
                        blocks = int(getattr(kv_cache, "v_tile_blocks", 0))
                        tokens = int(getattr(kv_cache, "v_quantized_tokens", 0))
                        c_seen = getattr(kv_cache, "last_v_tile_channels", None)
                        detail = (
                            f"{detail} k_quant_calls={k_calls} "
                            f"k_quantized_tokens={k_tokens} "
                            f"k_prompt_mean_layers={k_prompt_mean_layers} "
                            f"last_k_quant_mode={k_mode} "
                            f"v_quant_calls={calls} v_quantized_tokens={tokens} "
                            f"v_tile_blocks={blocks} last_v_quant_mode={mode} "
                            f"last_v_tile_channels={c_seen} context_length={context_length}"
                        )
                        engaged = (
                            engaged
                            and k_calls > 0
                            and k_tokens > 0
                            and k_prompt_mean_layers > 0
                            and k_mode == "per_token:qlut"
                        )
                        print(f"[vcache-2bit] {dataset} first-sample evidence: {detail}")
                        # Long prompts must engage at least one full tile; short
                        # prompts may legitimately leave blocks=0 (lazy calib).
                        ready = max(0, context_length - variant.sink_length - variant.buffer_length)
                        if ready >= 16:
                            engaged = engaged and blocks > 0 and mode == "tile16_rescued"
                            if variant.v_tile_channels is not None:
                                engaged = engaged and c_seen == variant.v_tile_channels
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
        "variant_semantic_hash": variant_semantic_hash(variant),
        "run_config_hash": expected_run_config_hash,
        "mask_sha256": variant_semantic_payload(variant)["mask_sha256"],
        "engagement": engagement_evidence,
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

    variant = _maybe_enable_quest_kernel(build_variant(args), args)
    model_family = args.model_family or infer_model_family(args.model_tag or args.model, args.model_path or args.model)
    validate_qlutattn_model_family(variant, model_family)
    validate_qlutattn_preload(args, variant, model_family)
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

    run_hashes = {
        dataset: longbench_run_config_hash(args, variant, dataset)
        for dataset in datasets
    }
    supplied_hash = getattr(args, "expected_run_config_hash", None)
    if supplied_hash is not None:
        if len(datasets) != 1:
            raise ValueError("--expected-run-config-hash requires exactly one --dataset")
        actual_hash = run_hashes[datasets[0]]
        if supplied_hash != actual_hash:
            raise RuntimeError(
                "Worker run_config_hash disagrees with shell preflight: "
                f"expected={supplied_hash}, recomputed={actual_hash}"
            )

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
    validate_qlutattn_model_config(variant, model_obj.config, model_obj.dtype)

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
    elif variant.quest_kernel:
        if legacy_cache_model:
            raise RuntimeError("QUEST_KERNEL=1 is not supported for legacy-cache/GLM-family models.")
        from kitty_sim.quest_kernel import QuestConfig as _QuestKernelConfig, install_quest_kernel

        quest_cfg = _QuestKernelConfig(
            page_size=16,
            token_budget=variant.quest_token_budget,
            skip_layers=variant.quest_skip_layers,
            sink_length=variant.sink_length,
            recent_length=variant.buffer_length,
        )
        kitty_stats = install_quest_kernel(model_obj, quest_cfg)
        print(
            f"[quest-kernel] installed Triton QUEST hook on {kitty_stats['installed']} layers "
            f"(variant={variant.tag}, budget={variant.quest_token_budget}, "
            f"skip_layers={variant.quest_skip_layers})"
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
                expected_run_config_hash=run_hashes[dataset],
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
        "variant_semantic_hash": variant_semantic_hash(variant),
        "mask_sha256": variant_semantic_payload(variant)["mask_sha256"],
        "run_config_hashes": run_hashes,
        "datasets": datasets,
        "manifests": manifests,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
