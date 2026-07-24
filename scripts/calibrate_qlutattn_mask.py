# -*- coding: utf-8 -*-
"""Offline codebook-mask calibration for the canonical qlutattn variant.

qlutattn quantizes the post-RoPE K cache per token on the per-channel-mean-
centered residual. Each channel is assigned ONE of two codebooks OFFLINE by a
per-(layer, kv-head, channel) ranking on a calibration corpus:

  ranking signal = sigma^2 x E|q|
    sigma^2 : residual variance of the post-RoPE K channel (how HARD the
              channel is to quantize)
    E|q|    : mean absolute post-RoPE query activation on that channel, query
              heads averaged within each GQA group (how much attention
              actually READS it)

Without ``--top-p``, the canonical artifact keeps the fixed assignment:

  - the 65% lowest-ranked channels  -> sign (1-bit codeword + per-token
    mean-|r| scale, ~1.25 bit nominal)
  - the 35% highest-ranked channels -> nf2 (symnf2-v1 fixed LUT + per-token
    absmax scale, ~2.25 bit nominal)

Its nominal K width is 1.60 bit/value. The opt-in research path ``--top-p P``
instead assigns nf2 to the shortest per-layer descending-score prefix whose
cumulative mass reaches P. Both statistics are collected in one model pass and
can be saved once for CPU-only threshold sweeps. There is no online ranking.

The per-channel MEAN is NOT calibrated here -- the runtime subtracts a
per-channel mean self-calibrated from each prompt at prefill (free for
attention, q.mu cancels in softmax). This script only fixes the sign-vs-nf2
CODEBOOK assignment. Note E|q| is corpus-dependent: with the default wikitext
corpus, gains concentrate on QA/retrieval-style tasks.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_qlutattn_mask.py \
    --model /path/to/model \
    --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
    --output /path/to/model.qlutattn_mask.pt
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from kitty_sim.qlut_quant import channel_sigma2  # noqa: E402

# The canonical no-top-p split and the qlutattn codebook/ranking identities are
# fixed; top-p is a separate artifact selection method.
CODEBOOKS = ("sign", "nf2")
SIGN_FRACTION = 0.65
BITS = {"sign": 1.25, "nf2": 2.25}
NOMINAL_K_BITS = SIGN_FRACTION * BITS["sign"] + (1 - SIGN_FRACTION) * BITS["nf2"]  # 1.60
NF2_IMPL = "symnf2-v1"
RANKING_SIGNAL = "sigma2_x_q"
TOP_P_SELECTION_METHOD = "layer_channel_top_p"
TOP_P_FORMAT_VERSION = 3
TOP_P_SELECTION_AXIS = "per_layer_flattened_kv_head_channel"
TOP_P_THRESHOLD_RULE = "minimal_desc_prefix_cumsum_ge_p"
TOP_P_TIE_RULE = "score_desc_flat_index_asc"
TOP_P_SCORE_DTYPE = "float64"
STATS_FORMAT = "qlutattn_calibration_statistics_v1"
CONTROL_SELECTION_METHOD = "layer_uniform_fixed_top_k_control"
CONTROL_FORMAT_VERSION = 2
REFERENCE_SEMANTIC_HASH_DOMAIN = "qlutattn_top_p_reference_semantics_v1"
CONTROL_KINDS = ("same_cardinality", "exact_packed_bits")



def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--calib-data", help="wikitext train parquet")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--sample-len", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128, help="sigma^2 submean group along tokens")
    p.add_argument("--skip-first", type=int, default=32, help="skip the sink window when measuring")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--top-p", type=float,
        help="per-layer cumulative ranking-score mass assigned to nf2")
    selection.add_argument(
        "--uniform-top-k-control-from",
        help="validated layer-channel top-p artifact used as the control reference")
    p.add_argument(
        "--control-kind", choices=CONTROL_KINDS,
        help="uniform top-k budget matched to the referenced top-p artifact")
    p.add_argument("--stats-input", help="reuse a saved CPU statistics artifact instead of running the model")
    p.add_argument("--stats-output", help="save reusable sigma2/q_absmean statistics from direct calibration")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if args.control_kind is not None and not args.uniform_top_k_control_from:
        p.error("--control-kind requires --uniform-top-k-control-from")
    if args.control_kind is None:
        args.control_kind = "exact_packed_bits"
    if args.stats_input:
        if args.model or args.calib_data:
            p.error("--stats-input cannot be combined with --model or --calib-data")
        if args.stats_output:
            p.error("--stats-output is only valid during direct calibration")
    else:
        missing = [flag for flag, value in (
            ("--model", args.model), ("--calib-data", args.calib_data)) if not value]
        if missing:
            p.error(f"{' and '.join(missing)} required unless --stats-input is used")
    return args


def load_calib_token_batches(tokenizer, parquet_path, num_samples, sample_len, seed):
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": parquet_path})["train"]
    texts = [t for t in ds["text"] if t and not t.isspace()]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]
    need = num_samples * sample_len
    if ids.numel() < need:
        raise RuntimeError(f"corpus too small: {ids.numel()} < {need}")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - sample_len, (num_samples,), generator=g)
    return torch.stack([ids[s:s + sample_len] for s in starts])


def cache_layer_keys(cache, layer_idx):
    try:
        keys, _ = cache[layer_idx]
        return keys
    except Exception:
        pass
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx]
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def rope_module_for(cfg):
    """The modeling module whose apply_rotary_pos_emb the q-tap wraps."""
    if cfg.model_type == "qwen3":
        import transformers.models.qwen3.modeling_qwen3 as ml
    elif cfg.model_type == "llama":
        import transformers.models.llama.modeling_llama as ml
    else:
        raise ValueError(f"unsupported model_type for calibration: {cfg.model_type}")
    return ml


@torch.no_grad()
def collect_statistics(model, rope_mod, batches, skip_first, group_size, device,
                       nl, n_kv, group, head_dim):
    """One pass over the corpus collecting BOTH per-channel statistics:
    sigma2 [nl, n_kv, D] from the cached post-RoPE keys, and q_absmean
    [nl, n_kv, D] from the post-RoPE queries. The rope wrap fires exactly once
    per layer per forward in layer order, so layer = call index % n_layers
    (verified by an exact call-count check)."""
    orig_rope = rope_mod.apply_rotary_pos_emb
    state = {"call": 0}
    q_accum = torch.zeros(nl, n_kv, head_dim, dtype=torch.float64)

    def tapped(q, k, cos, sin, *a, **kw):
        q_emb, k_emb = orig_rope(q, k, cos, sin, *a, **kw)
        li = state["call"] % nl
        state["call"] += 1
        qa = q_emb[0, :, skip_first:, :].abs().mean(dim=1)          # [n_q, D]
        q_accum[li] += qa.reshape(n_kv, group, -1).mean(dim=1).double().cpu()
        return q_emb, k_emb

    s2_accum = None
    n = 0
    rope_mod.apply_rotary_pos_emb = tapped
    try:
        for bi in range(batches.shape[0]):
            ids = batches[bi:bi + 1].to(device)
            cache = model(ids, use_cache=True).past_key_values
            per_layer = []
            for li in range(nl):
                keys = cache_layer_keys(cache, li)                  # [1, n_kv, T, D] post-RoPE
                x = keys[0, :, skip_first:, :].transpose(1, 2)      # [n_kv, D, T]
                per_layer.append(channel_sigma2(x.float().cpu(), group_size))  # [n_kv, D]
            per_layer = torch.stack(per_layer)                      # [nl, n_kv, D]
            s2_accum = per_layer if s2_accum is None else s2_accum + per_layer
            n += 1
            del cache
            if (bi + 1) % 16 == 0:
                print(f"  [calib] {bi + 1}/{batches.shape[0]} samples")
    finally:
        rope_mod.apply_rotary_pos_emb = orig_rope

    expected_calls = n * nl
    if state["call"] != expected_calls:
        raise RuntimeError(
            f"rope call count {state['call']} != expected {expected_calls}; "
            "layer attribution would be wrong (model not plain per-layer rope?)")
    return s2_accum / max(n, 1), (q_accum / max(n, 1)).float()


def build_mask(sigma2, q_absmean):
    """[nl,n_kv,D] stats -> codebook mask [nl,n_kv,D] uint8 (0=sign, 1=nf2).
    Per layer, channels are ranked ASCENDING by sigma2 * E|q| over the n_kv*D
    flattened channels (layer-global: louder heads may take more nf2 budget);
    the lowest round(0.65*N) -> sign, the rest -> nf2."""
    nl, n_kv, D = sigma2.shape
    signal = sigma2.float() * q_absmean.float()
    N = n_kv * D
    k_sign = int(round(SIGN_FRACTION * N))
    mask = torch.ones(nl, n_kv, D, dtype=torch.uint8)               # default nf2(1)
    for li in range(nl):
        order = torch.argsort(signal[li].reshape(-1))               # ascending
        m = mask[li].reshape(-1)
        m[order[:k_sign]] = 0
        mask[li] = m.reshape(n_kv, D)
    return mask

def _statistics_tensors(sigma2, q_absmean):
    if not isinstance(sigma2, torch.Tensor) or not isinstance(q_absmean, torch.Tensor):
        raise TypeError("sigma2 and q_absmean must be tensors")
    if sigma2.shape != q_absmean.shape:
        raise ValueError(
            f"sigma2 shape {tuple(sigma2.shape)} != q_absmean shape {tuple(q_absmean.shape)}")
    if sigma2.ndim != 3 or any(size <= 0 for size in sigma2.shape):
        raise ValueError(
            f"statistics must have non-empty [layer, kv_head, channel] shape, got {tuple(sigma2.shape)}")
    if not torch.is_floating_point(sigma2) or not torch.is_floating_point(q_absmean):
        raise TypeError("sigma2 and q_absmean must be floating-point tensors")
    return sigma2.detach().cpu(), q_absmean.detach().cpu()


def _validate_top_p_threshold(top_p_threshold):
    if isinstance(top_p_threshold, bool):
        raise ValueError("top-p threshold must be a finite float in (0, 1]")
    try:
        threshold = float(top_p_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("top-p threshold must be a finite float in (0, 1]") from exc
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError(f"top-p threshold must be in (0, 1], got {top_p_threshold!r}")
    return threshold


def _validated_ranking_score(ranking_score):
    if not isinstance(ranking_score, torch.Tensor):
        raise TypeError("ranking_score must be a tensor")
    if ranking_score.ndim != 3 or any(size <= 0 for size in ranking_score.shape):
        raise ValueError(
            "ranking_score must have non-empty [layer, kv_head, channel] shape")
    if not torch.is_floating_point(ranking_score):
        raise TypeError("ranking_score must be floating point")
    score = ranking_score.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(score).all()):
        raise ValueError("ranking_score contains NaN or infinity")
    if bool((score < 0).any()):
        raise ValueError("ranking_score contains negative values")
    layer_mass = score.reshape(score.shape[0], -1).sum(dim=1, dtype=torch.float64)
    if not bool(torch.isfinite(layer_mass).all()):
        raise ValueError("ranking_score layer mass overflowed FP64")
    zero_layers = torch.nonzero(layer_mass <= 0, as_tuple=False).flatten()
    if zero_layers.numel():
        indices = ", ".join(str(int(i)) for i in zero_layers)
        raise ValueError(f"ranking_score has zero total mass in layer(s): {indices}")
    return score


def ranking_score_from_statistics(sigma2, q_absmean):
    """Return the validated FP64 CPU sigma2 * E|q| ranking score."""
    sigma2, q_absmean = _statistics_tensors(sigma2, q_absmean)
    for name, statistic in (("sigma2", sigma2), ("q_absmean", q_absmean)):
        if not bool(torch.isfinite(statistic).all()):
            raise ValueError(f"{name} contains NaN or infinity")
        if bool((statistic < 0).any()):
            raise ValueError(f"{name} contains negative values")
    return _validated_ranking_score(sigma2.double() * q_absmean.double())


def select_layer_channel_top_p(ranking_score, top_p_threshold):
    """Select the shortest score-descending prefix reaching p in each layer.

    Selection is over flattened (kv_head, channel) entries. Stable CPU sorting
    makes equal scores resolve by ascending flattened index.
    """
    threshold = _validate_top_p_threshold(top_p_threshold)
    score = _validated_ranking_score(ranking_score)
    n_layers, n_kv, head_dim = score.shape
    mask = torch.zeros(n_layers, n_kv, head_dim, dtype=torch.uint8)
    nf2_count_per_layer = torch.empty(n_layers, dtype=torch.int32)
    captured_mass_per_layer = torch.empty(n_layers, dtype=torch.float64)
    previous_prefix_mass_per_layer = torch.empty(n_layers, dtype=torch.float64)
    boundary_score_per_layer = torch.empty(n_layers, dtype=torch.float64)

    for layer_idx in range(n_layers):
        flat_score = score[layer_idx].reshape(-1)
        order = torch.argsort(flat_score, descending=True, stable=True)
        ordered_score = flat_score[order]
        cumulative = torch.cumsum(ordered_score, dim=0, dtype=torch.float64)
        total_mass = cumulative[-1]
        target_mass = total_mass * threshold
        boundary = int(torch.searchsorted(cumulative, target_mass, right=False))
        if boundary >= order.numel():
            raise RuntimeError(f"top-p boundary escaped layer {layer_idx}")
        nf2_count = boundary + 1
        mask[layer_idx].reshape(-1)[order[:nf2_count]] = 1
        nf2_count_per_layer[layer_idx] = nf2_count
        captured_mass_per_layer[layer_idx] = cumulative[boundary] / total_mass
        previous_prefix_mass_per_layer[layer_idx] = (
            cumulative[boundary - 1] / total_mass if boundary else 0.0)
        boundary_score_per_layer[layer_idx] = ordered_score[boundary]

    return {
        "codebook_mask": mask,
        "nf2_count_per_layer": nf2_count_per_layer,
        "captured_mass_per_layer": captured_mass_per_layer,
        "previous_prefix_mass_per_layer": previous_prefix_mass_per_layer,
        "boundary_score_per_layer": boundary_score_per_layer,
    }


def build_head_local_reorder(codebook_mask):
    """Build new->old channel reorder and old->new inverse for every head."""
    if not isinstance(codebook_mask, torch.Tensor):
        raise TypeError("codebook_mask must be a tensor")
    if codebook_mask.dtype != torch.uint8 or codebook_mask.ndim != 3:
        raise ValueError("codebook_mask must be uint8 [layer, kv_head, channel]")
    if codebook_mask.device.type != "cpu":
        raise ValueError("codebook_mask must be on CPU")
    if not bool(((codebook_mask == 0) | (codebook_mask == 1)).all()):
        raise ValueError("codebook_mask values must be 0=sign or 1=nf2")

    n_layers, n_kv, head_dim = codebook_mask.shape
    original = torch.arange(head_dim, dtype=torch.int64)
    reorder = torch.empty(n_layers, n_kv, head_dim, dtype=torch.int64)
    inverse = torch.empty_like(reorder)
    nf2_count_per_head = codebook_mask.to(torch.int32).sum(dim=-1, dtype=torch.int32)
    for layer_idx in range(n_layers):
        for head_idx in range(n_kv):
            head_mask = codebook_mask[layer_idx, head_idx].bool()
            index = torch.cat((original[head_mask], original[~head_mask]))
            reorder[layer_idx, head_idx] = index
            inverse[layer_idx, head_idx, index] = original
    return reorder, inverse, nf2_count_per_head


def _require_exact_tensor(payload, key, expected, dtype):
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"invalid top-p artifact: {key} must be a tensor")
    if value.device.type != "cpu" or value.dtype != dtype:
        raise ValueError(
            f"invalid top-p artifact: {key} must be CPU {dtype}, got {value.device}/{value.dtype}")
    if not torch.equal(value, expected):
        raise ValueError(f"invalid top-p artifact: {key} metadata mismatch")


def validate_top_p_artifact(payload):
    """Recompute all selection/reorder metadata and reject any inconsistency."""
    if not isinstance(payload, dict):
        raise ValueError("invalid top-p artifact: payload must be a dict")
    required = {
        "format_version", "selection_method", "selection_axis",
        "threshold_rule", "tie_rule", "score_dtype_for_selection",
        "top_p_threshold", "codebook_mask", "codebooks", "reorder_index",
        "inverse_reorder_index", "nf2_count_per_head", "nf2_count_per_layer",
        "nf2_ratio_per_layer", "captured_mass_per_layer",
        "previous_prefix_mass_per_layer", "boundary_score_per_layer",
        "actual_nf2_frac", "low_frac", "actual_nf2_count",
        "packed_bits_total", "packed_k_values_total", "nf2_impl",
        "ranking_signal", "model", "group_size", "skip_first", "calib_data",
        "num_samples", "sample_len", "seed", "n_layers", "n_kv", "head_dim",
        "sigma2", "q_absmean", "ranking_score",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"invalid top-p artifact: missing fields {missing}")
    if payload["selection_method"] != TOP_P_SELECTION_METHOD:
        raise ValueError("invalid top-p artifact: selection_method mismatch")
    if (isinstance(payload["format_version"], bool)
            or not isinstance(payload["format_version"], int)
            or payload["format_version"] != TOP_P_FORMAT_VERSION):
        raise ValueError("invalid top-p artifact: format_version mismatch")
    identity_fields = {
        "selection_axis": TOP_P_SELECTION_AXIS,
        "threshold_rule": TOP_P_THRESHOLD_RULE,
        "tie_rule": TOP_P_TIE_RULE,
        "score_dtype_for_selection": TOP_P_SCORE_DTYPE,
    }
    for key, expected in identity_fields.items():
        if payload[key] != expected:
            raise ValueError(f"invalid top-p artifact: {key} mismatch")
    if payload["codebooks"] != list(CODEBOOKS):
        raise ValueError("invalid top-p artifact: codebooks must be ['sign', 'nf2']")
    if payload["nf2_impl"] != NF2_IMPL or payload["ranking_signal"] != RANKING_SIGNAL:
        raise ValueError("invalid top-p artifact: quantization identity mismatch")
    if not isinstance(payload["model"], str) or not payload["model"].strip():
        raise ValueError("invalid top-p artifact: model provenance is required")
    if (not isinstance(payload["calib_data"], str)
            or not payload["calib_data"].strip()):
        raise ValueError("invalid top-p artifact: calib_data provenance is required")
    for key, minimum in (
            ("group_size", 1), ("skip_first", 0), ("num_samples", 1),
            ("sample_len", 1)):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"invalid top-p artifact: {key} provenance is invalid")
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
        raise ValueError("invalid top-p artifact: seed provenance is invalid")

    sigma2_value = payload["sigma2"]
    q_absmean_value = payload["q_absmean"]
    if (not isinstance(sigma2_value, torch.Tensor)
            or not isinstance(q_absmean_value, torch.Tensor)
            or sigma2_value.device.type != "cpu"
            or q_absmean_value.device.type != "cpu"):
        raise ValueError("invalid top-p artifact: statistics must be CPU tensors")
    sigma2, q_absmean = _statistics_tensors(sigma2_value, q_absmean_value)
    n_layers, n_kv, head_dim = sigma2.shape
    for key, expected in (
            ("n_layers", n_layers), ("n_kv", n_kv), ("head_dim", head_dim)):
        if (isinstance(payload[key], bool)
                or not isinstance(payload[key], int)
                or payload[key] != expected):
            raise ValueError(f"invalid top-p artifact: {key} mismatch")

    expected_score = ranking_score_from_statistics(sigma2, q_absmean)
    _require_exact_tensor(payload, "ranking_score", expected_score, torch.float64)
    threshold_value = payload["top_p_threshold"]
    if not isinstance(threshold_value, float):
        raise ValueError("invalid top-p artifact: top_p_threshold must be a float")
    threshold = _validate_top_p_threshold(threshold_value)
    selected = select_layer_channel_top_p(expected_score, threshold)
    expected_mask = selected["codebook_mask"]
    _require_exact_tensor(payload, "codebook_mask", expected_mask, torch.uint8)

    reorder, inverse, count_per_head = build_head_local_reorder(expected_mask)
    _require_exact_tensor(payload, "reorder_index", reorder, torch.int64)
    _require_exact_tensor(payload, "inverse_reorder_index", inverse, torch.int64)
    _require_exact_tensor(payload, "nf2_count_per_head", count_per_head, torch.int32)
    _require_exact_tensor(
        payload, "nf2_count_per_layer", selected["nf2_count_per_layer"], torch.int32)

    ratio_per_layer = selected["nf2_count_per_layer"].double() / (n_kv * head_dim)
    _require_exact_tensor(payload, "nf2_ratio_per_layer", ratio_per_layer, torch.float64)
    for key in (
            "captured_mass_per_layer", "previous_prefix_mass_per_layer",
            "boundary_score_per_layer"):
        _require_exact_tensor(payload, key, selected[key], torch.float64)

    n_nf2 = int(expected_mask.sum())
    total = expected_mask.numel()
    expected_nf2_frac = n_nf2 / total
    expected_low_frac = (total - n_nf2) / total
    for key, expected in (
            ("actual_nf2_count", n_nf2),
            ("packed_bits_total", _packed_k_bits_total(expected_mask)),
            ("packed_k_values_total", total)):
        value = payload[key]
        if (isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected):
            raise ValueError(f"invalid top-p artifact: {key} metadata mismatch")
    for key, expected in (
            ("actual_nf2_frac", expected_nf2_frac),
            ("low_frac", expected_low_frac)):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid top-p artifact: {key} must be a scalar")
        if not math.isfinite(float(value)) or float(value) != expected:
            raise ValueError(f"invalid top-p artifact: {key} metadata mismatch")

    identity = torch.arange(head_dim, dtype=torch.int64).expand_as(reorder)
    if not torch.equal(torch.gather(reorder, -1, inverse), identity):
        raise ValueError("invalid top-p artifact: reorder inverse is not exact")
    reordered_mask = torch.gather(expected_mask, -1, reorder)
    for layer_idx in range(n_layers):
        for head_idx in range(n_kv):
            nf2_count = int(count_per_head[layer_idx, head_idx])
            if (not bool((reordered_mask[layer_idx, head_idx, :nf2_count] == 1).all())
                    or not bool((reordered_mask[layer_idx, head_idx, nf2_count:] == 0).all())):
                raise ValueError("invalid top-p artifact: nf2 channels are not a head-local prefix")
    return payload


def build_top_p_artifact(
        sigma2, q_absmean, top_p_threshold, *, group_size, skip_first, model,
        calib_data, num_samples, sample_len, seed):
    sigma2, q_absmean = _statistics_tensors(sigma2, q_absmean)
    ranking_score = ranking_score_from_statistics(sigma2, q_absmean)
    selected = select_layer_channel_top_p(ranking_score, top_p_threshold)
    mask = selected["codebook_mask"]
    reorder, inverse, count_per_head = build_head_local_reorder(mask)
    n_layers, n_kv, head_dim = mask.shape
    n_nf2 = int(mask.sum())
    total = mask.numel()
    actual_nf2_frac = n_nf2 / total
    low_frac = (total - n_nf2) / total
    payload = {
        "codebook_mask": mask,
        "codebooks": list(CODEBOOKS),
        "low_frac": low_frac,
        "actual_nf2_frac": actual_nf2_frac,
        "actual_nf2_count": n_nf2,
        "packed_bits_total": _packed_k_bits_total(mask),
        "packed_k_values_total": total,
        "nf2_impl": NF2_IMPL,
        "ranking_signal": RANKING_SIGNAL,
        "format_version": TOP_P_FORMAT_VERSION,
        "selection_axis": TOP_P_SELECTION_AXIS,
        "threshold_rule": TOP_P_THRESHOLD_RULE,
        "tie_rule": TOP_P_TIE_RULE,
        "score_dtype_for_selection": TOP_P_SCORE_DTYPE,
        "selection_method": TOP_P_SELECTION_METHOD,
        "top_p_threshold": float(top_p_threshold),
        "reorder_index": reorder,
        "inverse_reorder_index": inverse,
        "nf2_count_per_head": count_per_head,
        "nf2_count_per_layer": selected["nf2_count_per_layer"],
        "nf2_ratio_per_layer": selected["nf2_count_per_layer"].double() / (n_kv * head_dim),
        "captured_mass_per_layer": selected["captured_mass_per_layer"],
        "previous_prefix_mass_per_layer": selected["previous_prefix_mass_per_layer"],
        "boundary_score_per_layer": selected["boundary_score_per_layer"],
        "group_size": group_size,
        "skip_first": skip_first,
        "model": model,
        "calib_data": calib_data,
        "num_samples": num_samples,
        "sample_len": sample_len,
        "seed": seed,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "sigma2": sigma2,
        "q_absmean": q_absmean,
        "ranking_score": ranking_score,
    }
    validate_top_p_artifact(payload)
    return payload


def _packed_k_bits_total(codebook_mask):
    """Exact packed K bits for one token across a complete mask."""
    if not isinstance(codebook_mask, torch.Tensor):
        raise TypeError("codebook_mask must be a tensor")
    if codebook_mask.ndim != 3 or codebook_mask.dtype != torch.uint8:
        raise ValueError("codebook_mask must be uint8 [layer, kv_head, channel]")
    if not bool(((codebook_mask == 0) | (codebook_mask == 1)).all()):
        raise ValueError("codebook_mask values must be 0=sign or 1=nf2")
    head_dim = codebook_mask.shape[-1]
    nf2_per_head = codebook_mask.to(torch.int64).sum(dim=-1)
    code_bits = int(((head_dim - nf2_per_head) + 2 * nf2_per_head).sum())
    scale_bits = 16 * int((nf2_per_head > 0).sum())
    scale_bits += 16 * int((nf2_per_head < head_dim).sum())
    return code_bits + scale_bits


def _validated_uniform_control_inputs(ranking_score, reference_mask):
    score = _validated_ranking_score(ranking_score)
    _packed_k_bits_total(reference_mask)
    if tuple(score.shape) != tuple(reference_mask.shape):
        raise ValueError("ranking_score and reference_mask shapes must match")
    reference = reference_mask.detach().to(device="cpu")
    orders = [
        torch.argsort(score[layer].reshape(-1), descending=True, stable=True)
        for layer in range(score.shape[0])
    ]
    return score, reference, orders


def _uniform_layer_mask_for_total(score, reference, orders, total_nf2):
    n_layers, n_kv, head_dim = score.shape
    channels_per_layer = n_kv * head_dim
    floor_count, remainder = divmod(int(total_nf2), n_layers)
    if floor_count <= 0 or floor_count + bool(remainder) >= channels_per_layer:
        raise ValueError("NF2 cardinality cannot form nontrivial uniform layer counts")
    next_scores = torch.tensor(
        [
            float(score[layer].reshape(-1)[orders[layer][floor_count]])
            for layer in range(n_layers)
        ],
        dtype=torch.float64,
    )
    extra_layers = set(
        torch.argsort(next_scores, descending=True, stable=True)[:remainder].tolist()
    )
    mask = torch.zeros_like(reference)
    count_per_layer = torch.empty(n_layers, dtype=torch.int32)
    for layer in range(n_layers):
        count = floor_count + int(layer in extra_layers)
        mask[layer].reshape(-1)[orders[layer][:count]] = 1
        count_per_layer[layer] = count
    return mask, count_per_layer


def _uniform_layer_topk_same_cardinality(ranking_score, reference_mask):
    score, reference, orders = _validated_uniform_control_inputs(
        ranking_score, reference_mask)
    reference_nf2_count = int(reference.sum())
    mask, count_per_layer = _uniform_layer_mask_for_total(
        score, reference, orders, reference_nf2_count)
    return {
        "codebook_mask": mask,
        "nf2_count_per_layer": count_per_layer,
        "reference_nf2_count": reference_nf2_count,
        "actual_nf2_count": reference_nf2_count,
        "reference_packed_bits_total": _packed_k_bits_total(reference),
        "packed_bits_total": _packed_k_bits_total(mask),
    }


def _uniform_layer_topk_exact_bits(ranking_score, reference_mask):
    score, reference, orders = _validated_uniform_control_inputs(
        ranking_score, reference_mask)
    reference_packed_bits = _packed_k_bits_total(reference)
    reference_nf2_count = int(reference.sum())
    candidate_totals = sorted(
        range(1, score.numel()),
        key=lambda total: (abs(total - reference_nf2_count), total),
    )
    for total_nf2 in candidate_totals:
        try:
            mask, count_per_layer = _uniform_layer_mask_for_total(
                score, reference, orders, total_nf2)
        except ValueError:
            continue
        if _packed_k_bits_total(mask) == reference_packed_bits:
            return {
                "codebook_mask": mask,
                "nf2_count_per_layer": count_per_layer,
                "reference_nf2_count": reference_nf2_count,
                "actual_nf2_count": int(mask.sum()),
                "reference_packed_bits_total": reference_packed_bits,
                "packed_bits_total": reference_packed_bits,
            }
    raise ValueError(
        "no uniform per-layer top-k mask matches the reference packed bit cost")


def _select_uniform_layer_topk_control(ranking_score, reference_mask, control_kind):
    if control_kind == "same_cardinality":
        return _uniform_layer_topk_same_cardinality(ranking_score, reference_mask)
    if control_kind == "exact_packed_bits":
        return _uniform_layer_topk_exact_bits(ranking_score, reference_mask)
    raise ValueError(
        f"control_kind must be one of {CONTROL_KINDS}, got {control_kind!r}")


def load_top_p_reference_artifact(path):
    """Load, hash, and strictly validate the exact referenced file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    validate_top_p_artifact(payload)
    return payload, digest.hexdigest()


def _require_reference_matches_calibration(
        reference, sigma2, q_absmean, *, group_size, skip_first, model,
        calib_data, num_samples, sample_len, seed):
    validate_top_p_artifact(reference)
    sigma2, q_absmean = _statistics_tensors(sigma2, q_absmean)
    for key, active in (("sigma2", sigma2), ("q_absmean", q_absmean)):
        measured = reference[key]
        if measured.dtype != active.dtype or not torch.equal(measured, active):
            raise ValueError(
                f"reference top-p artifact {key} does not match active calibration")
    n_layers, n_kv, head_dim = sigma2.shape
    active_provenance = {
        "model": model,
        "calib_data": calib_data,
        "num_samples": num_samples,
        "sample_len": sample_len,
        "seed": seed,
        "group_size": group_size,
        "skip_first": skip_first,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
    }
    for key, active in active_provenance.items():
        if reference[key] != active:
            raise ValueError(
                f"reference top-p artifact {key} does not match active calibration")
    return sigma2, q_absmean


def _require_control_tensor(payload, key, expected, dtype):
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"invalid uniform top-k control artifact: {key} must be a tensor")
    if value.device.type != "cpu" or value.dtype != dtype:
        raise ValueError(
            f"invalid uniform top-k control artifact: {key} must be CPU "
            f"{dtype}, got {value.device}/{value.dtype}")
    if not torch.equal(value, expected):
        raise ValueError(
            f"invalid uniform top-k control artifact: {key} metadata mismatch")


def _require_control_int(payload, key, expected):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(
            f"invalid uniform top-k control artifact: {key} metadata mismatch")


def _reference_semantic_sha256(
        sigma2, q_absmean, reference_mask, *, top_p_threshold, group_size,
        skip_first, model, calib_data, num_samples, sample_len, seed):
    """Hash validated top-p semantics with a canonical, storage-free encoding."""
    n_layers, n_kv, head_dim = reference_mask.shape
    metadata = {
        "semantic_hash_domain": REFERENCE_SEMANTIC_HASH_DOMAIN,
        "top_p_format_version": TOP_P_FORMAT_VERSION,
        "selection_method": TOP_P_SELECTION_METHOD,
        "selection_axis": TOP_P_SELECTION_AXIS,
        "threshold_rule": TOP_P_THRESHOLD_RULE,
        "tie_rule": TOP_P_TIE_RULE,
        "score_dtype_for_selection": TOP_P_SCORE_DTYPE,
        "top_p_threshold_hex": float(top_p_threshold).hex(),
        "codebooks": list(CODEBOOKS),
        "nf2_impl": NF2_IMPL,
        "ranking_signal": RANKING_SIGNAL,
        "model": model,
        "calib_data": calib_data,
        "group_size": group_size,
        "skip_first": skip_first,
        "num_samples": num_samples,
        "sample_len": sample_len,
        "seed": seed,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "statistics_encoding": "ieee754_binary64_little_endian_c_order",
        "reference_mask_encoding": "uint8_c_order",
    }
    metadata_bytes = json.dumps(
        metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    sigma2_values = (
        sigma2.detach().to(device="cpu", dtype=torch.float64).contiguous()
        .numpy().astype("<f8", copy=False)
    )
    q_absmean_values = (
        q_absmean.detach().to(device="cpu", dtype=torch.float64).contiguous()
        .numpy().astype("<f8", copy=False)
    )
    reference_mask_values = (
        reference_mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        .numpy()
    )

    digest = hashlib.sha256()
    for label, values in (
            ("metadata_json_utf8", metadata_bytes),
            ("sigma2_float64_le", sigma2_values),
            ("q_absmean_float64_le", q_absmean_values),
            ("reference_mask_uint8", reference_mask_values)):
        label_bytes = label.encode("ascii")
        value_bytes = memoryview(values).cast("B")
        digest.update(len(label_bytes).to_bytes(4, "big"))
        digest.update(label_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def validate_uniform_top_k_control_artifact(payload):
    """Recompute the top-p reference and fixed-top-k control from measurements."""
    if not isinstance(payload, dict):
        raise ValueError(
            "invalid uniform top-k control artifact: payload must be a dict")
    required = {
        "selection_method", "format_version", "selection_axis", "tie_rule",
        "score_dtype_for_selection", "control_kind", "reference_mask_sha256",
        "reference_semantic_sha256",
        "reference_selection_method", "reference_top_p_threshold",
        "reference_codebook_mask", "codebook_mask", "codebooks", "low_frac",
        "actual_nf2_frac", "nf2_impl", "ranking_signal", "reorder_index",
        "inverse_reorder_index", "nf2_count_per_head", "nf2_count_per_layer",
        "nf2_ratio_per_layer", "group_size", "skip_first", "model",
        "calib_data", "num_samples", "sample_len", "seed", "n_layers",
        "n_kv", "head_dim", "sigma2", "q_absmean", "ranking_score",
        "reference_nf2_count", "actual_nf2_count",
        "reference_packed_bits_total", "packed_bits_total",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"invalid uniform top-k control artifact: missing fields {missing}")
    if payload["selection_method"] != CONTROL_SELECTION_METHOD:
        raise ValueError(
            "invalid uniform top-k control artifact: selection_method mismatch")
    if (isinstance(payload["format_version"], bool)
            or not isinstance(payload["format_version"], int)
            or payload["format_version"] != CONTROL_FORMAT_VERSION):
        raise ValueError(
            "invalid uniform top-k control artifact: format_version mismatch")
    for key, expected in (
            ("selection_axis", TOP_P_SELECTION_AXIS),
            ("tie_rule", TOP_P_TIE_RULE),
            ("score_dtype_for_selection", TOP_P_SCORE_DTYPE),
            ("reference_selection_method", TOP_P_SELECTION_METHOD)):
        if payload[key] != expected:
            raise ValueError(
                f"invalid uniform top-k control artifact: {key} mismatch")
    control_kind = payload["control_kind"]
    if control_kind not in CONTROL_KINDS:
        raise ValueError(
            "invalid uniform top-k control artifact: control_kind mismatch")
    reference_sha256 = payload["reference_mask_sha256"]
    if (not isinstance(reference_sha256, str)
            or len(reference_sha256) != 64
            or any(char not in "0123456789abcdef" for char in reference_sha256)):
        raise ValueError(
            "invalid uniform top-k control artifact: "
            "reference_mask_sha256 must be 64 lowercase hex characters")
    reference_semantic_sha256 = payload["reference_semantic_sha256"]
    if (not isinstance(reference_semantic_sha256, str)
            or len(reference_semantic_sha256) != 64
            or any(char not in "0123456789abcdef"
                   for char in reference_semantic_sha256)):
        raise ValueError(
            "invalid uniform top-k control artifact: "
            "reference_semantic_sha256 must be 64 lowercase hex characters")
    if payload["codebooks"] != list(CODEBOOKS):
        raise ValueError(
            "invalid uniform top-k control artifact: "
            "codebooks must be ['sign', 'nf2']")
    if payload["nf2_impl"] != NF2_IMPL or payload["ranking_signal"] != RANKING_SIGNAL:
        raise ValueError(
            "invalid uniform top-k control artifact: quantization identity mismatch")
    if not isinstance(payload["model"], str) or not payload["model"].strip():
        raise ValueError(
            "invalid uniform top-k control artifact: model provenance is required")
    if (not isinstance(payload["calib_data"], str)
            or not payload["calib_data"].strip()):
        raise ValueError(
            "invalid uniform top-k control artifact: "
            "calib_data provenance is required")
    for key, minimum in (
            ("group_size", 1), ("skip_first", 0), ("num_samples", 1),
            ("sample_len", 1)):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"invalid uniform top-k control artifact: {key} provenance is invalid")
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
        raise ValueError(
            "invalid uniform top-k control artifact: seed provenance is invalid")

    sigma2_value = payload["sigma2"]
    q_absmean_value = payload["q_absmean"]
    if (not isinstance(sigma2_value, torch.Tensor)
            or not isinstance(q_absmean_value, torch.Tensor)
            or sigma2_value.device.type != "cpu"
            or q_absmean_value.device.type != "cpu"):
        raise ValueError(
            "invalid uniform top-k control artifact: statistics must be CPU tensors")
    sigma2, q_absmean = _statistics_tensors(sigma2_value, q_absmean_value)
    n_layers, n_kv, head_dim = sigma2.shape
    for key, expected in (
            ("n_layers", n_layers), ("n_kv", n_kv), ("head_dim", head_dim)):
        _require_control_int(payload, key, expected)

    ranking_score = ranking_score_from_statistics(sigma2, q_absmean)
    _require_control_tensor(
        payload, "ranking_score", ranking_score, torch.float64)
    threshold_value = payload["reference_top_p_threshold"]
    if not isinstance(threshold_value, float):
        raise ValueError(
            "invalid uniform top-k control artifact: "
            "reference_top_p_threshold must be a float")
    threshold = _validate_top_p_threshold(threshold_value)
    reference_selection = select_layer_channel_top_p(ranking_score, threshold)
    reference_mask = reference_selection["codebook_mask"]
    _require_control_tensor(
        payload, "reference_codebook_mask", reference_mask, torch.uint8)
    expected_reference_semantic_sha256 = _reference_semantic_sha256(
        sigma2, q_absmean, reference_mask, top_p_threshold=threshold,
        group_size=payload["group_size"], skip_first=payload["skip_first"],
        model=payload["model"], calib_data=payload["calib_data"],
        num_samples=payload["num_samples"], sample_len=payload["sample_len"],
        seed=payload["seed"])
    if reference_semantic_sha256 != expected_reference_semantic_sha256:
        raise ValueError(
            "invalid uniform top-k control artifact: "
            "reference_semantic_sha256 mismatch")

    selected = _select_uniform_layer_topk_control(
        ranking_score, reference_mask, control_kind)
    mask = selected["codebook_mask"]
    _require_control_tensor(payload, "codebook_mask", mask, torch.uint8)
    _require_control_tensor(
        payload, "nf2_count_per_layer",
        selected["nf2_count_per_layer"], torch.int32)
    reorder, inverse, count_per_head = build_head_local_reorder(mask)
    _require_control_tensor(
        payload, "reorder_index", reorder, torch.int64)
    _require_control_tensor(
        payload, "inverse_reorder_index", inverse, torch.int64)
    _require_control_tensor(
        payload, "nf2_count_per_head", count_per_head, torch.int32)
    ratio_per_layer = selected["nf2_count_per_layer"].double() / (n_kv * head_dim)
    _require_control_tensor(
        payload, "nf2_ratio_per_layer", ratio_per_layer, torch.float64)
    for key in (
            "reference_nf2_count", "actual_nf2_count",
            "reference_packed_bits_total", "packed_bits_total"):
        _require_control_int(payload, key, selected[key])

    actual_nf2_count = selected["actual_nf2_count"]
    total = mask.numel()
    expected_nf2_frac = actual_nf2_count / total
    expected_low_frac = (total - actual_nf2_count) / total
    for key, expected in (
            ("actual_nf2_frac", expected_nf2_frac),
            ("low_frac", expected_low_frac)):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"invalid uniform top-k control artifact: {key} must be a scalar")
        if not math.isfinite(float(value)) or float(value) != expected:
            raise ValueError(
                f"invalid uniform top-k control artifact: {key} metadata mismatch")

    if (control_kind == "same_cardinality"
            and selected["actual_nf2_count"] != selected["reference_nf2_count"]):
        raise ValueError(
            "invalid uniform top-k control artifact: cardinality is not exact")
    if (control_kind == "exact_packed_bits"
            and selected["packed_bits_total"]
            != selected["reference_packed_bits_total"]):
        raise ValueError(
            "invalid uniform top-k control artifact: packed bit cost is not exact")
    identity = torch.arange(head_dim, dtype=torch.int64).expand_as(reorder)
    if not torch.equal(torch.gather(reorder, -1, inverse), identity):
        raise ValueError(
            "invalid uniform top-k control artifact: reorder inverse is not exact")
    reordered_mask = torch.gather(mask, -1, reorder)
    for layer_idx in range(n_layers):
        for head_idx in range(n_kv):
            nf2_count = int(count_per_head[layer_idx, head_idx])
            if (not bool((reordered_mask[layer_idx, head_idx, :nf2_count] == 1).all())
                    or not bool((reordered_mask[layer_idx, head_idx, nf2_count:] == 0).all())):
                raise ValueError(
                    "invalid uniform top-k control artifact: "
                    "nf2 channels are not a head-local prefix")
    return payload


def build_uniform_top_k_control_artifact(
        sigma2, q_absmean, reference_top_p_artifact, reference_mask_sha256, *,
        control_kind, group_size, skip_first, model, calib_data, num_samples,
        sample_len, seed):
    sigma2, q_absmean = _require_reference_matches_calibration(
        reference_top_p_artifact, sigma2, q_absmean, group_size=group_size,
        skip_first=skip_first, model=model, calib_data=calib_data,
        num_samples=num_samples, sample_len=sample_len, seed=seed)
    ranking_score = ranking_score_from_statistics(sigma2, q_absmean)
    reference_mask = reference_top_p_artifact["codebook_mask"].detach().clone(
        memory_format=torch.contiguous_format)
    reference_top_p_threshold = float(
        reference_top_p_artifact["top_p_threshold"])
    reference_semantic_sha256 = _reference_semantic_sha256(
        sigma2, q_absmean, reference_mask,
        top_p_threshold=reference_top_p_threshold, group_size=group_size,
        skip_first=skip_first, model=model, calib_data=calib_data,
        num_samples=num_samples, sample_len=sample_len, seed=seed)
    selected = _select_uniform_layer_topk_control(
        ranking_score, reference_mask, control_kind)
    mask = selected["codebook_mask"]
    reorder, inverse, count_per_head = build_head_local_reorder(mask)
    n_layers, n_kv, head_dim = mask.shape
    actual_nf2_count = selected["actual_nf2_count"]
    total = mask.numel()
    payload = {
        "selection_method": CONTROL_SELECTION_METHOD,
        "format_version": CONTROL_FORMAT_VERSION,
        "selection_axis": TOP_P_SELECTION_AXIS,
        "tie_rule": TOP_P_TIE_RULE,
        "score_dtype_for_selection": TOP_P_SCORE_DTYPE,
        "control_kind": control_kind,
        "reference_mask_sha256": reference_mask_sha256,
        "reference_semantic_sha256": reference_semantic_sha256,
        "reference_selection_method": TOP_P_SELECTION_METHOD,
        "reference_top_p_threshold": reference_top_p_threshold,
        "reference_codebook_mask": reference_mask,
        "codebook_mask": mask,
        "codebooks": list(CODEBOOKS),
        "low_frac": (total - actual_nf2_count) / total,
        "actual_nf2_frac": actual_nf2_count / total,
        "nf2_impl": NF2_IMPL,
        "ranking_signal": RANKING_SIGNAL,
        "reorder_index": reorder,
        "inverse_reorder_index": inverse,
        "nf2_count_per_head": count_per_head,
        "nf2_count_per_layer": selected["nf2_count_per_layer"],
        "nf2_ratio_per_layer": selected["nf2_count_per_layer"].double() / (
            n_kv * head_dim),
        "group_size": group_size,
        "skip_first": skip_first,
        "model": model,
        "calib_data": calib_data,
        "num_samples": num_samples,
        "sample_len": sample_len,
        "seed": seed,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "sigma2": sigma2,
        "q_absmean": q_absmean,
        "ranking_score": ranking_score,
        "reference_nf2_count": selected["reference_nf2_count"],
        "actual_nf2_count": actual_nf2_count,
        "reference_packed_bits_total": selected["reference_packed_bits_total"],
        "packed_bits_total": selected["packed_bits_total"],
    }
    validate_uniform_top_k_control_artifact(payload)
    return payload


def build_canonical_artifact(sigma2, q_absmean, *, group_size, skip_first, model):
    """Build the legacy fixed-65/35 payload without any top-p-only fields."""
    sigma2, q_absmean = _statistics_tensors(sigma2, q_absmean)
    mask = build_mask(sigma2, q_absmean)
    n_lo = int((mask == 0).sum())
    total = mask.numel()
    n_layers, n_kv, head_dim = mask.shape
    return {
        "codebook_mask": mask,
        "codebooks": list(CODEBOOKS),
        "low_frac": n_lo / total,
        "nominal_bits": NOMINAL_K_BITS,
        "nf2_impl": NF2_IMPL,
        "ranking_signal": RANKING_SIGNAL,
        "group_size": group_size,
        "skip_first": skip_first,
        "model": model,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "sigma2": sigma2,
        "q_absmean": q_absmean,
    }


def build_statistics_artifact(
        sigma2, q_absmean, *, model, group_size, skip_first, calib_data,
        num_samples, sample_len, seed):
    """Build the reusable, model-free calibration input artifact."""
    sigma2, q_absmean = _statistics_tensors(sigma2, q_absmean)
    n_layers, n_kv, head_dim = sigma2.shape
    return {
        "statistics_format": STATS_FORMAT,
        "ranking_signal": RANKING_SIGNAL,
        "sigma2": sigma2,
        "q_absmean": q_absmean,
        "model": model,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "group_size": group_size,
        "skip_first": skip_first,
        "calib_data": calib_data,
        "num_samples": num_samples,
        "sample_len": sample_len,
        "seed": seed,
    }


def load_statistics_artifact(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("statistics artifact must be a dict")
    required = {
        "sigma2", "q_absmean", "model", "n_layers", "n_kv", "head_dim",
        "group_size", "skip_first",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"statistics artifact missing fields: {missing}")
    if payload.get("ranking_signal", RANKING_SIGNAL) != RANKING_SIGNAL:
        raise ValueError(
            f"statistics artifact ranking_signal must be {RANKING_SIGNAL!r}")
    sigma2, q_absmean = _statistics_tensors(payload["sigma2"], payload["q_absmean"])
    n_layers, n_kv, head_dim = sigma2.shape
    for key, expected in (
            ("n_layers", n_layers), ("n_kv", n_kv), ("head_dim", head_dim)):
        if isinstance(payload[key], bool) or payload[key] != expected:
            raise ValueError(f"statistics artifact {key} mismatch")
    if not isinstance(payload["model"], str) or not payload["model"]:
        raise ValueError("statistics artifact model provenance is required")
    result = dict(payload)
    result["sigma2"] = sigma2
    result["q_absmean"] = q_absmean
    return result


def main():
    args = parse_args()
    if args.top_p is not None:
        _validate_top_p_threshold(args.top_p)
    if (args.stats_output
            and Path(args.stats_output).resolve() == Path(args.output).resolve()):
        raise ValueError("--stats-output and --output must be different paths")
    reference_top_p_artifact = None
    reference_mask_sha256 = None
    if args.uniform_top_k_control_from:
        reference_path = Path(args.uniform_top_k_control_from)
        if reference_path.resolve() == Path(args.output).resolve():
            raise ValueError(
                "--uniform-top-k-control-from and --output must be different paths")
        if (args.stats_output
                and reference_path.resolve() == Path(args.stats_output).resolve()):
            raise ValueError(
                "--uniform-top-k-control-from and --stats-output "
                "must be different paths")
        reference_top_p_artifact, reference_mask_sha256 = (
            load_top_p_reference_artifact(reference_path))
    torch.manual_seed(args.seed)
    t0 = time.time()

    if args.stats_input:
        stats = load_statistics_artifact(args.stats_input)
        sigma2 = stats["sigma2"]
        q_absmean = stats["q_absmean"]
        model_name = stats["model"]
        n_layers = stats["n_layers"]
        n_kv = stats["n_kv"]
        head_dim = stats["head_dim"]
        group_size = stats["group_size"]
        skip_first = stats["skip_first"]
        calib_data = stats.get("calib_data")
        num_samples = stats.get("num_samples")
        sample_len = stats.get("sample_len")
        seed = stats.get("seed")
        print(f"[calib] loaded reusable statistics from {args.stats_input}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="eager").to(args.device).eval()
        cfg = model.config
        n_q = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
        n_layers = cfg.num_hidden_layers
        batches = load_calib_token_batches(
            tok, args.calib_data, args.num_samples, args.sample_len, args.seed)
        sigma2, q_absmean = collect_statistics(
            model, rope_module_for(cfg), batches, args.skip_first, args.group_size,
            args.device, n_layers, n_kv, n_q // n_kv, head_dim)
        model_name = args.model
        group_size = args.group_size
        skip_first = args.skip_first
        calib_data = args.calib_data
        num_samples = args.num_samples
        sample_len = args.sample_len
        seed = args.seed

        if args.stats_output:
            stats_payload = build_statistics_artifact(
                sigma2, q_absmean, model=model_name, group_size=group_size,
                skip_first=skip_first, calib_data=args.calib_data,
                num_samples=args.num_samples, sample_len=args.sample_len,
                seed=args.seed)
            stats_out = Path(args.stats_output)
            stats_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stats_payload, stats_out)
            print(f"[calib] wrote reusable statistics to {stats_out}")

    print(
        f"[calib] model={model_name} layers={n_layers} kv={n_kv} D={head_dim}")
    if args.uniform_top_k_control_from:
        print(
            f"[calib] signal={RANKING_SIGNAL} selection={CONTROL_SELECTION_METHOD} "
            f"control_kind={args.control_kind}")
        payload = build_uniform_top_k_control_artifact(
            sigma2, q_absmean, reference_top_p_artifact,
            reference_mask_sha256, control_kind=args.control_kind,
            group_size=group_size, skip_first=skip_first, model=model_name,
            calib_data=calib_data, num_samples=num_samples,
            sample_len=sample_len, seed=seed)
    elif args.top_p is None:
        print(
            f"[calib] signal={RANKING_SIGNAL} low=sign({BITS['sign']}b)/"
            f"high=nf2({BITS['nf2']}b, {NF2_IMPL}) "
            f"sign_fraction={SIGN_FRACTION} -> nominal K ~{NOMINAL_K_BITS} bit/value")
        payload = build_canonical_artifact(
            sigma2, q_absmean, group_size=group_size, skip_first=skip_first,
            model=model_name)
    else:
        print(
            f"[calib] signal={RANKING_SIGNAL} selection={TOP_P_SELECTION_METHOD} "
            f"top_p={args.top_p}")
        payload = build_top_p_artifact(
            sigma2, q_absmean, args.top_p, group_size=group_size,
            skip_first=skip_first, model=model_name, calib_data=calib_data,
            num_samples=num_samples, sample_len=sample_len, seed=seed)

    mask = payload["codebook_mask"]
    n_sign = int((mask == 0).sum())
    total = mask.numel()
    print(
        f"[calib] sign={n_sign}/{total} ({n_sign / total:.2%}) "
        f"nf2={total - n_sign}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[calib] wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
