"""0-GPU wiring tests for the single canonical qlutattn variant.

Covers: variant resolution + slug, the fixed resolved config (K per-token
sign/nf2 f50 offline mask, symnf2-v1; V 2-bit tile16_rescued C=64), strict
mask validation, retired-knob rejection, QUEST rejection, GLM/FP16/head_dim
guards, mask-SHA/semantic-hash plumbing, shell/Python preflight parity,
manifest engagement, and the masked per-token K numerics.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from kitty_sim.cli.eval_longbench import _VARIANT_CHOICES, build_parser, finalize_args
from kitty_sim.kitty_simulate import KittyKVCacheConfig, KittyKVCache
from kitty_sim.longbench.runner import (
    NF2_IMPL_VERSION,
    QLUTATTN_VARIANT,
    QLUTATTN_CONTROL_SELECTION_METHOD,
    _cache_factory,
    _maybe_enable_quest_kernel,
    build_variant,
    generate_dataset,
    load_qlutattn_mask_blob,
    longbench_run_config_hash,
    method_layout_slug,
    resolve_longbench_preflight,
    run_longbench,
    validate_qlutattn_model_config,
    validate_qlutattn_model_family,
    validate_qlutattn_preload,
    variant_semantic_hash,
    variant_semantic_payload,
)
from kitty_sim.qlut_quant import nf2_symmetric_lastdim
from kitty_sim.v_tile_quant import (
    V_MSE_ITERS,
    V_RHT_SEED,
    V_TILE_ALGO_VERSION,
    V_TILE_TOKENS,
)

_GUARDED_ENV = (
    "QLUT_BIN_CODEBOOKS",
    "QLUT_CB_MASK",
    "V_TILE_CHANNELS",
    "VBITS",
    "PERTOKEN_BLOCK",
    "PERTOKEN_OUTLIER_K",
    "PROMOTE_RATIO_CONFIG",
)


def _args(variant: str = "qlutattn", **kwargs):
    ns = SimpleNamespace(
        variant=variant,
        kbits=2,
        vbits=2,
        promote_bit=4,
        promote_ratio=None,
        promote_ratio_config=None,
        sink_length=32,
        buffer_length=128,
        group_size=128,
        channel_selection=0,
        k_quant_mode="per_token",
        quest_kernel=False,
        quest_token_budget=None,
        quest_skip_layers=0,
        shadowkv_budget=None,
        shadowkv_rank=None,
        shadowkv_chunk_size=None,
        model="meta-llama/Llama-3.2-1B-Instruct",
        model_path=None,
        model_tag="llama32-1b-instruct",
        model_family="llama3",
        dataset=None,
        data_root=None,
        e=False,
        max_samples=-1,
        max_model_len=32768,
        default_max_model_len=3500,
        max_gen=256,
        prompt_token_reserve=0,
        torch_dtype="float16",
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _mask_payload(n_layers=1, n_kv=1, head_dim=64):
    """A valid canonical mask payload: sign/nf2, exactly round(0.65*N) sign
    channels per layer, low_frac = the ACTUAL fraction."""
    n = n_kv * head_dim
    k_sign = int(round(0.65 * n))
    mask = torch.zeros(n_layers, n_kv, head_dim, dtype=torch.uint8)
    flat = mask.reshape(n_layers, -1)
    flat[:, k_sign:] = 1
    return {
        "codebook_mask": mask,
        "codebooks": ["sign", "nf2"],
        "low_frac": k_sign / n,
    }


def _top_p_mask_payload(n_layers=1, n_kv=1, head_dim=64, threshold=0.55):
    """A valid top-p artifact whose increasing scores select the high suffix."""
    n_per_layer = n_kv * head_dim
    sigma2 = torch.arange(
        1, n_layers * n_per_layer + 1, dtype=torch.float32
    ).reshape(n_layers, n_kv, head_dim)
    q_absmean = torch.ones_like(sigma2)
    ranking = sigma2.double() * q_absmean.double()
    mask = torch.zeros(n_layers, n_kv, head_dim, dtype=torch.uint8)
    count_per_layer = torch.empty(n_layers, dtype=torch.int32)
    captured = torch.empty(n_layers, dtype=torch.float64)
    previous = torch.empty(n_layers, dtype=torch.float64)
    boundary = torch.empty(n_layers, dtype=torch.float64)
    for layer_idx in range(n_layers):
        flat = ranking[layer_idx].reshape(-1)
        order = torch.argsort(flat, descending=True, stable=True)
        ordered = flat[order]
        cumulative = ordered.cumsum(0)
        total = cumulative[-1]
        selected = int(torch.searchsorted(
            cumulative, total * threshold, right=False
        )) + 1
        mask[layer_idx].view(-1)[order[:selected]] = 1
        count_per_layer[layer_idx] = selected
        captured[layer_idx] = cumulative[selected - 1] / total
        previous[layer_idx] = (
            cumulative[selected - 2] / total if selected > 1 else 0.0
        )
        boundary[layer_idx] = ordered[selected - 1]

    count_per_head = (mask == 1).sum(-1).to(torch.int32)
    reorder = torch.empty(mask.shape, dtype=torch.int64)
    inverse = torch.empty_like(reorder)
    identity = torch.arange(head_dim, dtype=torch.int64)
    for layer_idx in range(n_layers):
        for head_idx in range(n_kv):
            head_mask = mask[layer_idx, head_idx]
            head_reorder = torch.cat((
                torch.nonzero(head_mask == 1, as_tuple=False).flatten(),
                torch.nonzero(head_mask == 0, as_tuple=False).flatten(),
            ))
            reorder[layer_idx, head_idx] = head_reorder
            inverse[layer_idx, head_idx, head_reorder] = identity

    nf2_count = int((mask == 1).sum())
    total_count = mask.numel()
    return {
        "format_version": 3,
        "selection_axis": "per_layer_flattened_kv_head_channel",
        "threshold_rule": "minimal_desc_prefix_cumsum_ge_p",
        "tie_rule": "score_desc_flat_index_asc",
        "score_dtype_for_selection": "float64",
        "nf2_impl": "symnf2-v1",
        "ranking_signal": "sigma2_x_q",
        "model": "test/model",
        "calib_data": "/test/calibration.parquet",
        "group_size": 128,
        "skip_first": 32,
        "num_samples": 8,
        "sample_len": 256,
        "seed": 0,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "selection_method": "layer_channel_top_p",
        "top_p_threshold": float(threshold),
        "codebook_mask": mask,
        "codebooks": ["sign", "nf2"],
        "reorder_index": reorder,
        "inverse_reorder_index": inverse,
        "nf2_count_per_head": count_per_head,
        "nf2_count_per_layer": count_per_layer,
        "sigma2": sigma2,
        "q_absmean": q_absmean,
        "ranking_score": ranking,
        "actual_nf2_frac": nf2_count / total_count,
        "low_frac": (total_count - nf2_count) / total_count,
        "nf2_ratio_per_layer": count_per_layer.double() / n_per_layer,
        "captured_mass_per_layer": captured,
        "previous_prefix_mass_per_layer": previous,
        "boundary_score_per_layer": boundary,
        "actual_nf2_count": nf2_count,
        "packed_bits_total": _packed_k_bits(mask),
        "packed_k_values_total": total_count,
    }


def _packed_k_bits(mask):
    """Independent packed-K accounting oracle for runtime control fixtures."""
    head_dim = mask.shape[-1]
    nf2_count = mask.to(torch.int64).sum(-1)
    code_bits = int(((head_dim - nf2_count) + 2 * nf2_count).sum())
    scale_bits = 16 * int((nf2_count > 0).sum())
    scale_bits += 16 * int((nf2_count < head_dim).sum())
    return code_bits + scale_bits


def _reference_semantic_sha256(reference):
    """Independent oracle for the control-v2 reference semantic digest."""
    reference_mask = reference["codebook_mask"]
    n_layers, n_kv, head_dim = reference_mask.shape
    metadata = {
        "semantic_hash_domain": "qlutattn_top_p_reference_semantics_v1",
        "top_p_format_version": 3,
        "selection_method": "layer_channel_top_p",
        "selection_axis": "per_layer_flattened_kv_head_channel",
        "threshold_rule": "minimal_desc_prefix_cumsum_ge_p",
        "tie_rule": "score_desc_flat_index_asc",
        "score_dtype_for_selection": "float64",
        "top_p_threshold_hex": float(reference["top_p_threshold"]).hex(),
        "codebooks": ["sign", "nf2"],
        "nf2_impl": "symnf2-v1",
        "ranking_signal": "sigma2_x_q",
        "model": reference["model"],
        "calib_data": reference["calib_data"],
        "group_size": reference["group_size"],
        "skip_first": reference["skip_first"],
        "num_samples": reference["num_samples"],
        "sample_len": reference["sample_len"],
        "seed": reference["seed"],
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "statistics_encoding": "ieee754_binary64_little_endian_c_order",
        "reference_mask_encoding": "uint8_c_order",
    }
    metadata_bytes = json.dumps(
        metadata,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sigma2_values = (
        reference["sigma2"].detach().to(dtype=torch.float64).contiguous()
        .numpy().astype("<f8", copy=False)
    )
    q_absmean_values = (
        reference["q_absmean"].detach().to(dtype=torch.float64).contiguous()
        .numpy().astype("<f8", copy=False)
    )
    reference_mask_values = (
        reference_mask.detach().to(dtype=torch.uint8).contiguous().numpy()
    )
    digest = hashlib.sha256()
    for label, values in (
        ("metadata_json_utf8", metadata_bytes),
        ("sigma2_float64_le", sigma2_values),
        ("q_absmean_float64_le", q_absmean_values),
        ("reference_mask_uint8", reference_mask_values),
    ):
        label_bytes = label.encode("ascii")
        value_bytes = memoryview(values).cast("B")
        digest.update(len(label_bytes).to_bytes(4, "big"))
        digest.update(label_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def _uniform_control_mask(ranking, reference_mask, total_nf2):
    n_layers, n_kv, head_dim = ranking.shape
    per_layer = n_kv * head_dim
    floor_count, remainder = divmod(total_nf2, n_layers)
    if floor_count <= 0 or floor_count + bool(remainder) >= per_layer:
        raise ValueError("nontrivial uniform layer counts required")
    flat = ranking.reshape(n_layers, -1)
    orders = [
        torch.argsort(flat[layer], descending=True, stable=True)
        for layer in range(n_layers)
    ]
    next_scores = torch.tensor([
        float(flat[layer, orders[layer][floor_count]])
        for layer in range(n_layers)
    ], dtype=torch.float64)
    extra_layers = set(
        torch.argsort(next_scores, descending=True, stable=True)[:remainder].tolist()
    )
    mask = torch.zeros_like(reference_mask)
    count_per_layer = torch.empty(n_layers, dtype=torch.int32)
    for layer in range(n_layers):
        count = floor_count + int(layer in extra_layers)
        mask[layer].reshape(-1)[orders[layer][:count]] = 1
        count_per_layer[layer] = count
    return mask, count_per_layer


def _control_mask_payload(
    control_kind="same_cardinality",
    n_layers=3,
    n_kv=2,
    head_dim=8,
    threshold=0.55,
):
    """Valid fixed-top-k control built independently of the runtime validator."""
    reference = _top_p_mask_payload(
        n_layers=n_layers,
        n_kv=n_kv,
        head_dim=head_dim,
        threshold=threshold,
    )
    reference_mask = reference["codebook_mask"]
    ranking = reference["ranking_score"]
    reference_nf2 = int(reference_mask.sum())
    if control_kind == "same_cardinality":
        mask, count_per_layer = _uniform_control_mask(
            ranking, reference_mask, reference_nf2
        )
    elif control_kind == "exact_packed_bits":
        target = _packed_k_bits(reference_mask)
        for total_nf2 in sorted(
            range(1, reference_mask.numel()),
            key=lambda value: (abs(value - reference_nf2), value),
        ):
            try:
                candidate, candidate_counts = _uniform_control_mask(
                    ranking, reference_mask, total_nf2
                )
            except ValueError:
                continue
            if _packed_k_bits(candidate) == target:
                mask, count_per_layer = candidate, candidate_counts
                break
        else:
            raise AssertionError("fixture has no exact packed-bit control")
    else:
        raise ValueError(control_kind)

    reorder = torch.empty_like(mask, dtype=torch.int64)
    inverse = torch.empty_like(reorder)
    identity = torch.arange(head_dim, dtype=torch.int64)
    count_per_head = mask.to(torch.int32).sum(-1, dtype=torch.int32)
    for layer in range(n_layers):
        for head in range(n_kv):
            selected = mask[layer, head].bool()
            index = torch.cat((identity[selected], identity[~selected]))
            reorder[layer, head] = index
            inverse[layer, head, index] = identity

    actual_nf2 = int(mask.sum())
    total = mask.numel()
    return {
        "selection_method": QLUTATTN_CONTROL_SELECTION_METHOD,
        "format_version": 2,
        "selection_axis": "per_layer_flattened_kv_head_channel",
        "tie_rule": "score_desc_flat_index_asc",
        "score_dtype_for_selection": "float64",
        "control_kind": control_kind,
        "reference_mask_sha256": "a" * 64,
        "reference_semantic_sha256": _reference_semantic_sha256(reference),
        "reference_selection_method": "layer_channel_top_p",
        "reference_top_p_threshold": float(threshold),
        "reference_codebook_mask": reference_mask.clone(),
        "codebook_mask": mask,
        "codebooks": ["sign", "nf2"],
        "low_frac": (total - actual_nf2) / total,
        "actual_nf2_frac": actual_nf2 / total,
        "nf2_impl": "symnf2-v1",
        "ranking_signal": "sigma2_x_q",
        "reorder_index": reorder,
        "inverse_reorder_index": inverse,
        "nf2_count_per_head": count_per_head,
        "nf2_count_per_layer": count_per_layer,
        "nf2_ratio_per_layer": count_per_layer.double() / (n_kv * head_dim),
        "group_size": reference["group_size"],
        "skip_first": reference["skip_first"],
        "model": reference["model"],
        "calib_data": reference["calib_data"],
        "num_samples": reference["num_samples"],
        "sample_len": reference["sample_len"],
        "seed": reference["seed"],
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "sigma2": reference["sigma2"],
        "q_absmean": reference["q_absmean"],
        "ranking_score": ranking,
        "reference_nf2_count": reference_nf2,
        "actual_nf2_count": actual_nf2,
        "reference_packed_bits_total": _packed_k_bits(reference_mask),
        "packed_bits_total": _packed_k_bits(mask),
    }


class _EnvIsolation(unittest.TestCase):
    def setUp(self):
        self._env_backup = {k: os.environ.get(k) for k in _GUARDED_ENV}
        for k in list(self._env_backup):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _write_mask(self, payload=None) -> str:
        path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        self.addCleanup(Path(path).unlink, missing_ok=True)
        torch.save(payload if payload is not None else _mask_payload(), path)
        return path

    def _masked_args(self, **kwargs):
        os.environ["QLUT_CB_MASK"] = self._write_mask()
        return _args(**kwargs)


class TestCanonicalResolution(_EnvIsolation):
    def test_public_qlut_surface_is_exactly_qlutattn(self):
        qlut = [x for x in _VARIANT_CHOICES if "qlut" in x.lower()]
        self.assertEqual(qlut, ["qlutattn"])

    def test_resolved_config_is_fixed(self):
        cfg = build_variant(self._masked_args())
        self.assertEqual(cfg.name, QLUTATTN_VARIANT)
        self.assertEqual(method_layout_slug(cfg), "qlutattn")
        self.assertEqual(method_layout_slug("qlutattn"), "qlutattn")
        self.assertEqual(cfg.tag, "qlutattn")
        self.assertTrue(cfg.use_kitty)
        self.assertEqual(cfg.k_quant_mode, "per_token")
        self.assertEqual(cfg.k_codebook, "qlut")
        self.assertEqual(cfg.bin_codebooks, ("sign", "nf2"))
        self.assertEqual(cfg.vbits, 2)
        self.assertEqual(cfg.v_codebook, "tile16_rescued")
        self.assertEqual(cfg.v_tile_tokens, V_TILE_TOKENS)
        self.assertEqual(cfg.v_tile_tokens, 16)
        self.assertEqual(cfg.v_tile_channels, 64)
        self.assertEqual(cfg.v_tile_algo_version, V_TILE_ALGO_VERSION)
        self.assertEqual(cfg.v_tile_algo_version, "rht-pcaff-mse1-bias-v1")
        self.assertEqual(cfg.v_rht_seed, V_RHT_SEED)
        self.assertEqual(cfg.v_rht_seed, 20260711)
        self.assertEqual(cfg.v_mse_iters, V_MSE_ITERS)
        self.assertEqual(cfg.v_mse_iters, 1)
        self.assertEqual(cfg.nf2_impl, NF2_IMPL_VERSION)
        self.assertEqual(cfg.nf2_impl, "symnf2-v1")
        self.assertEqual(cfg.sink_length, 32)
        self.assertEqual(cfg.buffer_length, 128)
        self.assertEqual(cfg.group_size, 128)
        self.assertEqual(cfg.promote_ratio, 0.0)
        # Retired mechanisms have no config surface at all.
        for retired in ("pertoken_rotate", "pertoken_block", "pertoken_pc_submean",
                        "pertoken_mixed", "pertoken_outlier_k", "n_bins"):
            self.assertFalse(hasattr(cfg, retired), retired)

    def test_mask_required(self):
        with self.assertRaises(FileNotFoundError):
            build_variant(_args())

    def test_semantic_payload_carries_mask_sha(self):
        cfg = build_variant(self._masked_args())
        payload = variant_semantic_payload(cfg)
        self.assertTrue(payload["mask_sha256"])
        self.assertNotIn("pertoken_cb_mask", payload)  # host path never hashed
        self.assertEqual(payload["nf2_impl"], NF2_IMPL_VERSION)
        self.assertNotIn("selection_method", payload)
        self.assertNotIn("top_p_threshold", payload)
        self.assertNotIn("actual_nf2_frac", payload)
        self.assertNotIn("actual_nf2_count", payload)
        self.assertNotIn("packed_bits_total", payload)
        self.assertNotIn("packed_k_values_total", payload)

    def test_cache_factory_threads_canonical_fields(self):
        cfg = build_variant(self._masked_args())
        cache = _cache_factory(cfg)
        self.assertEqual(cache.v_codebook, "tile16_rescued")
        self.assertEqual(cache.vbits, 2)
        self.assertEqual(cache.v_tile_tokens, 16)
        self.assertEqual(cache.v_tile_channels, 64)
        self.assertEqual(cache.v_tile_algo_version, V_TILE_ALGO_VERSION)
        self.assertEqual(cache.k_codebook, "qlut")
        self.assertEqual(cache.bin_codebooks, ["sign", "nf2"])
        self.assertTrue(cache.pertoken_offline)
        self.assertIn(0, cache.k_cb_mask)

    def test_top_p_resolution_is_threshold_qualified(self):
        mask_path = self._write_mask(_top_p_mask_payload())
        os.environ["QLUT_CB_MASK"] = mask_path
        digest = hashlib.sha256(Path(mask_path).read_bytes()).hexdigest()
        cfg = build_variant(_args())
        self.assertEqual(cfg.name, QLUTATTN_VARIANT)
        self.assertEqual(cfg.selection_method, "layer_channel_top_p")
        self.assertEqual(cfg.top_p_threshold, 0.55)
        self.assertEqual(
            cfg.actual_nf2_frac,
            _top_p_mask_payload()["actual_nf2_frac"],
        )
        self.assertEqual(cfg.actual_nf2_count, _top_p_mask_payload()["actual_nf2_count"])
        self.assertEqual(cfg.packed_bits_total, _top_p_mask_payload()["packed_bits_total"])
        self.assertEqual(
            cfg.packed_k_values_total,
            _top_p_mask_payload()["packed_k_values_total"],
        )
        expected_slug = f"qlutattn-topp-p0p55-m{digest}"
        self.assertEqual(cfg.tag, expected_slug)
        self.assertEqual(method_layout_slug(cfg), expected_slug)
        payload = variant_semantic_payload(cfg)
        self.assertEqual(payload["selection_method"], "layer_channel_top_p")
        self.assertEqual(payload["top_p_threshold"], 0.55)
        self.assertEqual(payload["actual_nf2_frac"], cfg.actual_nf2_frac)
        for key in (
            "actual_nf2_count",
            "packed_bits_total",
            "packed_k_values_total",
        ):
            self.assertEqual(payload[key], getattr(cfg, key), key)
        preflight = resolve_longbench_preflight(_args())
        for key in (
            "actual_nf2_count",
            "packed_bits_total",
            "packed_k_values_total",
        ):
            self.assertEqual(
                preflight["resolved_variant"][key], getattr(cfg, key), key
            )
        cache = _cache_factory(cfg)
        self.assertEqual(cache.qlut_selection_method, "layer_channel_top_p")
        self.assertIn(0, cache.k_reorder_index)
        self.assertIn(0, cache.k_inverse_reorder_index)
        self.assertIn(0, cache.k_nf2_count_per_head)


class TestRetiredKnobRejection(_EnvIsolation):
    def test_vbits_locked_to_two(self):
        with self.assertRaises(ValueError):
            build_variant(self._masked_args(vbits=4))
        os.environ["VBITS"] = "4"
        from_env = finalize_args(build_parser().parse_args(["model", "--variant", "qlutattn"]))
        with self.assertRaises(ValueError):
            build_variant(from_env)
        os.environ["VBITS"] = "2"
        ok = finalize_args(build_parser().parse_args(["model", "--variant", "qlutattn"]))
        self.assertEqual(build_variant(ok).vbits, 2)

    def test_pertoken_block_env(self):
        for ok_value in ("", "1"):
            os.environ["PERTOKEN_BLOCK"] = ok_value
            build_variant(self._masked_args())
        os.environ["PERTOKEN_BLOCK"] = "16"
        with self.assertRaises(ValueError):
            build_variant(self._masked_args())

    def test_retired_env_knobs_are_hard_errors(self):
        for env_name, value in (
            ("QLUT_BIN_CODEBOOKS", "sign"),
            ("V_TILE_CHANNELS", "64"),
            ("PERTOKEN_OUTLIER_K", "8"),
            ("PROMOTE_RATIO_CONFIG", "/tmp/nonexistent.json"),
        ):
            with self.subTest(env=env_name):
                args = self._masked_args()
                os.environ[env_name] = value
                try:
                    with self.assertRaises(ValueError):
                        build_variant(args)
                finally:
                    os.environ.pop(env_name, None)

    def test_empty_strings_are_unset_sentinels(self):
        for env_name in ("QLUT_BIN_CODEBOOKS", "V_TILE_CHANNELS",
                         "PERTOKEN_OUTLIER_K", "PROMOTE_RATIO_CONFIG"):
            os.environ[env_name] = ""
        cfg = build_variant(self._masked_args())
        self.assertEqual(cfg.name, "qlutattn")

    def test_quest_overlay_rejected(self):
        cfg = build_variant(self._masked_args())
        for env_name in ("QUEST_KERNEL", "QUEST_TRITON", "SIM_QUEST", "QUEST_SIM"):
            with self.subTest(env=env_name):
                with mock.patch.dict(os.environ, {env_name: "1"}, clear=False):
                    with self.assertRaisesRegex(ValueError, "qlutattn"):
                        _maybe_enable_quest_kernel(cfg, SimpleNamespace())

    def test_promote_ratio_config_cli_rejected(self):
        with self.assertRaises(ValueError):
            build_variant(self._masked_args(promote_ratio_config="/tmp/x.json"))


class TestMaskValidation(_EnvIsolation):
    def test_valid_mask_accepted(self):
        blob = load_qlutattn_mask_blob(self._write_mask())
        self.assertEqual(tuple(blob["codebooks"]), ("sign", "nf2"))

    def test_legacy_metadata_keys_are_ignored(self):
        payload = _mask_payload()
        payload["target_bits"] = 1.68
        payload["nominal_bits"] = 1.875
        blob = load_qlutattn_mask_blob(self._write_mask(payload))
        self.assertIsNotNone(blob)

    def test_bad_masks_rejected(self):
        good = _mask_payload()
        wrong_codebooks = dict(good, codebooks=["sign", "tern"])
        wrong_frac_meta = dict(good, low_frac=0.25)
        no_low_frac = {k: v for k, v in good.items() if k != "low_frac"}
        wrong_dtype = dict(good, codebook_mask=good["codebook_mask"].long())
        wrong_rank = dict(good, codebook_mask=good["codebook_mask"][0])
        all_zero = dict(good, codebook_mask=torch.zeros(1, 1, 64, dtype=torch.uint8))
        bad_values = dict(good, codebook_mask=(good["codebook_mask"] * 2))
        skew = _mask_payload()
        skew["codebook_mask"][..., :48] = 0  # 75% sign, metadata still claims 0.5
        for label, payload in (
            ("codebooks", wrong_codebooks),
            ("low_frac metadata", wrong_frac_meta),
            ("missing low_frac", no_low_frac),
            ("dtype", wrong_dtype),
            ("rank", wrong_rank),
            ("all-zero values", all_zero),
            ("value set", bad_values),
            ("actual sign fraction", skew),
        ):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    load_qlutattn_mask_blob(self._write_mask(payload))

    def test_build_variant_validates_mask_payload(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(
            dict(_mask_payload(), codebooks=["sign", "tern"])
        )
        with self.assertRaises(ValueError):
            build_variant(_args())

    def test_valid_top_p_mask_and_permutation_round_trip(self):
        payload = _top_p_mask_payload(n_layers=2, n_kv=2, head_dim=16)
        blob = load_qlutattn_mask_blob(self._write_mask(payload))
        x = torch.arange(2 * 2 * 16).reshape(2, 2, 16)
        reordered = torch.gather(x, -1, blob["reorder_index"])
        restored = torch.gather(reordered, -1, blob["inverse_reorder_index"])
        torch.testing.assert_close(restored, x, atol=0, rtol=0)
        reordered_mask = torch.gather(
            blob["codebook_mask"], -1, blob["reorder_index"]
        )
        expected_prefix = (
            torch.arange(16)[None, None, :]
            < blob["nf2_count_per_head"].unsqueeze(-1)
        ).to(torch.uint8)
        torch.testing.assert_close(reordered_mask, expected_prefix, atol=0, rtol=0)

    def test_top_p_mask_corruption_is_rejected(self):
        cases = []

        missing = _top_p_mask_payload()
        missing.pop("previous_prefix_mass_per_layer")
        cases.append(("missing statistics", missing))

        bad_permutation = _top_p_mask_payload()
        bad_permutation["reorder_index"] = bad_permutation["reorder_index"].clone()
        bad_permutation["reorder_index"][0, 0, 0] = (
            bad_permutation["reorder_index"][0, 0, 1]
        )
        cases.append(("non-permutation", bad_permutation))

        bad_inverse = _top_p_mask_payload()
        bad_inverse["inverse_reorder_index"] = torch.roll(
            bad_inverse["inverse_reorder_index"], 1, dims=-1
        )
        cases.append(("bad inverse", bad_inverse))

        bad_count_dtype = _top_p_mask_payload()
        bad_count_dtype["nf2_count_per_head"] = (
            bad_count_dtype["nf2_count_per_head"].long()
        )
        cases.append(("count dtype", bad_count_dtype))

        bad_count = _top_p_mask_payload()
        bad_count["nf2_count_per_layer"] = (
            bad_count["nf2_count_per_layer"].clone()
        )
        bad_count["nf2_count_per_layer"][0] += 1
        cases.append(("count value", bad_count))

        bad_ranking = _top_p_mask_payload()
        bad_ranking["ranking_score"] = bad_ranking["ranking_score"].clone()
        bad_ranking["ranking_score"][0, 0, 0] += 1.0
        cases.append(("ranking product", bad_ranking))

        bad_mass = _top_p_mask_payload()
        bad_mass["captured_mass_per_layer"] = (
            bad_mass["captured_mass_per_layer"].clone()
        )
        bad_mass["captured_mass_per_layer"][0] += 0.01
        cases.append(("captured mass", bad_mass))

        bad_fraction = _top_p_mask_payload()
        bad_fraction["actual_nf2_frac"] += 0.01
        cases.append(("actual fraction", bad_fraction))

        missing_method = _top_p_mask_payload()
        missing_method.pop("selection_method")
        cases.append(("missing selection method", missing_method))

        unknown_method = _top_p_mask_payload()
        unknown_method["selection_method"] = "other"
        cases.append(("selection method", unknown_method))

        bad_identity = _top_p_mask_payload()
        bad_identity["tie_rule"] = "unstable"
        cases.append(("identity metadata", bad_identity))
        for label, key, value in (
            ("nf2 implementation", "nf2_impl", "other"),
            ("ranking signal", "ranking_signal", "other"),
            ("model provenance", "model", ""),
            ("calibration provenance", "calib_data", " "),
            ("group size", "group_size", 0),
            ("sample count", "num_samples", 0),
            ("sample length", "sample_len", 0),
            ("skip first", "skip_first", -1),
            ("seed type", "seed", True),
            ("layer geometry", "n_layers", 2),
            ("head geometry", "n_kv", 2),
            ("channel geometry", "head_dim", 63),
        ):
            corrupted = _top_p_mask_payload()
            corrupted[key] = value
            cases.append((label, corrupted))

        float_format = _top_p_mask_payload()
        float_format["format_version"] = 3.0
        cases.append(("float format version", float_format))

        for key in (
            "actual_nf2_count",
            "packed_bits_total",
            "packed_k_values_total",
        ):
            missing_accounting = _top_p_mask_payload()
            missing_accounting.pop(key)
            cases.append((f"missing {key}", missing_accounting))

            wrong_accounting = _top_p_mask_payload()
            wrong_accounting[key] += 1
            cases.append((f"wrong {key}", wrong_accounting))

        for label, payload in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    load_qlutattn_mask_blob(self._write_mask(payload))


class TestModelGuards(_EnvIsolation):
    def test_glm_failfast(self):
        cfg = build_variant(self._masked_args())
        with self.assertRaisesRegex(ValueError, "GLM parity"):
            validate_qlutattn_model_family(cfg, "glm4")
        validate_qlutattn_model_family(cfg, "llama3")

    def test_model_dependent_validation(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload(1, 1, 64))
        cfg = build_variant(_args())
        good = SimpleNamespace(
            head_dim=64, num_hidden_layers=1, num_key_value_heads=1
        )
        validate_qlutattn_model_config(cfg, good, torch.float16)
        with self.assertRaisesRegex(ValueError, "FP16"):
            validate_qlutattn_model_config(cfg, good, torch.float32)
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_qlutattn_model_config(
                cfg,
                SimpleNamespace(head_dim=96, num_hidden_layers=1, num_key_value_heads=1),
                torch.float16,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_qlutattn_model_config(
                cfg,
                SimpleNamespace(head_dim=64, num_hidden_layers=2, num_key_value_heads=1),
                torch.float16,
            )

    def test_config_only_preload_validates_without_cuda(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload(1, 2, 64))
        cfg = build_variant(_args())
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({
                    "model_type": "llama",
                    "hidden_size": 128,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 1,
                    "intermediate_size": 128,
                    "vocab_size": 128,
                }),
                encoding="utf-8",
            )
            args = _args(model_path=str(model), local_files_only=True)
            cuda_before = torch.cuda.is_initialized()
            validate_qlutattn_preload(args, cfg, "llama3")
            self.assertEqual(torch.cuda.is_initialized(), cuda_before)
            # head_dim 64 with 2 kv heads matches the mask; shrink the mask to
            # prove a mismatch is caught before any weight load.
            os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload(3, 2, 64))
            bad_cfg = build_variant(_args())
            with self.assertRaisesRegex(ValueError, "shape"):
                validate_qlutattn_preload(args, bad_cfg, "llama3")

    def test_cache_config_requires_mask_for_qlut(self):
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(
                k_quant_mode="per_token", k_codebook="qlut",
                bin_codebooks=["sign", "nf2"], promote_ratio=0.0,
                channel_selection=0,
            )


class TestHashesAndPreflight(_EnvIsolation):
    def test_preflight_slug_and_hash(self):
        payload = resolve_longbench_preflight(self._masked_args())
        self.assertEqual(payload["canonical_variant"], "qlutattn")
        self.assertEqual(payload["method_slug"], "qlutattn")
        self.assertTrue(payload["mask_sha256"])
        self.assertEqual(payload["resolved_variant"]["nf2_impl"], NF2_IMPL_VERSION)
        self.assertEqual(payload["resolved_variant"]["v_tile_channels"], 64)

    def test_run_hash_changes_with_mask_content(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
            data_dir = root / "longbench" / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "narrativeqa.jsonl").write_text("{}\n", encoding="utf-8")
            mask = root / "mask.pt"
            os.environ["QLUT_CB_MASK"] = str(mask)
            torch.save(_mask_payload(1, 1, 64), mask)
            args = _args(
                model_path=str(model), dataset="narrativeqa",
                data_root=str(root / "longbench"),
            )
            h1 = longbench_run_config_hash(args, build_variant(args), "narrativeqa")
            # Different mask CONTENT with the same valid per-layer sign count:
            # rotate the channel assignment by one position.
            rolled = _mask_payload(1, 1, 64)
            rolled["codebook_mask"] = torch.roll(rolled["codebook_mask"], 1, dims=-1)
            torch.save(rolled, mask)
            h2 = longbench_run_config_hash(args, build_variant(args), "narrativeqa")
            self.assertNotEqual(h1, h2)

    def test_shell_and_python_slug_parity(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask()
        expected = resolve_longbench_preflight(_args())["method_slug"]
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update({
            "KITTY_ENV_FILE": str(repo / "does-not-exist.env"),
            "PYTHON_BIN": sys.executable,
            "PYTHONPATH": str(repo / "src"),
        })
        got = subprocess.run(
            ["bash", "scripts/run_exp.sh", "--variant", "qlutattn", "--print-method-slug"],
            cwd=repo, env=env, text=True, capture_output=True, check=True,
        ).stdout.strip()
        self.assertEqual(got, expected)
        self.assertEqual(got, "qlutattn")

    def test_shell_and_python_reject_stale_knob_consistently(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask()
        os.environ["QLUT_BIN_CODEBOOKS"] = "sign"
        with self.assertRaises(ValueError):
            build_variant(_args())
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update({
            "KITTY_ENV_FILE": str(repo / "does-not-exist.env"),
            "PYTHON_BIN": sys.executable,
            "PYTHONPATH": str(repo / "src"),
        })
        got = subprocess.run(
            ["bash", "scripts/run_exp.sh", "--variant", "qlutattn", "--print-method-slug"],
            cwd=repo, env=env, text=True, capture_output=True,
        )
        self.assertNotEqual(got.returncode, 0)
        self.assertIn("QLUT_BIN_CODEBOOKS", got.stderr)

    def test_worker_recomputes_expected_hash_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({
                    "model_type": "llama",
                    "hidden_size": 64,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_hidden_layers": 1,
                    "intermediate_size": 128,
                    "vocab_size": 128,
                }),
                encoding="utf-8",
            )
            data_dir = root / "longbench" / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "narrativeqa.jsonl").write_text("{}\n", encoding="utf-8")
            os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload(1, 1, 64))
            args = _args(
                model_path=str(model),
                dataset="narrativeqa",
                data_root=str(root / "longbench"),
                local_files_only=True,
                expected_run_config_hash="deliberately-wrong",
                require_gpu1=False,
                output_dir=str(root / "pred"),
                flat_output_dir=True,
                overwrite=False,
                strict_complete=True,
                report_json=None,
            )
            with mock.patch(
                "kitty_sim.longbench.runner.load_model_and_tokenizer"
            ) as loader:
                with self.assertRaisesRegex(RuntimeError, "disagrees with shell preflight"):
                    run_longbench(args)
                loader.assert_not_called()

    def test_completed_fastpath_preserves_and_checks_manifest(self):
        cfg = build_variant(self._masked_args())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "narrativeqa.jsonl"
            out.write_text("{}\n", encoding="utf-8")
            manifest = {
                "status": "ok", "expected_samples": 1, "written_samples": 1,
                "run_config_hash": "abc",
                "marker": "preserve-me",
            }
            out.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            got = generate_dataset(
                model_name="model", model=None, tokenizer=None, dataset="narrativeqa",
                records=[{}], prompt_format="{input}", max_model_len=32768, max_gen=1,
                out_path=out, variant=cfg, model_family="llama3",
                expected_run_config_hash="abc",
            )
            self.assertEqual(got["marker"], "preserve-me")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                generate_dataset(
                    model_name="model", model=None, tokenizer=None, dataset="narrativeqa",
                    records=[{}], prompt_format="{input}", max_model_len=32768, max_gen=1,
                    out_path=out, variant=cfg, model_family="llama3",
                    expected_run_config_hash="different",
                )


class TestMaskedKNumerics(_EnvIsolation):
    def test_prefill_is_pc_mean_plus_masked_sign_nf2(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload(1, 2, 16))
        cfg = build_variant(_args())
        cache = _cache_factory(cfg)
        torch.manual_seed(0)
        B, nh, T, D = 1, 2, 64, 16
        ks = torch.randn(B, nh, T, D, dtype=torch.float32) + 2.0
        out = cache._quant_k_pertoken(ks, 0)
        mu = ks[0].mean(dim=1)                                   # [nh,D] per-channel mean
        muB = mu[None, :, None, :]
        torch.testing.assert_close(cache.k_pc_mean[0], mu, atol=0.0, rtol=0.0)
        r = ks - muB
        m_sign = cache.k_cb_mask[0] == 0
        m_nf2 = cache.k_cb_mask[0] == 1
        expected = torch.zeros_like(r)
        for h in range(nh):
            sel = m_sign[h]
            if sel.any():
                sub = r[:, h, :, sel]
                expected[:, h, :, sel] = torch.sign(sub) * sub.abs().mean(-1, keepdim=True)
            sel = m_nf2[h]
            if sel.any():
                # A head may hold zero nf2 channels under the layer-global 65/35
                # ranking; the runtime reconstructs those rows as 0.
                expected[:, h, :, sel] = nf2_symmetric_lastdim(r[:, h, :, sel])
        # fp32 tolerance: the runtime's masked reductions sum zero-padded full
        # rows (vectorized across heads) while this reference sums the selected
        # slice — same math, different fp32 summation order (~2e-7).
        torch.testing.assert_close(out, (muB + expected).to(out.dtype), atol=1e-6, rtol=0.0)

    def test_top_p_reordered_quantization_matches_canonical_mask(self):
        canonical = _mask_payload(1, 2, 16)
        top_p = _top_p_mask_payload(1, 2, 16, threshold=0.55)
        torch.testing.assert_close(
            top_p["codebook_mask"], canonical["codebook_mask"], atol=0, rtol=0
        )

        os.environ["QLUT_CB_MASK"] = self._write_mask(canonical)
        canonical_cache = _cache_factory(build_variant(_args()))
        os.environ["QLUT_CB_MASK"] = self._write_mask(top_p)
        top_p_cache = _cache_factory(build_variant(_args()))

        torch.manual_seed(2)
        ks = (torch.randn(1, 2, 64, 16) + 1.5).to(torch.float16)
        canonical_out = canonical_cache._quant_k_pertoken(ks, 0)
        top_p_out = top_p_cache._quant_k_pertoken(ks, 0)

        self.assertEqual(top_p_out.dtype, ks.dtype)
        self.assertEqual(
            top_p_cache.k_nf2_count_per_head[0].tolist(), [0, 11]
        )
        torch.testing.assert_close(
            top_p_out, canonical_out, atol=torch.finfo(torch.float16).eps, rtol=0
        )

    def test_decode_reuses_cached_pc_mean(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload(1, 2, 16))
        cfg = build_variant(_args())
        cache = _cache_factory(cfg)
        torch.manual_seed(1)
        ks = torch.randn(1, 2, 64, 16, dtype=torch.float32)
        cache._quant_k_pertoken(ks, 0)                           # prefill sets mu_d
        mu = cache.k_pc_mean[0].clone()
        ks1 = torch.randn(1, 2, 1, 16, dtype=torch.float32)
        cache._quant_k_pertoken(ks1, 0)
        torch.testing.assert_close(cache.k_pc_mean[0], mu, atol=0.0, rtol=0.0)


class TestManifestEngagement(_EnvIsolation):
    def test_fresh_manifest_records_top_p_semantics_and_engagement(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(_top_p_mask_payload(1, 1, 64))
        cfg = build_variant(_args())

        class _Batch(dict):
            def __init__(self):
                super().__init__()
                self.input_ids = torch.zeros(1, 256, dtype=torch.long)
                self["input_ids"] = self.input_ids

            def to(self, _device):
                return self

        class _Tokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def __call__(self, *_args, **_kwargs):
                return _Batch()

            def decode(self, *_args, **_kwargs):
                return "prediction"

            def encode(self, *_args, **_kwargs):
                return [3]

        class _Model:
            device = torch.device("cpu")

            def generate(self, **kwargs):
                cache = kwargs["past_key_values"]
                value = torch.linspace(
                    -1, 1, 256 * 64, dtype=torch.float16
                ).reshape(1, 1, 256, 64)
                cache.update(value, value.clone(), 0, {})
                return torch.cat(
                    [kwargs["input_ids"], torch.ones(1, 1, dtype=torch.long)], dim=1
                )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "trec.jsonl"
            manifest = generate_dataset(
                model_name="model",
                model=_Model(),
                tokenizer=_Tokenizer(),
                dataset="trec",
                records=[{
                    "input": "question",
                    "answers": ["answer"],
                    "all_classes": [],
                    "length": 1,
                }],
                prompt_format="{input}",
                max_model_len=32768,
                max_gen=1,
                out_path=out,
                variant=cfg,
                model_family="default",
                expected_run_config_hash="run-hash",
            )
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["variant"]["name"], "qlutattn")
            self.assertEqual(manifest["variant"]["v_codebook"], "tile16_rescued")
            self.assertEqual(manifest["variant"]["vbits"], 2)
            self.assertEqual(
                manifest["variant"]["selection_method"], "layer_channel_top_p"
            )
            self.assertEqual(manifest["variant"]["top_p_threshold"], 0.55)
            self.assertEqual(
                manifest["variant"]["actual_nf2_frac"], cfg.actual_nf2_frac
            )
            for key in (
                "actual_nf2_count",
                "packed_bits_total",
                "packed_k_values_total",
            ):
                self.assertEqual(
                    manifest["variant"][key], getattr(cfg, key), key
                )
            self.assertEqual(manifest["run_config_hash"], "run-hash")
            self.assertEqual(manifest["variant_semantic_hash"], variant_semantic_hash(cfg))
            self.assertTrue(manifest["mask_sha256"])
            self.assertGreater(manifest["engagement"]["k_quant_calls"], 0)
            self.assertGreater(manifest["engagement"]["k_quantized_tokens"], 0)
            self.assertGreater(manifest["engagement"]["k_prompt_mean_layers"], 0)
            self.assertEqual(
                manifest["engagement"]["last_k_quant_mode"], "per_token:qlut"
            )
            self.assertGreater(manifest["engagement"]["v_quant_calls"], 0)
            self.assertGreater(manifest["engagement"]["v_quantized_tokens"], 0)
            self.assertGreater(manifest["engagement"]["v_tile_blocks"], 0)
            self.assertEqual(manifest["engagement"]["last_v_quant_mode"], "tile16_rescued")
            self.assertEqual(manifest["engagement"]["last_v_tile_channels"], 64)


    def test_control_refuses_missing_k_engagement_evidence(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(
            _control_mask_payload(
                control_kind="same_cardinality",
                n_layers=1,
                n_kv=1,
                head_dim=64,
            )
        )
        cfg = build_variant(_args())

        class _Batch(dict):
            def __init__(self):
                super().__init__()
                self.input_ids = torch.zeros(1, 256, dtype=torch.long)
                self["input_ids"] = self.input_ids

            def to(self, _device):
                return self

        class _Tokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def __call__(self, *_args, **_kwargs):
                return _Batch()

            def decode(self, *_args, **_kwargs):
                return "prediction"

            def encode(self, *_args, **_kwargs):
                return [3]

        class _Model:
            device = torch.device("cpu")

            def __init__(self, missing):
                self.missing = missing

            def generate(self, **kwargs):
                cache = kwargs["past_key_values"]
                value = torch.linspace(
                    -1, 1, 256 * 64, dtype=torch.float16
                ).reshape(1, 1, 256, 64)
                cache.update(value, value.clone(), 0, {})
                if self.missing == "k_prompt_mean_layers":
                    cache.k_pc_mean.clear()
                elif self.missing == "last_k_quant_mode":
                    cache.last_k_quant_mode = "wrong"
                else:
                    setattr(cache, self.missing, 0)
                return torch.cat(
                    [kwargs["input_ids"], torch.ones(1, 1, dtype=torch.long)],
                    dim=1,
                )

        for field, evidence in (
            ("k_quant_calls", "k_quant_calls=0"),
            ("k_quantized_tokens", "k_quantized_tokens=0"),
            ("k_prompt_mean_layers", "k_prompt_mean_layers=0"),
            ("last_k_quant_mode", "last_k_quant_mode=wrong"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(RuntimeError, evidence):
                    generate_dataset(
                        model_name="model",
                        model=_Model(field),
                        tokenizer=_Tokenizer(),
                        dataset="trec",
                        records=[{
                            "input": "question",
                            "answers": ["answer"],
                            "all_classes": [],
                            "length": 1,
                        }],
                        prompt_format="{input}",
                        max_model_len=32768,
                        max_gen=1,
                        out_path=Path(td) / "trec.jsonl",
                        variant=cfg,
                        model_family="default",
                        expected_run_config_hash="run-hash",
                    )


class TestFixedTopKControlRuntime(_EnvIsolation):
    def test_control_kinds_have_distinct_slugs_hashes_and_metadata(self):
        configs = {}
        slugs = {}
        hashes = {}
        for kind in ("same_cardinality", "exact_packed_bits"):
            artifact = _control_mask_payload(control_kind=kind)
            path = self._write_mask(artifact)
            os.environ["QLUT_CB_MASK"] = path
            cfg = build_variant(_args())
            configs[kind] = cfg
            slugs[kind] = method_layout_slug(cfg)
            hashes[kind] = variant_semantic_hash(cfg)

            file_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            self.assertEqual(cfg.selection_method, QLUTATTN_CONTROL_SELECTION_METHOD)
            self.assertEqual(cfg.control_kind, kind)
            self.assertEqual(cfg.reference_selection_method, "layer_channel_top_p")
            self.assertEqual(cfg.reference_top_p_threshold, 0.55)
            self.assertEqual(cfg.reference_mask_sha256, "a" * 64)
            self.assertEqual(
                cfg.reference_semantic_sha256,
                artifact["reference_semantic_sha256"],
            )
            self.assertEqual(
                cfg.reference_nf2_count, artifact["reference_nf2_count"]
            )
            self.assertEqual(cfg.actual_nf2_count, artifact["actual_nf2_count"])
            self.assertEqual(
                cfg.reference_packed_bits_total,
                artifact["reference_packed_bits_total"],
            )
            self.assertEqual(cfg.packed_bits_total, artifact["packed_bits_total"])
            self.assertEqual(cfg.packed_k_values_total, artifact["codebook_mask"].numel())
            self.assertIn(kind.replace("_", "-"), slugs[kind])
            self.assertIn(
                f"kbpv{artifact['packed_bits_total']}of"
                f"{artifact['codebook_mask'].numel()}",
                slugs[kind],
            )
            self.assertTrue(slugs[kind].endswith(f"-m{file_digest}"))
            self.assertEqual(cfg.tag, slugs[kind])

            semantic = variant_semantic_payload(cfg)
            for key in (
                "control_kind",
                "reference_selection_method",
                "reference_top_p_threshold",
                "reference_mask_sha256",
                "reference_semantic_sha256",
                "reference_nf2_count",
                "actual_nf2_count",
                "reference_packed_bits_total",
                "packed_bits_total",
                "packed_k_values_total",
            ):
                self.assertEqual(semantic[key], getattr(cfg, key), key)

            preflight = resolve_longbench_preflight(_args())
            self.assertEqual(preflight["method_slug"], slugs[kind])
            self.assertEqual(preflight["variant_semantic_hash"], hashes[kind])
            resolved = preflight["resolved_variant"]
            for key in (
                "selection_method",
                "control_kind",
                "reference_selection_method",
                "reference_top_p_threshold",
                "reference_mask_sha256",
                "reference_semantic_sha256",
                "reference_nf2_count",
                "actual_nf2_count",
                "reference_packed_bits_total",
                "packed_bits_total",
                "packed_k_values_total",
            ):
                self.assertEqual(resolved[key], getattr(cfg, key), key)

        self.assertEqual(len(set(slugs.values())), 2)
        self.assertEqual(len(set(hashes.values())), 2)
        self.assertEqual(
            configs["same_cardinality"].actual_nf2_count,
            configs["same_cardinality"].reference_nf2_count,
        )
        self.assertEqual(
            configs["exact_packed_bits"].packed_bits_total,
            configs["exact_packed_bits"].reference_packed_bits_total,
        )

    def test_control_metadata_does_not_change_canonical_or_top_p_identity(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload())
        canonical = build_variant(_args())
        self.assertEqual(method_layout_slug(canonical), "qlutattn")
        os.environ["QLUT_CB_MASK"] = self._write_mask(_top_p_mask_payload())
        top_p = build_variant(_args())
        self.assertTrue(method_layout_slug(top_p).startswith("qlutattn-topp-p0p55-m"))

        reference_only_keys = (
            "control_kind",
            "reference_selection_method",
            "reference_top_p_threshold",
            "reference_mask_sha256",
            "reference_semantic_sha256",
            "reference_nf2_count",
            "reference_packed_bits_total",
        )
        accounting_keys = (
            "actual_nf2_count",
            "packed_bits_total",
            "packed_k_values_total",
        )
        canonical_semantic = variant_semantic_payload(canonical)
        for key in reference_only_keys + accounting_keys:
            self.assertNotIn(key, canonical_semantic)
        top_p_semantic = variant_semantic_payload(top_p)
        for key in reference_only_keys:
            self.assertNotIn(key, top_p_semantic)
        for key in accounting_keys:
            self.assertEqual(top_p_semantic[key], getattr(top_p, key), key)

    def test_control_tampering_is_rejected(self):
        cases = {}

        identity = _control_mask_payload()
        identity["tie_rule"] = "unstable"
        cases["identity"] = identity

        provenance = _control_mask_payload()
        provenance["reference_mask_sha256"] = "A" * 64
        cases["provenance"] = provenance

        semantic_digest = _control_mask_payload()
        semantic_digest["reference_semantic_sha256"] = "b" * 64
        cases["valid lowercase semantic digest"] = semantic_digest

        semantic_provenance = _control_mask_payload()
        semantic_provenance["model"] = "tampered/model"
        cases["semantic provenance"] = semantic_provenance

        reference_mask = _control_mask_payload()
        reference_mask["reference_codebook_mask"] = (
            reference_mask["reference_codebook_mask"].clone()
        )
        reference_mask["reference_codebook_mask"][0, 0, 0] ^= 1
        cases["reference mask"] = reference_mask

        selected_mask = _control_mask_payload()
        selected_mask["codebook_mask"] = selected_mask["codebook_mask"].clone()
        selected_mask["codebook_mask"][0, 0, 0] ^= 1
        cases["selected mask"] = selected_mask

        count = _control_mask_payload()
        count["actual_nf2_count"] += 1
        cases["count"] = count

        cost = _control_mask_payload()
        cost["packed_bits_total"] += 1
        cases["cost"] = cost

        reorder = _control_mask_payload()
        reorder["reorder_index"] = torch.roll(
            reorder["reorder_index"], 1, dims=-1
        )
        cases["reorder"] = reorder

        inverse = _control_mask_payload()
        inverse["inverse_reorder_index"] = torch.roll(
            inverse["inverse_reorder_index"], 1, dims=-1
        )
        cases["inverse"] = inverse

        ranking = _control_mask_payload()
        ranking["ranking_score"] = ranking["ranking_score"].clone()
        ranking["ranking_score"][0, 0, 0] += 1.0
        cases["ranking"] = ranking

        overflow = _control_mask_payload()
        overflow["sigma2"] = torch.full(
            overflow["sigma2"].shape, 1e308, dtype=torch.float64
        )
        overflow["q_absmean"] = torch.ones_like(overflow["sigma2"])
        overflow["ranking_score"] = (
            overflow["sigma2"] * overflow["q_absmean"]
        )
        cases["ranking layer-mass overflow"] = overflow

        for label, payload in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    load_qlutattn_mask_blob(self._write_mask(payload))

    def test_tampering_fails_before_model_execution(self):
        artifact = _control_mask_payload()
        artifact["reference_packed_bits_total"] += 1
        os.environ["QLUT_CB_MASK"] = self._write_mask(artifact)
        args = _args(require_gpu1=False)
        with mock.patch(
            "kitty_sim.longbench.runner.AutoConfig.from_pretrained"
        ) as config_loader, mock.patch(
            "kitty_sim.longbench.runner.AutoModelForCausalLM.from_pretrained"
        ) as model_loader:
            with self.assertRaises(ValueError):
                run_longbench(args)
            config_loader.assert_not_called()
            model_loader.assert_not_called()

    def test_control_reordered_quantization_matches_direct_mask(self):
        for kind in ("same_cardinality", "exact_packed_bits"):
            with self.subTest(kind=kind):
                control = _control_mask_payload(
                    control_kind=kind,
                    n_layers=1,
                    n_kv=2,
                    head_dim=16,
                )
                direct = {
                    "codebook_mask": control["codebook_mask"].clone(),
                    "codebooks": ["sign", "nf2"],
                    "low_frac": control["low_frac"],
                }
                os.environ["QLUT_CB_MASK"] = self._write_mask(direct)
                direct_cache = _cache_factory(build_variant(_args()))
                os.environ["QLUT_CB_MASK"] = self._write_mask(control)
                control_cache = _cache_factory(build_variant(_args()))

                torch.manual_seed(23)
                keys = (
                    torch.randn(1, 2, 64, 16, dtype=torch.float32) + 1.25
                ).to(torch.float16)
                direct_out = direct_cache._quant_k_pertoken(keys, 0)
                with mock.patch.object(
                    control_cache,
                    "_pt_apply_reordered_cb_segments",
                    wraps=control_cache._pt_apply_reordered_cb_segments,
                ) as reordered:
                    control_out = control_cache._quant_k_pertoken(keys, 0)
                reordered.assert_called_once()
                self.assertEqual(
                    control_cache.qlut_selection_method,
                    QLUTATTN_CONTROL_SELECTION_METHOD,
                )
                self.assertIn(0, control_cache.k_reorder_index)
                self.assertIn(0, control_cache.k_inverse_reorder_index)
                self.assertIn(0, control_cache.k_nf2_count_per_head)
                torch.testing.assert_close(
                    control_out,
                    direct_out,
                    atol=torch.finfo(torch.float16).eps,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
