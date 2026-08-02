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
    VariantConfig,
    _cache_factory,
    _ThinkingAnswerBudgetStoppingCriteria,
    _longbench_generation_policy,
    _maybe_enable_quest_kernel,
    _thinking_final_answer,
    _thinking_dataset_generation_policy,
    _thinking_sample_seed,
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
    """Independent packed-K accounting oracle for top-p fixtures."""
    head_dim = mask.shape[-1]
    nf2_count = mask.to(torch.int64).sum(-1)
    code_bits = int(((head_dim - nf2_count) + 2 * nf2_count).sum())
    scale_bits = 16 * int((nf2_count > 0).sum())
    scale_bits += 16 * int((nf2_count < head_dim).sum())
    return code_bits + scale_bits


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

    def test_retired_control_selector_cannot_use_canonical_slug(self):
        config = VariantConfig(
            name=QLUTATTN_VARIANT,
            use_kitty=True,
            k_codebook="qlut",
            selection_method="layer_uniform_fixed_top_k_control",
        )
        with self.assertRaisesRegex(ValueError, "unsupported qlutattn selection_method"):
            _ = config.tag
        with self.assertRaisesRegex(ValueError, "unsupported qlutattn selection_method"):
            method_layout_slug(config)

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

    def test_uniform_control_artifact_is_rejected(self):
        payload = _mask_payload()
        payload["selection_method"] = "layer_uniform_fixed_top_k_control"
        with self.assertRaisesRegex(ValueError, "unsupported qlutattn selection_method"):
            load_qlutattn_mask_blob(self._write_mask(payload))

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


    def test_top_p_refuses_missing_k_engagement_evidence(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(
            _top_p_mask_payload(n_layers=1, n_kv=1, head_dim=64)
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




class TestQwenThinkingGeneration(_EnvIsolation):
    def test_policy_changes_run_hash_and_records_fixed_settings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({
                    "model_type": "qwen3",
                    "head_dim": 64,
                    "hidden_size": 64,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_hidden_layers": 1,
                    "intermediate_size": 128,
                    "vocab_size": 128,
                }),
                encoding="utf-8",
            )
            (model / "tokenizer_config.json").write_text(
                json.dumps({
                    "chat_template": (
                        "enable_thinking {% if enable_thinking %}"
                        "<think>{{ content }}</think>{% endif %}"
                    )
                }),
                encoding="utf-8",
            )
            (model / "generation_config.json").write_text(
                json.dumps({"eos_token_id": [2, 3]}),
                encoding="utf-8",
            )
            data_dir = root / "longbench" / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "narrativeqa.jsonl").write_text("{}\n", encoding="utf-8")
            os.environ["QLUT_CB_MASK"] = self._write_mask()
            args = _args(
                model="Qwen/Qwen3-8B",
                model_path=str(model),
                model_tag="qwen3-8b",
                model_family="qwen",
                dataset="narrativeqa",
                data_root=str(root / "longbench"),
                max_samples=1,
                max_gen=4096,
                qwen_thinking=False,
            )
            variant = build_variant(args)
            greedy_hash = longbench_run_config_hash(
                args, variant, "narrativeqa"
            )
            args.qwen_thinking = True
            thinking_hash = longbench_run_config_hash(
                args, variant, "narrativeqa"
            )
            payload = resolve_longbench_preflight(args)
            policy = payload["run_config"]["generation_policy"]

            self.assertNotEqual(greedy_hash, thinking_hash)
            self.assertEqual(payload["run_config_hash"], thinking_hash)
            self.assertEqual(policy["name"], "qwen3-thinking-sampling-v5")
            self.assertEqual(
                policy["answer_extraction"], "prompt-mode-final-answer-v3"
            )
            self.assertTrue(policy["do_sample"])
            self.assertEqual(policy["temperature"], 0.6)
            self.assertEqual(policy["top_p"], 0.95)
            self.assertEqual(policy["top_k"], 20)
            self.assertEqual(policy["seed_base"], 0)
            self.assertTrue(policy["chat_template_sha256"])
            self.assertEqual(
                policy["answer_stopping"],
                "canonical-answer-budget-by-prompt-mode-v2",
            )
            self.assertEqual(
                policy["invalid_answer_policy"],
                "empty-prediction-score-zero-v1",
            )
            self.assertEqual(policy["answer_token_budget"], 128)
            self.assertEqual(policy["max_new_tokens"], 4096)
            self.assertEqual(policy["prompt_mode"], "chat-thinking")
            self.assertEqual(policy["eos_token_ids"], [2, 3])
            self.assertTrue(policy["generation_config_sha256"])

    def test_policy_rejects_non_qwen_family(self):
        args = _args(qwen_thinking=True)
        with self.assertRaisesRegex(ValueError, "model_family='qwen'"):
            _longbench_generation_policy(args, "llama3")

    def test_seed_derivation_is_stable_and_paired(self):
        policy = {"seed_base": 0}
        first = _thinking_sample_seed(policy, "qasper", 7)
        self.assertEqual(first, _thinking_sample_seed(policy, "qasper", 7))
        self.assertNotEqual(first, _thinking_sample_seed(policy, "qasper", 8))
        self.assertNotEqual(first, _thinking_sample_seed(policy, "trec", 7))

    def test_answer_budget_stops_only_after_generated_close(self):
        criterion = _ThinkingAnswerBudgetStoppingCriteria(
            prompt_length=2,
            prompt_mode="chat-thinking",
            closing_tag_ids=(7, 8),
            answer_token_budget=2,
        )

        def stopped(tokens):
            return bool(
                criterion(torch.tensor([tokens], dtype=torch.long), None).item()
            )

        self.assertFalse(stopped([7, 8, 5, 5, 5]))
        self.assertFalse(stopped([7, 8, 5, 7, 8, 11]))
        self.assertTrue(stopped([7, 8, 5, 7, 8, 11, 12]))
        with self.assertRaisesRegex(ValueError, "must exceed"):
            _thinking_dataset_generation_policy(
                {
                    "answer_stopping": (
                        "canonical-answer-budget-by-prompt-mode-v2"
                    )
                },
                dataset="narrativeqa",
                answer_token_budget=128,
                max_new_tokens=128,
            )
        raw_policy = _thinking_dataset_generation_policy(
            {"answer_stopping": "canonical-answer-budget-by-prompt-mode-v2"},
            dataset="lcc",
            answer_token_budget=2,
            max_new_tokens=2,
        )
        self.assertEqual(raw_policy["prompt_mode"], "raw-no-chat")
        raw_criterion = _ThinkingAnswerBudgetStoppingCriteria(
            prompt_length=2,
            prompt_mode=raw_policy["prompt_mode"],
            closing_tag_ids=(),
            answer_token_budget=raw_policy["answer_token_budget"],
        )
        self.assertFalse(
            raw_criterion(torch.tensor([[1, 2, 9]], dtype=torch.long), None).item()
        )
        self.assertTrue(
            raw_criterion(
                torch.tensor([[1, 2, 9, 10]], dtype=torch.long), None
            ).item()
        )

    def test_final_answer_extraction(self):
        self.assertEqual(
            _thinking_final_answer(
                "<think>reason</think>\n</think>\n final answer ",
                "chat-thinking",
            ),
            ("final answer", "ok"),
        )
        self.assertEqual(
            _thinking_final_answer(" direct answer ", "raw-no-chat"),
            ("direct answer", "ok"),
        )
        self.assertEqual(
            _thinking_final_answer("reasoning without tags", "chat-thinking"),
            ("", "no_closing_think"),
        )
        self.assertEqual(
            _thinking_final_answer("<think>unfinished", "raw-no-chat"),
            ("", "no_closing_think"),
        )
        self.assertEqual(
            _thinking_final_answer(
                "<think>reason</think>", "chat-thinking"
            ),
            ("", "empty_final_answer"),
        )
        self.assertEqual(
            _thinking_final_answer(
                "<think>reason</think> answer <think>again",
                "chat-thinking",
            ),
            ("", "reopened_think_after_final_close"),
        )

    def test_generate_uses_sampling_and_writes_only_final_answer(self):
        class _Batch(dict):
            def __init__(self):
                super().__init__(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, _device):
                return self

        class _Tokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def __call__(self, *_args, **_kwargs):
                return _Batch()

            def apply_chat_template(self, messages, **_kwargs):
                return messages[0]["content"]

            def encode(self, *_args, **_kwargs):
                return [7, 8]

            def decode(self, *_args, **_kwargs):
                return "<think>reasoning</think>\n final answer "

        class _Model:
            device = torch.device("cpu")

            def __init__(self):
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.cat(
                    [kwargs["input_ids"], torch.ones(1, 1, dtype=torch.long)],
                    dim=1,
                )

        policy = {
            "name": "qwen3-thinking-sampling-v5",
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed_base": 0,
            "seed_derivation": "sha256-base-dataset-sample-v1",
            "answer_extraction": "prompt-mode-final-answer-v3",
            "answer_stopping": "canonical-answer-budget-by-prompt-mode-v2",
            "invalid_answer_policy": "empty-prediction-score-zero-v1",
            "eos_token_ids": [2, 3],
            "generation_config_sha256": "fixture",
            "prompt_mode": "chat-thinking",
            "answer_token_budget": 128,
            "max_new_tokens": 4096,
            "chat_template_sha256": "fixture",
        }
        model = _Model()
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "kitty_sim.longbench.runner.torch.cuda.is_available",
            return_value=False,
        ):
            out = Path(td) / "qasper.jsonl"
            manifest = generate_dataset(
                model_name="model",
                model=model,
                tokenizer=_Tokenizer(),
                dataset="qasper",
                records=[{
                    "input": "code",
                    "answers": ["answer"],
                    "all_classes": [],
                    "length": 1,
                }],
                prompt_format="{input}",
                max_model_len=32768,
                max_gen=4096,
                out_path=out,
                variant=build_variant(_args("fp16")),
                model_family="qwen",
                expected_run_config_hash="thinking-run-hash",
                generation_policy=policy,
            )
            row = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(row["pred"], "final answer")
        self.assertEqual(manifest["generation_policy"], policy)
        self.assertEqual(manifest["run_config_hash"], "thinking-run-hash")
        self.assertTrue(model.kwargs["do_sample"])
        self.assertEqual(model.kwargs["temperature"], 0.6)
        self.assertEqual(model.kwargs["top_p"], 0.95)
        self.assertEqual(model.kwargs["top_k"], 20)
        self.assertEqual(model.kwargs["eos_token_id"], [2, 3])
        criteria = model.kwargs["stopping_criteria"]
        self.assertEqual(len(criteria), 1)
        self.assertEqual(criteria[0].prompt_length, 2)
        self.assertEqual(criteria[0].closing_tag_ids, (7, 8))
        self.assertEqual(criteria[0].answer_token_budget, 128)
        self.assertEqual(criteria[0].prompt_mode, "chat-thinking")

    def test_no_chat_generation_stops_without_a_think_tag(self):
        class _Batch(dict):
            def __init__(self):
                super().__init__(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, _device):
                return self

        class _Tokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def __call__(self, *_args, **_kwargs):
                return _Batch()

            def encode(self, *_args, **_kwargs):
                raise AssertionError("raw-no-chat mode must not tokenize </think>")

            def decode(self, *_args, **_kwargs):
                return "direct answer"

        class _Model:
            device = torch.device("cpu")

            def __init__(self):
                self.generated_tokens = None
                self.criteria = None
                self.eos_token_id = None

            def generate(self, **kwargs):
                output = kwargs["input_ids"]
                self.criteria = kwargs["stopping_criteria"]
                self.eos_token_id = kwargs["eos_token_id"]
                for token_id in (9, 10, 11):
                    output = torch.cat(
                        [output, torch.tensor([[token_id]], dtype=torch.long)],
                        dim=1,
                    )
                    if bool(self.criteria(output, None).all()):
                        break
                self.generated_tokens = output.shape[1] - kwargs["input_ids"].shape[1]
                return output

        base_policy = {
            "name": "qwen3-thinking-sampling-v5",
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed_base": 0,
            "seed_derivation": "sha256-base-dataset-sample-v1",
            "answer_extraction": "prompt-mode-final-answer-v3",
            "answer_stopping": "canonical-answer-budget-by-prompt-mode-v2",
            "invalid_answer_policy": "empty-prediction-score-zero-v1",
            "eos_token_ids": [2, 3],
            "generation_config_sha256": "fixture",
            "chat_template_sha256": "fixture",
        }
        policy = _thinking_dataset_generation_policy(
            base_policy,
            dataset="lcc",
            answer_token_budget=2,
            max_new_tokens=4096,
        )
        model = _Model()
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "kitty_sim.longbench.runner.torch.cuda.is_available",
            return_value=False,
        ):
            out = Path(td) / "lcc.jsonl"
            manifest = generate_dataset(
                model_name="model",
                model=model,
                tokenizer=_Tokenizer(),
                dataset="lcc",
                records=[{
                    "input": "code",
                    "answers": ["answer"],
                    "all_classes": [],
                    "length": 1,
                }],
                prompt_format="{input}",
                max_model_len=32768,
                max_gen=4096,
                out_path=out,
                variant=build_variant(_args("fp16")),
                model_family="qwen",
                expected_run_config_hash="raw-thinking-run-hash",
                generation_policy=policy,
            )
            row = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(row["pred"], "direct answer")
        self.assertEqual(row["answer_extraction_status"], "ok")
        self.assertEqual(manifest["generation_outcomes"]["invalid_answer_count"], 0)
        self.assertEqual(model.generated_tokens, 2)
        self.assertEqual(policy["prompt_mode"], "raw-no-chat")
        self.assertEqual(manifest["generation_policy"], policy)
        self.assertEqual(model.criteria[0].closing_tag_ids, ())
        self.assertEqual(model.eos_token_id, [2, 3])

    def test_invalid_chat_answer_is_zero_score_row_with_manifest_evidence(self):
        class _Batch(dict):
            def __init__(self):
                super().__init__(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, _device):
                return self

        class _Tokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def __call__(self, *_args, **_kwargs):
                return _Batch()

            def apply_chat_template(self, messages, **_kwargs):
                return messages[0]["content"]

            def encode(self, *_args, **_kwargs):
                return [7, 8]

            def decode(self, *_args, **_kwargs):
                return "reasoning without a close"

        class _Model:
            device = torch.device("cpu")

            def generate(self, **kwargs):
                return torch.cat(
                    [
                        kwargs["input_ids"],
                        torch.tensor([[9, 10]], dtype=torch.long),
                    ],
                    dim=1,
                )

        base_policy = {
            "name": "qwen3-thinking-sampling-v5",
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed_base": 0,
            "seed_derivation": "sha256-base-dataset-sample-v1",
            "answer_extraction": "prompt-mode-final-answer-v3",
            "answer_stopping": "canonical-answer-budget-by-prompt-mode-v2",
            "invalid_answer_policy": "empty-prediction-score-zero-v1",
            "eos_token_ids": [2, 3],
            "generation_config_sha256": "fixture",
            "chat_template_sha256": "fixture",
        }
        policy = _thinking_dataset_generation_policy(
            base_policy,
            dataset="qasper",
            answer_token_budget=1,
            max_new_tokens=2,
        )
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "kitty_sim.longbench.runner.torch.cuda.is_available",
            return_value=False,
        ):
            out = Path(td) / "qasper.jsonl"
            manifest = generate_dataset(
                model_name="model",
                model=_Model(),
                tokenizer=_Tokenizer(),
                dataset="qasper",
                records=[{
                    "input": "question",
                    "answers": ["answer"],
                    "all_classes": [],
                    "length": 1,
                }],
                prompt_format="{input}",
                max_model_len=32768,
                max_gen=2,
                out_path=out,
                variant=build_variant(_args("fp16")),
                model_family="qwen",
                expected_run_config_hash="invalid-thinking-run-hash",
                generation_policy=policy,
            )
            row = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["written_samples"], 1)
        self.assertEqual(manifest["failed_sample_ids"], [])
        self.assertEqual(row["pred"], "")
        self.assertEqual(row["answer_extraction_status"], "no_closing_think")
        self.assertEqual(row["generation_termination"], "max_new_tokens")
        self.assertEqual(row["generated_tokens"], 2)
        self.assertEqual(manifest["generation_outcomes"], {
            "invalid_answer_count": 1,
            "answer_extraction_status_counts": {"no_closing_think": 1},
            "termination_counts": {"max_new_tokens": 1},
            "max_generated_tokens": 2,
        })


if __name__ == "__main__":
    unittest.main()
