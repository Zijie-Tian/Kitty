"""0-GPU wiring tests for the single canonical qlutattn variant.

Covers: variant resolution + slug, the fixed resolved config (K per-token
sign/nf2 f50 offline mask, symnf2-v1; V 2-bit tile16_rescued C=64), strict
mask validation, retired-knob rejection, QUEST rejection, GLM/FP16/head_dim
guards, mask-SHA/semantic-hash plumbing, shell/Python preflight parity,
manifest engagement, and the masked per-token K numerics.
"""

from __future__ import annotations

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
    def test_fresh_manifest_records_canonical_semantics_and_engagement(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(_mask_payload(1, 1, 64))
        cfg = build_variant(_args())

        class _Batch(dict):
            def __init__(self):
                super().__init__()
                self.input_ids = torch.zeros(1, 192, dtype=torch.long)
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
                    -1, 1, 192 * 64, dtype=torch.float16
                ).reshape(1, 1, 192, 64)
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
            self.assertEqual(manifest["run_config_hash"], "run-hash")
            self.assertEqual(manifest["variant_semantic_hash"], variant_semantic_hash(cfg))
            self.assertTrue(manifest["mask_sha256"])
            self.assertGreater(manifest["engagement"]["v_quant_calls"], 0)
            self.assertGreater(manifest["engagement"]["v_quantized_tokens"], 0)
            self.assertGreater(manifest["engagement"]["v_tile_blocks"], 0)
            self.assertEqual(manifest["engagement"]["last_v_quant_mode"], "tile16_rescued")
            self.assertEqual(manifest["engagement"]["last_v_tile_channels"], 64)


if __name__ == "__main__":
    unittest.main()
