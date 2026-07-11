"""0-GPU wiring tests: variants, slugs, GLM fail-fast, env conflicts."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from kitty_sim.cli.eval_longbench import build_parser, finalize_args
from kitty_sim.longbench.runner import (
    NEW_V2_VARIANTS,
    _cache_factory,
    build_variant,
    generate_dataset,
    longbench_run_config_hash,
    method_layout_slug,
    resolve_longbench_preflight,
    run_longbench,
    validate_new_v2_model_config,
    validate_new_v2_model_family,
    validate_new_v2_preload,
    variant_semantic_hash,
)
from kitty_sim.v_tile_quant import V_TILE_ALGO_VERSION


def _args(variant: str, **kwargs):
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
        v_tile_channels=None,
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


class TestV2Wiring(unittest.TestCase):
    def setUp(self):
        self._env_backup = {
            k: os.environ.get(k)
            for k in (
                "QLUT_BIN_CODEBOOKS",
                "QLUT_CB_MASK",
                "V_TILE_CHANNELS",
                "VBITS",
                "PERTOKEN_BLOCK",
            )
        }
        for k in list(self._env_backup):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _write_mask(self) -> str:
        path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        self.addCleanup(Path(path).unlink, missing_ok=True)
        blob = {
            "codebook_mask": torch.zeros(2, 2, 64, dtype=torch.uint8),
            "codebooks": ["sign", "nf2"],
            "nominal_bits": 1.875,
        }
        torch.save(blob, path)
        return path

    def test_k125v2_inherits_sign_k(self):
        parent = build_variant(_args("qlutattn_k125v4_pt"))
        child = build_variant(_args("qlutattn_k125v2_pt"))
        self.assertEqual(child.k_quant_mode, parent.k_quant_mode)
        self.assertEqual(child.pertoken_pc_submean, parent.pertoken_pc_submean)
        self.assertEqual(child.bin_codebooks, ("sign",))
        self.assertEqual(child.vbits, 2)
        self.assertEqual(child.v_codebook, "per_token2")
        self.assertIsNone(child.v_tile_channels)

    def test_k188v2_inherits_snf_k(self):
        mask = self._write_mask()
        os.environ["QLUT_CB_MASK"] = mask
        parent = build_variant(_args("qlutattn_k188v4_pt"))
        child = build_variant(_args("qlutattn_k188v2_pt"))
        self.assertEqual(child.pertoken_cb_mask, parent.pertoken_cb_mask)
        self.assertEqual(child.bin_codebooks, ("sign", "nf2"))
        self.assertEqual(child.vbits, 2)
        self.assertEqual(child.v_codebook, "per_token2")

    def test_snf_variants_require_mask(self):
        with self.assertRaises(FileNotFoundError):
            build_variant(_args("qlutattn_k188v2_pt"))
        with self.assertRaises(FileNotFoundError):
            build_variant(_args(
                "qlutattn_k188v2_pt_vtile16", v_tile_channels=16
            ))

    def test_tile_slug_unique_by_c(self):
        v16 = build_variant(_args("qlutattn_k125v2_pt_vtile16", v_tile_channels=16))
        v32 = build_variant(_args("qlutattn_k125v2_pt_vtile16", v_tile_channels=32))
        s16 = method_layout_slug(v16)
        s32 = method_layout_slug(v32)
        self.assertEqual(s16, "qlutattn-k125v2-pt-vtile16c16-rv1")
        self.assertEqual(s32, "qlutattn-k125v2-pt-vtile16c32-rv1")
        self.assertNotEqual(s16, s32)
        self.assertNotEqual(v16.tag, v32.tag)
        with self.assertRaises(ValueError):
            method_layout_slug("qlutattn_k125v2_pt_vtile16")

    def test_cache_factory_threads_v2_fields(self):
        pt = build_variant(_args("qlutattn_k125v2_pt"))
        pt_cache = _cache_factory(pt)
        self.assertEqual(pt_cache.v_codebook, "per_token2")
        self.assertEqual(pt_cache.vbits, 2)
        self.assertIsNone(pt_cache.v_tile_channels)

        tile = build_variant(_args(
            "qlutattn_k125v2_pt_vtile16", v_tile_channels=32
        ))
        tile_cache = _cache_factory(tile)
        self.assertEqual(tile_cache.v_codebook, "tile16_rescued")
        self.assertEqual(tile_cache.vbits, 2)
        self.assertEqual(tile_cache.v_tile_tokens, 16)
        self.assertEqual(tile_cache.v_tile_channels, 32)
        self.assertEqual(tile_cache.v_tile_algo_version, V_TILE_ALGO_VERSION)

    def test_hyphen_aliases(self):
        a = build_variant(_args("qlutattn-k125v2-pt"))
        b = build_variant(_args("qlutattn_k125v2_pt"))
        self.assertEqual(a.name, b.name)
        self.assertEqual(method_layout_slug(a), method_layout_slug(b))

    def test_sign_rejects_tern_codebook(self):
        os.environ["QLUT_BIN_CODEBOOKS"] = "tern"
        with self.assertRaises(ValueError):
            build_variant(_args("qlutattn_k125v2_pt"))

    def test_sign_rejects_pertoken_block(self):
        os.environ["PERTOKEN_BLOCK"] = "16"
        with self.assertRaises(ValueError):
            build_variant(_args("qlutattn_k125v2_pt"))

    def test_missing_c_on_tile(self):
        with self.assertRaises(ValueError):
            build_variant(_args("qlutattn_k125v2_pt_vtile16"))

    def test_stale_c_on_nontile(self):
        os.environ["V_TILE_CHANNELS"] = "16"
        with self.assertRaises(ValueError):
            build_variant(_args("qlutattn_k125v2_pt"))

    def test_empty_c_sentinel_on_nontile(self):
        os.environ["V_TILE_CHANNELS"] = ""
        cfg = build_variant(_args("qlutattn_k125v2_pt"))
        self.assertIsNone(cfg.v_tile_channels)

    def test_vbits_cli_value_wins_over_environment(self):
        os.environ["VBITS"] = "4"
        cfg = build_variant(_args("qlutattn_k125v2_pt", vbits=2))
        self.assertEqual(cfg.vbits, 2)
        with self.assertRaises(ValueError):
            build_variant(_args("qlutattn_k125v2_pt", vbits=4))

        from_env = finalize_args(build_parser().parse_args([
            "model", "--variant", "qlutattn_k125v2_pt",
        ]))
        with self.assertRaises(ValueError):
            build_variant(from_env)
        explicit_cli = finalize_args(build_parser().parse_args([
            "model", "--variant", "qlutattn_k125v2_pt", "--vbits", "2",
        ]))
        self.assertEqual(build_variant(explicit_cli).vbits, 2)

    def test_glm_failfast_message(self):
        cfg = build_variant(_args("qlutattn_k125v2_pt_vtile16", v_tile_channels=16))
        self.assertIn(cfg.name, NEW_V2_VARIANTS)
        with self.assertRaises(ValueError) as ctx:
            validate_new_v2_model_family(cfg, "glm4")
        self.assertIn("GLM parity", str(ctx.exception))
        validate_new_v2_model_family(cfg, "llama3")

    def test_model_dependent_validation(self):
        cfg = build_variant(_args("qlutattn_k125v2_pt_vtile16", v_tile_channels=16))
        validate_new_v2_model_config(cfg, SimpleNamespace(head_dim=64), torch.float16)
        with self.assertRaisesRegex(ValueError, "divisible"):
            bad_c = build_variant(_args("qlutattn_k125v2_pt_vtile16", v_tile_channels=24))
            validate_new_v2_model_config(bad_c, SimpleNamespace(head_dim=64), torch.float16)
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            validate_new_v2_model_config(cfg, SimpleNamespace(head_dim=48), torch.float16)
        with self.assertRaisesRegex(ValueError, "FP16"):
            validate_new_v2_model_config(cfg, SimpleNamespace(head_dim=64), torch.float32)

    def test_stale_c_rejected_for_all_non_tile_variants(self):
        with self.assertRaises(ValueError):
            build_variant(_args("fp16", v_tile_channels=16))

    def test_old_unsupported_block_rejected(self):
        os.environ["PERTOKEN_BLOCK"] = "16"
        with self.assertRaises(ValueError):
            build_variant(_args("qlutattn_k125v4_pt"))

    def test_preflight_slug(self):
        payload = resolve_longbench_preflight(
            _args("qlutattn_k125v2_pt_vtile16", v_tile_channels=32)
        )
        self.assertEqual(payload["method_slug"], "qlutattn-k125v2-pt-vtile16c32-rv1")
        self.assertEqual(payload["canonical_variant"], "qlutattn_k125v2_pt_vtile16")
        self.assertEqual(
            payload["resolved_variant"]["v_tile_algo_version"], V_TILE_ALGO_VERSION
        )
        h1 = variant_semantic_hash(build_variant(_args("qlutattn_k125v2_pt_vtile16", v_tile_channels=16)))
        h2 = variant_semantic_hash(build_variant(_args("qlutattn_k125v2_pt_vtile16", v_tile_channels=32)))
        self.assertNotEqual(h1, h2)

    def test_tile_block_quest_suffix_order(self):
        mask = self._write_mask()
        os.environ["QLUT_CB_MASK"] = mask
        os.environ["PERTOKEN_BLOCK"] = "16"
        payload = resolve_longbench_preflight(_args(
            "qlutattn_k188v2_pt_vtile16",
            v_tile_channels=32,
            quest_kernel=True,
            quest_token_budget=2048,
            quest_skip_layers=0,
        ))
        self.assertEqual(
            payload["method_slug"],
            "qlutattn-k188v2-pt-vtile16c32-rv1-blk16-quest-kernel",
        )

    def test_shell_and_python_slug_parity(self):
        expected = resolve_longbench_preflight(
            _args("qlutattn_k125v2_pt_vtile16", v_tile_channels=32)
        )["method_slug"]
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update({
            "KITTY_ENV_FILE": str(repo / "does-not-exist.env"),
            "PYTHON_BIN": sys.executable,
            "PYTHONPATH": str(repo / "src"),
            "VBITS": "2",
            "PERTOKEN_BLOCK": "1",
            "QLUT_BIN_CODEBOOKS": "sign",
            # Explicit CLI must beat a conflicting environment value.
            "V_TILE_CHANNELS": "16",
        })
        got = subprocess.run(
            [
                "bash", "scripts/run_exp.sh", "--variant", "qlutattn_k125v2_pt_vtile16",
                "--v-tile-channels", "32", "--print-method-slug",
            ],
            cwd=repo, env=env, text=True, capture_output=True, check=True,
        ).stdout.strip()
        self.assertEqual(got, expected)

    def test_shell_and_python_reject_unsupported_block_consistently(self):
        os.environ["PERTOKEN_BLOCK"] = "16"
        with self.assertRaises(ValueError):
            build_variant(_args("qlutattn_k125v4_pt"))
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update({
            "KITTY_ENV_FILE": str(repo / "does-not-exist.env"),
            "PYTHON_BIN": sys.executable,
            "PYTHONPATH": str(repo / "src"),
            "V_TILE_CHANNELS": "",
        })
        got = subprocess.run(
            [
                "bash", "scripts/run_exp.sh", "--variant", "qlutattn_k125v4_pt",
                "--print-method-slug",
            ],
            cwd=repo, env=env, text=True, capture_output=True,
        )
        self.assertNotEqual(got.returncode, 0)
        self.assertIn("does not support PERTOKEN_BLOCK>1", got.stderr)

    def test_empty_c_environment_blocks_stale_dotenv_value(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / ".env"
            env_file.write_text("V_TILE_CHANNELS=16\n", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "KITTY_ENV_FILE": str(env_file),
                "PYTHON_BIN": sys.executable,
                "PYTHONPATH": str(repo / "src"),
                "V_TILE_CHANNELS": "",
                "PERTOKEN_BLOCK": "1",
            })
            got = subprocess.run(
                [
                    "bash", "scripts/run_exp.sh", "--variant", "fp16",
                    "--print-method-slug",
                ],
                cwd=repo, env=env, text=True, capture_output=True, check=True,
            )
            self.assertEqual(got.stdout.strip(), "fp16")

    def test_config_only_preload_validates_without_cuda(self):
        cfg = build_variant(
            _args("qlutattn_k125v2_pt_vtile16", v_tile_channels=16)
        )
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({
                    "model_type": "llama",
                    "hidden_size": 96,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                    "intermediate_size": 128,
                    "vocab_size": 128,
                }),
                encoding="utf-8",
            )
            args = _args(
                "qlutattn_k125v2_pt_vtile16",
                v_tile_channels=16,
                model_path=str(model),
                local_files_only=True,
            )
            cuda_before = torch.cuda.is_initialized()
            with self.assertRaisesRegex(ValueError, "power-of-two"):
                validate_new_v2_preload(args, cfg, "llama3")
            self.assertEqual(torch.cuda.is_initialized(), cuda_before)

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
            torch.save({"codebook_mask": torch.zeros(1, 1, 64, dtype=torch.uint8),
                        "codebooks": ["sign", "nf2"]}, mask)
            args = _args(
                "qlutattn_k188v2_pt", model_path=str(model), dataset="narrativeqa",
                data_root=str(root / "longbench"),
            )
            cfg1 = build_variant(args)
            h1 = longbench_run_config_hash(args, cfg1, "narrativeqa")
            torch.save({"codebook_mask": torch.ones(1, 1, 64, dtype=torch.uint8),
                        "codebooks": ["sign", "nf2"]}, mask)
            cfg2 = build_variant(args)
            h2 = longbench_run_config_hash(args, cfg2, "narrativeqa")
            self.assertNotEqual(h1, h2)

    def test_run_hash_distinguishes_local_model_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "longbench" / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "narrativeqa.jsonl").write_text("{}\n", encoding="utf-8")
            model_paths = []
            for name in ("model-a", "model-b"):
                model = root / name
                model.mkdir()
                (model / "config.json").write_text(
                    '{"model_type":"llama"}', encoding="utf-8"
                )
                model_paths.append(model)
            hashes = []
            for model in model_paths:
                args = _args(
                    "qlutattn_k125v2_pt",
                    model_path=str(model),
                    dataset="narrativeqa",
                    data_root=str(root / "longbench"),
                )
                cfg = build_variant(args)
                hashes.append(longbench_run_config_hash(args, cfg, "narrativeqa"))
            self.assertNotEqual(hashes[0], hashes[1])

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
                    "num_hidden_layers": 1,
                    "intermediate_size": 128,
                    "vocab_size": 128,
                }),
                encoding="utf-8",
            )
            data_dir = root / "longbench" / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "narrativeqa.jsonl").write_text("{}\n", encoding="utf-8")
            args = _args(
                "qlutattn_k125v2_pt",
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
        cfg = build_variant(_args("qlutattn_k125v2_pt"))
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

    def test_fresh_manifest_records_v2_semantics_and_engagement(self):
        cfg = build_variant(_args("qlutattn_k125v2_pt"))

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
            self.assertEqual(manifest["variant"]["v_codebook"], "per_token2")
            self.assertEqual(manifest["variant"]["vbits"], 2)
            self.assertEqual(manifest["run_config_hash"], "run-hash")
            self.assertEqual(manifest["variant_semantic_hash"], variant_semantic_hash(cfg))
            self.assertIsNone(manifest["mask_sha256"])
            self.assertGreater(manifest["engagement"]["v_quant_calls"], 0)
            self.assertGreater(manifest["engagement"]["v_quantized_tokens"], 0)
            self.assertEqual(
                manifest["engagement"]["last_v_quant_mode"], "per_token2"
            )

    def test_parent_v4_slugs_unchanged(self):
        self.assertEqual(
            method_layout_slug(build_variant(_args("qlutattn_k125v4_pt"))),
            "qlutattn-k125v4-pt",
        )
        mask = self._write_mask()
        os.environ["QLUT_CB_MASK"] = mask
        self.assertEqual(
            method_layout_slug(build_variant(_args("qlutattn_k188v4_pt"))),
            "qlutattn-k188v4-pt",
        )


if __name__ == "__main__":
    unittest.main()
