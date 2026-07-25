"""Focused CPU tests for offline QLUTATTN layer-channel top-p artifacts."""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from kitty_sim.longbench import runner as longbench_runner


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibrate_qlutattn_mask.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_qlutattn_mask", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
calibration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibration)


class TestCanonicalMaskCompatibility(unittest.TestCase):
    def test_fixed_65_35_mask_and_payload_are_unchanged(self):
        sigma2 = torch.tensor([[[8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]]])
        q_absmean = torch.ones_like(sigma2)
        expected = torch.tensor([[[1, 0, 1, 0, 1, 0, 0, 0]]], dtype=torch.uint8)

        self.assertTrue(torch.equal(calibration.build_mask(sigma2, q_absmean), expected))
        payload = calibration.build_canonical_artifact(
            sigma2, q_absmean, group_size=128, skip_first=32, model="synthetic")
        self.assertTrue(torch.equal(payload["codebook_mask"], expected))
        self.assertEqual(
            list(payload),
            [
                "codebook_mask", "codebooks", "low_frac", "nominal_bits",
                "nf2_impl", "ranking_signal", "group_size", "skip_first",
                "model", "n_layers", "n_kv", "head_dim", "sigma2",
                "q_absmean",
            ],
        )
        self.assertEqual(payload["low_frac"], 5 / 8)
        self.assertEqual(payload["nominal_bits"], calibration.NOMINAL_K_BITS)
        self.assertNotIn("selection_method", payload)
        self.assertNotIn("reorder_index", payload)
        self.assertNotIn("ranking_score", payload)


class TestLayerChannelTopPSelection(unittest.TestCase):
    def test_uniform_scores_use_ascending_flat_index_for_ties(self):
        selected = calibration.select_layer_channel_top_p(
            torch.ones(1, 2, 2, dtype=torch.float64), 0.5)
        expected = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.uint8)
        self.assertTrue(torch.equal(selected["codebook_mask"], expected))
        self.assertEqual(selected["nf2_count_per_layer"].dtype, torch.int32)
        self.assertTrue(torch.equal(
            selected["nf2_count_per_layer"], torch.tensor([2], dtype=torch.int32)))
        self.assertTrue(torch.equal(
            selected["captured_mass_per_layer"], torch.tensor([0.5], dtype=torch.float64)))
        self.assertTrue(torch.equal(
            selected["previous_prefix_mass_per_layer"],
            torch.tensor([0.25], dtype=torch.float64)))
        self.assertTrue(torch.equal(
            selected["boundary_score_per_layer"], torch.tensor([1.0], dtype=torch.float64)))

    def test_dominant_score_needs_only_one_channel(self):
        scores = torch.tensor([[[90.0, 5.0, 3.0, 2.0]]], dtype=torch.float64)
        selected = calibration.select_layer_channel_top_p(scores, 0.9)
        self.assertTrue(torch.equal(
            selected["codebook_mask"],
            torch.tensor([[[1, 0, 0, 0]]], dtype=torch.uint8)))
        self.assertEqual(int(selected["nf2_count_per_layer"][0]), 1)
        self.assertEqual(float(selected["captured_mass_per_layer"][0]), 0.9)
        self.assertEqual(float(selected["previous_prefix_mass_per_layer"][0]), 0.0)
        self.assertEqual(float(selected["boundary_score_per_layer"][0]), 90.0)

    def test_boundary_ties_are_resolved_by_ascending_flat_index(self):
        scores = torch.tensor([[[4.0, 4.0], [4.0, 4.0]]], dtype=torch.float64)
        selected = calibration.select_layer_channel_top_p(scores, 0.51)
        self.assertTrue(torch.equal(
            selected["codebook_mask"],
            torch.tensor([[[1, 1], [1, 0]]], dtype=torch.uint8)))
        self.assertEqual(float(selected["previous_prefix_mass_per_layer"][0]), 0.5)
        self.assertEqual(float(selected["captured_mass_per_layer"][0]), 0.75)

    def test_layers_receive_independent_prefixes(self):
        scores = torch.tensor(
            [
                [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
                [[1.0, 1.0, 1.0], [9.0, 1.0, 1.0]],
            ],
            dtype=torch.float64,
        )
        selected = calibration.select_layer_channel_top_p(scores, 0.5)
        expected = torch.tensor(
            [
                [[1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0]],
            ],
            dtype=torch.uint8,
        )
        self.assertTrue(torch.equal(selected["codebook_mask"], expected))
        self.assertTrue(torch.equal(
            selected["nf2_count_per_layer"], torch.tensor([2, 1], dtype=torch.int32)))
        self.assertGreaterEqual(float(selected["captured_mass_per_layer"].min()), 0.5)
        self.assertLess(float(selected["previous_prefix_mass_per_layer"].max()), 0.5)

    def test_invalid_scores_and_thresholds_hard_fail(self):
        invalid_scores = {
            "nan": torch.tensor([[[1.0, float("nan")]]]),
            "infinity": torch.tensor([[[1.0, float("inf")]]]),
            "negative": torch.tensor([[[1.0, -0.1]]]),
            "zero_mass": torch.zeros(1, 1, 2),
        }
        for name, scores in invalid_scores.items():
            with self.subTest(score=name), self.assertRaises(ValueError):
                calibration.select_layer_channel_top_p(scores, 0.5)

        valid_scores = torch.ones(1, 1, 2)
        for threshold in (0.0, -0.1, 1.01, float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                calibration.select_layer_channel_top_p(valid_scores, threshold)

        for name, sigma2, q_absmean in (
            ("negative_sigma2", torch.tensor([[[-1.0, 2.0]]]), torch.ones(1, 1, 2)),
            ("negative_q", torch.ones(1, 1, 2), torch.tensor([[[1.0, -1.0]]])),
        ):
            with self.subTest(statistic=name), self.assertRaises(ValueError):
                calibration.ranking_score_from_statistics(sigma2, q_absmean)


class TestHeadLocalReorder(unittest.TestCase):
    def setUp(self):
        self.mask = torch.tensor(
            [[[0, 1, 0, 1, 1], [1, 0, 1, 0, 0]]], dtype=torch.uint8)
        self.reorder, self.inverse, self.counts = calibration.build_head_local_reorder(
            self.mask)

    def test_permutation_and_inverse_are_exact(self):
        self.assertEqual(self.reorder.dtype, torch.int64)
        self.assertEqual(self.inverse.dtype, torch.int64)
        self.assertEqual(self.counts.dtype, torch.int32)
        self.assertTrue(torch.equal(
            self.reorder,
            torch.tensor([[[1, 3, 4, 0, 2], [0, 2, 1, 3, 4]]], dtype=torch.int64)))
        self.assertTrue(torch.equal(
            self.counts, torch.tensor([[3, 2]], dtype=torch.int32)))

        original = torch.arange(10).reshape(1, 2, 5)
        reordered = torch.gather(original, -1, self.reorder)
        restored = torch.gather(reordered, -1, self.inverse)
        self.assertTrue(torch.equal(restored, original))

    def test_nf2_prefix_matches_original_codebook_assignment(self):
        reordered_mask = torch.gather(self.mask, -1, self.reorder)
        for head_idx in range(self.mask.shape[1]):
            count = int(self.counts[0, head_idx])
            self.assertTrue(bool((reordered_mask[0, head_idx, :count] == 1).all()))
            self.assertTrue(bool((reordered_mask[0, head_idx, count:] == 0).all()))

        original = torch.arange(10, dtype=torch.float64).reshape(1, 2, 5)
        direct = torch.where(self.mask.bool(), original + 1000.0, original - 1000.0)
        reordered = torch.gather(original, -1, self.reorder)
        prefix_quantized = torch.empty_like(reordered)
        for head_idx in range(self.mask.shape[1]):
            count = int(self.counts[0, head_idx])
            prefix_quantized[0, head_idx, :count] = reordered[0, head_idx, :count] + 1000.0
            prefix_quantized[0, head_idx, count:] = reordered[0, head_idx, count:] - 1000.0
        restored = torch.gather(prefix_quantized, -1, self.inverse)
        self.assertTrue(torch.equal(restored, direct))


class TestTopPOutputIdentity(unittest.TestCase):
    def test_full_digest_distinguishes_matching_eight_hex_prefixes(self):
        config = SimpleNamespace(
            pertoken_cb_mask="/unused/mask.pt",
            top_p_threshold=0.55,
        )
        prefix = "89abcdef"
        first_digest = prefix + "0" * 56
        second_digest = prefix + "f" * 56
        self.assertEqual(first_digest[:8], second_digest[:8])
        self.assertNotEqual(first_digest, second_digest)

        with mock.patch.object(
            longbench_runner,
            "_sha256_file",
            side_effect=(first_digest, second_digest),
        ):
            first_slug = longbench_runner._qlutattn_top_p_slug(config)
            second_slug = longbench_runner._qlutattn_top_p_slug(config)

        self.assertEqual(
            first_slug,
            f"qlutattn-topp-p0p55-m{first_digest}",
        )
        self.assertEqual(
            second_slug,
            f"qlutattn-topp-p0p55-m{second_digest}",
        )
        self.assertNotEqual(first_slug, second_slug)


class TestArtifactValidationAndStatsReuse(unittest.TestCase):
    @staticmethod
    def _artifact():
        sigma2 = torch.tensor(
            [[[1.0, 8.0, 2.0, 7.0], [6.0, 3.0, 5.0, 4.0]]],
            dtype=torch.float32,
        )
        return calibration.build_top_p_artifact(
            sigma2, torch.ones_like(sigma2), 0.6,
            group_size=128, skip_first=32, model="synthetic-model",
            calib_data="synthetic.parquet", num_samples=4, sample_len=64,
            seed=7)

    def test_artifact_fields_and_dtypes(self):
        payload = self._artifact()
        self.assertIs(calibration.validate_top_p_artifact(payload), payload)
        self.assertEqual(payload["selection_method"], "layer_channel_top_p")
        self.assertEqual(payload["format_version"], 3)
        self.assertEqual(
            payload["selection_axis"], "per_layer_flattened_kv_head_channel")
        self.assertEqual(
            payload["threshold_rule"], "minimal_desc_prefix_cumsum_ge_p")
        self.assertEqual(payload["tie_rule"], "score_desc_flat_index_asc")
        self.assertEqual(payload["score_dtype_for_selection"], "float64")
        self.assertEqual(payload["calib_data"], "synthetic.parquet")
        self.assertEqual(payload["num_samples"], 4)
        self.assertEqual(payload["sample_len"], 64)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["codebooks"], ["sign", "nf2"])
        self.assertEqual(payload["ranking_score"].dtype, torch.float64)
        self.assertEqual(payload["codebook_mask"].dtype, torch.uint8)
        self.assertEqual(payload["reorder_index"].dtype, torch.int64)
        self.assertEqual(payload["inverse_reorder_index"].dtype, torch.int64)
        self.assertEqual(payload["nf2_count_per_head"].dtype, torch.int32)
        self.assertEqual(payload["nf2_count_per_layer"].dtype, torch.int32)
        self.assertEqual(payload["nf2_ratio_per_layer"].dtype, torch.float64)
        self.assertEqual(
            payload["low_frac"] + payload["actual_nf2_frac"], 1.0)
        self.assertNotIn("nominal_bits", payload)
        self.assertIs(type(payload["actual_nf2_count"]), int)
        self.assertIs(type(payload["packed_bits_total"]), int)
        self.assertIs(type(payload["packed_k_values_total"]), int)
        self.assertEqual(
            payload["actual_nf2_count"], int(payload["codebook_mask"].sum()))
        self.assertEqual(
            payload["packed_k_values_total"], payload["codebook_mask"].numel())
        head_dim = payload["codebook_mask"].shape[-1]
        nf2_per_head = payload["codebook_mask"].to(torch.int64).sum(dim=-1)
        expected_packed_bits = int(
            ((head_dim - nf2_per_head) + 2 * nf2_per_head).sum())
        expected_packed_bits += 16 * int((nf2_per_head > 0).sum())
        expected_packed_bits += 16 * int((nf2_per_head < head_dim).sum())
        self.assertEqual(payload["packed_bits_total"], expected_packed_bits)

    def test_metadata_tampering_is_rejected(self):
        mutators = {
            "mask": lambda p: p["codebook_mask"].view(-1).__setitem__(0, 1 - p["codebook_mask"].view(-1)[0]),
            "ranking_score": lambda p: p["ranking_score"].view(-1).__setitem__(0, p["ranking_score"].view(-1)[0] + 0.25),
            "reorder": lambda p: p["reorder_index"].view(-1).__setitem__(0, p["reorder_index"].view(-1)[1]),
            "inverse": lambda p: p["inverse_reorder_index"].view(-1).__setitem__(0, p["inverse_reorder_index"].view(-1)[1]),
            "head_count": lambda p: p["nf2_count_per_head"].view(-1).__setitem__(0, p["nf2_count_per_head"].view(-1)[0] + 1),
            "layer_count": lambda p: p["nf2_count_per_layer"].view(-1).__setitem__(0, p["nf2_count_per_layer"].view(-1)[0] + 1),
            "layer_ratio": lambda p: p["nf2_ratio_per_layer"].add_(0.01),
            "captured_mass": lambda p: p["captured_mass_per_layer"].add_(1e-12),
            "previous_mass": lambda p: p["previous_prefix_mass_per_layer"].add_(1e-12),
            "boundary_score": lambda p: p["boundary_score_per_layer"].add_(1e-12),
            "actual_fraction": lambda p: p.__setitem__("actual_nf2_frac", p["actual_nf2_frac"] + 0.01),
            "low_fraction": lambda p: p.__setitem__("low_frac", p["low_frac"] - 0.01),
            "selection_method": lambda p: p.__setitem__("selection_method", "other"),
            "format_version": lambda p: p.__setitem__("format_version", 4),
            "selection_axis": lambda p: p.__setitem__("selection_axis", "per_head"),
            "threshold_rule": lambda p: p.__setitem__("threshold_rule", "other"),
            "tie_rule": lambda p: p.__setitem__("tie_rule", "other"),
            "score_dtype": lambda p: p.__setitem__("score_dtype_for_selection", "float32"),
            "missing_provenance": lambda p: p.pop("calib_data"),
            "format_version_float": lambda p: p.__setitem__("format_version", 3.0),
            "actual_count": lambda p: p.__setitem__("actual_nf2_count", p["actual_nf2_count"] + 1),
            "packed_bits": lambda p: p.__setitem__("packed_bits_total", p["packed_bits_total"] + 1),
            "packed_values": lambda p: p.__setitem__("packed_k_values_total", p["packed_k_values_total"] + 1),
        }
        original = self._artifact()
        for name, mutate in mutators.items():
            with self.subTest(field=name):
                tampered = copy.deepcopy(original)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    calibration.validate_top_p_artifact(tampered)


    def test_uniform_control_cli_is_removed(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--stats-input",
                "unused.pt",
                "--uniform-top-k-control-from",
                "unused-control.pt",
                "--output",
                "unused-output.pt",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unrecognized arguments", proc.stderr)

    def test_saved_statistics_cli_builds_top_p_without_model_arguments(self):
        sigma2 = torch.tensor(
            [
                [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
                [[1.0, 1.0, 1.0], [9.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        stats = calibration.build_statistics_artifact(
            sigma2, torch.ones_like(sigma2), model="synthetic-model",
            group_size=128, skip_first=32, calib_data="synthetic.parquet",
            num_samples=2, sample_len=16, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "stats.pt"
            output_path = Path(tmp) / "top_p.pt"
            torch.save(stats, stats_path)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            proc = subprocess.run(
                [
                    sys.executable, str(SCRIPT_PATH), "--stats-input", str(stats_path),
                    "--top-p", "0.5", "--output", str(output_path),
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["top_p_threshold"], 0.5)
            self.assertEqual(payload["model"], "synthetic-model")
            self.assertEqual(payload["calib_data"], "synthetic.parquet")
            self.assertEqual(payload["num_samples"], 2)
            self.assertEqual(payload["sample_len"], 16)
            self.assertEqual(payload["seed"], 7)
            self.assertEqual(payload["format_version"], 3)
            self.assertEqual(
                payload["actual_nf2_count"], int(payload["codebook_mask"].sum()))
            self.assertEqual(
                payload["packed_k_values_total"], payload["codebook_mask"].numel())
            calibration.validate_top_p_artifact(payload)


if __name__ == "__main__":
    unittest.main()
