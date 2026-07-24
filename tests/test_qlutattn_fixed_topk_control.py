"""Focused CPU tests for QLUTATTN fixed-top-k control artifacts."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibrate_qlutattn_mask.py"
_SPEC = importlib.util.spec_from_file_location(
    "calibrate_qlutattn_mask_control", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
calibration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibration)

CONTROL_FIELDS = {
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


class ControlFixture(unittest.TestCase):
    @staticmethod
    def statistics():
        head_dim = 32
        alternating_heads = [
            index
            for channel in range(head_dim)
            for index in (channel, head_dim + channel)
        ]
        sigma2 = torch.empty(2, 2, head_dim, dtype=torch.float64)
        layer0 = sigma2[0].reshape(-1)
        layer1 = sigma2[1].reshape(-1)
        for rank, index in enumerate(alternating_heads):
            layer0[index] = (
                1e9 if rank == 0 else 1.0 + (len(alternating_heads) - rank) * 1e-6)
            layer1[index] = 1.0 + (len(alternating_heads) - rank) * 1e-6
        return sigma2, torch.ones_like(sigma2)

    @classmethod
    def reference(cls):
        sigma2, q_absmean = cls.statistics()
        return calibration.build_top_p_artifact(
            sigma2, q_absmean, 0.98, group_size=8, skip_first=2,
            model="synthetic-model", calib_data="synthetic.parquet",
            num_samples=3, sample_len=16, seed=5)

    @classmethod
    def build(
            cls, kind, reference=None, reference_mask_sha256="a" * 64,
            **overrides):
        sigma2, q_absmean = cls.statistics()
        arguments = {
            "control_kind": kind,
            "group_size": 8,
            "skip_first": 2,
            "model": "synthetic-model",
            "calib_data": "synthetic.parquet",
            "num_samples": 3,
            "sample_len": 16,
            "seed": 5,
        }
        arguments.update(overrides)
        return calibration.build_uniform_top_k_control_artifact(
            sigma2, q_absmean, reference or cls.reference(),
            reference_mask_sha256,
            **arguments)

    def assert_artifacts_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            with self.subTest(field=key):
                if isinstance(first[key], torch.Tensor):
                    self.assertTrue(torch.equal(first[key], second[key]))
                else:
                    self.assertEqual(first[key], second[key])


class TestUniformTopKConstruction(ControlFixture):
    def test_both_controls_are_deterministic_exact_topk_artifacts(self):
        reference = self.reference()
        self.assertTrue(torch.equal(
            reference["nf2_count_per_layer"],
            torch.tensor([1, 63], dtype=torch.int32)))
        artifacts = {}
        for kind in calibration.CONTROL_KINDS:
            with self.subTest(kind=kind):
                first = self.build(kind, reference=reference)
                second = self.build(kind, reference=reference)
                self.assertIs(
                    calibration.validate_uniform_top_k_control_artifact(first),
                    first)
                self.assert_artifacts_equal(first, second)
                self.assertEqual(set(first), CONTROL_FIELDS)
                self.assertEqual(
                    first["selection_method"],
                    "layer_uniform_fixed_top_k_control")
                self.assertEqual(first["format_version"], 2)
                self.assertEqual(
                    first["selection_axis"],
                    "per_layer_flattened_kv_head_channel")
                self.assertEqual(first["tie_rule"], "score_desc_flat_index_asc")
                self.assertEqual(first["score_dtype_for_selection"], "float64")
                self.assertEqual(first["control_kind"], kind)
                self.assertEqual(
                    first["reference_selection_method"], "layer_channel_top_p")
                self.assertEqual(first["reference_top_p_threshold"], 0.98)
                self.assertTrue(torch.equal(
                    first["reference_codebook_mask"],
                    reference["codebook_mask"]))

                for layer in range(first["n_layers"]):
                    count = int(first["nf2_count_per_layer"][layer])
                    order = torch.argsort(
                        first["ranking_score"][layer].reshape(-1),
                        descending=True, stable=True)
                    expected = torch.zeros(
                        first["n_kv"] * first["head_dim"], dtype=torch.uint8)
                    expected[order[:count]] = 1
                    self.assertTrue(torch.equal(
                        first["codebook_mask"][layer].reshape(-1), expected))
                artifacts[kind] = first

        same = artifacts["same_cardinality"]
        exact = artifacts["exact_packed_bits"]
        self.assertEqual(same["reference_nf2_count"], 64)
        self.assertEqual(same["actual_nf2_count"], 64)
        self.assertTrue(torch.equal(
            same["nf2_count_per_layer"],
            torch.tensor([32, 32], dtype=torch.int32)))
        self.assertEqual(same["reference_packed_bits_total"], 288)
        self.assertEqual(same["packed_bits_total"], 320)

        self.assertEqual(exact["reference_packed_bits_total"], 288)
        self.assertEqual(exact["packed_bits_total"], 288)
        self.assertEqual(exact["reference_nf2_count"], 64)
        self.assertEqual(exact["actual_nf2_count"], 32)
        self.assertTrue(torch.equal(
            exact["nf2_count_per_layer"],
            torch.tensor([16, 16], dtype=torch.int32)))
        self.assertFalse(torch.equal(same["codebook_mask"], exact["codebook_mask"]))

    def test_reference_semantic_digest_is_storage_and_source_independent(self):
        reference = self.reference()
        equivalent_reference = copy.deepcopy(reference)
        for key in ("sigma2", "q_absmean", "codebook_mask"):
            value = equivalent_reference[key]
            equivalent_reference[key] = (
                value.transpose(1, 2).contiguous().transpose(1, 2))
            self.assertFalse(equivalent_reference[key].is_contiguous())

        first = self.build(
            "same_cardinality", reference=reference,
            reference_mask_sha256="a" * 64)
        second = self.build(
            "same_cardinality", reference=equivalent_reference,
            reference_mask_sha256="b" * 64)
        self.assertNotEqual(
            first["reference_mask_sha256"], second["reference_mask_sha256"])
        self.assertRegex(first["reference_semantic_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            first["reference_semantic_sha256"],
            second["reference_semantic_sha256"])


    def test_uniform_ties_use_flat_and_layer_index_order(self):
        scores = torch.ones(3, 1, 4, dtype=torch.float64)
        reference = torch.zeros(3, 1, 4, dtype=torch.uint8)
        reference.reshape(-1)[:7] = 1
        selected = calibration._uniform_layer_topk_same_cardinality(
            scores, reference)
        self.assertTrue(torch.equal(
            selected["nf2_count_per_layer"],
            torch.tensor([3, 2, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(
            selected["codebook_mask"],
            torch.tensor(
                [[[1, 1, 1, 0]], [[1, 1, 0, 0]], [[1, 1, 0, 0]]],
                dtype=torch.uint8)))


class TestControlValidationAndProvenance(ControlFixture):
    def test_recomputed_metadata_tampering_is_rejected_for_both_kinds(self):
        mutators = {
            "mask": lambda p: p["codebook_mask"].view(-1).__setitem__(
                0, 1 - p["codebook_mask"].view(-1)[0]),
            "reference_mask": lambda p: p["reference_codebook_mask"].view(-1).__setitem__(
                0, 1 - p["reference_codebook_mask"].view(-1)[0]),
            "ranking_score": lambda p: p["ranking_score"].view(-1).__setitem__(
                0, p["ranking_score"].view(-1)[0] + 0.25),
            "reorder": lambda p: p["reorder_index"].view(-1).__setitem__(
                0, p["reorder_index"].view(-1)[1]),
            "head_count": lambda p: p["nf2_count_per_head"].view(-1).__setitem__(
                0, p["nf2_count_per_head"].view(-1)[0] + 1),
            "layer_count": lambda p: p["nf2_count_per_layer"].view(-1).__setitem__(
                0, p["nf2_count_per_layer"].view(-1)[0] + 1),
            "actual_count": lambda p: p.__setitem__(
                "actual_nf2_count", p["actual_nf2_count"] + 1),
            "reference_count": lambda p: p.__setitem__(
                "reference_nf2_count", p["reference_nf2_count"] + 1),
            "packed_bits": lambda p: p.__setitem__(
                "packed_bits_total", p["packed_bits_total"] + 1),
            "fraction": lambda p: p.__setitem__(
                "actual_nf2_frac", p["actual_nf2_frac"] + 0.01),
            "selection_method": lambda p: p.__setitem__(
                "selection_method", "other"),
            "reference_hash": lambda p: p.__setitem__(
                "reference_mask_sha256", "A" * 64),
            "reference_semantic_hash_format": lambda p: p.__setitem__(
                "reference_semantic_sha256", "A" * 64),
            "reference_threshold_type": lambda p: p.__setitem__(
                "reference_top_p_threshold", "0.98"),
            "blank_model": lambda p: p.__setitem__("model", " \t"),
            "blank_calib_data": lambda p: p.__setitem__("calib_data", "\n"),
        }
        for kind in calibration.CONTROL_KINDS:
            original = self.build(kind)
            for name, mutate in mutators.items():
                with self.subTest(kind=kind, field=name):
                    tampered = copy.deepcopy(original)
                    mutate(tampered)
                    with self.assertRaises(ValueError):
                        calibration.validate_uniform_top_k_control_artifact(
                            tampered)

    def test_valid_lowercase_semantic_digest_tampering_is_rejected(self):
        for kind in calibration.CONTROL_KINDS:
            with self.subTest(kind=kind):
                tampered = self.build(kind)
                original = tampered["reference_semantic_sha256"]
                tampered["reference_semantic_sha256"] = (
                    "0" * 64 if original != "0" * 64 else "1" * 64)
                self.assertRegex(
                    tampered["reference_semantic_sha256"], r"^[0-9a-f]{64}$")
                with self.assertRaisesRegex(
                        ValueError, "reference_semantic_sha256 mismatch"):
                    calibration.validate_uniform_top_k_control_artifact(
                        tampered)



    def test_reference_file_hash_and_strict_top_p_validation(self):
        reference = self.reference()
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.pt"
            torch.save(reference, reference_path)
            loaded, digest = calibration.load_top_p_reference_artifact(
                reference_path)
            self.assertEqual(
                digest, hashlib.sha256(reference_path.read_bytes()).hexdigest())
            calibration.validate_top_p_artifact(loaded)

            source_mutators = {
                "mask": lambda p: p["codebook_mask"].view(-1).__setitem__(
                    0, 1 - p["codebook_mask"].view(-1)[0]),
                "threshold_type": lambda p: p.__setitem__(
                    "top_p_threshold", "0.98"),
                "blank_model": lambda p: p.__setitem__("model", " \t"),
                "blank_calib_data": lambda p: p.__setitem__("calib_data", "\n"),
            }
            for name, mutate in source_mutators.items():
                with self.subTest(field=name):
                    tampered = copy.deepcopy(reference)
                    mutate(tampered)
                    torch.save(tampered, reference_path)
                    with self.assertRaises(ValueError):
                        calibration.load_top_p_reference_artifact(reference_path)

    def test_active_statistics_and_provenance_must_match_for_both_kinds(self):
        reference = self.reference()
        provenance_mismatches = {
            "group_size": 16,
            "skip_first": 3,
            "model": "other-model",
            "calib_data": "other.parquet",
            "num_samples": 4,
            "sample_len": 32,
            "seed": 6,
        }
        for kind in calibration.CONTROL_KINDS:
            for field, value in provenance_mismatches.items():
                with self.subTest(kind=kind, field=field):
                    with self.assertRaisesRegex(
                            ValueError, f"{field} does not match"):
                        self.build(kind, reference=reference, **{field: value})

            sigma2, q_absmean = self.statistics()
            sigma2[0, 0, 0] += 1.0
            with self.subTest(kind=kind, field="sigma2"):
                with self.assertRaisesRegex(ValueError, "sigma2 does not match"):
                    calibration.build_uniform_top_k_control_artifact(
                        sigma2, q_absmean, reference, "a" * 64,
                        control_kind=kind, group_size=8, skip_first=2,
                        model="synthetic-model", calib_data="synthetic.parquet",
                        num_samples=3, sample_len=16, seed=5)


class TestControlCLI(ControlFixture):
    def test_stats_input_generates_both_kinds_without_model_arguments(self):
        sigma2, q_absmean = self.statistics()
        stats = calibration.build_statistics_artifact(
            sigma2, q_absmean, model="synthetic-model", group_size=8,
            skip_first=2, calib_data="synthetic.parquet", num_samples=3,
            sample_len=16, seed=5)
        reference = self.reference()
        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "stats.pt"
            reference_path = Path(tmp) / "reference.pt"
            torch.save(stats, stats_path)
            torch.save(reference, reference_path)
            reference_digest = hashlib.sha256(
                reference_path.read_bytes()).hexdigest()
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PYTHONPATH"] = str(REPO_ROOT / "src")

            for kind in calibration.CONTROL_KINDS:
                with self.subTest(kind=kind):
                    output_path = Path(tmp) / f"{kind}.pt"
                    command = [
                        sys.executable, str(SCRIPT_PATH), "--stats-input",
                        str(stats_path), "--uniform-top-k-control-from",
                        str(reference_path), "--output", str(output_path),
                    ]
                    if kind != "exact_packed_bits":
                        command.extend(["--control-kind", kind])
                    proc = subprocess.run(
                        command, cwd=REPO_ROOT, env=env, text=True,
                        capture_output=True, check=False)
                    self.assertEqual(
                        proc.returncode, 0, proc.stdout + proc.stderr)
                    payload = torch.load(
                        output_path, map_location="cpu", weights_only=True)
                    self.assertEqual(payload["control_kind"], kind)
                    self.assertEqual(
                        payload["reference_mask_sha256"], reference_digest)
                    calibration.validate_uniform_top_k_control_artifact(payload)

    def test_top_p_and_control_reference_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                calibration.parse_args([
                    "--stats-input", "stats.pt", "--top-p", "0.5",
                    "--uniform-top-k-control-from", "reference.pt",
                    "--output", "control.pt",
                ])


if __name__ == "__main__":
    unittest.main()
