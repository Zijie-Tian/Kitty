"""0-GPU unit tests for the symmetric-NF2 codebook (symnf2-v1).

nf2 is a FIXED symmetric LUT (IR-QLoRA appendix B.2): normalized levels
{-1, -c, +c, +1}, c = NF2_INNER. The canonical qlutattn K path feeds it an
ALREADY-CENTERED residual (per-channel mu_d subtracted upstream): NO second
mean, absmax-only side-info (s = 1 fp16 / group -> ~2.25 bit at head_dim 64).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest tests.test_nf2_symmetric -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from kitty_sim.kitty_simulate import KittyKVCache
from kitty_sim.qlut_quant import (
    NF2_INNER,
    NF2_THRESH,
    nf2_symmetric_lastdim,
)
from kitty_sim.longbench.runner import (
    NF2_IMPL_VERSION,
    build_variant,
    variant_semantic_payload,
)

LUT = torch.tensor([-1.0, -NF2_INNER, NF2_INNER, 1.0], dtype=torch.float32)


def _brute_lut_lastdim(r: torch.Tensor) -> torch.Tensor:
    """Independent reference: absmax-normalize, nearest LUT level by argmin."""
    s = r.abs().amax(-1, keepdim=True)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    idx = ((r / safe).unsqueeze(-1) - LUT.to(r.device)).abs().argmin(-1)
    rec = LUT.to(r.device)[idx] * s
    return torch.where(s > 0, rec, torch.zeros_like(rec))


class TestNF2Constants(unittest.TestCase):
    def test_ir_qlora_values(self):
        self.assertAlmostEqual(NF2_INNER, 0.25256848335266113, places=12)
        self.assertAlmostEqual(NF2_THRESH, (1.0 + NF2_INNER) / 2.0, places=12)


class TestNF2CoreSnap(unittest.TestCase):
    def test_closed_form_matches_brute_force_argmin(self):
        torch.manual_seed(0)
        r = torch.randn(4, 3, 17, 64, dtype=torch.float32)
        out = nf2_symmetric_lastdim(r)
        ref = _brute_lut_lastdim(r)
        self.assertTrue(torch.equal(out, ref))

    def test_reconstruction_value_set(self):
        torch.manual_seed(1)
        r = torch.randn(5, 32, dtype=torch.float32)
        out = nf2_symmetric_lastdim(r)
        s = r.abs().amax(-1, keepdim=True)
        cand = torch.stack([-s.expand_as(r), -NF2_INNER * s.expand_as(r),
                            NF2_INNER * s.expand_as(r), s.expand_as(r)], dim=-1)
        hit = (out.unsqueeze(-1) == cand).any(-1)
        self.assertTrue(bool(hit.all()))

    def test_all_zero_group_reconstructs_zero(self):
        r = torch.zeros(2, 8, dtype=torch.float32)
        out = nf2_symmetric_lastdim(r)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertFalse(bool(out.isnan().any()))


class TestMaskedNF2(unittest.TestCase):
    def test_masked_matches_perhead_extract(self):
        """Vectorized masked core == per-head channel-extract reference (zero
        outside the bin), i.e. the decode path == the prefill _pure_pt_codebook."""
        torch.manual_seed(4)
        B, nh, T, D = 2, 3, 9, 16
        r = torch.randn(B, nh, T, D, dtype=torch.float32)
        m = torch.rand(nh, D) > 0.5
        m[1] = False  # one head entirely outside this bin (empty-mask row)
        out = KittyKVCache._pt_codebook_masked(r, m, "nf2")
        ref = torch.zeros_like(r)
        for h in range(nh):
            sel = m[h]
            if not sel.any():
                continue
            ref[:, h, :, sel] = nf2_symmetric_lastdim(r[:, h, :, sel])
        self.assertTrue(torch.equal(out, ref))

    def test_empty_mask_row_zero_no_nan(self):
        r = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        m = torch.zeros(2, 8, dtype=torch.bool)
        out = KittyKVCache._pt_codebook_masked(r, m, "nf2")
        self.assertFalse(bool(out.isnan().any()))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_no_second_mean_on_centered_path(self):
        """The masked path must NOT subtract another mean: a residual with a
        deliberate DC offset keeps absmax-only semantics (2 levels +/- around 0,
        NOT around the group mean)."""
        r = torch.full((1, 1, 1, 8), 1.0, dtype=torch.float32)
        r[0, 0, 0, 0] = 3.0
        m = torch.ones(1, 8, dtype=torch.bool)
        out = KittyKVCache._pt_codebook_masked(r, m, "nf2")
        # absmax-only: s = 3, so values 1.0 (< thresh*3=1.879) snap to +c*3, the
        # 3.0 snaps to +3. A mean-subtracted version would instead center at 1.25.
        expected = torch.full_like(r, NF2_INNER * 3.0)
        expected[0, 0, 0, 0] = 3.0
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


class TestNF2ImplHashMarker(unittest.TestCase):
    def setUp(self):
        self._env_backup = {
            k: os.environ.get(k)
            for k in ("QLUT_BIN_CODEBOOKS", "QLUT_CB_MASK", "V_TILE_CHANNELS",
                      "VBITS", "PERTOKEN_BLOCK", "PERTOKEN_OUTLIER_K")
        }
        for k in list(self._env_backup):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _args(self, variant: str, **kwargs):
        ns = SimpleNamespace(
            variant=variant, kbits=2, vbits=2, promote_bit=4, promote_ratio=None,
            promote_ratio_config=None, sink_length=32, buffer_length=128,
            group_size=128, channel_selection=0, k_quant_mode="per_token",
            quest_kernel=False, quest_token_budget=None,
            quest_skip_layers=0, shadowkv_budget=None, shadowkv_rank=None,
            shadowkv_chunk_size=None,
        )
        for k, v in kwargs.items():
            setattr(ns, k, v)
        return ns

    def _write_mask(self) -> str:
        """A valid canonical qlutattn mask: sign/nf2, round(0.65*N) sign per layer."""
        path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        self.addCleanup(Path(path).unlink, missing_ok=True)
        n_layers, n_kv, head_dim = 2, 2, 64
        n = n_kv * head_dim
        k_sign = int(round(0.65 * n))
        mask = torch.zeros(n_layers, n_kv, head_dim, dtype=torch.uint8)
        flat = mask.reshape(n_layers, -1)
        flat[:, k_sign:] = 1
        torch.save({
            "codebook_mask": mask,
            "codebooks": ["sign", "nf2"],
            "low_frac": k_sign / n,
        }, path)
        return path

    def test_qlutattn_gets_marker(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask()
        cfg = build_variant(self._args("qlutattn"))
        self.assertEqual(cfg.nf2_impl, NF2_IMPL_VERSION)
        self.assertEqual(variant_semantic_payload(cfg).get("nf2_impl"), NF2_IMPL_VERSION)

    def test_non_nf2_variants_keep_historical_payload(self):
        for name in ("fp16", "kivi", "kitty"):
            cfg = build_variant(self._args(name))
            self.assertIsNone(cfg.nf2_impl, name)
            self.assertNotIn("nf2_impl", variant_semantic_payload(cfg), name)


if __name__ == "__main__":
    unittest.main()
