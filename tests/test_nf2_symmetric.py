"""0-GPU unit tests for the symmetric-NF2 codebook (the Lloyd-Max replacement).

nf2 is now a FIXED symmetric LUT (IR-QLoRA appendix B.2): normalized levels
{-1, -c, +c, +1}, c = NF2_INNER. Two entry points share the semantics:

  * apply_codebook(x, G, "nf2")     -- UNCENTERED input: group mean mu is
    subtracted first, the residual gets an absmax scale s and the LUT snap,
    reconstruction = mu + s * C[q]  (side-info: mu + s = 2 fp16 / group).
  * _pt_codebook_masked / _pure_pt_codebook nf2 -- ALREADY-CENTERED residual
    (per-channel mu_d subtracted upstream): NO second mean, absmax-only
    (side-info: s = 1 fp16 / group -> ~2.25 bit at head_dim 64).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest tests.test_nf2_symmetric -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig
from kitty_sim.qlut_quant import (
    NF2_INNER,
    NF2_THRESH,
    apply_codebook,
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


class TestApplyCodebookNF2(unittest.TestCase):
    def test_mu_plus_absmax_lut(self):
        """apply_codebook nf2 on UNCENTERED data == mu + symmetric snap of (x-mu)."""
        torch.manual_seed(2)
        H, D, T, G = 2, 4, 256, 128
        x = torch.randn(H, D, T, dtype=torch.float32) + 3.0  # strong DC offset
        out = apply_codebook(x, G, "nf2")
        xg = x.reshape(H, D, T // G, G)
        mu = xg.mean(-1, keepdim=True)
        ref = (mu + nf2_symmetric_lastdim(xg - mu)).reshape(H, D, T)
        self.assertTrue(torch.equal(out, ref))

    def test_group_value_set_is_four_levels_around_mu(self):
        torch.manual_seed(3)
        H, D, G = 1, 2, 64
        x = torch.randn(H, D, G, dtype=torch.float32) * 2.5
        out = apply_codebook(x, G, "nf2")
        for h in range(H):
            for d in range(D):
                g = x[h, d].float()
                mu = g.mean()
                s = (g - mu).abs().max()
                cand = torch.stack([mu - s, mu - NF2_INNER * s, mu + NF2_INNER * s, mu + s])
                dmin = (out[h, d].unsqueeze(-1) - cand).abs().min(-1).values
                self.assertTrue(bool((dmin < 1e-5).all()))

    def test_constant_group_exact(self):
        x = torch.full((1, 1, 128), 7.25, dtype=torch.float32)
        out = apply_codebook(x, 128, "nf2")
        self.assertTrue(torch.allclose(out, x, atol=0.0))


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

    def test_pure_pt_codebook_consistency(self):
        torch.manual_seed(5)
        sub = torch.randn(2, 7, 24, dtype=torch.float32)
        out = KittyKVCache._pure_pt_codebook(sub, "nf2")
        self.assertTrue(torch.equal(out, nf2_symmetric_lastdim(sub)))


class TestBlockedNF2(unittest.TestCase):
    def _cache(self, block: int) -> KittyKVCache:
        cfg = KittyKVCacheConfig(
            sink_length=32, buffer_length=128, group_size=128, kbits=2, vbits=4,
            promote_ratio=0.0, channel_selection=0, k_quant_mode="per_token",
            k_codebook="qlut", bin_codebooks=["sign", "nf2"], n_bins=2,
            pertoken_mixed=True, pertoken_block=block)
        return KittyKVCache(cfg)

    def test_block1_bit_identical_to_masked(self):
        torch.manual_seed(6)
        r = torch.randn(1, 2, 33, 16, dtype=torch.float32)
        m = torch.rand(2, 16) > 0.4
        out = self._cache(1)._pt_codebook_blocked(r, m, "nf2")
        ref = KittyKVCache._pt_codebook_masked(r, m, "nf2")
        self.assertTrue(torch.equal(out, ref))

    def test_block16_matches_manual_blocking(self):
        torch.manual_seed(7)
        B, nh, T, D, blk = 1, 2, 40, 8, 16  # 2 full blocks + tail of 8
        r = torch.randn(B, nh, T, D, dtype=torch.float32)
        m = torch.rand(nh, D) > 0.4
        out = self._cache(blk)._pt_codebook_blocked(r, m, "nf2")
        ref = torch.empty_like(r)
        start = 0
        while start < T:
            width = min(blk, T - start)
            rt = r[:, :, start:start + width, :].reshape(B, nh, 1, width * D)
            mf = m[:, None, :].expand(nh, width, D).reshape(nh, width * D)
            rec = KittyKVCache._pt_codebook_masked(rt, mf, "nf2")
            ref[:, :, start:start + width, :] = rec.reshape(B, nh, width, D)
            start += width
        self.assertTrue(torch.equal(out, ref))


class TestPertokenPcSubmeanNF2(unittest.TestCase):
    """qlutattn_pertoken now routes through pertoken_pc_submean (per-channel
    mean mu_d, the sign recipe axis) with the general codebook dispatch."""

    def _cache(self, bins: list[str]) -> KittyKVCache:
        cfg = KittyKVCacheConfig(
            sink_length=32, buffer_length=128, group_size=128, kbits=2, vbits=4,
            promote_ratio=0.0, channel_selection=0, k_quant_mode="per_token",
            k_codebook="qlut", bin_codebooks=bins, n_bins=len(bins),
            pertoken_pc_submean=True)
        return KittyKVCache(cfg)

    def test_nf2_prefill_is_pc_mean_plus_symmetric_snap(self):
        torch.manual_seed(8)
        B, nh, T, D = 1, 2, 128, 16
        ks = torch.randn(B, nh, T, D, dtype=torch.float32) + 2.0  # channel DC offsets
        cache = self._cache(["nf2"])
        out = cache._quant_k_pertoken(ks, 0)
        mu = ks[0].mean(dim=1)                                     # [nh,D] per-CHANNEL mean
        muB = mu[None, :, None, :]
        self.assertTrue(torch.equal(cache.k_pc_mean[0], mu))
        self.assertTrue(torch.equal(out, muB + nf2_symmetric_lastdim(ks - muB)))

    def test_nf2_decode_reuses_cached_pc_mean(self):
        torch.manual_seed(9)
        B, nh, T, D = 1, 2, 128, 16
        ks = torch.randn(B, nh, T, D, dtype=torch.float32)
        cache = self._cache(["nf2"])
        cache._quant_k_pertoken(ks, 0)                             # prefill sets mu_d
        muB = cache.k_pc_mean[0][None, :, None, :]
        ks1 = torch.randn(B, nh, 1, D, dtype=torch.float32)
        out1 = cache._quant_k_pertoken(ks1, 0)
        self.assertTrue(torch.equal(out1, muB + nf2_symmetric_lastdim(ks1 - muB)))

    def test_sign_numerics_preserved_by_dispatch_refactor(self):
        torch.manual_seed(10)
        B, nh, T, D = 1, 2, 64, 16
        ks = torch.randn(B, nh, T, D, dtype=torch.float32)
        out = self._cache(["sign"])._quant_k_pertoken(ks, 0)
        muB = ks[0].mean(dim=1)[None, :, None, :]
        r = ks - muB
        mag = r.abs().mean(dim=3, keepdim=True)
        self.assertTrue(torch.equal(out, muB + torch.sign(r) * mag))


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
            v_tile_channels=None, quest_kernel=False, quest_token_budget=None,
            quest_skip_layers=0, shadowkv_budget=None, shadowkv_rank=None,
            shadowkv_chunk_size=None,
        )
        for k, v in kwargs.items():
            setattr(ns, k, v)
        return ns

    def _write_mask(self, codebooks: list[str]) -> str:
        path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        self.addCleanup(Path(path).unlink, missing_ok=True)
        torch.save({
            "codebook_mask": torch.zeros(2, 2, 64, dtype=torch.uint8),
            "codebooks": codebooks,
            "nominal_bits": 1.875,
        }, path)
        return path

    def test_static_nf2_bins_get_marker(self):
        cfg = build_variant(self._args("qlutattn_k1v4"))
        self.assertEqual(cfg.nf2_impl, NF2_IMPL_VERSION)
        self.assertEqual(variant_semantic_payload(cfg).get("nf2_impl"), NF2_IMPL_VERSION)

    def test_mask_nf2_gets_marker(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(["sign", "nf2"])
        cfg = build_variant(self._args("qlutattn_k188v4_pt"))
        self.assertEqual(cfg.nf2_impl, NF2_IMPL_VERSION)

    def test_mask_override_detection_on_sign_tern_variant(self):
        """k168 declares static sign/tern bins, but the runtime trusts the mask
        blob; a sign,nf2 mask must therefore stamp the marker too."""
        os.environ["QLUT_CB_MASK"] = self._write_mask(["sign", "nf2"])
        cfg = build_variant(self._args("qlutattn_k168v4_pt"))
        self.assertEqual(cfg.nf2_impl, NF2_IMPL_VERSION)

    def test_sign_tern_mask_no_marker(self):
        os.environ["QLUT_CB_MASK"] = self._write_mask(["sign", "tern"])
        cfg = build_variant(self._args("qlutattn_k168v4_pt"))
        self.assertIsNone(cfg.nf2_impl)
        self.assertNotIn("nf2_impl", variant_semantic_payload(cfg))

    def test_non_nf2_variants_keep_historical_payload(self):
        for name in ("fp16", "kivi", "qlutattn_k125v4_pt", "qlutattn_k184v4"):
            cfg = build_variant(self._args(name))
            self.assertIsNone(cfg.nf2_impl, name)
            self.assertNotIn("nf2_impl", variant_semantic_payload(cfg), name)

    def test_qlutattn_pertoken_routes_to_pc_submean(self):
        cfg = build_variant(self._args("qlutattn_pertoken"))
        self.assertTrue(cfg.pertoken_pc_submean)
        self.assertEqual(cfg.bin_codebooks, ("nf2",))
        self.assertEqual(cfg.nf2_impl, NF2_IMPL_VERSION)
        self.assertEqual(cfg.pertoken_outlier_k, 0)

    def test_qlutattn_pertoken_rejects_legacy_outlier_env(self):
        os.environ["PERTOKEN_OUTLIER_K"] = "8"
        with self.assertRaises(ValueError):
            build_variant(self._args("qlutattn_pertoken"))


if __name__ == "__main__":
    unittest.main()
