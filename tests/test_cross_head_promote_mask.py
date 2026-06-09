"""Cross-head (layer-global) channel selection (channel_selection=3) — sim-path unit tests.

These tests pin the behaviour added for cross-head K-channel boost allocation:
  * build_promote_mask(cs=3) spends EXACTLY the same total budget as the uniform
    per-head strategy (cs=1): nh * int(D * ratio) promoted channels per sample.
  * Per-head counts are free to differ (important heads take more channels) and
    a head may be starved to zero.
  * Allocation is per batch sample (independent across B).
  * fake_quant_groupwise_lastdim consumes a non-uniform mask correctly.
  * KittyKVCacheConfig accepts channel_selection=3 (and still rejects 2 / -1).
  * build_variant wires kitty_k1v4_xhead (channel_selection=3, otherwise
    identical to kitty_k1v4, including --promote-ratio-config support).

Run with: python -m unittest tests.test_cross_head_promote_mask -v
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig
from kitty_sim.longbench.runner import build_variant, method_layout_slug
from kitty_sim.utils_quant import build_promote_mask, fake_quant_groupwise_lastdim


class CrossHeadBudgetTests(unittest.TestCase):
    """Strategy 3 must be bit-identical in total budget to strategy 1."""

    def test_total_budget_matches_uniform_for_any_ratio(self):
        torch.manual_seed(0)
        B, nh, D, T = 2, 8, 64, 128
        key = torch.randn(B, nh, D, T)
        for ratio in (0.125, 0.25, 0.3, 0.5, 0.6875, 0.875):
            k_chan = int(D * ratio + 1e-6)
            uniform = build_promote_mask(key, ratio, 1)
            xhead = build_promote_mask(key, ratio, 3)
            for b in range(B):
                self.assertEqual(int(uniform[b].sum()), nh * k_chan, f"ratio={ratio}")
                self.assertEqual(int(xhead[b].sum()), nh * k_chan, f"ratio={ratio}")

    def test_zero_and_full_ratio_corner_cases(self):
        key = torch.randn(1, 4, 8, 16)
        self.assertEqual(int(build_promote_mask(key, 0.0, 3).sum()), 0)
        self.assertTrue(build_promote_mask(key, 1.0, 3).all())

    def test_shape_and_dtype(self):
        key = torch.randn(2, 4, 8, 16)
        mask = build_promote_mask(key, 0.5, 3)
        self.assertEqual(mask.shape, (2, 4, 8))
        self.assertEqual(mask.dtype, torch.bool)


class CrossHeadAllocationTests(unittest.TestCase):
    """The point of cs=3: per-head counts follow magnitude, not a fixed quota."""

    def test_louder_head_takes_more_channels(self):
        torch.manual_seed(0)
        B, nh, D, T = 1, 8, 64, 128
        key = torch.randn(B, nh, D, T)
        key[:, 0] *= 10.0  # head 0 dominates
        ratio = 0.5
        k_chan = int(D * ratio)
        mask = build_promote_mask(key, ratio, 3)
        per_head = mask[0].sum(dim=-1)  # (nh,)
        self.assertGreater(int(per_head[0]), k_chan, f"per_head={per_head.tolist()}")
        self.assertLess(int(per_head.min()), k_chan, f"per_head={per_head.tolist()}")
        self.assertEqual(int(per_head.sum()), nh * k_chan)

    def test_head_can_be_starved_to_zero(self):
        torch.manual_seed(0)
        B, nh, D, T = 1, 4, 16, 32
        # Heads 0..2 dwarf head 3 entirely; at ratio 0.25 the budget
        # (4*4=16 channels) fits inside the three loud heads (3*16=48 channels).
        key = torch.randn(B, nh, D, T)
        key[:, :3] *= 100.0
        mask = build_promote_mask(key, 0.25, 3)
        per_head = mask[0].sum(dim=-1)
        self.assertEqual(int(per_head[3]), 0, f"per_head={per_head.tolist()}")
        self.assertEqual(int(per_head.sum()), 4 * 4)

    def test_allocation_is_per_batch_sample(self):
        torch.manual_seed(0)
        nh, D, T = 4, 16, 32
        key = torch.randn(2, nh, D, T)
        key[0, 0] *= 50.0  # sample 0: head 0 loud
        key[1, 3] *= 50.0  # sample 1: head 3 loud
        mask = build_promote_mask(key, 0.25, 3)
        per_head0 = mask[0].sum(dim=-1)
        per_head1 = mask[1].sum(dim=-1)
        self.assertEqual(int(per_head0.argmax()), 0)
        self.assertEqual(int(per_head1.argmax()), 3)

    def test_uniform_strategy_unchanged(self):
        torch.manual_seed(0)
        key = torch.randn(1, 8, 64, 128)
        key[:, 0] *= 10.0
        mask = build_promote_mask(key, 0.5, 1)
        per_head = mask[0].sum(dim=-1)
        self.assertTrue(all(int(c) == 32 for c in per_head), f"per_head={per_head.tolist()}")


class FakeQuantNonUniformMaskTests(unittest.TestCase):
    def test_promoted_channels_get_lower_error(self):
        torch.manual_seed(0)
        B, nh, D, T = 1, 4, 16, 32
        data = torch.randn(B, nh, D, T)
        data[:, 0] *= 10.0
        mask = build_promote_mask(data, 0.5, 3)
        dq = fake_quant_groupwise_lastdim(data, 16, 1, mask, 2)
        self.assertEqual(dq.shape, data.shape)
        err = (dq - data).abs().mean(dim=-1)  # (B, nh, D)
        # Promoted channels (2-bit) must reconstruct better than 1-bit channels
        # on average; compare within each head that has both kinds.
        for h in range(nh):
            m = mask[0, h]
            if m.any() and (~m).any():
                rel_promoted = (err[0, h][m] / data[0, h][m].abs().mean(dim=-1).clamp(min=1e-3)).mean()
                rel_base = (err[0, h][~m] / data[0, h][~m].abs().mean(dim=-1).clamp(min=1e-3)).mean()
                self.assertLess(float(rel_promoted), float(rel_base), f"head={h}")


class ConfigValidationTests(unittest.TestCase):
    def _config(self, channel_selection):
        return KittyKVCacheConfig(
            sink_length=2, buffer_length=2, group_size=2,
            kbits=1, vbits=4, promote_bit=2, promote_ratio=0.5,
            channel_selection=channel_selection,
        )

    def test_accepts_cross_head(self):
        cfg = self._config(3)
        self.assertEqual(cfg.channel_selection, 3)

    def test_still_rejects_variance_and_unspecified(self):
        for bad in (2, -1):
            with self.assertRaises(ValueError):
                self._config(bad)


class CacheUpdateWithCrossHeadTests(unittest.TestCase):
    def test_update_runs_end_to_end_with_cs3(self):
        cfg = KittyKVCacheConfig(
            sink_length=2, buffer_length=2, group_size=2,
            kbits=1, vbits=4, promote_bit=2, promote_ratio=0.5,
            channel_selection=3,
        )
        cache = KittyKVCache(cfg)
        k, v = torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4)
        out_k, out_v = cache.update(k, v, 0)
        self.assertEqual(out_k.shape, k.shape)
        self.assertEqual(out_v.shape, v.shape)


class BuildVariantXheadTests(unittest.TestCase):
    def _write(self, payload):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        self.addCleanup(os.remove, path)
        return path

    def test_xhead_differs_from_k1v4_only_in_channel_selection(self):
        base = build_variant(SimpleNamespace(variant="kitty_k1v4", promote_ratio_config=None))
        xhead = build_variant(SimpleNamespace(variant="kitty_k1v4_xhead", promote_ratio_config=None))
        self.assertEqual(xhead.name, "kitty_k1v4_xhead")
        self.assertEqual(xhead.channel_selection, 3)
        self.assertEqual(base.channel_selection, 1)
        for field in ("kbits", "vbits", "promote_bit", "promote_ratio",
                      "sink_length", "buffer_length", "group_size", "use_kitty"):
            self.assertEqual(getattr(xhead, field), getattr(base, field), field)

    def test_xhead_accepts_promote_ratio_config(self):
        path = self._write({"default": 0.5})
        v = build_variant(SimpleNamespace(variant="kitty_k1v4_xhead", promote_ratio_config=path))
        self.assertEqual(v.promote_ratio, 0.5)
        self.assertEqual(v.channel_selection, 3)
        self.assertIsNone(v.promote_ratio_per_layer)

    def test_xhead_layout_slug(self):
        self.assertEqual(method_layout_slug("kitty_k1v4_xhead"), "kitty-k1v4-xhead")

    def test_tag_distinguishes_selection_strategy(self):
        base = build_variant(SimpleNamespace(variant="kitty_k1v4", promote_ratio_config=None))
        xhead = build_variant(SimpleNamespace(variant="kitty_k1v4_xhead", promote_ratio_config=None))
        self.assertIn("_sel1_", base.tag)
        self.assertIn("_sel3_", xhead.tag)


if __name__ == "__main__":
    unittest.main()
