"""Per-layer promote_ratio (JSON-controlled, kitty_k1v4) — sim-path unit tests.

These tests pin the behaviour added for per-layer K-channel boost:
  * KittyKVCache._layer_pr resolves per-layer overrides with scalar fallback.
  * KittyKVCache.update() actually feeds each layer its own promote_ratio.
  * _load_promote_ratio_config parses both JSON forms and validates ratios.
  * build_variant wires kitty_k1v4 to the JSON and rejects the flag elsewhere.

Run with: python -m unittest tests.test_kitty_per_layer_promote_ratio -v
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from kitty_sim import kitty_simulate
from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig, get_kvcache_kitty
from kitty_sim.longbench.runner import _load_promote_ratio_config, build_variant


def _k1v4_config(promote_ratio=0.25, per_layer=None, **overrides):
    kwargs = dict(
        sink_length=2, buffer_length=2, group_size=2,
        kbits=1, vbits=4, promote_bit=2, promote_ratio=promote_ratio,
        channel_selection=1, promote_ratio_per_layer=per_layer,
    )
    kwargs.update(overrides)
    return KittyKVCacheConfig(**kwargs)


class LayerPrLookupTests(unittest.TestCase):
    def test_scalar_when_no_per_layer(self):
        cache = KittyKVCache(_k1v4_config(promote_ratio=0.3, per_layer=None))
        self.assertEqual(cache._layer_pr(0), 0.3)
        self.assertEqual(cache._layer_pr(7), 0.3)

    def test_override_with_scalar_fallback(self):
        cache = KittyKVCache(_k1v4_config(promote_ratio=0.5, per_layer={0: 0.75, 3: 1.0}))
        self.assertEqual(cache._layer_pr(0), 0.75)   # explicit
        self.assertEqual(cache._layer_pr(3), 1.0)    # explicit
        self.assertEqual(cache._layer_pr(1), 0.5)    # fallback to scalar default
        self.assertEqual(cache._layer_pr(99), 0.5)   # fallback

    def test_string_keys_are_normalised_to_int(self):
        # get_kvcache_kitty / config accept whatever the loader produced; keys
        # must be coerced to int so layer_idx (int) lookups hit.
        cache = KittyKVCache(_k1v4_config(per_layer={"2": 0.875}))
        self.assertEqual(cache._layer_pr(2), 0.875)


class UpdateAppliesPerLayerRatioTests(unittest.TestCase):
    """update() must call build_promote_mask with each layer's own ratio."""

    def setUp(self):
        self._real = kitty_simulate.build_promote_mask
        self.calls = []  # list of (layer_marker, ratio)

    def tearDown(self):
        kitty_simulate.build_promote_mask = self._real

    def test_prefill_uses_layer_specific_ratio(self):
        per_layer = {0: 0.75, 1: 0.25}
        cache = KittyKVCache(_k1v4_config(promote_ratio=0.5, per_layer=per_layer))

        recorded = self.calls
        real = self._real

        def spy(key_states, promote_ratio, channel_selection):
            recorded.append(round(float(promote_ratio), 6))
            return real(key_states, promote_ratio, channel_selection)

        kitty_simulate.build_promote_mask = spy

        # B=1, nh=1, T=6 (> sink2+buffer2 so quantization triggers), D=4.
        def kv():
            return torch.randn(1, 1, 6, 4), torch.randn(1, 1, 6, 4)

        # layer 0 -> 0.75
        recorded.clear()
        k, v = kv(); cache.update(k, v, 0)
        self.assertTrue(recorded, "build_promote_mask never ran on prefill")
        self.assertTrue(all(r == 0.75 for r in recorded), f"layer0 ratios={recorded}")

        # layer 1 -> 0.25 (explicit override)
        recorded.clear()
        k, v = kv(); cache.update(k, v, 1)
        self.assertTrue(all(r == 0.25 for r in recorded), f"layer1 ratios={recorded}")

        # layer 2 -> 0.5 (falls back to scalar default)
        recorded.clear()
        k, v = kv(); cache.update(k, v, 2)
        self.assertTrue(all(r == 0.5 for r in recorded), f"layer2 ratios={recorded}")


class LoaderTests(unittest.TestCase):
    def _write(self, payload):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        self.addCleanup(os.remove, path)
        return path

    def test_dict_form_default_and_layers(self):
        path = self._write({"default": 0.3, "layers": {"0": 0.75, "5": 0.5}})
        default, per = _load_promote_ratio_config(path, 0.25)
        self.assertEqual(default, 0.3)
        self.assertEqual(dict(per), {0: 0.75, 5: 0.5})

    def test_dict_form_default_only(self):
        path = self._write({"default": 0.6875})
        default, per = _load_promote_ratio_config(path, 0.25)
        self.assertEqual(default, 0.6875)
        self.assertIsNone(per)  # no explicit per-layer -> None (scalar everywhere)

    def test_list_form(self):
        path = self._write([0.75, 0.5, 0.25])
        default, per = _load_promote_ratio_config(path, 0.25)
        self.assertEqual(default, 0.25)  # variant default kept; list covers layers
        self.assertEqual(dict(per), {0: 0.75, 1: 0.5, 2: 0.25})

    def test_rejects_out_of_range_ratio(self):
        path = self._write({"default": 0.25, "layers": {"0": 1.5}})
        with self.assertRaises(ValueError):
            _load_promote_ratio_config(path, 0.25)

    def test_rejects_bad_default(self):
        path = self._write({"default": -0.1})
        with self.assertRaises(ValueError):
            _load_promote_ratio_config(path, 0.25)


class BuildVariantTests(unittest.TestCase):
    def _write(self, payload):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        self.addCleanup(os.remove, path)
        return path

    def test_k1v4_without_config_keeps_legacy_scalar(self):
        v = build_variant(SimpleNamespace(variant="kitty_k1v4", promote_ratio_config=None))
        self.assertEqual(v.kbits, 1)
        self.assertEqual(v.vbits, 4)
        self.assertEqual(v.promote_bit, 2)
        self.assertEqual(v.promote_ratio, 0.25)
        self.assertIsNone(v.promote_ratio_per_layer)

    def test_k1v4_with_config_sets_per_layer(self):
        path = self._write({"default": 0.5, "layers": {"0": 1.0, "1": 1.0}})
        v = build_variant(SimpleNamespace(variant="kitty_k1v4", promote_ratio_config=path))
        self.assertEqual(v.promote_ratio, 0.5)
        self.assertEqual(dict(v.promote_ratio_per_layer), {0: 1.0, 1: 1.0})
        self.assertEqual(v.promote_ratio_config_path, path)

    def test_config_rejected_for_non_k1v4(self):
        path = self._write({"default": 0.5})
        for bad in ("kitty", "kitty_pro", "custom", "fp16", "kivi"):
            with self.assertRaises(ValueError):
                build_variant(SimpleNamespace(
                    variant=bad, promote_ratio_config=path,
                    sink_length=32, buffer_length=128, group_size=128,
                    kbits=2, vbits=2, promote_ratio=0.0, promote_bit=4, channel_selection=1,
                ))

    def test_deleted_variants_are_gone(self):
        for gone in ("kitty_k1v2", "kitty_k1v2_pr50", "kitty_k1v2_pr75",
                     "kitty_k1v4_pr50", "kitty_k1v4_pr75"):
            with self.assertRaises(ValueError):
                build_variant(SimpleNamespace(variant=gone, promote_ratio_config=None))


class GetKvcacheKittyTests(unittest.TestCase):
    def test_namespace_threads_per_layer_through(self):
        ns = SimpleNamespace(
            sink_length=2, buffer_length=2, group_size=2,
            kbits=1, vbits=4, promote_ratio=0.5, promote_bit=2,
            promote_ratio_per_layer={0: 0.75}, channel_selection=1,
        )
        cache = get_kvcache_kitty(ns)
        self.assertEqual(cache._layer_pr(0), 0.75)
        self.assertEqual(cache._layer_pr(1), 0.5)


if __name__ == "__main__":
    unittest.main()
