import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import kitty.kvcache.kitty as kitty_module
except Exception as exc:  # pragma: no cover - depends on optional runtime deps
    kitty_module = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class FakeLayer:
    calls = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        FakeLayer.calls.append(kwargs)


def _config(*, head_dim: int = 64, layers: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=layers,
        head_dim=head_dim,
        num_key_value_heads=1,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"Kitty runtime import failed: {_IMPORT_ERROR}")
class KittyPromoteRatioTests(unittest.TestCase):
    def setUp(self):
        FakeLayer.calls = []

    def _new_cache(self, **kwargs):
        assert kitty_module is not None
        with patch.object(kitty_module, "KVCache_Layer", FakeLayer):
            return kitty_module.get_kvcache_kitty(
                _config(head_dim=kwargs.pop("head_dim", 64)),
                max_batch_size=1,
                max_length=256,
                **kwargs,
            )

    def test_default_promote_ratio_matches_paper_kitty(self):
        cache = self._new_cache(page_size=128)

        self.assertEqual(cache.page_size, 128)
        self.assertEqual(cache.promote_ratio, 0.125)
        self.assertEqual(cache.d_boosted, int(64 * 0.125 + 1e-6))
        self.assertEqual(len(FakeLayer.calls), 2)
        self.assertTrue(all(call["D_BOOSTED"] == 8 for call in FakeLayer.calls))
        self.assertTrue(all(call["PAGE_SIZE"] == 128 for call in FakeLayer.calls))

    def test_explicit_quarter_promote_ratio_enables_kitty_pro(self):
        cache = self._new_cache(page_size=16, promote_ratio=0.25)

        self.assertEqual(cache.page_size, 16)
        self.assertEqual(cache.promote_ratio, 0.25)
        self.assertEqual(cache.d_boosted, int(64 * 0.25 + 1e-6))
        self.assertTrue(all(call["D_BOOSTED"] == 16 for call in FakeLayer.calls))
        self.assertTrue(all(call["PAGE_SIZE"] == 16 for call in FakeLayer.calls))

    def test_d_boosted_uses_floor_formula_from_kitty_sim(self):
        cache = self._new_cache(head_dim=127, promote_ratio=0.125)

        self.assertEqual(cache.d_boosted, int(127 * 0.125 + 1e-6))
        self.assertEqual(FakeLayer.calls[0]["D_BOOSTED"], 15)

    def test_invalid_promote_ratio_is_rejected_before_layer_allocation(self):
        with self.assertRaisesRegex(ValueError, "promote_ratio"):
            self._new_cache(promote_ratio=1.25)

        self.assertEqual(FakeLayer.calls, [])


if __name__ == "__main__":
    unittest.main()
