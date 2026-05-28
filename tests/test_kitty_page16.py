import math
import unittest
from types import SimpleNamespace

import torch

try:
    from kitty.kvcache import get_kvcache_kitty
    from kitty.kvcache.kernels.kitty_attention import kitty_attention_forward
except Exception as exc:  # pragma: no cover - exercised only on missing optional runtime deps
    get_kvcache_kitty = None
    kitty_attention_forward = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=1,
        head_dim=16,
        num_key_value_heads=1,
    )


def _skip_reason() -> str | None:
    if _IMPORT_ERROR is not None:
        return f"real Kitty runtime import failed: {_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return "CUDA is required for real Kitty cache tests"
    return None


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class RealKittyPage16Tests(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def _new_cache(self, *, max_length: int = 96, page_size: int = 16):
        assert get_kvcache_kitty is not None
        return get_kvcache_kitty(
            _tiny_config(),
            max_batch_size=1,
            max_length=max_length,
            page_size=page_size,
        )

    def _random_kv(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = torch.randn(1, 1, length, 16, dtype=torch.float16, device="cuda")
        value = torch.randn(1, 1, length, 16, dtype=torch.float16, device="cuda")
        return key, value

    def test_default_page_size_remains_128_and_page16_is_opt_in(self):
        default_cache = self._new_cache(max_length=96, page_size=128)
        default_layer = default_cache.kv_cache[0]
        self.assertEqual(default_cache.page_size, 128)
        self.assertEqual(default_layer.PAGE_SIZE, 128)
        self.assertEqual(default_layer.MAX_PAGE, math.ceil(96 / 128))

        page16_cache = self._new_cache(max_length=96, page_size=16)
        page16_layer = page16_cache.kv_cache[0]
        self.assertEqual(page16_cache.page_size, 16)
        self.assertEqual(page16_layer.PAGE_SIZE, 16)
        self.assertEqual(page16_layer.MAX_PAGE, math.ceil(96 / 16))
        self.assertEqual(page16_layer.Q_Buffer_K.shape, (1, 1, 16, 16))
        self.assertEqual(page16_layer.Local_Buffer_V.shape, (1, 1, 16, 16))
        self.assertEqual(page16_cache.promote_ratio, 0.125)
        self.assertEqual(page16_cache.d_boosted, 2)
        self.assertEqual(page16_layer.D_BOOSTED, 2)
        self.assertEqual(page16_layer.bytes_per_page_K, 88)
        self.assertEqual(page16_layer.bytes_per_page_V, 64)

        pro_cache = get_kvcache_kitty(
            _tiny_config(),
            max_batch_size=1,
            max_length=96,
            page_size=16,
            promote_ratio=0.25,
        )
        pro_layer = pro_cache.kv_cache[0]
        self.assertEqual(pro_cache.promote_ratio, 0.25)
        self.assertEqual(pro_cache.d_boosted, 4)
        self.assertEqual(pro_layer.D_BOOSTED, 4)
        self.assertEqual(pro_layer.bytes_per_page_K, 96)
        self.assertEqual(pro_layer.bytes_per_page_V, 64)
        self.assertEqual(page16_cache.max_batch_size, 1)
        self.assertEqual(page16_cache.max_cache_len, 96)
        self.assertEqual(page16_cache.get_max_cache_shape(), 96)
        self.assertEqual(page16_cache.get_mask_sizes(torch.arange(5, device="cuda"), 0), (5, 0))

    def test_real_kittycache_promote_ratio_defaults_to_0p125(self):
        cache = self._new_cache(max_length=96, page_size=16)
        self.assertEqual(cache.promote_ratio, 0.125)
        self.assertEqual(cache.d_boosted, int(16 * 0.125 + 1e-6))
        self.assertEqual(cache.kv_cache[0].D_BOOSTED, cache.d_boosted)

    def test_real_kittycache_kitty_pro_promote_ratio_0p25(self):
        assert get_kvcache_kitty is not None
        cache = get_kvcache_kitty(
            _tiny_config(),
            max_batch_size=1,
            max_length=96,
            page_size=16,
            promote_ratio=0.25,
        )
        self.assertEqual(cache.promote_ratio, 0.25)
        self.assertEqual(cache.d_boosted, int(16 * 0.25 + 1e-6))
        self.assertEqual(cache.kv_cache[0].D_BOOSTED, cache.d_boosted)

    def test_invalid_promote_ratio_is_rejected(self):
        assert get_kvcache_kitty is not None
        with self.assertRaisesRegex(ValueError, "promote_ratio"):
            get_kvcache_kitty(_tiny_config(), max_batch_size=1, max_length=96, promote_ratio=1.5)

    def test_invalid_page_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of 4"):
            self._new_cache(page_size=3)

    def test_page16_prefill_boundary_accounting(self):
        sink = 32
        page_size = 16
        for length in [sink + 15, sink + 16, sink + 17, sink + 31, sink + 32, sink + 33]:
            cache = self._new_cache(max_length=128, page_size=page_size)
            layer = cache.kv_cache[0]
            key, value = self._random_kv(length)

            self.assertTrue(cache.update(key, value, layer_idx=0))
            cache.quantize_prefill(layer_idx=0)
            torch.cuda.synchronize()

            tail = length - sink
            expected_q_k = tail % page_size
            expected_pages_k = (tail - expected_q_k) // page_size
            expected_local_v = min(page_size, tail)
            expected_q_v = (tail - expected_local_v) % page_size
            expected_pages_v = (tail - expected_local_v - expected_q_v) // page_size

            self.assertEqual(layer.Sink_Count, sink)
            self.assertEqual(layer.PageCount_K, expected_pages_k)
            self.assertEqual(layer.Q_Buffer_Count_K, expected_q_k)
            self.assertEqual(layer.PageCount_V, expected_pages_v)
            self.assertEqual(layer.Q_Buffer_Count_V, expected_q_v)
            self.assertEqual(layer.Local_Count_V, expected_local_v)
            self.assertEqual(cache.get_seq_length(0), length)

    def test_page16_decode_rollover_and_local_wrap(self):
        sink = 32
        page_size = 16
        cache = self._new_cache(max_length=96, page_size=page_size)
        layer = cache.kv_cache[0]
        key, value = self._random_kv(sink)

        self.assertTrue(cache.update(key, value, layer_idx=0))
        cache.quantize_prefill(layer_idx=0)

        for _ in range(2 * page_size + 2):
            key, value = self._random_kv(1)
            self.assertFalse(cache.update(key, value, layer_idx=0))
            cache.quantize_decode(layer_idx=0)
        torch.cuda.synchronize()

        self.assertEqual(layer.PageCount_K, 2)
        self.assertEqual(layer.Q_Buffer_Count_K, 2)
        self.assertEqual(layer.PageCount_V, 1)
        self.assertEqual(layer.Q_Buffer_Count_V, 2)
        self.assertEqual(layer.Local_Count_V, page_size)
        self.assertEqual(layer.Write_Offset_Local_V, 2)
        self.assertEqual(cache.get_seq_length(0), sink + 2 * page_size + 2)

    def test_page16_attention_forward_smoke(self):
        assert kitty_attention_forward is not None
        cache = self._new_cache(max_length=96, page_size=16)
        key, value = self._random_kv(65)
        self.assertTrue(cache.update(key, value, layer_idx=0))
        cache.quantize_prefill(layer_idx=0)

        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        output, attn_weights = kitty_attention_forward(
            module,
            query,
            cache.kv_cache[0],
            scaling=16 ** -0.5,
        )
        torch.cuda.synchronize()

        self.assertIsNone(attn_weights)
        self.assertEqual(output.shape, (1, 1, 1, 16))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
