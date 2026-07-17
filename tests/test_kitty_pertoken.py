"""Unit tests for the generic per-token K quant schedule.

CPU-only, no model downloads:
  PYTHONPATH=src python -m unittest tests.test_kitty_pertoken -v
"""

import unittest
from types import SimpleNamespace

import torch

from kitty_sim.kitty_simulate import KittyKVCacheConfig, get_kvcache_kitty
from kitty_sim.utils_quant import fake_quant_groupwise_lastdim


def _pertoken_cache(sink=4, buffer=8, group=8, kbits=2, vbits=4):
    args = SimpleNamespace(
        sink_length=sink, buffer_length=buffer, group_size=group,
        kbits=kbits, vbits=vbits, promote_ratio=0.0, promote_bit=4,
        channel_selection=0, k_quant_mode="per_token",
    )
    return get_kvcache_kitty(args)


class TestPerTokenKQuant(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_validate_rejects_promote_with_per_token(self):
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_quant_mode="per_token", promote_ratio=0.125)
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_quant_mode="per_token", promote_ratio=0.0,
                               promote_ratio_per_layer={0: 0.5})
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_quant_mode="bogus", promote_ratio=0.0)
        # Valid configuration constructs fine.
        KittyKVCacheConfig(k_quant_mode="per_token", promote_ratio=0.0)

    def test_prefill_window_and_axis(self):
        sink, buffer, t = 4, 8, 20
        cache = _pertoken_cache(sink=sink, buffer=buffer)
        k = torch.randn(1, 2, t, 16)
        v = torch.randn(1, 2, t, 16)
        cache.update(k.clone(), v.clone(), 0)
        stored_k = cache.key_cache[0]
        # Sink and the most recent buffer tokens stay fp16.
        torch.testing.assert_close(stored_k[:, :, :sink, :], k[:, :, :sink, :])
        torch.testing.assert_close(stored_k[:, :, t - buffer:, :], k[:, :, t - buffer:, :])
        # The middle window is quantized along head_dim, exactly like the V path.
        expected = fake_quant_groupwise_lastdim(k[:, :, sink : t - buffer, :].clone(), 8, 2)
        torch.testing.assert_close(stored_k[:, :, sink : t - buffer, :], expected)
        self.assertFalse(torch.allclose(stored_k[:, :, sink : t - buffer, :],
                                        k[:, :, sink : t - buffer, :]))
        # V window matches its own (vbits=4) quantization, unchanged behavior.
        expected_v = fake_quant_groupwise_lastdim(v[:, :, sink : t - buffer, :].clone(), 8, 4)
        torch.testing.assert_close(cache.value_cache[0][:, :, sink : t - buffer, :], expected_v)

    def test_decode_quantizes_sliding_token(self):
        sink, buffer, t = 4, 8, 20
        cache = _pertoken_cache(sink=sink, buffer=buffer)
        k = torch.randn(1, 2, t, 16)
        v = torch.randn(1, 2, t, 16)
        cache.update(k.clone(), v.clone(), 0)
        # After prefill, token index t - buffer = 12 is still fp16.
        torch.testing.assert_close(cache.key_cache[0][:, :, 12, :], k[:, :, 12, :])
        k_new = torch.randn(1, 2, 1, 16)
        v_new = torch.randn(1, 2, 1, 16)
        cache.update(k_new, v_new, 0)
        # Decode step quantizes exactly the token sliding out of the window.
        expected = fake_quant_groupwise_lastdim(k[:, :, 12:13, :].clone(), 8, 2)
        torch.testing.assert_close(cache.key_cache[0][:, :, 12:13, :], expected)
        # Newest token and the rest of the recent window remain fp16.
        torch.testing.assert_close(cache.key_cache[0][:, :, 13:t, :], k[:, :, 13:, :])
        torch.testing.assert_close(cache.key_cache[0][:, :, t:, :], k_new)

    def test_per_channel_mode_untouched(self):
        """Default mode still quantizes along the token axis (regression guard)."""
        args = SimpleNamespace(
            sink_length=4, buffer_length=8, group_size=8, kbits=2, vbits=2,
            promote_ratio=0.0, promote_bit=4, channel_selection=0,
            k_quant_mode="per_channel",
        )
        cache = get_kvcache_kitty(args)
        k = torch.randn(1, 2, 20, 16)
        cache.update(k.clone(), k.clone(), 0)
        ks = k[:, :, 4:12, :].transpose(2, 3).contiguous()
        from kitty_sim.utils_quant import build_promote_mask
        mask = build_promote_mask(ks, 0.0, 0)
        expected = fake_quant_groupwise_lastdim(ks, 8, 2, mask, 4).transpose(2, 3)
        torch.testing.assert_close(cache.key_cache[0][:, :, 4:12, :], expected)


if __name__ == "__main__":
    unittest.main()
