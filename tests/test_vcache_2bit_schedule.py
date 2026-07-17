"""0-GPU schedule tests for the rescued V tile16cC cache integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from kitty_sim.kitty_simulate import KittyKVCacheConfig, get_kvcache_kitty
from kitty_sim.utils_quant import fake_quant_groupwise_lastdim
from kitty_sim.v_tile_quant import (
    V_MSE_ITERS,
    V_RHT_SEED,
    V_TILE_ALGO_VERSION,
    V_TILE_TOKENS,
    calibrate_and_quantize_first_v_tile_block,
    calibrate_and_quantize_v_tile_prompt,
    quantize_v_tile_blocks_with_frozen_stats,
)


def _tile_cache(sink=32, recent=128, C=16, D=64, layers=1):
    """Rescued tile16cC V cache with a generic per-token K (the canonical
    qlutattn masked-K numerics are covered by tests/test_qlutattn_wiring.py
    and tests/test_nf2_symmetric.py; here the subject is the V schedule)."""
    args = SimpleNamespace(
        sink_length=sink,
        buffer_length=recent,
        group_size=min(128, recent) if recent > 0 else 128,
        kbits=2,
        vbits=2,
        promote_ratio=0.0,
        promote_bit=4,
        channel_selection=0,
        k_quant_mode="per_token",
        k_codebook="kivi",
        v_codebook="tile16_rescued",
        v_tile_tokens=V_TILE_TOKENS,
        v_tile_channels=C,
        v_tile_algo_version=V_TILE_ALGO_VERSION,
        v_rht_seed=V_RHT_SEED,
        v_mse_iters=V_MSE_ITERS,
    )
    return get_kvcache_kitty(args)


def _legacy_v2_cache(sink=4, recent=8, group=8):
    """Historical kivi V2 path: configured grouping must remain unchanged."""
    args = SimpleNamespace(
        sink_length=sink,
        buffer_length=recent,
        group_size=group,
        kbits=2,
        vbits=2,
        promote_ratio=0.0,
        promote_bit=4,
        channel_selection=0,
        k_quant_mode="per_token",
        k_codebook="kivi",
        v_codebook="kivi",
    )
    return get_kvcache_kitty(args)


class TestVTileSchedule(unittest.TestCase):
    def test_short_prefill_no_tile(self):
        sink, recent = 4, 32
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 15  # 51: settled=15 < 16
        k = torch.randn(1, 2, T, 64).half()
        v = torch.randn(1, 2, T, 64).half()
        v_before = v.clone()
        cache.update(k, v, 0)
        self.assertNotIn(0, cache.v_pc_mean)
        self.assertEqual(cache.v_tile_blocks, 0)
        # Settled region still fp16 (identical to input clone path after PostQuant mutate).
        torch.testing.assert_close(
            cache.value_cache[0][:, :, sink:sink + 15, :],
            v_before[:, :, sink:sink + 15, :],
            atol=0.0,
            rtol=0.0,
        )

    def test_prefill_full_blocks_only(self):
        sink, recent = 4, 32
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        # settled = 37 -> two full tiles (32) + 5 pending
        T = sink + recent + 37
        torch.manual_seed(10)
        k = torch.randn(1, 2, T, 64).half()
        v = torch.randn(1, 2, T, 64).half()
        v_orig = v.clone()
        cache.update(k, v, 0)
        self.assertEqual(cache.v_tile_quant_end[0], sink + 32)
        self.assertEqual(cache.v_tile_blocks, 2)
        self.assertEqual(cache.last_v_quant_mode, "tile16_rescued")
        # pending 5 settled tokens remain fp16
        torch.testing.assert_close(
            cache.value_cache[0][:, :, sink + 32:sink + 37, :],
            v_orig[:, :, sink + 32:sink + 37, :],
            atol=0.0,
            rtol=0.0,
        )
        # sink untouched
        torch.testing.assert_close(
            cache.value_cache[0][:, :, :sink, :],
            v_orig[:, :, :sink, :],
            atol=0.0,
            rtol=0.0,
        )

    def test_decode_flush_and_postquant(self):
        sink, recent = 4, 32
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 37
        torch.manual_seed(11)
        k = torch.randn(1, 2, T, 64).half()
        v = torch.randn(1, 2, T, 64).half()
        k_ret, v_ret = cache.update(k, v, 0)
        # PostQuant: returned V is pre-mutation
        self.assertTrue(torch.equal(v_ret, v))
        qend = cache.v_tile_quant_end[0]
        # 10 more decode steps: still no new tile (need 11 to fill pending 5 -> 16)
        for i in range(10):
            ki = torch.randn(1, 2, 1, 64).half()
            vi = torch.randn(1, 2, 1, 64).half()
            cache.update(ki, vi, 0)
        self.assertEqual(cache.v_tile_quant_end[0], qend)
        # 11th step flushes one tile
        cache.update(torch.randn(1, 2, 1, 64).half(), torch.randn(1, 2, 1, 64).half(), 0)
        self.assertEqual(cache.v_tile_quant_end[0], qend + 16)
        self.assertEqual(cache.v_tile_blocks, 3)

    def test_reset_clears_state(self):
        cache = _tile_cache(sink=4, recent=32, C=16)
        T = 4 + 32 + 37
        cache.update(torch.randn(1, 1, T, 64).half(), torch.randn(1, 1, T, 64).half(), 0)
        self.assertGreater(cache.v_tile_blocks, 0)
        cache.reset()
        self.assertEqual(cache.v_tile_blocks, 0)
        self.assertEqual(len(cache.v_pc_mean), 0)
        self.assertEqual(len(cache.v_pc_rms), 0)
        self.assertEqual(len(cache.v_error_bias), 0)
        self.assertEqual(len(cache.v_tile_quant_end), 0)
        self.assertEqual(len(cache.k_pt_quant_end), 0)
        self.assertEqual(cache.v_quant_calls, 0)
        self.assertEqual(cache.v_quantized_tokens, 0)
        self.assertIsNone(cache.last_v_quant_mode)
        self.assertIsNone(cache.last_v_tile_channels)
        self.assertEqual(len(cache.value_cache), 0)

    def test_no_requant_old_blocks(self):
        sink, recent = 4, 32
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 37
        torch.manual_seed(12)
        k = torch.randn(1, 1, T, 64).half()
        v = torch.randn(1, 1, T, 64).half()
        cache.update(k, v, 0)
        snap = cache.value_cache[0][:, :, sink:sink + 32, :].clone()
        for _ in range(20):
            cache.update(torch.randn(1, 1, 1, 64).half(), torch.randn(1, 1, 1, 64).half(), 0)
        torch.testing.assert_close(
            cache.value_cache[0][:, :, sink:sink + 32, :], snap, atol=0.0, rtol=0.0
        )

    def test_lazy_calibration(self):
        sink, recent = 4, 32
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 15
        cache.update(torch.randn(1, 1, T, 64).half(), torch.randn(1, 1, T, 64).half(), 0)
        self.assertFalse(cache._v_tile_stats_ready(0))
        # One more token -> settled=16 -> calibrate
        cache.update(torch.randn(1, 1, 1, 64).half(), torch.randn(1, 1, 1, 64).half(), 0)
        self.assertTrue(cache._v_tile_stats_ready(0))
        self.assertEqual(cache.v_tile_quant_end[0], sink + 16)
        self.assertEqual(cache.v_tile_blocks, 1)

    def test_crop_mid_tile_failfast(self):
        sink, recent = 4, 32
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 37
        cache.update(torch.randn(1, 1, T, 64).half(), torch.randn(1, 1, T, 64).half(), 0)
        k_before = cache.key_cache[0].clone()
        v_before = cache.value_cache[0].clone()
        qend_before = dict(cache.v_tile_quant_end)
        mu_before = {i: x.clone() for i, x in cache.v_pc_mean.items()}
        rms_before = {i: x.clone() for i, x in cache.v_pc_rms.items()}
        bias_before = {i: x.clone() for i, x in cache.v_error_bias.items()}
        with self.assertRaises(ValueError):
            cache.crop(sink + 8)  # mid-tile into quantized region
        torch.testing.assert_close(cache.key_cache[0], k_before, atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.value_cache[0], v_before, atol=0.0, rtol=0.0)
        self.assertEqual(cache.v_tile_quant_end, qend_before)
        for i in mu_before:
            torch.testing.assert_close(cache.v_pc_mean[i], mu_before[i], atol=0.0, rtol=0.0)
            torch.testing.assert_close(cache.v_pc_rms[i], rms_before[i], atol=0.0, rtol=0.0)
            torch.testing.assert_close(cache.v_error_bias[i], bias_before[i], atol=0.0, rtol=0.0)

    def test_multi_token_append_decode(self):
        sink, recent = 4, 32
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 5
        cache.update(torch.randn(1, 1, T, 64).half(), torch.randn(1, 1, T, 64).half(), 0)
        # Append 20 tokens at once -> should flush at least one full tile
        cache.update(torch.randn(1, 1, 20, 64).half(), torch.randn(1, 1, 20, 64).half(), 0)
        self.assertGreaterEqual(cache.v_tile_blocks, 1)
        self.assertTrue(cache._v_tile_stats_ready(0))

    def test_prefill_t32_t48_uses_full_prompt_oracle_vectorized(self):
        sink, recent = 4, 32
        for settled in (32, 48):
            for C in (16, 32, 64):
                with self.subTest(settled=settled, C=C):
                    torch.manual_seed(1000 + settled + C)
                    T = sink + recent + settled
                    k = torch.randn(1, 2, T, 64).half()
                    v = torch.randn(1, 2, T, 64).half()
                    cache = _tile_cache(sink=sink, recent=recent, C=C)
                    cache.update(k, v, 0)
                    expected, mu, rms, bias = calibrate_and_quantize_v_tile_prompt(
                        v[:, :, sink:sink + settled, :],
                        v_tile_channels=C,
                    )
                    torch.testing.assert_close(
                        cache.value_cache[0][:, :, sink:sink + settled, :],
                        expected,
                        atol=0.0,
                        rtol=0.0,
                    )
                    torch.testing.assert_close(cache.v_pc_mean[0], mu, atol=0.0, rtol=0.0)
                    torch.testing.assert_close(cache.v_pc_rms[0], rms, atol=0.0, rtol=0.0)
                    torch.testing.assert_close(cache.v_error_bias[0], bias, atol=0.0, rtol=0.0)
                    # Full prefill is one vectorized quantizer invocation.
                    self.assertEqual(cache.v_quant_calls, 1)
                    self.assertEqual(cache.v_tile_blocks, settled // V_TILE_TOKENS)
                    torch.testing.assert_close(
                        cache.value_cache[0][:, :, -recent:, :],
                        v[:, :, -recent:, :],
                        atol=0.0,
                        rtol=0.0,
                    )

    def test_legacy_kivi_v2_keeps_configured_group_size(self):
        torch.manual_seed(21)
        cache = _legacy_v2_cache(group=8)
        x = torch.randn(1, 2, 3, 16).half()
        got = cache._quant_v(0, x)
        expected = fake_quant_groupwise_lastdim(x, 8, 2)
        whole_head = fake_quant_groupwise_lastdim(x, x.shape[-1], 2)
        torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)
        self.assertFalse(torch.equal(got, whole_head))
        self.assertIsNone(cache.last_v_quant_mode)

    def test_tile_multi_token_append_exact_and_postquant(self):
        sink, recent = 4, 32
        torch.manual_seed(22)
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 5
        k0 = torch.randn(1, 1, T, 64).half()
        v0 = torch.randn(1, 1, T, 64).half()
        cache.update(k0, v0, 0)
        k1 = torch.randn(1, 1, 27, 64).half()
        v1 = torch.randn(1, 1, 27, 64).half()
        _, returned_v = cache.update(k1, v1, 0)
        raw = torch.cat((v0, v1), dim=2)

        first, mu, rms, bias = calibrate_and_quantize_first_v_tile_block(
            raw[:, :, sink:sink + 16, :], v_tile_channels=16
        )
        second = quantize_v_tile_blocks_with_frozen_stats(
            raw[:, :, sink + 16:sink + 32, :],
            mu=mu,
            rms=rms,
            bias=bias,
            v_tile_channels=16,
        )
        expected = torch.cat((first, second), dim=2)
        # PostQuant returns the just-settled region before mutation.
        torch.testing.assert_close(
            returned_v[:, :, sink:sink + 32, :],
            raw[:, :, sink:sink + 32, :],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            cache.value_cache[0][:, :, sink:sink + 32, :],
            expected,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(cache.v_tile_quant_end[0], sink + 32)
        self.assertEqual(cache.v_tile_blocks, 2)
        self.assertEqual(cache.k_pt_quant_end[0], sink + 32)

    def test_per_token_k_multi_token_append_is_exact(self):
        sink, recent, D = 4, 8, 16
        torch.manual_seed(230)
        cache = _legacy_v2_cache(sink=sink, recent=recent, group=8)
        T = sink + recent + 5
        k0 = torch.randn(1, 1, T, D).half()
        v0 = torch.randn(1, 1, T, D).half()
        cache.update(k0, v0, 0)
        old_qend = cache.k_pt_quant_end[0]
        k1 = torch.randn(1, 1, 7, D).half()
        v1 = torch.randn(1, 1, 7, D).half()
        returned_k, _ = cache.update(k1, v1, 0)
        raw = torch.cat((k0, k1), dim=2)
        ready_end = raw.shape[2] - recent
        expected = fake_quant_groupwise_lastdim(
            raw[:, :, old_qend:ready_end, :], 8, 2
        )
        torch.testing.assert_close(
            returned_k[:, :, old_qend:ready_end, :],
            raw[:, :, old_qend:ready_end, :],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            cache.key_cache[0][:, :, old_qend:ready_end, :],
            expected,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(cache.k_pt_quant_end[0], ready_end)

    def test_decode_boundary_postquant_returns_unquantized_tile(self):
        sink, recent = 4, 32
        torch.manual_seed(24)
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 37
        k0 = torch.randn(1, 1, T, 64).half()
        v0 = torch.randn(1, 1, T, 64).half()
        cache.update(k0, v0, 0)
        qend = cache.v_tile_quant_end[0]
        raw = v0.clone()
        for _ in range(10):
            ki = torch.randn(1, 1, 1, 64).half()
            vi = torch.randn(1, 1, 1, 64).half()
            cache.update(ki, vi, 0)
            raw = torch.cat((raw, vi), dim=2)
        ki = torch.randn(1, 1, 1, 64).half()
        vi = torch.randn(1, 1, 1, 64).half()
        raw = torch.cat((raw, vi), dim=2)
        _, returned_v = cache.update(ki, vi, 0)
        expected = quantize_v_tile_blocks_with_frozen_stats(
            raw[:, :, qend:qend + 16, :],
            mu=cache.v_pc_mean[0],
            rms=cache.v_pc_rms[0],
            bias=cache.v_error_bias[0],
            v_tile_channels=16,
        )
        torch.testing.assert_close(
            returned_v[:, :, qend:qend + 16, :],
            raw[:, :, qend:qend + 16, :],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            cache.value_cache[0][:, :, qend:qend + 16, :],
            expected,
            atol=0.0,
            rtol=0.0,
        )

    def test_reset_prompt_b_matches_fresh_cache(self):
        sink, recent = 4, 32
        torch.manual_seed(25)
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        T = sink + recent + 32
        cache.update(torch.randn(1, 1, T, 64).half(), torch.randn(1, 1, T, 64).half(), 0)
        cache.reset()
        kb = torch.randn(1, 1, T, 64).half()
        vb = torch.randn(1, 1, T, 64).half()
        cache.update(kb, vb, 0)
        fresh = _tile_cache(sink=sink, recent=recent, C=16)
        fresh.update(kb, vb, 0)
        torch.testing.assert_close(cache.value_cache[0], fresh.value_cache[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_mean[0], fresh.v_pc_mean[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_rms[0], fresh.v_pc_rms[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_error_bias[0], fresh.v_error_bias[0], atol=0.0, rtol=0.0)
        self.assertEqual(cache.v_tile_quant_end, fresh.v_tile_quant_end)
        self.assertEqual(cache.v_tile_blocks, fresh.v_tile_blocks)

    def test_multilayer_state_is_independent(self):
        sink, recent, settled = 4, 32, 32
        T = sink + recent + settled
        torch.manual_seed(26)
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        k0 = torch.randn(1, 1, T, 64).half()
        v0 = torch.randn(1, 1, T, 64).half()
        k1 = torch.randn(1, 1, T, 64).half()
        v1 = (torch.randn(1, 1, T, 64) * 3 + 2).half()
        cache.update(k0, v0, 0)
        cache.update(k1, v1, 1)
        self.assertEqual(cache.v_tile_quant_end, {0: sink + settled, 1: sink + settled})
        self.assertFalse(torch.equal(cache.v_pc_mean[0], cache.v_pc_mean[1]))
        exp0, mu0, _, _ = calibrate_and_quantize_v_tile_prompt(
            v0[:, :, sink:sink + settled, :], v_tile_channels=16
        )
        exp1, mu1, _, _ = calibrate_and_quantize_v_tile_prompt(
            v1[:, :, sink:sink + settled, :], v_tile_channels=16
        )
        torch.testing.assert_close(cache.value_cache[0][:, :, sink:sink + settled], exp0, atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.value_cache[1][:, :, sink:sink + settled], exp1, atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_mean[0], mu0, atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_mean[1], mu1, atol=0.0, rtol=0.0)

    def test_batch_reorder_repeat_and_select_transform_stats(self):
        sink, recent, settled = 4, 32, 32
        T = sink + recent + settled
        torch.manual_seed(27)
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        k = torch.randn(2, 1, T, 64).half()
        v = torch.randn(2, 1, T, 64).half()
        cache.update(k, v, 0)
        value0 = cache.value_cache[0].clone()
        mu0 = cache.v_pc_mean[0].clone()
        rms0 = cache.v_pc_rms[0].clone()
        bias0 = cache.v_error_bias[0].clone()
        order = torch.tensor([1, 0])
        cache.reorder_cache(order)
        torch.testing.assert_close(cache.value_cache[0], value0.index_select(0, order), atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_mean[0], mu0.index_select(0, order), atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_rms[0], rms0.index_select(0, order), atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_error_bias[0], bias0.index_select(0, order), atol=0.0, rtol=0.0)
        value1 = cache.value_cache[0].clone()
        mu1 = cache.v_pc_mean[0].clone()
        cache.batch_repeat_interleave(2)
        torch.testing.assert_close(cache.value_cache[0], value1.repeat_interleave(2, 0), atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_mean[0], mu1.repeat_interleave(2, 0), atol=0.0, rtol=0.0)
        value2 = cache.value_cache[0].clone()
        mu2 = cache.v_pc_mean[0].clone()
        select = torch.tensor([3, 0])
        cache.batch_select_indices(select)
        torch.testing.assert_close(cache.value_cache[0], value2.index_select(0, select), atol=0.0, rtol=0.0)
        torch.testing.assert_close(cache.v_pc_mean[0], mu2.index_select(0, select), atol=0.0, rtol=0.0)

    def test_aligned_crop_updates_all_quant_pointers(self):
        sink, recent = 4, 32
        T = sink + recent + 37
        cache = _tile_cache(sink=sink, recent=recent, C=16)
        cache.update(torch.randn(1, 1, T, 64).half(), torch.randn(1, 1, T, 64).half(), 0)
        self.assertGreater(cache.k_pt_quant_end[0], sink + 16)
        cache.crop(sink + 16)
        self.assertEqual(cache.get_seq_length(), sink + 16)
        self.assertEqual(cache.v_tile_quant_end[0], sink + 16)
        self.assertEqual(cache.k_pt_quant_end[0], sink + 16)
        self.assertTrue(cache._v_tile_stats_ready(0))


class TestV2ConfigStrictness(unittest.TestCase):
    def _tile_kwargs(self):
        return dict(
            promote_ratio=0.0,
            channel_selection=0,
            k_quant_mode="per_token",
            k_codebook="kivi",
            v_codebook="tile16_rescued",
            vbits=2,
            v_tile_tokens=V_TILE_TOKENS,
            v_tile_channels=16,
            v_tile_algo_version=V_TILE_ALGO_VERSION,
            v_rht_seed=V_RHT_SEED,
            v_mse_iters=V_MSE_ITERS,
        )

    def test_rv1_fields_are_strict(self):
        KittyKVCacheConfig(**self._tile_kwargs())
        for key, bad in (
            ("vbits", 4),
            ("v_tile_tokens", 8),
            ("v_tile_algo_version", None),
            ("v_rht_seed", V_RHT_SEED + 1),
            ("v_mse_iters", 2),
        ):
            with self.subTest(key=key):
                kwargs = self._tile_kwargs()
                kwargs[key] = bad
                with self.assertRaises(ValueError):
                    KittyKVCacheConfig(**kwargs)

    def test_tile_rejects_non_fp16_cache_tensor(self):
        cache = _tile_cache(sink=4, recent=32, C=16)
        T = 4 + 32 + 16
        with self.assertRaisesRegex(ValueError, "FP16"):
            cache.update(torch.randn(1, 1, T, 64), torch.randn(1, 1, T, 64), 0)


if __name__ == "__main__":
    unittest.main()
