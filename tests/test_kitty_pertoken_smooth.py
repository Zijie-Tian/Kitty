"""Unit tests for the per-token K quant mode and SmoothAttention calibration math.

CPU-only, no model downloads:
  PYTHONPATH=src python -m unittest tests.test_kitty_pertoken_smooth -v
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from kitty_sim.kitty_simulate import KittyKVCacheConfig, get_kvcache_kitty
from kitty_sim.utils_quant import fake_quant_groupwise_lastdim

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "calibrate_smooth_qk", REPO_ROOT / "scripts" / "calibrate_smooth_qk.py"
)
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)


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


class TestTernaryK(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _tern_cache(self, sink=4, buffer=8, group=8):
        args = SimpleNamespace(
            sink_length=sink, buffer_length=buffer, group_size=group,
            kbits=2, vbits=4, promote_ratio=0.0, promote_bit=4,
            channel_selection=0, k_quant_mode="per_channel",
            k_codebook="ternary", k_tern_threshold=0.5, k_tern_submean=True,
        )
        return get_kvcache_kitty(args)

    def test_validate_rejects_bad_combos(self):
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_codebook="ternary", promote_ratio=0.125)
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_codebook="ternary", k_quant_mode="per_token", promote_ratio=0.0)
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_codebook="bogus", promote_ratio=0.0)
        KittyKVCacheConfig(k_codebook="ternary", promote_ratio=0.0)

    def test_prefill_matches_reference_ternary(self):
        from kitty_sim.utils_quant import fake_quant_ternary_lastdim
        sink, buffer, t = 4, 8, 20
        cache = self._tern_cache(sink=sink, buffer=buffer)
        k = torch.randn(1, 2, t, 16)
        v = torch.randn(1, 2, t, 16)
        cache.update(k.clone(), v.clone(), 0)
        stored = cache.key_cache[0]
        # sink + residual fp16 (per-channel path quantizes whole buffer blocks)
        torch.testing.assert_close(stored[:, :, :sink, :], k[:, :, :sink, :])
        expected = fake_quant_ternary_lastdim(
            k[:, :, 4:12, :].transpose(2, 3).contiguous(), 8, 0.5, True
        ).transpose(2, 3)
        torch.testing.assert_close(stored[:, :, 4:12, :], expected)
        # V still uniform 4-bit per-token
        expected_v = fake_quant_groupwise_lastdim(v[:, :, 4:12, :].clone(), 8, 4)
        torch.testing.assert_close(cache.value_cache[0][:, :, 4:12, :], expected_v)


class TestSmoothScales(unittest.TestCase):
    def test_rope_pair_constraint(self):
        absmax = torch.rand(2, 16) * 10 + 0.1
        lam = calib.compute_smooth_scales(absmax, alpha=0.5)
        self.assertEqual(lam.shape, (2, 16))
        torch.testing.assert_close(lam[:, :8], lam[:, 8:])
        expected = torch.maximum(absmax[:, :8], absmax[:, 8:]).pow(0.5)
        torch.testing.assert_close(lam[:, :8], expected)

    def test_zero_channel_is_noop(self):
        absmax = torch.zeros(1, 4)
        lam = calib.compute_smooth_scales(absmax, alpha=0.5)
        torch.testing.assert_close(lam, torch.ones(1, 4))


class TestFoldEquivalence(unittest.TestCase):
    """End-to-end math check on a tiny fp32 Llama: folding lam into W_q/W_k
    must leave logits unchanged and divide the post-RoPE K cache by lam."""

    def _tiny_model(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=128, max_position_embeddings=256, tie_word_embeddings=False,
            attn_implementation="eager",
        )
        torch.manual_seed(42)
        return LlamaForCausalLM(cfg).float().eval()

    def test_fold_preserves_logits_and_scales_keys(self):
        model = self._tiny_model()
        ids = torch.randint(0, 128, (1, 24))
        with torch.no_grad():
            out0 = model(ids, use_cache=True)
        k0 = [calib.cache_layer_keys(out0.past_key_values, li).clone() for li in range(2)]

        scales = {}
        torch.manual_seed(7)
        for li in range(2):
            absmax = torch.rand(2, 16) * 8 + 0.2
            scales[li] = calib.compute_smooth_scales(absmax, alpha=0.5)
        calib.fold_scales(model, scales)

        with torch.no_grad():
            out1 = model(ids, use_cache=True)
        torch.testing.assert_close(out1.logits, out0.logits, atol=1e-4, rtol=1e-4)
        for li in range(2):
            k1 = calib.cache_layer_keys(out1.past_key_values, li)
            lam = scales[li].view(1, 2, 1, 16)
            torch.testing.assert_close(k1, k0[li] / lam, atol=1e-5, rtol=1e-4)

    def test_fold_rejects_qk_norm_arch(self):
        model = self._tiny_model()
        model.model.layers[0].self_attn.k_norm = torch.nn.Identity()
        with self.assertRaises(RuntimeError):
            calib.fold_scales(model, {0: torch.ones(2, 16), 1: torch.ones(2, 16)})


if __name__ == "__main__":
    unittest.main()
