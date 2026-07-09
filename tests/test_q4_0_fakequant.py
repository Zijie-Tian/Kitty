"""Unit tests for the llama.cpp Q4_0 KV-cache fake-quant port.

Oracle: a literal scalar port of ggml-quants.c quantize_row_q4_0_ref +
dequantize_row_q4_0 (incl. the fp16 storage rounding of the scale d), compared
bit-exactly against the vectorized torch implementation. Plus schedule
invariants for the llamacpp_q40 (sink=0, buffer=0, quantize-on-write) and
llamacpp_q40_star (Kitty policy) variants.

CPU-only, no model downloads:
  PYTHONPATH=src python -m unittest tests.test_q4_0_fakequant -v
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from kitty_sim.kitty_simulate import KittyKVCacheConfig, get_kvcache_kitty
from kitty_sim.utils_quant import fake_quant_q4_0_lastdim

QK = 32


def _q4_0_oracle_block(block: np.ndarray) -> np.ndarray:
    """One 32-element block, straight from the C reference (fp32 in/out).

    quantize_row_q4_0_ref: amax scan with strict '>', d = signed_max / -8,
    id = 1/d from the fp32 d, codes MIN(15, (int8_t)(x*id + 8.5)); the stored
    d is fp16. dequantize_row_q4_0: (q - 8) * fp16(d).
    """
    amax, mx = 0.0, 0.0
    for v in block:
        if abs(v) > amax:
            amax, mx = abs(v), v
    d = np.float32(mx) / np.float32(-8.0)
    id_ = np.float32(1.0) / d if d != 0.0 else np.float32(0.0)
    d16 = np.float32(np.float16(d))
    out = np.empty_like(block)
    for j, v in enumerate(block):
        xi = min(15, int(np.float32(v) * id_ + np.float32(8.5)))  # C int8 cast truncates
        out[j] = np.float32(xi - 8) * d16
    return out


def _q4_0_oracle(x: torch.Tensor) -> torch.Tensor:
    flat = x.float().reshape(-1, QK).cpu().numpy().astype(np.float32)
    out = np.stack([_q4_0_oracle_block(b) for b in flat])
    return torch.from_numpy(out.astype(np.float32)).reshape(x.shape).to(x.dtype)


def _q40_cache(sink=0, buffer=0, group=32):
    args = SimpleNamespace(
        sink_length=sink, buffer_length=buffer, group_size=group,
        kbits=4, vbits=4, promote_ratio=0.0, promote_bit=4,
        channel_selection=0, k_quant_mode="per_token",
        k_codebook="q4_0", v_codebook="q4_0",
    )
    return get_kvcache_kitty(args)


class TestQ40FakeQuantCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_c_reference_random(self):
        for shape, scale in (((1, 2, 5, 64), 1.0), ((2, 3, 4, 128), 30.0), ((1, 1, 7, 64), 0.01)):
            x = (torch.randn(*shape) * scale).half()
            got = fake_quant_q4_0_lastdim(x)
            expected = _q4_0_oracle(x)
            torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)

    def test_absmax_element_lands_on_minus8_code(self):
        # The signed absmax element must round-trip to exactly (0-8)*fp16(d) = fp16(max/-8)*-8.
        for sign in (+1.0, -1.0):
            block = torch.zeros(1, QK)
            block[0, 3] = 5.3 * sign
            block[0, 10] = 1.1
            got = fake_quant_q4_0_lastdim(block)
            d16 = torch.tensor(sign * 5.3 / -8.0).half().float()
            self.assertEqual(got[0, 3].item(), (d16 * -8.0).item())
            torch.testing.assert_close(got, _q4_0_oracle(block), atol=0.0, rtol=0.0)

    def test_zero_block_is_zero(self):
        x = torch.zeros(2, 3, QK, dtype=torch.float16)
        got = fake_quant_q4_0_lastdim(x)
        self.assertTrue(torch.equal(got, x))

    def test_truncation_boundary(self):
        # d = 1 exactly (absmax element = -8 -> d = -8/-8 = 1, fp16-exact), so
        # codes are int(x + 8.5): 3.5 -> 12 -> +4; 3.4995 -> 11 -> +3.
        block = torch.zeros(1, QK)
        block[0, 0] = -8.0
        block[0, 1] = 3.5
        block[0, 2] = 3.4995
        block[0, 3] = -3.5
        got = fake_quant_q4_0_lastdim(block)
        self.assertEqual(got[0, 0].item(), -8.0)
        self.assertEqual(got[0, 1].item(), 4.0)
        self.assertEqual(got[0, 2].item(), 3.0)
        self.assertEqual(got[0, 3].item(), -3.0)
        torch.testing.assert_close(got, _q4_0_oracle(block), atol=0.0, rtol=0.0)

    def test_tie_keeps_first_absmax(self):
        # Two elements share |absmax| with opposite signs; C's strict '>' keeps
        # the FIRST, so d's sign follows it. torch argmax must match.
        block = torch.zeros(1, QK)
        block[0, 4] = 6.0
        block[0, 20] = -6.0
        got = fake_quant_q4_0_lastdim(block)
        torch.testing.assert_close(got, _q4_0_oracle(block), atol=0.0, rtol=0.0)
        self.assertEqual(got[0, 4].item(), 6.0)   # first extreme sits on the -8 code exactly

    def test_rejects_bad_head_dim(self):
        with self.assertRaises(AssertionError):
            fake_quant_q4_0_lastdim(torch.randn(1, 2, 3, 48))


class TestQ40ConfigValidation(unittest.TestCase):
    def test_q4_0_requires_per_token(self):
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_codebook="q4_0", k_quant_mode="per_channel")

    def test_buffer0_only_for_per_token(self):
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(buffer_length=0, k_quant_mode="per_channel")
        # per_token + buffer 0 constructs (group/buffer coupling skipped).
        KittyKVCacheConfig(
            sink_length=0, buffer_length=0, group_size=32, promote_ratio=0.0,
            k_quant_mode="per_token", k_codebook="q4_0", v_codebook="q4_0")

    def test_bogus_codebooks_rejected(self):
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(k_codebook="bogus")
        with self.assertRaises(ValueError):
            KittyKVCacheConfig(v_codebook="bogus")


class TestQ40FaithfulSchedule(unittest.TestCase):
    """llamacpp_q40 semantics: quantize-on-write, no sink, no recent window.
    Per-token blocks are independent, so prefill + N decode steps must equal
    one-shot quantization of the full K/V -- bit-identical."""

    def setUp(self):
        torch.manual_seed(1)

    def test_prefill_plus_decode_equals_oneshot(self):
        cache = _q40_cache(sink=0, buffer=0)
        t0, steps, nh, d = 37, 5, 2, 64
        k = torch.randn(1, nh, t0, d).half()
        v = torch.randn(1, nh, t0, d).half()
        ret_k, ret_v = cache.update(k.clone(), v.clone(), 0)
        # PostQuant: the CURRENT step still sees the unquantized values.
        torch.testing.assert_close(ret_k, k)
        torch.testing.assert_close(ret_v, v)
        # The stored prefill region is quantized immediately (no fp16 tail).
        torch.testing.assert_close(
            cache.key_cache[0], fake_quant_q4_0_lastdim(k), atol=0.0, rtol=0.0)
        ks, vs = [k], [v]
        for _ in range(steps):
            k_new = torch.randn(1, nh, 1, d).half()
            v_new = torch.randn(1, nh, 1, d).half()
            cache.update(k_new, v_new, 0)
            ks.append(k_new)
            vs.append(v_new)
        full_k = torch.cat(ks, dim=2)
        full_v = torch.cat(vs, dim=2)
        torch.testing.assert_close(
            cache.key_cache[0], fake_quant_q4_0_lastdim(full_k), atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            cache.value_cache[0], fake_quant_q4_0_lastdim(full_v), atol=0.0, rtol=0.0)

    def test_star_schedule_keeps_sink_and_recent(self):
        sink, buffer, t = 32, 128, 200
        cache = _q40_cache(sink=sink, buffer=buffer, group=32)
        k = torch.randn(1, 2, t, 64).half()
        v = torch.randn(1, 2, t, 64).half()
        cache.update(k.clone(), v.clone(), 0)
        end = sink + (t - sink - buffer)  # 72: quantized region is [sink, end)
        stored_k, stored_v = cache.key_cache[0], cache.value_cache[0]
        torch.testing.assert_close(stored_k[:, :, :sink, :], k[:, :, :sink, :])
        torch.testing.assert_close(stored_k[:, :, end:, :], k[:, :, end:, :])
        torch.testing.assert_close(
            stored_k[:, :, sink:end, :],
            fake_quant_q4_0_lastdim(k[:, :, sink:end, :]), atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            stored_v[:, :, sink:end, :],
            fake_quant_q4_0_lastdim(v[:, :, sink:end, :]), atol=0.0, rtol=0.0)
        torch.testing.assert_close(stored_v[:, :, end:, :], v[:, :, end:, :])


class TestQ40VariantWiring(unittest.TestCase):
    def test_build_variant_and_slugs(self):
        from kitty_sim.longbench.runner import _cache_factory, build_variant, method_layout_slug

        v = build_variant(SimpleNamespace(variant="llamacpp_q40"))
        self.assertEqual(v.k_codebook, "q4_0")
        self.assertEqual(v.v_codebook, "q4_0")
        self.assertEqual(v.k_quant_mode, "per_token")
        self.assertEqual((v.sink_length, v.buffer_length), (0, 0))
        self.assertEqual(method_layout_slug(v), "llamacpp-q40")
        # _cache_factory must pass v_codebook through to the cache object.
        cache = _cache_factory(v)
        self.assertEqual(cache.k_codebook, "q4_0")
        self.assertEqual(cache.v_codebook, "q4_0")

        star = build_variant(SimpleNamespace(variant="llamacpp_q40_star"))
        self.assertEqual((star.sink_length, star.buffer_length), (32, 128))
        self.assertEqual(star.k_codebook, "q4_0")
        self.assertEqual(method_layout_slug(star), "llamacpp-q40-star")


if __name__ == "__main__":
    unittest.main()
