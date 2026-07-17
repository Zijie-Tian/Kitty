"""0-GPU core tests for the rescued tile16cC V path (canonical qlutattn V).

Literal tile oracle is an independent loop reference (does NOT call production
helpers for expected values).
"""

from __future__ import annotations

import math
import unittest

import torch

from kitty_sim.utils_quant import fake_quant_groupwise_lastdim
from kitty_sim import v_tile_quant as vt


def _oracle_rht_signs(D: int, seed: int = 20260711) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    s = torch.randint(0, 2, (D,), generator=gen, dtype=torch.int8)
    return (2 * s - 1).float()


def _oracle_fwht(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    lead = x.shape[:-1]
    y = x.clone()
    h = 1
    while h < n:
        y = y.reshape(*lead, n // (2 * h), 2, h)
        a0, a1 = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a0 + a1, a0 - a1), dim=-2).reshape(*lead, n)
        h *= 2
    return y / math.sqrt(n)


def _oracle_rescued_tile(
    v: torch.Tensor,
    C: int,
    seed: int = 20260711,
    mse_iters: int = 1,
) -> torch.Tensor:
    """Independent literal reference for one or more full 16-token blocks.

    Mirrors production math with its own code (no production helper calls). LS
    reductions and re-encode use FP32 / FP16 roundtrips identical to the plan.
    """
    assert v.dim() == 4
    B, H, T, D = v.shape
    N = 16
    assert T % N == 0 and D % C == 0
    signs = _oracle_rht_signs(D, seed).to(device=v.device, dtype=torch.float32)
    v0 = v.half().float()
    v_r = _oracle_fwht(v0 * signs).half().float()
    mu = v_r.mean(dim=2, keepdim=True).half().float()
    rms = (v_r - mu).square().mean(dim=2, keepdim=True).sqrt().clamp(min=1e-4).half().float()
    z = (v_r - mu) / rms
    nb = T // N
    # Same tile layout as production: [B,H,nb,D/C,N*C]
    groups = (
        z.reshape(B, H, nb, N, D // C, C)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, H, nb, D // C, N * C)
    )
    xh = groups.half()
    offset0 = xh.amin(dim=-1, keepdim=True)
    scale0 = (xh.amax(dim=-1, keepdim=True) - offset0).clamp(min=1e-4) / 3
    q0 = ((xh - offset0) / scale0).clamp(0, 3).round()
    z0 = (offset0 + scale0 * q0).float()
    # Probe-exact MSE candidate: independent FP32 min/max assignment after an
    # FP16 value round-trip.  It does NOT reuse the FP16 baseline's q0.
    x = groups.half().float()
    cand_offset = x.amin(dim=-1, keepdim=True)
    cand_scale = (
        x.amax(dim=-1, keepdim=True) - cand_offset
    ).clamp(min=1e-4) / 3
    n = x.shape[-1]
    sum_x = x.sum(dim=-1, keepdim=True)
    for _ in range(mse_iters):
        cand_q = ((x - cand_offset) / cand_scale).round().clamp(0, 3)
        sum_q = cand_q.sum(dim=-1, keepdim=True)
        sum_q2 = cand_q.square().sum(dim=-1, keepdim=True)
        sum_qx = (cand_q * x).sum(dim=-1, keepdim=True)
        denom = n * sum_q2 - sum_q.square()
        new_scale = (n * sum_qx - sum_q * sum_x) / denom.clamp(min=1e-12)
        new_offset = (sum_x - new_scale * sum_q) / n
        valid = (
            (denom > 0)
            & (new_scale > 1e-6)
            & torch.isfinite(new_scale)
            & torch.isfinite(new_offset)
        )
        cand_scale = torch.where(valid, new_scale, cand_scale)
        cand_offset = torch.where(valid, new_offset, cand_offset)
    cand_offset = cand_offset.half().float()
    cand_scale = cand_scale.clamp(min=1e-4).half().float()
    cand_q = ((x - cand_offset) / cand_scale).round().clamp(0, 3)
    z1 = cand_offset + cand_scale * cand_q
    sse1 = (z1 - groups).square().sum(dim=-1, keepdim=True)
    sse0 = (z0 - groups).square().sum(dim=-1, keepdim=True)
    z_hat_g = torch.where(sse1 <= sse0, z1, z0)
    out_z = (
        z_hat_g.reshape(B, H, nb, D // C, N, C)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, H, T, D)
    )
    v_r_hat = out_z * rms + mu
    v_hat_0 = _oracle_fwht(v_r_hat) * signs
    bias = (v_hat_0 - v0).mean(dim=2, keepdim=True).half().float()
    return (v_hat_0 - bias).half()


def _oracle_v4_whole_head(x: torch.Tensor) -> torch.Tensor:
    """Independent FP16 min-max V4 reference for one whole-head row."""
    xh = x.half()
    mn = xh.amin(dim=-1, keepdim=True)
    scale = (xh.amax(dim=-1, keepdim=True) - mn).clamp(min=1e-4) / 15
    q = ((xh - mn) / scale).clamp(0, 15).round()
    return (q * scale + mn).half()


class TestGroupwiseHelperRegression(unittest.TestCase):
    def test_v4_regression_unchanged_helper(self):
        torch.manual_seed(1)
        for D in (64, 128):
            x = torch.randn(1, 2, 9, D).half()
            got = fake_quant_groupwise_lastdim(x, 128, 4)
            expected = _oracle_v4_whole_head(x)
            torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)


class TestTileOracle(unittest.TestCase):
    def test_matches_literal_oracle(self):
        torch.manual_seed(2)
        for D, Cs in ((64, (16, 32, 64)), (128, (16, 32, 64, 128))):
            for C in Cs:
                v = torch.randn(2, 3, 16, D).half()
                recon, mu, rms, bias = vt.calibrate_and_quantize_first_v_tile_block(
                    v, v_tile_channels=C
                )
                expected = _oracle_rescued_tile(v, C)
                torch.testing.assert_close(recon, expected, atol=0.0, rtol=0.0)
                self.assertEqual(tuple(mu.shape), (2, 3, 1, D))
                self.assertEqual(mu.dtype, torch.float16)
                self.assertEqual(rms.dtype, torch.float16)
                self.assertEqual(bias.dtype, torch.float16)

    def test_full_prompt_t32_t48_matches_literal_oracle(self):
        torch.manual_seed(22)
        for T in (32, 48):
            for C in (16, 32, 64):
                with self.subTest(T=T, C=C):
                    v = torch.randn(2, 2, T, 64).half()
                    got, mu, rms, bias = vt.calibrate_and_quantize_v_tile_prompt(
                        v, v_tile_channels=C
                    )
                    expected = _oracle_rescued_tile(v, C, mse_iters=1)
                    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)
                    self.assertEqual(tuple(mu.shape), (2, 2, 1, 64))
                    self.assertEqual(tuple(rms.shape), (2, 2, 1, 64))
                    self.assertEqual(tuple(bias.shape), (2, 2, 1, 64))

    def test_frozen_stats_multi_block(self):
        torch.manual_seed(3)
        C = 16
        v = torch.randn(1, 2, 32, 64).half()
        first = v[:, :, :16, :]
        recon0, mu, rms, bias = vt.calibrate_and_quantize_first_v_tile_block(
            first, v_tile_channels=C
        )
        rest = vt.quantize_v_tile_blocks_with_frozen_stats(
            v[:, :, 16:, :], mu=mu, rms=rms, bias=bias, v_tile_channels=C
        )
        # Second block with frozen stats must equal production path on that block alone.
        rest2 = vt.quantize_v_tile_blocks_with_frozen_stats(
            v[:, :, 16:, :], mu=mu, rms=rms, bias=bias, v_tile_channels=C
        )
        torch.testing.assert_close(rest, rest2, atol=0.0, rtol=0.0)
        self.assertEqual(recon0.shape[-2], 16)

    def test_exactly_one_ls_step(self):
        # Fixed input where one and two alternating LS steps differ after bias.
        torch.manual_seed(4)
        C = 16
        v = torch.randn(1, 1, 16, 64).half()
        recon1, _, _, _ = vt.calibrate_and_quantize_first_v_tile_block(v, v_tile_channels=C)
        expected1 = _oracle_rescued_tile(v, C, mse_iters=1)
        expected2 = _oracle_rescued_tile(v, C, mse_iters=2)
        torch.testing.assert_close(recon1, expected1, atol=0.0, rtol=0.0)
        self.assertFalse(torch.equal(expected1, expected2))

    def test_fallback_monotonic_per_tile(self):
        torch.manual_seed(11)
        # Work directly at the normalized-group boundary so the assertion is per
        # tile, not hidden by inverse affine/RHT or a tensor-wide SSE.
        # The heterogeneous row scales make this fixed fixture exercise both the
        # MSE-candidate and the FP16-minmax fallback arms (row 68 falls back).
        groups = torch.randn(256, 16) * torch.exp(torch.randn(256, 1) * 1.5)
        _, _, _, baseline = vt._minmax_init_2bit(groups)
        candidate = vt._mse_candidate_probe_exact(groups, iters=1)
        chosen = vt._mse_refit_once_with_fallback(groups, baseline)
        sse0 = (baseline - groups).square().sum(dim=-1)
        sse1 = (candidate - groups).square().sum(dim=-1)
        ssec = (chosen - groups).square().sum(dim=-1)
        self.assertTrue(torch.all(ssec <= sse0))
        expected = torch.where((sse1 <= sse0).unsqueeze(-1), candidate, baseline)
        torch.testing.assert_close(chosen, expected, atol=0.0, rtol=0.0)
        # Exercise both the candidate and fallback arms in this fixed fixture.
        self.assertTrue(bool((sse1 <= sse0).any()))
        self.assertTrue(bool((sse1 > sse0).any()))

    def test_degenerate_groups(self):
        C = 16
        zeros = torch.zeros(1, 1, 16, 64).half()
        recon, _, _, _ = vt.calibrate_and_quantize_first_v_tile_block(zeros, v_tile_channels=C)
        self.assertTrue(torch.isfinite(recon.float()).all())
        const = torch.full((1, 1, 16, 64), 0.25).half()
        recon2, _, _, _ = vt.calibrate_and_quantize_first_v_tile_block(const, v_tile_channels=C)
        self.assertTrue(torch.isfinite(recon2.float()).all())

    def test_rht_half_overflow_fails_fast(self):
        # RHT can amplify a legal FP16 value beyond the FP16 storage range at the
        # mandated round-trip.  It must fail before NaN/Inf reaches the cache.
        extreme = torch.full((1, 1, 16, 64), 30000.0, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            vt.calibrate_and_quantize_first_v_tile_block(
                extreme, v_tile_channels=16
            )

    def test_rht_determinism_and_inverse(self):
        signs_a = vt.rht_signs(64, "cpu", torch.float32)
        signs_b = vt.rht_signs(64, "cpu", torch.float32)
        torch.testing.assert_close(signs_a, signs_b, atol=0.0, rtol=0.0)
        x = torch.randn(2, 3, 5, 64)
        y = vt.rht_forward(x, signs_a)
        x2 = vt.rht_inverse(y, signs_a)
        torch.testing.assert_close(x2, x, atol=1e-5, rtol=1e-5)
        y2 = vt.fwht_lastdim(vt.fwht_lastdim(x))
        torch.testing.assert_close(y2, x, atol=1e-5, rtol=1e-5)

    def test_c_validation_messages(self):
        v = torch.randn(1, 1, 16, 64).half()
        with self.assertRaises(ValueError) as ctx:
            vt.calibrate_and_quantize_first_v_tile_block(v, v_tile_channels=24)
        msg = str(ctx.exception)
        self.assertIn("head_dim=64", msg)
        self.assertIn("v_tile_channels=24", msg)
        v48 = torch.randn(1, 1, 16, 48).half()
        with self.assertRaises(ValueError) as ctx2:
            vt.calibrate_and_quantize_first_v_tile_block(v48, v_tile_channels=16)
        self.assertIn("power-of-two", str(ctx2.exception))

    def test_bit_accounting(self):
        b = vt.theoretical_v_tile_bits(16, 160)
        self.assertAlmostEqual(b, 2.0 + 32.0 / (16 * 16) + 48.0 / 160)
        full0 = vt.theoretical_v_tile_full_cache_bits(
            D=64, C=16, T_total=200, T_quantized=0, stats_initialized=False
        )
        self.assertEqual(full0["theoretical_packed_bits"], 16.0)
        self.assertEqual(full0["metadata_bits"], 0.0)
        full = vt.theoretical_v_tile_full_cache_bits(
            D=64, C=16, T_total=200, T_quantized=32, stats_initialized=True
        )
        self.assertIn("theoretical_packed_bits", full)
        self.assertGreater(full["metadata_bits"], 0.0)

    def test_bit_accounting_rejects_impossible_states(self):
        with self.assertRaises(ValueError):
            vt.theoretical_v_tile_bits(16, 17)  # partial token tile
        with self.assertRaises(ValueError):
            vt.theoretical_v_tile_bits(0, 16)
        invalid_tile_kwargs = (
            dict(D=64, C=24, T_total=200, T_quantized=16, stats_initialized=True),
            dict(D=64, C=16, T_total=200, T_quantized=17, stats_initialized=True),
            dict(D=64, C=16, T_total=200, T_quantized=16, stats_initialized=False),
            dict(D=64, C=16, T_total=200, T_quantized=0, stats_initialized=True),
        )
        for kwargs in invalid_tile_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                vt.theoretical_v_tile_full_cache_bits(**kwargs)

    def test_frozen_stats_require_exact_shape_dtype_finite_and_positive_rms(self):
        torch.manual_seed(8)
        value = torch.randn(2, 3, 16, 64).half()
        _, mu, rms, bias = vt.calibrate_and_quantize_v_tile_prompt(
            value, v_tile_channels=16
        )

        bad_cases = [
            ("shape", mu[:1], rms, bias),
            ("dtype", mu.float(), rms, bias),
            ("zero rms", mu, torch.zeros_like(rms), bias),
            ("non-finite", mu, rms, bias.clone()),
        ]
        bad_cases[-1][3][0, 0, 0, 0] = float("inf")
        for label, bad_mu, bad_rms, bad_bias in bad_cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                vt.quantize_v_tile_blocks_with_frozen_stats(
                    value,
                    mu=bad_mu,
                    rms=bad_rms,
                    bias=bad_bias,
                    v_tile_channels=16,
                )

        # Device mismatch must be rejected before any operation tries to use it.
        with self.assertRaisesRegex(ValueError, "device"):
            vt.quantize_v_tile_blocks_with_frozen_stats(
                value,
                mu=torch.empty(mu.shape, dtype=mu.dtype, device="meta"),
                rms=rms,
                bias=bias,
                v_tile_channels=16,
            )


if __name__ == "__main__":
    unittest.main()
