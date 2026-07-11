"""Standalone proof for accumulate-then-restore RHT V-cache attention.

This file is intentionally independent of ``kitty_sim``.  It implements an
RV1-inspired, CPU-only *ideal affine packed* model of the rescued V-cache path:

    V -> fixed RHT -> prompt mean/RMS -> affine tile 2-bit codes
      -> inverse affine/RHT -> reconstruction-bias correction.

It then evaluates the same quantized cache in two ways:

1. restore every cached V token to the original basis, then compute ``P @ V``;
2. accumulate the packed affine 2-bit codes in the normalized RHT basis, then
   apply mean/RMS, inverse RHT and bias exactly once per query/head.

The tests prove that the two orders are algebraically equivalent for such a
packed representation (up to normal floating-point reduction order), including
the harder case where sink/recent tokens remain FP16.  This is a feasibility
proof, not a bit-exact replay of today's dense RV1 fake-quant helper: production
uses mixed FP16/FP32 arithmetic for its fallback and materializes every restored
token through a final FP16 roundtrip.  Run directly with:

    python tests/test_vcache_rht_accumulate_then_restore.py

No model, dataset, CUDA device, or project import is required.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import torch


TILE_TOKENS = 16
RHT_SEED = 20260711
EPS = 1e-4


def _fp16_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Store as FP16, then continue the standalone oracle in FP64."""

    return x.to(torch.float16).to(torch.float64)


def _fwht_normalized(x: torch.Tensor) -> torch.Tensor:
    """Normalized, self-inverse Walsh-Hadamard transform on the last axis."""

    d = x.shape[-1]
    if d <= 0 or d & (d - 1):
        raise ValueError(f"RHT requires power-of-two head_dim, got D={d}")
    lead = x.shape[:-1]
    y = x.clone()
    width = 1
    while width < d:
        y = y.reshape(*lead, d // (2 * width), 2, width)
        lo, hi = y[..., 0, :], y[..., 1, :]
        y = torch.stack((lo + hi, lo - hi), dim=-2).reshape(*lead, d)
        width *= 2
    return y / math.sqrt(d)


def _explicit_hadamard(d: int) -> torch.Tensor:
    """Dense normalized Sylvester matrix used only as an independent oracle."""

    if d <= 0 or d & (d - 1):
        raise ValueError(f"Hadamard order must be a power of two, got D={d}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < d:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return h / math.sqrt(d)


def _rht_signs(d: int, *, device: torch.device) -> torch.Tensor:
    """Seed-stable Rademacher signs matching the rescued-V design."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(RHT_SEED)
    signs = torch.randint(0, 2, (d,), generator=generator, dtype=torch.int8)
    return (2 * signs - 1).to(device=device, dtype=torch.float64)


def _rht_forward(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    # Code convention: multiply by random signs first, then normalized FWHT.
    return _fwht_normalized(x * signs)


def _rht_inverse(y: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    # H_n is self-inverse and signs**2 == 1.
    return _fwht_normalized(y) * signs


@dataclass(frozen=True)
class PackedAffine2Bit:
    """A logical affine-code representation, unlike a dense fake-quant cache.

    ``codes`` has logical shape [B,H,G,M,16,C], where G is the token-tile
    count and M=D/C is the channel-tile count.  Each [16,C] tile owns one FP16
    offset and scale.  Codes are stored in uint8 here for clarity, although a
    real kernel would pack four 2-bit codes per byte.
    """

    codes: torch.Tensor  # uint8 [B,H,G,M,16,C]
    offset: torch.Tensor  # FP16-roundtripped FP64 [B,H,G,M,1]
    scale: torch.Tensor  # FP16-roundtripped FP64 [B,H,G,M,1]
    token_count: int
    head_dim: int
    channel_tile: int


@dataclass(frozen=True)
class RescuedVState:
    """Prompt statistics and both dense/packed views of one quantized region."""

    original_fp16: torch.Tensor
    signs: torch.Tensor
    mean: torch.Tensor
    rms: torch.Tensor
    bias: torch.Tensor
    packed: PackedAffine2Bit
    z_hat: torch.Tensor
    restored_unrounded: torch.Tensor
    restored_fp16: torch.Tensor


def _reshape_to_groups(z: torch.Tensor, c: int) -> torch.Tensor:
    """[B,H,T,D] -> [B,H,T/16,D/C,16,C]."""

    b, h, t, d = z.shape
    if t <= 0 or t % TILE_TOKENS:
        raise ValueError(f"T must be a positive multiple of 16, got T={t}")
    if c <= 0 or d % c:
        raise ValueError(f"channel tile C={c} must divide head_dim D={d}")
    groups = t // TILE_TOKENS
    return (
        z.reshape(b, h, groups, TILE_TOKENS, d // c, c)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )


def _groups_to_tensor(groups: torch.Tensor, *, t: int, d: int) -> torch.Tensor:
    """Inverse layout transform for [B,H,G,M,16,C]."""

    b, h, g, m, n, c = groups.shape
    if n != TILE_TOKENS or g * n != t or m * c != d:
        raise ValueError("invalid grouped V-cache shape")
    return (
        groups.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(b, h, t, d)
    )


def _pack_affine_2bit_mse1(z: torch.Tensor, c: int) -> PackedAffine2Bit:
    """RV1-inspired affine 2-bit tile encoder with one LS refit.

    The exact quantizer is not needed for the linearity proof, but using
    per-tile FP16 offset/scale, four codes, one least-squares refit and a
    min-max fallback prevents the test from proving only a trivial global-scale
    special case.
    """

    b, h, t, d = z.shape
    groups = _reshape_to_groups(z, c)
    x = _fp16_roundtrip(groups)

    # FP16-side-information min-max baseline.
    offset0 = _fp16_roundtrip(x.amin(dim=(-2, -1), keepdim=True))
    range0 = (x.amax(dim=(-2, -1), keepdim=True) - offset0).clamp(min=EPS)
    scale0 = _fp16_roundtrip(range0 / 3.0)
    q0 = ((x - offset0) / scale0).round().clamp(0, 3)
    recon0 = offset0 + scale0 * q0

    # Exactly one alternating assignment / closed-form affine LS refit.
    offset = x.amin(dim=(-2, -1), keepdim=True)
    scale = (x.amax(dim=(-2, -1), keepdim=True) - offset).clamp(min=EPS) / 3.0
    q = ((x - offset) / scale).round().clamp(0, 3)
    n = TILE_TOKENS * c
    sum_x = x.sum(dim=(-2, -1), keepdim=True)
    sum_q = q.sum(dim=(-2, -1), keepdim=True)
    sum_q2 = q.square().sum(dim=(-2, -1), keepdim=True)
    sum_qx = (q * x).sum(dim=(-2, -1), keepdim=True)
    denom = n * sum_q2 - sum_q.square()
    new_scale = (n * sum_qx - sum_q * sum_x) / denom.clamp(min=1e-12)
    new_offset = (sum_x - new_scale * sum_q) / float(n)
    valid = (
        (denom > 0)
        & (new_scale > 1e-6)
        & torch.isfinite(new_scale)
        & torch.isfinite(new_offset)
    )
    offset1 = _fp16_roundtrip(torch.where(valid, new_offset, offset))
    scale1 = _fp16_roundtrip(
        torch.where(valid, new_scale, scale).clamp(min=EPS)
    )
    q1 = ((x - offset1) / scale1).round().clamp(0, 3)
    recon1 = offset1 + scale1 * q1

    # Fallback is decided independently for every [16,C] tile.
    sse0 = (recon0 - groups).square().sum(dim=(-2, -1), keepdim=True)
    sse1 = (recon1 - groups).square().sum(dim=(-2, -1), keepdim=True)
    use_refit = sse1 <= sse0
    codes = torch.where(use_refit, q1, q0).to(torch.uint8)
    offset_final = torch.where(use_refit, offset1, offset0).squeeze(-2)
    scale_final = torch.where(use_refit, scale1, scale0).squeeze(-2)

    return PackedAffine2Bit(
        codes=codes,
        offset=offset_final,
        scale=scale_final,
        token_count=t,
        head_dim=d,
        channel_tile=c,
    )


def _dequantize_packed(packed: PackedAffine2Bit) -> torch.Tensor:
    """Return normalized-RHT values [B,H,T,D] from packed tile metadata."""

    groups = (
        packed.offset.unsqueeze(-2)
        + packed.scale.unsqueeze(-2) * packed.codes.to(torch.float64)
    )
    return _groups_to_tensor(
        groups, t=packed.token_count, d=packed.head_dim
    )


def _build_rescued_state(v: torch.Tensor, c: int) -> RescuedVState:
    """Calibrate one prompt region and produce packed plus dense reconstructions."""

    if v.dim() != 4:
        raise ValueError(f"expected [B,H,T,D], got {tuple(v.shape)}")
    original = _fp16_roundtrip(v)
    signs = _rht_signs(v.shape[-1], device=v.device)
    y = _fp16_roundtrip(_rht_forward(original, signs))
    mean = _fp16_roundtrip(y.mean(dim=2, keepdim=True))
    rms = _fp16_roundtrip(
        (y - mean).square().mean(dim=2, keepdim=True).sqrt().clamp(min=EPS)
    )
    z = (y - mean) / rms
    packed = _pack_affine_2bit_mse1(z, c)
    z_hat = _dequantize_packed(packed)
    y_hat = mean + rms * z_hat
    v_hat0 = _rht_inverse(y_hat, signs)
    bias = _fp16_roundtrip((v_hat0 - original).mean(dim=2, keepdim=True))
    restored = v_hat0 - bias
    return RescuedVState(
        original_fp16=original,
        signs=signs,
        mean=mean,
        rms=rms,
        bias=bias,
        packed=packed,
        z_hat=z_hat,
        restored_unrounded=restored,
        restored_fp16=_fp16_roundtrip(restored),
    )


def _attention_weights(
    *, b: int, h: int, q: int, t: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scores = torch.randn(b, h, q, t, generator=generator, dtype=torch.float64)
    return scores.softmax(dim=-1)


def _weighted_sum(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Reference attention value reduction: [B,H,Q,T] @ [B,H,T,D]."""

    return torch.einsum("bhqt,bhtd->bhqd", weights, values)


def _accumulate_packed_normalized(
    weights: torch.Tensor, packed: PackedAffine2Bit
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse P @ dequant(int2) without restoring individual V tokens.

    For token tile g, broadcast its channel-group parameters to vectors a_g and
    s_g, and write its codes as Q_g.  The block-matrix identity is:

        Zhat_g = 1 a_g^T + Q_g diag(s_g)
        p_g Zhat_g
          = (p_g 1) a_g^T + (p_g Q_g) diag(s_g)
          = alpha_g a_g + s_g * (p_g Q_g).

    Equivalently, for tile g and channel block m:

        sum_t p_t (offset_gm + scale_gm * q_td)
          = offset_gm * alpha_g + scale_gm * sum_t p_t q_td.

    Returns normalized-RHT accumulation [B,H,Q,D] and total probability mass
    [B,H,Q,1].  ``weights`` may also be unnormalized softmax-numerator weights.
    Blockwise local-max merging/rescaling is deliberately outside this algebra
    test and would need a separate kernel-level oracle.
    """

    b, h, queries, t = weights.shape
    codes = packed.codes.to(torch.float64)
    pb, ph, groups, channel_groups, n, c = codes.shape
    if (b, h, t) != (pb, ph, packed.token_count) or n != TILE_TOKENS:
        raise ValueError("attention weights and packed V-cache shapes do not match")
    grouped_weights = weights.reshape(b, h, queries, groups, TILE_TOKENS)
    code_acc = torch.einsum(
        "bhqgn,bhgmnc->bhqgmc", grouped_weights, codes
    )
    tile_mass = grouped_weights.sum(dim=-1, keepdim=True).unsqueeze(-1)
    offset = packed.offset[:, :, None, :, :, :]
    scale = packed.scale[:, :, None, :, :, :]
    normalized_groups = offset * tile_mass + scale * code_acc
    normalized = normalized_groups.sum(dim=3).reshape(
        b, h, queries, channel_groups * c
    )
    total_mass = weights.sum(dim=-1, keepdim=True)
    return normalized, total_mass


def _restore_after_accumulation(
    normalized_acc: torch.Tensor,
    probability_mass: torch.Tensor,
    state: RescuedVState,
) -> torch.Tensor:
    """Apply prompt affine, inverse RHT and bias once per query/head."""

    y_acc = state.rms * normalized_acc + state.mean * probability_mass
    return (
        _rht_inverse(y_acc, state.signs)
        - state.bias * probability_mass
    )


def _all_quantized_outputs(
    *, c: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, RescuedVState, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    v = torch.randn(2, 3, 48, 64, generator=generator, dtype=torch.float64)
    state = _build_rescued_state(v, c)
    weights = _attention_weights(b=2, h=3, q=5, t=48, seed=seed + 100)
    restore_first = _weighted_sum(weights, state.restored_unrounded)
    normalized_acc, mass = _accumulate_packed_normalized(weights, state.packed)
    accumulate_first = _restore_after_accumulation(normalized_acc, mass, state)
    return restore_first, accumulate_first, state, weights


class TestRHTAccumulateThenRestore(unittest.TestCase):
    def test_fwht_matches_independent_dense_matrix_and_rht_order(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        x = torch.randn(3, 5, 64, generator=generator, dtype=torch.float64)
        h = _explicit_hadamard(64)
        signs = _rht_signs(64, device=x.device)
        torch.testing.assert_close(
            _fwht_normalized(x), x @ h, atol=2e-12, rtol=2e-12
        )
        torch.testing.assert_close(
            _rht_forward(x, signs), (x * signs) @ h, atol=2e-12, rtol=2e-12
        )
        torch.testing.assert_close(
            _rht_inverse(x, signs), (x @ h) * signs, atol=2e-12, rtol=2e-12
        )
        # The normalized Hadamard is self-inverse; randomized S*H generally is
        # not.  Forward must therefore not be reused as the inverse.
        self.assertGreater(
            float((_rht_forward(_rht_forward(x, signs), signs) - x).abs().max()),
            1e-2,
        )

    def test_rht_roundtrip_and_energy(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1)
        x = torch.randn(4, 7, 64, generator=generator, dtype=torch.float64)
        signs = _rht_signs(64, device=x.device)
        y = _rht_forward(x, signs)
        restored = _rht_inverse(y, signs)
        torch.testing.assert_close(restored, x, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
            y.square().sum(dim=-1),
            x.square().sum(dim=-1),
            atol=2e-12,
            rtol=2e-12,
        )

    def test_packed_accumulator_matches_explicit_dequantization(self):
        # This separately checks the important offset*mass + scale*code_sum
        # identity over several tiles with different affine parameters.
        for c in (16, 32, 64):
            with self.subTest(channel_tile=c):
                _, _, state, weights = _all_quantized_outputs(c=c, seed=2 + c)
                self.assertEqual(state.packed.codes.dtype, torch.uint8)
                self.assertGreaterEqual(int(state.packed.codes.min()), 0)
                self.assertLessEqual(int(state.packed.codes.max()), 3)
                expected_side_shape = (*state.packed.codes.shape[:4], 1)
                self.assertEqual(tuple(state.packed.offset.shape), expected_side_shape)
                self.assertEqual(tuple(state.packed.scale.shape), expected_side_shape)
                self.assertTrue(bool((state.packed.scale > 0).all()))
                torch.testing.assert_close(
                    state.packed.offset,
                    _fp16_roundtrip(state.packed.offset),
                    atol=0.0,
                    rtol=0.0,
                )
                torch.testing.assert_close(
                    state.packed.scale,
                    _fp16_roundtrip(state.packed.scale),
                    atol=0.0,
                    rtol=0.0,
                )
                for metadata in (state.mean, state.rms, state.bias):
                    self.assertEqual(tuple(metadata.shape), (2, 3, 1, 64))
                    torch.testing.assert_close(
                        metadata, _fp16_roundtrip(metadata), atol=0.0, rtol=0.0
                    )
                self.assertGreater(torch.unique(state.packed.scale).numel(), 1)
                self.assertGreater(torch.unique(state.packed.offset).numel(), 1)
                explicit = _weighted_sum(weights, state.z_hat)
                fused, mass = _accumulate_packed_normalized(weights, state.packed)
                torch.testing.assert_close(fused, explicit, atol=2e-12, rtol=2e-12)
                torch.testing.assert_close(
                    mass,
                    torch.ones_like(mass),
                    atol=2e-12,
                    rtol=2e-12,
                )

    def test_restore_each_token_equals_restore_after_accumulation(self):
        for c in (16, 32, 64):
            with self.subTest(channel_tile=c):
                before, after, _, _ = _all_quantized_outputs(c=c, seed=10 + c)
                torch.testing.assert_close(before, after, atol=3e-12, rtol=3e-12)

    def test_fp16_sink_recent_plus_quantized_middle(self):
        # Real policy shape: sink32 | quantized 48 | recent128.
        sink, quantized, recent = 32, 48, 128
        total = sink + quantized + recent
        generator = torch.Generator(device="cpu")
        generator.manual_seed(30)
        full_v = _fp16_roundtrip(
            torch.randn(1, 2, total, 64, generator=generator, dtype=torch.float64)
        )
        q0, q1 = sink, sink + quantized
        state = _build_rescued_state(full_v[:, :, q0:q1, :], c=64)
        self.assertGreater(float(state.bias.abs().max()), 0.0)
        weights = _attention_weights(b=1, h=2, q=4, t=total, seed=31)
        # Query 0 models a causal/masked row with zero mass on the quantized
        # middle; the remaining rows retain mixed FP16/quantized attention.
        weights[:, :, 0, q0:q1] = 0.0
        weights[:, :, 0, :] /= weights[:, :, 0, :].sum(dim=-1, keepdim=True)

        # Algebraic reference before the current cache's final FP16 storage
        # roundtrip: inverse every quantized token first.
        effective_v = full_v.clone()
        effective_v[:, :, q0:q1, :] = state.restored_unrounded
        restore_first = _weighted_sum(weights, effective_v)

        # Proposed packed-kernel semantics: keep only codes in the middle,
        # accumulate there, inverse once, then add the original-basis FP16 arms.
        middle_weights = weights[:, :, :, q0:q1]
        normalized_acc, quantized_mass = _accumulate_packed_normalized(
            middle_weights, state.packed
        )
        quantized_out = _restore_after_accumulation(
            normalized_acc, quantized_mass, state
        )
        fp16_out = _weighted_sum(weights[:, :, :, :q0], full_v[:, :, :q0, :])
        fp16_out += _weighted_sum(weights[:, :, :, q1:], full_v[:, :, q1:, :])
        accumulate_first = quantized_out + fp16_out

        torch.testing.assert_close(
            restore_first, accumulate_first, atol=3e-12, rtol=3e-12
        )
        # The middle is only a subset, so mu and bias must be weighted by its
        # probability mass rather than being added once as if mass==1.
        self.assertTrue(bool((quantized_mass < 1.0).all()))
        torch.testing.assert_close(
            quantized_mass[:, :, 0, :],
            torch.zeros_like(quantized_mass[:, :, 0, :]),
            atol=0.0,
            rtol=0.0,
        )
        self.assertTrue(bool((quantized_mass[:, :, 1:, :] > 0.0).all()))
        wrong = _restore_after_accumulation(
            normalized_acc, torch.ones_like(quantized_mass), state
        ) + fp16_out
        self.assertGreater(float((wrong - restore_first).abs().max()), 1e-3)

    def test_unnormalized_softmax_numerator_accumulation(self):
        # A fused attention epilogue may receive exp(score-global_max)
        # numerators and divide by the softmax denominator only afterward.  This
        # does not claim to test blockwise local-max online-softmax merging.
        _, _, state, _ = _all_quantized_outputs(c=64, seed=40)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(41)
        scores = torch.randn(2, 3, 5, 48, generator=generator, dtype=torch.float64)
        unnormalized = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        denominator = unnormalized.sum(dim=-1, keepdim=True)
        probabilities = unnormalized / denominator

        reference = _weighted_sum(probabilities, state.restored_unrounded)
        z_numerator, mass_numerator = _accumulate_packed_normalized(
            unnormalized, state.packed
        )
        # Unnormalized-numerator epilogue: restore once, then divide by the
        # shared denominator.
        after = _restore_after_accumulation(
            z_numerator, mass_numerator, state
        ) / denominator
        torch.testing.assert_close(reference, after, atol=3e-12, rtol=3e-12)

    def test_final_per_token_fp16_roundtrip_is_close_but_not_bit_exact(self):
        # The algebraic proof uses the unrounded reconstructed value.  Today's
        # dense fake-quant path performs a final per-token FP16 roundtrip before
        # P@V.  Within this oracle the measured delta isolates that extra
        # roundtrip; production's mixed arithmetic and a future kernel's chosen
        # accumulation precision remain separate compatibility questions.
        before, after, state, weights = _all_quantized_outputs(c=64, seed=50)
        dense_fp16_result = _weighted_sum(weights, state.restored_fp16)
        observed = (dense_fp16_result - after).abs()
        roundtrip_error = float(observed.max())
        per_token_rounding = (
            state.restored_fp16 - state.restored_unrounded
        ).abs()
        triangle_bound = _weighted_sum(weights, per_token_rounding)
        self.assertGreater(roundtrip_error, 0.0)
        # |sum p_t e_t| <= sum p_t |e_t| for non-negative softmax weights.
        self.assertTrue(bool((observed <= triangle_bound + 2e-12).all()))
        self.assertLess(float(triangle_bound.max()), 2e-3)
        # Sanity: the ideal restore-first path remains the exact algebraic target.
        torch.testing.assert_close(before, after, atol=3e-12, rtol=3e-12)


def _print_demo() -> None:
    before, after, state, weights = _all_quantized_outputs(c=64, seed=123)
    fp16_dense = _weighted_sum(weights, state.restored_fp16)
    print("RHT V-cache accumulate-then-restore standalone demo")
    print(f"  algebraic max_abs_error : {(before - after).abs().max().item():.3e}")
    print(f"  dense-FP16 order delta  : {(fp16_dense - after).abs().max().item():.3e}")
    print("  result                  : algebraically valid for this ideal packed model")


if __name__ == "__main__":
    _print_demo()
    unittest.main(verbosity=2)
