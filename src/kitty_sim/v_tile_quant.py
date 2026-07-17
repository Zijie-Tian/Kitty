"""Rescued V-cache tile16cC 2-bit fake-quant (rht-pcaff-mse1-bias-v1).

Production V path for the canonical qlutattn variant (C=64). Algorithm
documented in docs/qlutattn.md. Do NOT import probe scripts under .claude/.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

V_TILE_TOKENS = 16
V_TILE_ALGO_VERSION = "rht-pcaff-mse1-bias-v1"
V_TILE_ALGO_SLUG = "rv1"
V_RHT_SEED = 20260711
V_MSE_ITERS = 1
_EPS = 1e-4
_LEVELS = 3  # 2-bit: codes 0..3


def _require_positive_int(name: str, value: int) -> int:
    """Return ``value`` as an int, rejecting bool/non-integral/invalid inputs."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_finite(name: str, value: torch.Tensor) -> None:
    """Fail before a non-finite intermediate can be written into the KV cache."""
    if not bool(torch.isfinite(value).all()):
        raise ValueError(
            f"rescued V-tile quantization produced non-finite values at {name}; "
            "refusing to write NaN/Inf into the cache"
        )


def rht_signs(
    D: int,
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int = V_RHT_SEED,
) -> torch.Tensor:
    """Fixed Rademacher signs from a CPU generator (seed-stable across devices)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    s_int8 = torch.randint(0, 2, (D,), generator=gen, dtype=torch.int8)
    return (2 * s_int8 - 1).to(device=device, dtype=dtype)


def fwht_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Normalized self-inverse Walsh-Hadamard transform on the last dim."""
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(
            f"Hadamard/RHT requires power-of-two head_dim, got head_dim={n}"
        )
    lead = x.shape[:-1]
    y = x
    h = 1
    while h < n:
        y = y.reshape(*lead, n // (2 * h), 2, h)
        a0, a1 = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a0 + a1, a0 - a1), dim=-2).reshape(*lead, n)
        h *= 2
    return y / math.sqrt(n)


def rht_forward(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """R(V) = H_norm(V * s)."""
    return fwht_lastdim(x * signs)


def rht_inverse(y: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """R_inverse(Y) = H_norm(Y) * s."""
    return fwht_lastdim(y) * signs


def theoretical_v_tile_bits(C: int, T_quantized: int) -> float:
    """Theoretical packed V bit/value in the quantized region (tile + metadata)."""
    C = _require_positive_int("C", C)
    T_quantized = _require_positive_int("T_quantized", T_quantized)
    if T_quantized % V_TILE_TOKENS != 0:
        raise ValueError(
            f"T_quantized must be a multiple of {V_TILE_TOKENS}, "
            f"got T_quantized={T_quantized}"
        )
    return 2.0 + 32.0 / (float(V_TILE_TOKENS) * float(C)) + 48.0 / float(T_quantized)


def theoretical_v_tile_full_cache_bits(
    *,
    D: int,
    C: int,
    T_total: int,
    T_quantized: int,
    stats_initialized: bool,
) -> dict:
    """Full-cache theoretical packed bits (sink/recent/pending + codes + side)."""
    D = _require_positive_int("D", D)
    C = _require_positive_int("C", C)
    T_total = _require_positive_int("T_total", T_total)
    if D % C != 0:
        raise ValueError(f"D={D} must be divisible by C={C}")
    if D & (D - 1):
        raise ValueError(f"D must be a power of two for RHT, got D={D}")
    if isinstance(T_quantized, bool) or not isinstance(T_quantized, int):
        raise ValueError(f"T_quantized must be an integer, got {T_quantized!r}")
    if T_quantized < 0 or T_quantized > T_total:
        raise ValueError(
            f"T_quantized must be in [0, T_total], got T_quantized={T_quantized}, "
            f"T_total={T_total}"
        )
    if T_quantized % V_TILE_TOKENS != 0:
        raise ValueError(
            f"T_quantized must be a multiple of {V_TILE_TOKENS}, "
            f"got T_quantized={T_quantized}"
        )
    if not isinstance(stats_initialized, bool):
        raise ValueError(
            f"stats_initialized must be bool, got {stats_initialized!r}"
        )
    if stats_initialized != (T_quantized > 0):
        raise ValueError(
            "stats_initialized must be true iff at least one full tile is quantized; "
            f"got stats_initialized={stats_initialized}, T_quantized={T_quantized}"
        )
    T_fp16 = T_total - T_quantized
    meta = (48 * D) if stats_initialized else 0
    n_tiles = (T_quantized * D) // (V_TILE_TOKENS * C) if T_quantized > 0 else 0
    total_bits = (
        16 * T_fp16 * D
        + 2 * T_quantized * D
        + 32 * n_tiles
        + meta
    )
    return {
        "theoretical_packed_bits": total_bits / float(T_total * D),
        "total_bits": float(total_bits),
        "T_fp16": T_fp16,
        "T_quantized": T_quantized,
        "metadata_bits": float(meta),
    }


def _validate_tile_shape(value_slice: torch.Tensor, v_tile_channels: int) -> Tuple[int, int, int, int]:
    if value_slice.dim() != 4:
        raise ValueError(
            f"expected V shape [B,H,T,D], got {tuple(value_slice.shape)}"
        )
    B, H, T, D = value_slice.shape
    if D <= 0:
        raise ValueError(f"head_dim must be > 0, got head_dim={D}")
    if v_tile_channels <= 0:
        raise ValueError(f"v_tile_channels must be > 0, got {v_tile_channels}")
    if D % v_tile_channels != 0:
        raise ValueError(
            f"head_dim={D} must be divisible by "
            f"v_tile_channels={v_tile_channels}"
        )
    # Keep the requested invariant visible during ordinary debug execution,
    # while the explicit ValueError above still protects optimized ``python -O``.
    assert D % v_tile_channels == 0, (
        f"head_dim={D} must be divisible by v_tile_channels={v_tile_channels}"
    )
    if D & (D - 1):
        raise ValueError(
            f"Hadamard/RHT requires power-of-two head_dim, got head_dim={D}"
        )
    return B, H, T, D


def _reshape_to_tiles(z: torch.Tensor, C: int) -> torch.Tensor:
    """[B,H,T,D] -> [B,H,n_blocks,D/C,16*C] with T multiple of 16."""
    B, H, T, D = z.shape
    N = V_TILE_TOKENS
    if T % N != 0:
        raise ValueError(
            f"tile quant requires T multiple of {N}, got T={T}"
        )
    nb = T // N
    return (
        z.reshape(B, H, nb, N, D // C, C)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, H, nb, D // C, N * C)
    )


def _tiles_to_tensor(tiles: torch.Tensor, C: int, T: int, D: int) -> torch.Tensor:
    """Inverse of _reshape_to_tiles."""
    B, H, nb, n_chan, flat = tiles.shape
    N = V_TILE_TOKENS
    return (
        tiles.reshape(B, H, nb, D // C, N, C)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, H, T, D)
    )


def _minmax_init_2bit(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP16 min-max init on last dim. Returns q_0, offset_0, scale_0, Z_0 (float)."""
    xh = g.half()
    offset_0 = xh.amin(dim=-1, keepdim=True)
    scale_0 = (xh.amax(dim=-1, keepdim=True) - offset_0).clamp(min=_EPS) / _LEVELS
    q_0 = ((xh - offset_0) / scale_0).clamp(0, _LEVELS).round()
    Z_0 = (offset_0 + scale_0 * q_0).float()
    return q_0, offset_0.float(), scale_0.float(), Z_0


def _mse_candidate_probe_exact(g: torch.Tensor, *, iters: int = V_MSE_ITERS) -> torch.Tensor:
    """Probe-exact affine-MSE candidate, independent of the FP16 baseline.

    The research probe first round-trips the values through FP16, then performs
    its *own* FP32 min/max assignment before one alternating LS refit.  This is
    deliberately separate from :func:`_minmax_init_2bit`, whose assignment uses
    FP16 arithmetic and is only the fallback baseline.
    """
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    x = g.half().float()
    _require_finite("MSE candidate input FP16 round-trip", x)
    offset = x.amin(dim=-1, keepdim=True)
    scale = (x.amax(dim=-1, keepdim=True) - offset).clamp(min=_EPS) / _LEVELS
    n = x.shape[-1]
    sum_x = x.sum(dim=-1, keepdim=True)
    for _ in range(iters):
        q = ((x - offset) / scale).round().clamp(0, _LEVELS)
        sum_q = q.sum(dim=-1, keepdim=True)
        sum_q2 = q.square().sum(dim=-1, keepdim=True)
        sum_qx = (q * x).sum(dim=-1, keepdim=True)
        denom = n * sum_q2 - sum_q.square()
        new_scale = (n * sum_qx - sum_q * sum_x) / denom.clamp(min=1e-12)
        new_offset = (sum_x - new_scale * sum_q) / n
        valid = (
            (denom > 0)
            & (new_scale > 1e-6)
            & torch.isfinite(new_scale)
            & torch.isfinite(new_offset)
        )
        scale = torch.where(valid, new_scale, scale)
        offset = torch.where(valid, new_offset, offset)
    offset = offset.half().float()
    scale = scale.clamp(min=_EPS).half().float()
    _require_finite("MSE candidate affine parameters", offset)
    _require_finite("MSE candidate affine scale", scale)
    q = ((x - offset) / scale).round().clamp(0, _LEVELS)
    candidate = q * scale + offset
    _require_finite("MSE candidate reconstruction", candidate)
    return candidate


def _mse_refit_once_with_fallback(g: torch.Tensor, Z_0: torch.Tensor) -> torch.Tensor:
    """Choose probe-exact one-step MSE candidate or FP16 min-max per tile."""
    candidate = _mse_candidate_probe_exact(g, iters=V_MSE_ITERS)
    sse1 = (candidate - g).square().sum(dim=-1, keepdim=True)
    sse0 = (Z_0 - g).square().sum(dim=-1, keepdim=True)
    use_cand = sse1 <= sse0
    return torch.where(use_cand, candidate, Z_0)


def _quantize_normalized_tiles(z: torch.Tensor, C: int) -> torch.Tensor:
    """Quantize normalized Z [B,H,T,D] with T%16==0 into tile16cC 2-bit."""
    B, H, T, D = z.shape
    groups = _reshape_to_tiles(z, C)
    _require_finite("normalized tile input", groups)
    _, _, _, Z_0 = _minmax_init_2bit(groups)
    _require_finite("FP16 min-max baseline", Z_0)
    Z_hat = _mse_refit_once_with_fallback(groups, Z_0)
    _require_finite("selected tile reconstruction", Z_hat)
    return _tiles_to_tensor(Z_hat, C, T, D)


def calibrate_and_quantize_v_tile_prompt(
    value_blocks: torch.Tensor,
    *,
    v_tile_channels: int,
    v_rht_seed: int = V_RHT_SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibrate on and quantize all complete settled prompt blocks.

    ``value_blocks`` is ``[B,H,T,D]`` with positive ``T % 16 == 0``.  RHT
    mean/RMS and reconstruction bias are computed over the *entire* T-token
    prompt region; tile quantization itself remains vectorized by 16-token block.
    Returns ``(recon_fp16, mu_fp16, rms_fp16, bias_fp16)``.
    """
    C = int(v_tile_channels)
    _, _, T, D = _validate_tile_shape(value_blocks, C)
    if T <= 0 or T % V_TILE_TOKENS != 0:
        raise ValueError(
            f"prompt tile calibration requires positive T multiple of "
            f"{V_TILE_TOKENS}, got T={T}"
        )
    v0 = value_blocks.half().float()
    _require_finite("input FP16 round-trip", v0)
    signs = rht_signs(D, v0.device, v0.dtype, seed=v_rht_seed)
    v_r = rht_forward(v0, signs).half().float()
    _require_finite("forward RHT FP16 round-trip", v_r)
    mu = v_r.mean(dim=2, keepdim=True).half().float()
    _require_finite("prompt per-channel mean", mu)
    rms = (
        (v_r - mu).square().mean(dim=2, keepdim=True).sqrt()
        .clamp(min=_EPS)
        .half()
        .float()
    )
    _require_finite("prompt per-channel RMS", rms)
    if not bool((rms > 0).all()):
        raise ValueError("prompt per-channel RMS must be strictly positive")
    z = (v_r - mu) / rms
    _require_finite("prompt affine-normalized values", z)
    z_hat = _quantize_normalized_tiles(z, C)
    v_r_hat = z_hat * rms + mu
    _require_finite("inverse affine reconstruction", v_r_hat)
    v_hat_0 = rht_inverse(v_r_hat, signs)
    _require_finite("inverse RHT reconstruction", v_hat_0)
    bias = (v_hat_0 - v0).mean(dim=2, keepdim=True).half().float()
    _require_finite("prompt reconstruction bias", bias)
    recon = (v_hat_0 - bias).half()
    _require_finite("final FP16 cache reconstruction", recon)
    return recon, mu.half(), rms.half(), bias.half()


def calibrate_and_quantize_first_v_tile_block(
    value_block: torch.Tensor,
    *,
    v_tile_channels: int,
    v_rht_seed: int = V_RHT_SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibrate mu/rms/bias on the first full 16-token block and quantize it.

    value_block: [B,H,16,D] FP16/float. Returns
      (recon_fp16, mu_fp16, rms_fp16, bias_fp16)
    where mu/rms/bias have shape [B,H,1,D].
    """
    C = int(v_tile_channels)
    _, _, T, _ = _validate_tile_shape(value_block, C)
    if T != V_TILE_TOKENS:
        raise ValueError(
            f"first tile block must have T={V_TILE_TOKENS}, got T={T}"
        )
    return calibrate_and_quantize_v_tile_prompt(
        value_block,
        v_tile_channels=C,
        v_rht_seed=v_rht_seed,
    )


def quantize_v_tile_blocks_with_frozen_stats(
    value_blocks: torch.Tensor,
    *,
    mu: torch.Tensor,
    rms: torch.Tensor,
    bias: torch.Tensor,
    v_tile_channels: int,
    v_rht_seed: int = V_RHT_SEED,
) -> torch.Tensor:
    """Quantize one or more full 16-token blocks with frozen prompt stats.

    value_blocks: [B,H,T,D] with T % 16 == 0.
    mu/rms/bias: [B,H,1,D] (FP16 storage ok).
    """
    C = int(v_tile_channels)
    B, H, T, D = _validate_tile_shape(value_blocks, C)
    if T <= 0 or T % V_TILE_TOKENS != 0:
        raise ValueError(
            f"frozen-stats tile quant requires positive T multiple of "
            f"{V_TILE_TOKENS}, got T={T}"
        )
    expected_shape = (B, H, 1, D)
    for name, stat in (("mu", mu), ("rms", rms), ("bias", bias)):
        if tuple(stat.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(stat.shape)}"
            )
        if stat.device != value_blocks.device:
            raise ValueError(
                f"{name} must be on device {value_blocks.device}, got {stat.device}"
            )
        if stat.dtype != torch.float16:
            raise ValueError(
                f"{name} must use FP16 storage, got dtype={stat.dtype}"
            )
        _require_finite(f"frozen {name}", stat)
    if not bool((rms > 0).all()):
        raise ValueError("frozen rms must be strictly positive")
    v0 = value_blocks.half().float()
    _require_finite("input FP16 round-trip", v0)
    signs = rht_signs(D, v0.device, v0.dtype, seed=v_rht_seed)
    mu_f = mu.half().float()
    rms_f = rms.half().float()
    bias_f = bias.half().float()
    v_r = rht_forward(v0, signs).half().float()
    _require_finite("forward RHT FP16 round-trip", v_r)
    z = (v_r - mu_f) / rms_f
    _require_finite("frozen-stat affine-normalized values", z)
    z_hat = _quantize_normalized_tiles(z, C)
    v_r_hat = z_hat * rms_f + mu_f
    _require_finite("inverse affine reconstruction", v_r_hat)
    v_hat_0 = rht_inverse(v_r_hat, signs)
    _require_finite("inverse RHT reconstruction", v_hat_0)
    recon = (v_hat_0 - bias_f).half()
    _require_finite("final FP16 cache reconstruction", recon)
    return recon


def algo_slug_for_version(version: Optional[str] = None) -> str:
    """Map algorithm version string to slug suffix (rv1)."""
    ver = version or V_TILE_ALGO_VERSION
    if ver == V_TILE_ALGO_VERSION:
        return V_TILE_ALGO_SLUG
    raise ValueError(f"unknown v_tile_algo_version={ver!r}")
