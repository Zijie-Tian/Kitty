# -*- coding: utf-8 -*-
"""Codebook primitives for the canonical qlutattn K-cache path.

qlutattn quantizes the post-RoPE K cache per token on the per-channel-mean-
centered residual. Each channel uses ONE of two codebooks, fixed OFFLINE by a
per-(layer, kv-head, channel) mask (scripts/calibrate_qlutattn_mask.py ranks
channels by residual sigma^2 on a calibration corpus):

  * sign  -- 1-bit binary with a per-token mean-|r| scale (low-sigma^2 channels)
  * nf2   -- fixed symmetric NF2 LUT with a per-token absmax scale
             (high-sigma^2 channels)

This module holds the symmetric-NF2 constants/snap shared by the runtime cache
(kitty_sim.kitty_simulate) and the residual sigma^2 statistic used by the
offline calibration. Pure fake-quant accuracy proxy; no packed storage.
"""
import torch

# Symmetric NF2 (IR-QLoRA appendix B.2, Table 11): fixed normalized levels
# {-1, -c, +c, +1}. The published +/- inner entries differ in the 8th decimal
# (float artifacts); we symmetrize to the fp32 magnitude of the negative entry.
# Nearest-neighbor boundary between inner and outer level is t = (1 + c) / 2;
# a strict `>` sends the exact boundary to the INNER level, matching argmin's
# first-hit tie rule over the sorted LUT. The codebook is a FIXED LUT ("symnf2-v1");
# only the absmax scale s is data-dependent (one fp16 side value per group).
NF2_INNER = 0.25256848335266113
NF2_THRESH = (1.0 + NF2_INNER) / 2.0


def nf2_symmetric_lastdim(r):
    """Symmetric-NF2 snap of a ZERO-CENTERED group along the last axis:
    per-group absmax scale s, closed-form nearest level sign(r)*s*(c or 1)
    (no argmin tensor, no division -- s==0 all-zero groups reconstruct 0).
    sign(0)==0 maps an exact-zero value to 0 (measure-zero event; the strict
    LUT would give -c*s)."""
    s = r.abs().amax(-1, keepdim=True)
    level = torch.where(r.abs() > NF2_THRESH * s, s, NF2_INNER * s)
    return torch.sign(r) * level


def _grouped(x, G):
    H, D, T = x.shape
    ng = T // G
    return x[:, :, : ng * G].reshape(H, D, ng, G), ng


def channel_sigma2(x_quant, G):
    """x_quant:[H,D,Tq] -> per-channel residual sigma^2 [H,D] (submean over G-groups).

    The offline mask calibration ranks channels by this statistic (low sigma^2
    -> sign, high sigma^2 -> nf2)."""
    xg, _ = _grouped(x_quant.float(), G)
    return (xg - xg.mean(-1, keepdim=True)).pow(2).mean(dim=(-1, -2))
