# -*- coding: utf-8 -*-
"""CPU tests for the research token-tier K path (QLUT_TOKEN_TIER=1).

Covers the pure numerics: observation-window scoring, per-head top-rho tier
selection, and the tiered _quant_k_pertoken blend (hi rows == all-nf2 result,
lo rows == low-mask result). No model, no GPU, no env dependence.
"""
import torch

from kitty_sim.kitty_simulate import KittyKVCache
from kitty_sim.token_tier import observation_scores, tier_from_scores


def test_tier_from_scores_per_head_topk():
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8], [0.5, 0.1, 0.6, 0.2]])
    tier = tier_from_scores(scores, rho=0.5)
    assert tier.tolist() == [[False, True, False, True], [True, False, True, False]]
    assert tier_from_scores(scores, rho=0.0).sum() == 0


def test_observation_scores_prefers_aligned_key():
    torch.manual_seed(0)
    n_q, n_kv, T, D = 4, 2, 32, 16
    q = torch.randn(1, n_q, T, D)
    keys = torch.randn(1, n_kv, T, D) * 0.1
    # Make key position 5 of head 0 align with head 0's window queries.
    probe = torch.randn(D)
    q[0, 0:2, -8:, :] = probe
    keys[0, 0, 5, :] = probe * 3
    sc = observation_scores(q, keys, settled_end=20, window=8, n_kv=n_kv)
    assert sc.shape == (n_kv, 20)
    assert sc[0].argmax().item() == 5


class _TierCache:
    """Minimal host for the tiered _quant_k_pertoken numerics."""

    def __init__(self, lo_id, hi_id, bin_codebooks=("sign", "nf2")):
        self.k_codebook = "qlut"
        self.bin_codebooks = list(bin_codebooks)
        self.k_pc_mean = {}
        self.k_cb_mask = {0: lo_id}
        self.k_tier_hi_mask = {0: hi_id}

    _quant_k_pertoken = KittyKVCache._quant_k_pertoken
    _pt_apply_cb_mask = KittyKVCache._pt_apply_cb_mask
    _pt_codebook_masked = staticmethod(KittyKVCache._pt_codebook_masked)
    _masked_nf2sym_lastdim = staticmethod(KittyKVCache._masked_nf2sym_lastdim)


def test_tiered_quant_blends_lo_and_hi_rows():
    torch.manual_seed(0)
    nh, T, D = 2, 96, 64
    ks = torch.randn(1, nh, T, D, dtype=torch.float16)
    lo_id = torch.zeros(nh, D, dtype=torch.long)
    lo_id[:, D // 2:] = 1                                   # half sign, half nf2
    hi_id = torch.ones(nh, D, dtype=torch.long)             # all-nf2 HIGH mask

    tier = torch.zeros(nh, T, dtype=torch.bool)
    tier[0, ::3] = True                                     # promote every 3rd token of head 0

    tiered = _TierCache(lo_id, hi_id)._quant_k_pertoken(ks.clone(), 0, tier=tier)
    all_lo = _TierCache(lo_id, hi_id)._quant_k_pertoken(ks.clone(), 0, tier=None)
    all_hi = _TierCache(hi_id, hi_id)._quant_k_pertoken(ks.clone(), 0, tier=None)

    hi_rows = tier[0].nonzero(as_tuple=True)[0]
    lo_rows = (~tier[0]).nonzero(as_tuple=True)[0]
    assert torch.equal(tiered[0, 0, hi_rows], all_hi[0, 0, hi_rows])
    assert torch.equal(tiered[0, 0, lo_rows], all_lo[0, 0, lo_rows])
    assert torch.equal(tiered[0, 1], all_lo[0, 1])          # head 1 fully lo
    # Tier path must not disturb the cached per-channel mean state semantics.
    assert not tiered.isnan().any()
