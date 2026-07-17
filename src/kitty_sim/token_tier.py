# -*- coding: utf-8 -*-
"""Research-only token-tier support for qlutattn K quantization (QLUT_TOKEN_TIER=1).

Direction: token-level mixed precision INSIDE the fixed sign/nf2 per-token K
path. At prefill, an observation window of the prompt's last W post-RoPE
queries scores every settled key token (SnapKV/H2O-style accumulated
attention, GQA-aggregated per kv head); the top rho fraction of tokens per
head are quantized with a HIGH channel mask (all-nf2), the rest with the
regular LOW mask from the research blob. Decode-time tokens (past the recent
FP16 window) fall back to the LOW mask in this v1.

The per-token quant mechanism itself (per-channel mean subtraction, per-token
sign/nf2 codebooks, protection windows) is unchanged; only the channel->codebook
assignment becomes token-dependent.

Post-RoPE queries are captured by wrapping transformers' llama
apply_rotary_pos_emb. Within one attention forward the call order is strictly
rope -> cache.update, so a single "latest q" slot is always this layer's q.
"""
from typing import Optional

import torch

_STATE: dict = {"q": None, "installed": False, "orig": None}


def install_rope_q_tap() -> None:
    """Wrap llama apply_rotary_pos_emb to stash the post-RoPE query tensor."""
    if _STATE["installed"]:
        return
    import transformers.models.llama.modeling_llama as ml

    orig = ml.apply_rotary_pos_emb

    def tapped(q, k, cos, sin, *a, **kw):
        q_emb, k_emb = orig(q, k, cos, sin, *a, **kw)
        _STATE["q"] = q_emb.detach()
        return q_emb, k_emb

    ml.apply_rotary_pos_emb = tapped
    _STATE["orig"] = orig
    _STATE["installed"] = True


def take_q() -> Optional[torch.Tensor]:
    """Pop the most recently stashed post-RoPE q ([B, n_q, T, D]) or None."""
    q = _STATE["q"]
    _STATE["q"] = None
    return q


def observation_scores(q: torch.Tensor, keys: torch.Tensor, settled_end: int,
                       window: int, n_kv: int) -> torch.Tensor:
    """Accumulated-attention score of every settled key token.

    q:    [1, n_q, T, D] post-RoPE queries of the full prompt forward
    keys: [1, n_kv, T, D] FP16 keys currently in the cache (pre-quant)
    settled_end: number of leading key positions being quantized (scores are
                 returned for positions [0, settled_end))
    window: number of trailing prompt queries to use as the observation window
    Returns [n_kv, settled_end] float32 scores (softmax over ALL T keys, summed
    over window queries, mean over the GQA query group).
    """
    B, n_q, T, D = q.shape
    if B != 1:
        raise ValueError(f"token-tier scoring expects batch 1; got {B}")
    group = n_q // n_kv
    W = min(window, T)
    scale = D ** -0.5
    kf = keys[0].float()                                            # [n_kv, T, D]
    scores = torch.zeros(n_kv, settled_end, dtype=torch.float32, device=q.device)
    for h in range(n_kv):
        qh = q[0, h * group:(h + 1) * group, T - W:, :].float()     # [g, W, D]
        logits = torch.einsum("gwd,td->gwt", qh, kf[h]) * scale     # [g, W, T]
        attn = torch.softmax(logits, dim=-1)
        scores[h] = attn[:, :, :settled_end].sum(dim=1).mean(dim=0)
    return scores


def tier_from_scores(scores: torch.Tensor, rho: float) -> torch.Tensor:
    """Per-head top-rho boolean tier mask. scores [n_kv, S] -> bool [n_kv, S]."""
    n_kv, S = scores.shape
    k_hi = int(round(rho * S))
    tier = torch.zeros(n_kv, S, dtype=torch.bool, device=scores.device)
    if k_hi <= 0:
        return tier
    idx = scores.topk(k_hi, dim=-1).indices                         # [n_kv, k_hi]
    tier.scatter_(1, idx, True)
    return tier
