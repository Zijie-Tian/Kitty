# Author: Haojun Xia (xhjustc@gmail.com)

import torch
from typing import Optional

__all__ = [
    "build_promote_mask",
    "fake_quant_groupwise_lastdim",
    "fake_quant_q4_0_lastdim",
]


def build_promote_mask(
        key_states: torch.Tensor, 
        promote_ratio: float, 
        channel_selection: int,
    ) -> torch.Tensor:
    """
    Generating a mask to select channels based on importance scores.
    args:
        key_states : (B, nh, D, T),     key cache after RoPE but before quantization
        promote_ratio : float,          the ratio of channels to promote, in [0, 1]
        channel_selection : int,        channel selection strategy
                                        (-1) for Unspecified, raise an error;
                                        (0)  for Random Selection;
                                        (1)  Magnitude-based Channel Selection;
                                        (2)  Variance-based Channel Selection;
                                        (3)  Cross-head Magnitude-based Selection: the layer's total
                                             budget (nh * k_chan, identical to strategy 1) is allocated
                                             jointly across all heads, so per-head counts may differ;
    returns:
        promote_mask  : (B, nh, D) bool,      promote_mask[i][j] == True -> j-th channel of i-th head is selected for promotion
    """
    assert key_states.dim() == 4
    B, nh, D, _ = key_states.shape
    assert 0. <= promote_ratio <= 1.0, f"promote_ratio must be in [0, 1], got {promote_ratio}"
    # corner cases
    k_chan = int(D * promote_ratio + 1e-6)  # number of channels to promote
    k_chan = max(0, min(k_chan, D))
    if k_chan == 0:  # promote no channel
        return torch.zeros((B, nh, D), dtype=torch.bool, device=key_states.device)
    if k_chan >= D:  # promote all
        return torch.ones((B, nh, D), dtype=torch.bool, device=key_states.device)
    #
    promote_mask = torch.zeros((B, nh, D), dtype=torch.bool, device=key_states.device)
    ##############################################channel selection strategies###############################################
    if channel_selection == 0:                                                            # (0)  for Random Selection;
        for i in range(nh):
            rand_idx = torch.randperm(D, device=key_states.device)[:k_chan]         # (k_chan,)
            promote_mask[:, i, rand_idx] = True                                     # (B, k_chan)
    elif channel_selection == 1:                                                            # (1)  Magnitude-based Channel Selection
        score = key_states.abs()
        score = score.mean(dim=-1)  # (B, nh, D, T) → (B, nh, D)
        _, top_idx = score.topk(k_chan, dim=-1)     # (B, nh, k_chan)
        promote_mask.scatter_(-1, top_idx, True)   # True → promote channel, (B, nh, D)
    elif channel_selection == 2:                                                            # (2)  Variance-based Channel Selection
        diff = key_states - key_states.mean(dim=-1, keepdim=True)
        score = diff.pow(2).mean(dim=-1)  # (B, nh, D, T) → (B, nh, D)
        _, top_idx = score.topk(k_chan, dim=-1)
        promote_mask.scatter_(-1, top_idx, True)   # True → promote channel
    elif channel_selection == 3:                                                            # (3)  Cross-head Magnitude-based Selection
        # The layer budget is exactly nh * k_chan -- the same total as strategy (1)
        # at any ratio -- but the topk runs over the flattened (nh*D) channels of
        # the whole layer, so heads with larger-magnitude channels take more of
        # the budget and others take less (possibly zero).
        score = key_states.abs().mean(dim=-1)                       # (B, nh, D)
        flat_score = score.reshape(B, nh * D)                       # (B, nh*D)
        _, top_idx = flat_score.topk(nh * k_chan, dim=-1)           # (B, nh*k_chan)
        flat_mask = torch.zeros((B, nh * D), dtype=torch.bool, device=key_states.device)
        flat_mask.scatter_(-1, top_idx, True)
        promote_mask = flat_mask.view(B, nh, D)
    ########################################################################################################################
    else:
        raise ValueError(f"Invalid channel_selection strategy: {channel_selection}")
    #
    return promote_mask


def fake_quant_q4_0_lastdim(data: torch.Tensor) -> torch.Tensor:
    """
    Simulate llama.cpp/ggml Q4_0 quantization along the last dim (fake quant).

    Faithful to quantize_row_q4_0_ref + dequantize_row_q4_0 (ggml-quants.c):
      * 32-element blocks along the last dim (QK4_0), per-token when the last
        dim is head_dim -- symmetric, NO zero-point.
      * scale d = signed_absmax / -8 (the max-|x| element lands exactly on the
        -8 code); 1/d is computed from the fp32 d, while dequantization uses
        the fp16-STORED d (round-tripped through half precision).
      * codes q = min(15, int(x/d + 8.5)) -> dequant (q - 8) * d.
    Effective 4.5 bit/value (16B nibbles + 2B fp16 scale per 32 values).
    Args:
        data: (..., D) float tensor with D % 32 == 0.
    Returns:
        Tensor of the same shape/dtype with fake-quantized values.
    """
    QK = 32
    shape, dtype = data.shape, data.dtype
    assert shape[-1] % QK == 0, f"Q4_0 needs last dim % {QK} == 0, got {shape[-1]}"
    x = data.float().reshape(*shape[:-1], shape[-1] // QK, QK)
    # Signed value of the max-|x| element per block. C uses strict '>' so ties
    # keep the FIRST occurrence; torch argmax also returns the first max index.
    idx = x.abs().argmax(dim=-1, keepdim=True)
    mx = torch.gather(x, -1, idx)
    d32 = mx / -8.0
    inv = torch.where(d32 == 0, torch.zeros_like(d32), 1.0 / d32)  # id from fp32 d
    # (int8_t)(x*id + 8.5) with MIN(15, .). The operand is always >= 0.5 by
    # construction (|x/d| <= 8), so C's trunc-toward-zero equals floor here.
    q = (x * inv + 8.5).floor().clamp_(0.0, 15.0)
    d16 = d32.half().float()  # d is stored as fp16 -> dequant uses the rounded d
    return ((q - 8.0) * d16).reshape(shape).to(dtype)


def fake_quant_groupwise_lastdim(
        data: torch.Tensor,
        group_size: int,
        bit: int,
        promote_mask: Optional[torch.Tensor] = None,
        promote_bit: int = 4,
) -> torch.Tensor:
    """
    Simulate the numerical effect of group-wise quantization along the last dim.
    Input and output are both fp16 tensors, used for 'fake quantization'.
    Args:
        data: (B, nh, D, T) - input float tensor
        group_size: int - number of elements per quant group along last dim
        bit: int - quantization bit width (e.g. 2 or 4)
    Optional Args:
        promote_mask: (B, nh, D) - bool tensor, True → use promote_bit, False → use bit
        promote_bit:  int - bit width for channels that are promoted to (e.g. 4)
    Returns:
        dequantized_data: same shape as input, fp16 tensor with fake quantized values
    """
    assert data.dim() == 4, "Expected input shape [B, nh, D, T]"
    assert group_size > 0, "group_size must be positive"
    B, nh, D, T = data.shape
    if bit >= 16:   # No quantization needed, return the original data
        return data
    if T == 0:
        return data
    if T % group_size != 0:
        # Newer/smaller Llama-family checkpoints can have head_dim < the
        # paper-default group_size=128 for value-cache quantization. Quantize
        # complete groups and use one smaller final group instead of failing.
        chunks = [
            fake_quant_groupwise_lastdim(
                chunk,
                min(group_size, chunk.shape[-1]),
                bit,
                promote_mask,
                promote_bit,
            )
            for chunk in data.split(group_size, dim=-1)
        ]
        return torch.cat(chunks, dim=-1)
    G = T // group_size
    data = data.contiguous()  # Ensure contiguous memory layout
    x = data.view(B, nh, D, G, group_size)
    # Compute min and max per group
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    eps = 1e-4 if data.dtype in (torch.float16, torch.bfloat16) else 1e-6  # numerical stability
    
    if promote_mask is not None:
        assert promote_mask.shape == (B, nh, D), f"Expected mask shape (B, nh, D), got {promote_mask.shape}"
        promote_mask = promote_mask.view(B, nh, D, 1, 1)
        # promote_bit>=16 means "keep promoted channels in fp16" (realized by the
        # passthrough below). Use the base bit as a placeholder for their scale so
        # the 2**16-1 = 65535 constant is never cast to fp16, whose max is 65504
        # ("value cannot be converted to type at::Half without overflow").
        eff_promote_bit = bit if promote_bit >= 16 else promote_bit
        scale_base = (mx - mn).clamp(min=eps) / (2 ** bit - 1)
        scale_promote = (mx - mn).clamp(min=eps) / (2 ** eff_promote_bit - 1)
        scale = torch.where(promote_mask, scale_promote, scale_base)
        max_val = torch.where(
            promote_mask,
            torch.full_like(scale, 2 ** eff_promote_bit - 1),
            torch.full_like(scale, 2 ** bit - 1)
        )
    else:
        scale = (mx - mn).clamp(min=eps) / (2 ** bit - 1)
        max_val = torch.full_like(scale, 2 ** bit - 1)
    # fake quantization
    q = ((x - mn) / scale).clamp(torch.zeros_like(max_val), max_val).round()
    dq = q * scale + mn
    if promote_mask is not None and promote_bit >= 16:
        # "F16 channels": the magnitude-selected promoted channels are kept in
        # full precision (no quantization); the rest stay at `bit`. Realizes the
        # 1-bit-base + fp16-boost K regime (promote_bit=16) for the low-bit K
        # study. promote_mask is already (B, nh, D, 1, 1) and x is the grouped
        # (B, nh, D, G, group_size) view, so this broadcasts per channel.
        dq = torch.where(promote_mask, x, dq)
    return dq.view(B, nh, D, T).to(data.dtype)
