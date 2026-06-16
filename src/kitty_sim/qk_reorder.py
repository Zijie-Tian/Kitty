# -*- coding: utf-8 -*-
"""Runtime patch for QK-channel-reordered checkpoints.

scripts/preprocess_qlutattn_model.py folds a RoPE-pair permutation into W_q/W_k
(saved in the checkpoint weights) and reorders RoPE inv_freq to match. HF's
LlamaRotaryEmbedding.inv_freq is model-level and persistent=False, so it is not
saved with the checkpoint -- we re-apply inv_freq = inv_freq[pair_perm] here
after from_pretrained. With both folds in place QK^T is identical to the
original model (per-channel quant on the reordered ckpt reproduces the score).
"""
import os

import torch


def apply_qk_reorder(model, model_path) -> bool:
    """If model_path holds reorder_qk.pt, patch model.model.rotary_emb.inv_freq
    by pair_perm. W_q/W_k are already permuted in the saved weights. Returns
    True if a reorder was applied, False if the checkpoint is not reordered."""
    pf = os.path.join(str(model_path), "reorder_qk.pt")
    if not os.path.exists(pf):
        return False
    d = torch.load(pf, map_location="cpu", weights_only=True)
    pair_perm = d["pair_perm"].long()
    base = getattr(model, "model", model)
    rotary = getattr(base, "rotary_emb", None)
    if rotary is None or not hasattr(rotary, "inv_freq"):
        raise RuntimeError(
            "apply_qk_reorder: model has no model.rotary_emb.inv_freq; route-a "
            "reorder only supports model-level RoPE (Llama family).")
    if rotary.inv_freq.numel() != pair_perm.numel():
        raise RuntimeError(
            f"apply_qk_reorder: inv_freq size {rotary.inv_freq.numel()} != "
            f"pair_perm size {pair_perm.numel()}")
    dev = rotary.inv_freq.device
    rotary.inv_freq = rotary.inv_freq[pair_perm.to(dev)].contiguous()
    rotary.original_inv_freq = rotary.inv_freq
    return True
