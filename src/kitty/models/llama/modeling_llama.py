# src/kitty/models/llama/modeling_llama.py
#
# Real Triton Kitty dense kernel path for the Llama architecture (e.g. Llama-3.2-1B).
#
# The real Kitty Triton kernels are architecture-agnostic at the
# attention-compute level: they operate on the projected, RoPE-applied Q/K/V
# tensors and the KittyCache paged store. The only model-specific piece is the
# attention forward, which must drive the KittyCache protocol (update -> prefill
# dense / decode dense kernel -> quantize) instead of the stock HF cache protocol.
#
# Rather than fork ~600 lines of Hugging Face's Llama modeling (and risk drifting
# from the installed transformers version), we reuse the stock, already-validated
# `LlamaForCausalLM` for everything (embeddings, RoPE incl. llama3 scaling, masks,
# RMSNorm, MLP, the model/decoder loops) and surgically replace each attention
# layer's `forward` with the Kitty version below. This mirrors the upstream
# `Qwen3ForCausalLM_Kitty` integration (kitty/models/qwen3) but for Llama, which
# is simpler: no q_norm/k_norm and no sliding-window layers.

from __future__ import annotations

import types
from typing import Any, Callable, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# The real Triton Kitty dense decode kernel.
from kitty.kvcache.kernels.kitty_attention import kitty_attention_forward


def _llama_kitty_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Any = None,
    cache_position: Optional[torch.LongTensor] = None,
    past_key_values: Any = None,
    **kwargs,
):
    """Kitty attention forward for a stock Llama attention module.

    Identical in spirit to `Qwen3ForCausalLM_Kitty`'s attention, minus the
    Qwen3-only q_norm/k_norm and sliding-window handling that Llama does not use.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # (B, H_Q, T, D)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)     # (B, H_KV, T, D)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # (B, H_KV, T, D)

    # RoPE is applied to Q/K here; cos/sin are also forwarded to the KittyCache via
    # cache_kwargs below, so the prefill attention_interface does not need them.
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Transformers attention modules receive the cache as `past_key_value`
    # (singular) on some versions/call sites, while model/generate APIs use
    # `past_key_values` (plural). Accept both so the KittyCache is never lost
    # (otherwise a singular-keyword call site silently passes None and decode would
    # be mislabelled).
    kv_cache = past_key_value if past_key_value is not None else past_key_values
    assert kv_cache is not None, (
        "LlamaForCausalLM_Kitty requires a KittyCache passed via past_key_values; "
        "got None. Build it with kitty.kvcache.get_kvcache_kitty(...)."
    )
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    # KittyCache.update() returns a bool IsPrefill flag (NOT the (k, v) tuple a
    # stock HF cache returns), which is exactly why the attention forward must be
    # replaced rather than reused.
    is_prefill = kv_cache.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if is_prefill:  # Prefill: dense attention over the full freshly-computed K/V.
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        kv_cache.quantize_prefill(self.layer_idx)
    else:  # Decode: real Triton Kitty dense kernel.
        attn_output, attn_weights = kitty_attention_forward(
            self,
            query_states,
            kv_cache.kv_cache[self.layer_idx],
            scaling=self.scaling,
        )
        kv_cache.quantize_decode(self.layer_idx)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def convert_llama_attention_to_kitty(model: torch.nn.Module) -> int:
    """Patch every Llama attention layer in `model` to use the Kitty kernel path.

    Returns the number of attention layers patched. `kitty_attention_forward`
    reads `module.num_attention_heads` / `module.num_key_value_heads`, which the
    stock Llama attention does not store as instance attributes, so we add them.
    """
    config = model.config
    patched = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.num_attention_heads = config.num_attention_heads
        attn.num_key_value_heads = config.num_key_value_heads
        attn.forward = types.MethodType(_llama_kitty_attention_forward, attn)
        patched += 1
    return patched


class LlamaForCausalLM_Kitty(LlamaForCausalLM):
    """Stock Llama for causal LM with the Kitty dense attention kernel wired in.

    Use exactly like the stock class, but pass a KittyCache via
    `past_key_values` to `generate()` / `forward()`:

        from kitty.models.llama import LlamaForCausalLM_Kitty
        from kitty.kvcache import get_kvcache_kitty
        model = LlamaForCausalLM_Kitty.from_pretrained(path, attn_implementation="sdpa")
        cache = get_kvcache_kitty(model.config, 1, ctx_len + max_gen,
                                  page_size=16)
        model.generate(**inputs, past_key_values=cache, disable_compile=True)
    """

    def __init__(self, config):
        super().__init__(config)
        convert_llama_attention_to_kitty(self)
