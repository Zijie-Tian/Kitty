# src/kitty/models/phi3/modeling_phi3.py
#
# Real Triton Kitty dense kernel path for the Phi-3 / Phi-4-mini architecture.
#
# Mirrors kitty/models/llama (LlamaForCausalLM_Kitty): reuse the stock,
# already-validated `Phi3ForCausalLM` for everything (embeddings, partial RoPE +
# longrope scaling, masks, RMSNorm, MLP, the model/decoder loops) and surgically
# replace each attention layer's `forward` with the Kitty version below.
#
# Two Phi-3-specific differences from the Llama port, both confined to the
# attention forward (the Triton Kitty kernel and KittyCache are
# architecture-agnostic: they consume already-projected, already-RoPE'd Q/K/V):
#   - fused `qkv_proj` (a single [Q|K|V] Linear) instead of separate q/k/v_proj;
#   - partial RoPE (`partial_rotary_factor` 0.75 -> rotary_dim 96 of head_dim 128)
#     + longrope, handled by Phi-3's own partial-aware `apply_rotary_pos_emb`
#     (cos/sin are rotary_dim-wide; it rotates the first rotary_dim dims and
#     passes the tail through). The kernel then sees a full 128-dim post-RoPE K
#     exactly as for Llama, so no kernel/cache change is needed.

from __future__ import annotations

import types
from typing import Any, Callable, Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.phi3.modeling_phi3 import (
    Phi3ForCausalLM,
    apply_rotary_pos_emb,  # partial-RoPE-aware: rotary_dim = cos.shape[-1]
    eager_attention_forward,
)

# The real Triton Kitty dense decode kernel.
from kitty.kvcache.kernels.kitty_attention import kitty_attention_forward


def _phi3_kitty_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Any = None,
    cache_position: Optional[torch.LongTensor] = None,
    past_key_values: Any = None,
    **kwargs,
):
    """Kitty attention forward for a stock Phi-3 attention module.

    Same protocol as `LlamaForCausalLM_Kitty`'s attention, with Phi-3's fused
    `qkv_proj` split and partial-RoPE apply. Phi-3 has no q_norm/k_norm.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Phi-3 fused qkv_proj: a single [Q | K | V] projection, split by head counts.
    qkv = self.qkv_proj(hidden_states)
    q_pos = self.config.num_attention_heads * self.head_dim
    kv_pos = self.config.num_key_value_heads * self.head_dim
    query_states = qkv[..., :q_pos].view(hidden_shape).transpose(1, 2)                # (B, H_Q, T, D)
    key_states = qkv[..., q_pos : q_pos + kv_pos].view(hidden_shape).transpose(1, 2)  # (B, H_KV, T, D)
    value_states = qkv[..., q_pos + kv_pos :].view(hidden_shape).transpose(1, 2)      # (B, H_KV, T, D)

    # Partial RoPE: Phi-3's apply_rotary_pos_emb rotates the first rotary_dim
    # (=cos.shape[-1], 96) dims and passes the tail (32) through, so the cached K
    # is a full 128-dim post-RoPE vector exactly as Llama produces for the kernel.
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Accept the cache via either keyword (singular past_key_value / plural
    # past_key_values), mirroring the Llama port, so the KittyCache is never lost.
    kv_cache = past_key_value if past_key_value is not None else past_key_values
    assert kv_cache is not None, (
        "Phi3ForCausalLM_Kitty requires a KittyCache passed via past_key_values; "
        "got None. Build it with kitty.kvcache.get_kvcache_kitty(...)."
    )
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    # KittyCache.update() returns a bool IsPrefill flag (NOT a (k, v) tuple).
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


def convert_phi3_attention_to_kitty(model: torch.nn.Module) -> int:
    """Patch every Phi-3 attention layer in `model` to use the Kitty kernel path.

    Returns the number of attention layers patched. `kitty_attention_forward`
    reads `module.num_attention_heads` / `module.num_key_value_heads`, which the
    stock Phi-3 attention does not store as instance attributes, so we add them.
    """
    config = model.config
    patched = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.num_attention_heads = config.num_attention_heads
        attn.num_key_value_heads = config.num_key_value_heads
        attn.forward = types.MethodType(_phi3_kitty_attention_forward, attn)
        patched += 1
    return patched


class Phi3ForCausalLM_Kitty(Phi3ForCausalLM):
    """Stock Phi-3 / Phi-4-mini causal LM with the Kitty dense kernel wired in.

    Use exactly like the stock class, but pass a KittyCache via `past_key_values`
    to `generate()` / `forward()`:

        from kitty.models.phi3 import Phi3ForCausalLM_Kitty
        from kitty.kvcache import get_kvcache_kitty
        model = Phi3ForCausalLM_Kitty.from_pretrained(path, attn_implementation="sdpa")
        cache = get_kvcache_kitty(model.config, 1, ctx_len + max_gen,
                                  page_size=16)
        model.generate(**inputs, past_key_values=cache, disable_compile=True)
    """

    def __init__(self, config):
        super().__init__(config)
        convert_phi3_attention_to_kitty(self)
