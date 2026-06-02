"""Pure-PyTorch (no-Triton) QUEST + Kitty sim path for LongBench.

This wires the existing pure-torch QUEST oracle (``kitty_sim.quest_sparse``) into
a real HF model as an ACCURACY proxy that, unlike the removed ``kitty_page16``
fake-quant proxy, performs genuine query-aware page selection.

How it works (and why it is not in the cache object):
- The Kitty fake-quant lives in the sim ``KittyKVCache`` (passed via
  ``past_key_values``); its ``update()`` returns the fake-quantized FULL K/V.
- QUEST page selection needs the QUERY, which the HF ``Cache.update()`` interface
  never receives. So selection must run in the attention forward, after RoPE.
- We therefore replace each attention layer's ``forward`` with a hook that:
  projects Q/K/V (+RoPE), pushes K/V through the sim Kitty cache (fake-quant),
  then on DECODE runs the gather-based QUEST oracle over the fake-quant K/V
  (bounded to the token budget), and on PREFILL runs dense attention.

Because the oracle is gather-based (it index-selects only the selected pages +
sink + recent), decode attention compute is bounded by the QUEST budget rather
than the context length. That is what makes a 16k-vs-128k decode-timing check a
valid way to confirm QUEST is engaged (near-flat) vs pure Kitty (grows with
context). It models accuracy + relative timing; it does NOT save KV memory and
is not a kernel-speed proof (that is the real Triton path,
``quest_kitty_page16_kernel``).

Architecture coverage: only Q/K/V projection + optional q_norm/k_norm + RoPE are
architecture-specific; the hook handles Llama (no q/k norm) and Qwen3 (q/k norm)
via an attribute check, and the QUEST oracle is architecture-agnostic.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .quest_sparse import QuestConfig, quest_sparse_attention_page16


def _sim_quest_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Q/K/V projection (+ Qwen3 q_norm/k_norm if present) — the only arch-specific part.
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if getattr(self, "q_norm", None) is not None:
        query_states = self.q_norm(query_states)
    if getattr(self, "k_norm", None) is not None:
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    assert past_key_values is not None, (
        "sim QUEST requires a sim KittyKVCache via past_key_values (kitty_sim fake-quant)."
    )
    # Kitty fake-quant: returns the fake-quantized FULL K/V history for this layer.
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_q, value_q = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    stats = self._sim_quest_stats
    q_len = query_states.shape[2]
    cfg: QuestConfig = self._sim_quest_cfg
    is_decode = q_len == 1 and self.layer_idx >= cfg.skip_layers

    if not is_decode:
        # Prefill (or an intentionally skipped layer): dense attention over the
        # fake-quant K/V. QUEST is decode-only, matching the real kernel.
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_q,
            value_q,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        stats["prefill_calls"] += 1
    else:
        # Decode: gather-based QUEST sparse selection over the fake-quant K/V.
        result = quest_sparse_attention_page16(
            query_states.contiguous(),
            key_q,
            value_q,
            config=cfg,
            scale=self.scaling,
            return_metadata=True,
        )
        attn_output = result.output.transpose(1, 2)  # (B, Hq, 1, D) -> (B, 1, Hq, D)
        meta = result.metadata
        sel = meta.selected_pages
        stats["decode_calls"] += 1
        stats["last_selected_pages"] = int(sel.shape[-1]) if sel is not None else 0
        stats["last_page_count"] = int(meta.page_bounds.page_count)
        stats["last_sink_end"] = int(meta.sink_end)
        stats["last_recent_start"] = int(meta.recent_start)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_sim_quest(model: torch.nn.Module, quest_cfg: QuestConfig) -> dict[str, Any]:
    """Patch every attention layer of a stock HF model to run sim QUEST + Kitty.

    The Kitty fake-quant is applied by the sim KittyKVCache passed at generate
    time via ``past_key_values``; this patch adds the query-aware QUEST selection
    that the cache interface cannot do. Returns a shared stats dict used by the
    runner guardrail to prove the QUEST decode path actually engaged.
    """
    stats: dict[str, Any] = {
        "installed": 0,
        "prefill_calls": 0,
        "decode_calls": 0,
        "last_selected_pages": None,
        "last_page_count": None,
        "last_sink_end": None,
        "last_recent_start": None,
    }
    for layer in model.model.layers:
        attn = layer.self_attn
        attn._sim_quest_cfg = quest_cfg
        attn._sim_quest_stats = stats
        attn.forward = types.MethodType(_sim_quest_attention_forward, attn)
        stats["installed"] += 1
    return stats
