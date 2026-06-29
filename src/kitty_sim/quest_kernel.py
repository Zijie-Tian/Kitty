"""Triton QUEST sparse decode overlay for qlutattn/Kitty fake-quant caches.

The qlutattn family in ``kitty_sim`` stores a dense fake-quantized K/V history in
``KittyKVCache``.  QUEST page selection needs the post-RoPE query, so it cannot
live inside ``Cache.update()``.  This module patches the HF attention forward:

* prefill stays dense over the fake-quantized K/V cache;
* decode builds QUEST page scores from page min/max metadata;
* selected-page attention is computed by a Triton kernel over the dense
  fake-quantized K/V tensors (no Python gather/softmax oracle).
"""

from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any, Callable

import torch
import triton
import triton.language as tl
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .quest_sparse import (
    QuestConfig,
    _middle_ranges,
    build_page_minmax,
    reduce_gqa_scores_to_kv_heads,
    score_pages_minmax_bound,
    select_topk_pages,
)


@dataclass(frozen=True)
class QuestKernelResult:
    output: torch.Tensor | None
    path: str
    selected_pages: torch.Tensor | None
    page_count: int
    sink_end: int
    recent_start: int
    topk_pages: int


@triton.jit
def _quest_decode_dense_kv_kernel(
    q_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_ptr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_ptr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_d: tl.constexpr,
    selected_ptr,
    selected_stride_b: tl.constexpr,
    selected_stride_h: tl.constexpr,
    selected_stride_p: tl.constexpr,
    out_ptr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_d: tl.constexpr,
    scale: tl.constexpr,
    seq_len: tl.constexpr,
    sink_end: tl.constexpr,
    recent_start: tl.constexpr,
    page_size: tl.constexpr,
    topk_pages: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    group_size: tl.constexpr = num_query_heads // num_kv_heads
    pid_hkv = pid_hq // group_size

    offs_d = tl.arange(0, block_d)
    offs_n = tl.arange(0, block_n)
    d_mask = offs_d < head_dim

    q = tl.load(
        q_ptr
        + pid_b * q_stride_b
        + pid_hq * q_stride_h
        + 0 * q_stride_t
        + offs_d * q_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((), -3.4028234663852886e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((block_d,), tl.float32)

    # Sink prefix.
    pos = 0
    while pos < sink_end:
        toks = pos + offs_n
        tok_mask = toks < sink_end
        k = tl.load(
            k_ptr
            + pid_b * k_stride_b
            + pid_hkv * k_stride_h
            + toks[:, None] * k_stride_t
            + offs_d[None, :] * k_stride_d,
            mask=tok_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale
        logits = tl.where(tok_mask, logits, -3.4028234663852886e38)
        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        p = tl.exp(logits - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr
            + pid_b * v_stride_b
            + pid_hkv * v_stride_h
            + toks[:, None] * v_stride_t
            + offs_d[None, :] * v_stride_d,
            mask=tok_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        pos += block_n

    # Selected middle pages.
    i = 0
    while i < topk_pages:
        page = tl.load(
            selected_ptr
            + pid_b * selected_stride_b
            + pid_hkv * selected_stride_h
            + i * selected_stride_p
        )
        toks = sink_end + page * page_size + offs_n
        tok_mask = toks < recent_start
        k = tl.load(
            k_ptr
            + pid_b * k_stride_b
            + pid_hkv * k_stride_h
            + toks[:, None] * k_stride_t
            + offs_d[None, :] * k_stride_d,
            mask=tok_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale
        logits = tl.where(tok_mask, logits, -3.4028234663852886e38)
        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        p = tl.exp(logits - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr
            + pid_b * v_stride_b
            + pid_hkv * v_stride_h
            + toks[:, None] * v_stride_t
            + offs_d[None, :] * v_stride_d,
            mask=tok_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        i += 1

    # Recent suffix.
    pos = recent_start
    while pos < seq_len:
        toks = pos + offs_n
        tok_mask = toks < seq_len
        k = tl.load(
            k_ptr
            + pid_b * k_stride_b
            + pid_hkv * k_stride_h
            + toks[:, None] * k_stride_t
            + offs_d[None, :] * k_stride_d,
            mask=tok_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale
        logits = tl.where(tok_mask, logits, -3.4028234663852886e38)
        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        p = tl.exp(logits - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(
            v_ptr
            + pid_b * v_stride_b
            + pid_hkv * v_stride_h
            + toks[:, None] * v_stride_t
            + offs_d[None, :] * v_stride_d,
            mask=tok_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        pos += block_n

    out = acc / l_i
    out = tl.where(l_i > 0.0, out, 0.0)
    tl.store(
        out_ptr
        + pid_b * out_stride_b
        + pid_hq * out_stride_h
        + 0 * out_stride_t
        + offs_d * out_stride_d,
        out,
        mask=d_mask,
    )


def _quest_sparse_attention_triton_dense_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    config: QuestConfig,
    scale: float,
) -> QuestKernelResult:
    if query_states.dim() != 4 or query_states.shape[2] != 1:
        return QuestKernelResult(None, "dense_prefill", None, 0, 0, 0, 0)
    batch, query_heads, _q_len, head_dim = (int(x) for x in query_states.shape)
    key_batch, kv_heads, seq_len, key_dim = (int(x) for x in key_states.shape)
    if batch != key_batch or value_states.shape[:3] != key_states.shape[:3] or head_dim != key_dim:
        raise ValueError("query/key/value shapes are incompatible for QUEST kernel")
    if query_heads % kv_heads != 0:
        raise ValueError(f"query_heads ({query_heads}) must be a multiple of kv_heads ({kv_heads})")

    sink_end, recent_start = _middle_ranges(
        seq_len,
        keep_sink=config.keep_sink,
        sink_length=config.sink_length,
        keep_recent=config.keep_recent,
        recent_length=config.recent_length,
    )
    bounds = build_page_minmax(key_states, page_size=config.page_size, start=sink_end, end=recent_start)
    page_count = int(bounds.page_count)
    if page_count <= 0:
        return QuestKernelResult(None, "dense_no_shared_pages", None, page_count, sink_end, recent_start, 0)

    topk = int(config.resolve_topk_pages(page_count))
    if topk >= page_count:
        return QuestKernelResult(None, "dense_full_budget", None, page_count, sink_end, recent_start, topk)

    raw_scores = score_pages_minmax_bound(query_states, bounds.page_min, bounds.page_max)
    reduced_scores = reduce_gqa_scores_to_kv_heads(
        raw_scores,
        num_key_value_heads=kv_heads,
        reduce=config.score_reduce,
    )
    selected_pages = select_topk_pages(
        reduced_scores,
        page_size=config.page_size,
        token_budget=config.token_budget,
        topk_pages=config.topk_pages,
    ).contiguous()

    query_states = query_states.contiguous()
    output = torch.empty((batch, query_heads, 1, head_dim), dtype=query_states.dtype, device=query_states.device)
    block_n = int(config.page_size)
    block_d = triton.next_power_of_2(head_dim)
    grid = (batch, query_heads)
    _quest_decode_dense_kv_kernel[grid](
        query_states,
        query_states.stride(0), query_states.stride(1), query_states.stride(2), query_states.stride(3),
        key_states,
        key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
        value_states,
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        selected_pages,
        selected_pages.stride(0), selected_pages.stride(1), selected_pages.stride(2),
        output,
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale,
        seq_len,
        sink_end,
        recent_start,
        int(config.page_size),
        int(selected_pages.shape[-1]),
        query_heads,
        kv_heads,
        head_dim,
        block_n,
        block_d,
        num_warps=4,
    )
    return QuestKernelResult(
        output=output,
        path="triton_sparse_reduced_budget",
        selected_pages=selected_pages,
        page_count=page_count,
        sink_end=sink_end,
        recent_start=recent_start,
        topk_pages=int(selected_pages.shape[-1]),
    )


def _dense_attention_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    kwargs: dict[str, Any],
):
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    return attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        **kwargs,
    )


def _quest_kernel_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any = None,
    cache_position: torch.LongTensor | None = None,
    past_key_values: Any = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

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

    kv_cache = past_key_value if past_key_value is not None else past_key_values
    assert kv_cache is not None, (
        "QUEST kernel requires a KittyKVCache via past_key_values (qlutattn fake-quant)."
    )
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_q, value_q = kv_cache.update(key_states, value_states, self.layer_idx, cache_kwargs)

    stats = self._quest_kernel_stats
    cfg: QuestConfig = self._quest_kernel_cfg
    q_len = int(query_states.shape[2])
    is_decode = q_len == 1 and int(self.layer_idx) >= int(cfg.skip_layers)

    if not is_decode:
        attn_output, _ = _dense_attention_forward(
            self, query_states, key_q, value_q, attention_mask, kwargs
        )
        stats["prefill_calls"] += 1
        stats["last_path"] = "dense_prefill"
    else:
        result = _quest_sparse_attention_triton_dense_kv(
            query_states.contiguous(),
            key_q,
            value_q,
            config=cfg,
            scale=float(self.scaling),
        )
        stats["last_path"] = result.path
        stats["last_page_count"] = int(result.page_count)
        stats["last_selected_pages"] = int(result.topk_pages)
        stats["last_sink_end"] = int(result.sink_end)
        stats["last_recent_start"] = int(result.recent_start)
        if result.selected_pages is not None:
            stats["last_selected_pages_shape"] = tuple(result.selected_pages.shape)

        if result.output is None:
            attn_output, _ = _dense_attention_forward(
                self, query_states, key_q, value_q, attention_mask, kwargs
            )
            stats["dense_fallback_calls"] += 1
        else:
            attn_output = result.output.transpose(1, 2)
            stats["decode_calls"] += 1

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_quest_kernel(model: torch.nn.Module, quest_cfg: QuestConfig) -> dict[str, Any]:
    """Patch attention layers to run Triton QUEST over qlutattn fake-quant K/V."""
    stats: dict[str, Any] = {
        "installed": 0,
        "prefill_calls": 0,
        "decode_calls": 0,
        "dense_fallback_calls": 0,
        "last_path": None,
        "last_selected_pages": None,
        "last_selected_pages_shape": None,
        "last_page_count": None,
        "last_sink_end": None,
        "last_recent_start": None,
    }
    for layer in model.model.layers:
        attn = layer.self_attn
        attn._quest_kernel_cfg = quest_cfg
        attn._quest_kernel_stats = stats
        attn.forward = types.MethodType(_quest_kernel_attention_forward, attn)
        stats["installed"] += 1
    return stats


__all__ = ["QuestConfig", "QuestKernelResult", "install_quest_kernel"]
