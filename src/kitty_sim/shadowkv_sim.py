"""Pure-PyTorch (no-CUTLASS) ShadowKV sim path for LongBench.

This is a faithful port of ShadowKV's *accuracy* cache class
(``ShadowKVCache`` in the upstream repo, the pure-torch / GPU-resident variant
used for accuracy measurement, NOT the CUTLASS+CPU-offload ``ShadowKVCache_CPU``
throughput class). The KV-cache *algorithm* is copied verbatim:

- Prefill: SVD-compress the **pre-RoPE** keys to rank ``r`` (U, SV), segment the
  **post-RoPE** keys into chunks, keep the lowest-cosine-similarity chunks as
  exact "outliers" + a local window, and register the per-chunk means as
  landmarks.
- Decode: landmark attention selects the top-``budget/chunk`` chunks for the
  current query, reconstruct their keys on the fly from the low-rank ``U @ SV``
  (+ RoPE at the selected absolute positions), gather their values, and attend
  over the compact ``[local | outlier | selected | generated]`` buffer.

Only the *engine* and the *RoPE op* differ from upstream, mirroring how Kitty
already adapted QUEST into ``kitty_sim.sim_quest``:
- engine: ShadowKV's hand-written transformer + flash-attn/vLLM/CUTLASS kernels
  -> a stock HuggingFace model whose attention ``forward`` is method-patched and
  driven by HF ``generate`` (this module);
- RoPE: ShadowKV's CUDA ``apply_rotary_pos_emb_single`` -> the mathematically
  identical pure-torch ``_rope_at_positions`` here, using the model's own rotary
  ``inv_freq`` so Llama-3.1 rope scaling stays exact.

Like ``sim_quest``, this is an ACCURACY + relative-timing proxy: it does NOT
save KV memory (the full value cache and the low-rank key factors stay resident
on the GPU) and it is NOT a kernel-speed proof.

Architecture coverage: Llama, Qwen3, and Phi-3 / Phi-4-mini. The only
arch-specific parts are handled via attribute checks: Q/K/V projection (separate
q/k/v_proj vs Phi's fused ``qkv_proj``), optional Qwen3 q_norm/k_norm, and RoPE
(full vs Phi's ``partial_rotary_factor`` 0.75 + longrope, via
``_apply_rotary_partial`` and a rotary_emb-built cos/sin table). GLM (legacy
tuple cache) is intentionally out of scope.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    repeat_kv,
)


@dataclass
class ShadowKVSimConfig:
    """ShadowKV sim hyper-parameters (upstream paper-aligned defaults)."""

    sparse_budget: int = 2048
    rank: int = 160
    chunk_size: int = 8
    local_chunk: int = 4
    # outlier chunk count scales with the budget, matching upstream
    # ShadowKVCache_CPU ((budget//1024)*24 -> 48 at budget 2048).
    outlier_chunk: int | None = None

    def resolved_outlier_chunk(self) -> int:
        if self.outlier_chunk is not None:
            return int(self.outlier_chunk)
        return max(1, (self.sparse_budget // 1024) * 24)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial-RoPE-aware apply (rotary_dim = ``cos.shape[-1]`` <= head_dim).

    Mirrors HF's modern ``apply_rotary_pos_emb``: rotate the first ``rotary_dim``
    dims of q/k and pass the remainder through unchanged. Correct for full RoPE
    (Llama / Qwen3, rotary_dim == head_dim so the pass slice is empty) and for
    partial RoPE (Phi-3 / Phi-4-mini, partial_rotary_factor 0.75 -> rotary_dim
    96 of 128). Replaces the stock llama ``apply_rotary_pos_emb`` (full-rotate
    only), whose broadcast would fail on Phi's 96-dim cos against the 128-dim head.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


class ShadowKVSimCache(DynamicCache):
    """Faithful pure-torch port of ShadowKV's accuracy cache (batch size 1).

    One instance per LongBench sample, sized to ``max_length`` = context + gen.
    The attention hook calls the ShadowKV-specific methods directly (get_svd /
    prefill_kv_cache on prefill; update_kv_cache / get_retrieval_position_ids /
    get_key_cache / get_value_cache on decode) instead of the generic
    ``Cache.update`` -- exactly like ShadowKV's own ``layer_compute`` does. The
    DynamicCache API surface below is only what HF ``generate`` touches so it
    can thread this object via ``past_key_values`` and advance positions.
    """

    def __init__(
        self,
        config: Any,
        *,
        sparse_budget: int = 2048,
        rank: int = 160,
        chunk_size: int = 8,
        local_chunk: int = 4,
        outlier_chunk: int | None = None,
        max_length: int = 32 * 1024,
        max_gen: int = 256,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        rotary_emb: Any = None,
    ) -> None:
        super().__init__()
        # DynamicCache legacy list surface kept empty; ShadowKV state is custom.
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_layers = int(config.num_hidden_layers)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_attention_heads = int(config.num_attention_heads)
        self.head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.batch_size = 1
        self.sparse_budget = int(sparse_budget)
        self.chunk_size = int(chunk_size)
        self.rank = int(rank)
        self.local_chunk = int(local_chunk)
        self.outlier_chunk = (
            int(outlier_chunk) if outlier_chunk is not None else max(1, (self.sparse_budget // 1024) * 24)
        )
        self.select_sets = self.sparse_budget // self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget, (
            f"sparse_budget ({self.sparse_budget}) must be divisible by chunk_size ({self.chunk_size})"
        )
        # Per-prompt active (clamped) budgets; set in prefill_kv_cache so short
        # LongBench prompts with few chunks never ask top-k for more than exist.
        self.active_outlier_chunk = self.outlier_chunk
        self.active_select_sets = self.select_sets
        self.active_sparse_budget = self.sparse_budget

        self.max_length = int(max_length)
        self.max_gen = int(max_gen)
        # The compact decode buffers only hold [local | outlier | selected |
        # generated], so they scale with budget + max_gen (NOT context length).
        # Only the full value store v_cache_cpu is context-sized (for gathering).
        buf_len = self.sparse_budget + (self.outlier_chunk + self.local_chunk + 2) * self.chunk_size + self.max_gen
        self._buf_len = int(buf_len)
        self.selected_chunk_idx = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets,
            device=self.device, dtype=torch.long,
        )
        self.v_cache_cpu = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads, self.max_length, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        self.k_cache_buffer = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads, self._buf_len, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        self.v_cache_buffer = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads, self._buf_len, self.head_dim,
            device=self.device, dtype=self.dtype,
        )

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
        self.chunks = 0
        self.sparse_start = 0
        self.sparse_end = 0
        self.incoming_q_len = 0
        self.last_selected_chunks = 0

        self.k_landmark: list[torch.Tensor] | None = None
        self.k_landmark_idx: list[torch.Tensor] | None = None
        self.U: torch.Tensor | None = None
        self.SV: torch.Tensor | None = None

        # Build a RoPE cos/sin table from the model's own rotary embedding so the
        # reconstructed keys are RoPE'd at their absolute positions exactly like
        # the query path (handles Llama-3.1 rope scaling).
        self._build_rope_table(rotary_emb)

    # ------------------------------------------------------------------ RoPE
    def _build_rope_table(self, rotary_emb: Any) -> None:
        # Prefer the model's own rotary embedding to build the [0, max_length)
        # cos/sin table: this reproduces partial RoPE (Phi: rotary_dim 96 < 128)
        # and rope-scaling (Llama-3.1 / Phi longrope, whose length-dependent
        # short/long factor is selected here by this sample's max_length, and
        # whose attention_scaling is already folded into the returned cos/sin)
        # exactly, so reconstructed keys match the query path. The table's last
        # dim becomes rotary_dim, which _rope_at_positions rotates (passing the tail).
        if rotary_emb is not None:
            pos = torch.arange(self.max_length, device=self.device, dtype=torch.long).unsqueeze(0)
            dummy = torch.zeros(1, 1, 1, device=self.device, dtype=self.dtype)
            with torch.no_grad():
                cos, sin = rotary_emb(dummy, pos)
            self.cos_table = cos[0].to(self.dtype)
            self.sin_table = sin[0].to(self.dtype)
        else:  # fallback: standard full RoPE from config theta
            theta = float(getattr(self.config, "rope_theta", 10000.0))
            half = self.head_dim // 2
            inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=self.device) / half))
            t = torch.arange(self.max_length, dtype=torch.float32, device=self.device)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_table = emb.cos().to(self.dtype)
            self.sin_table = emb.sin().to(self.dtype)
        self.rotary_dim = int(self.cos_table.shape[-1])

    def _rope_at_positions(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: [bsz, kv_heads, n, head_dim]; position_ids: [bsz, kv_heads, n].
        # Rotate only the first rotary_dim dims (partial RoPE for Phi; full for
        # Llama/Qwen where rotary_dim == head_dim and the pass slice is empty).
        cos = self.cos_table[position_ids]
        sin = self.sin_table[position_ids]
        rotary_dim = cos.shape[-1]
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        return torch.cat([(x_rot * cos) + (_rotate_half(x_rot) * sin), x_pass], dim=-1)

    @staticmethod
    def _layer(value: Any, layer_idx: int) -> torch.Tensor:
        return value[layer_idx]

    # ------------------------------------------------- HF generate cache API
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.kv_offset

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        kv_length = self.get_seq_length(layer_idx) + cache_position.shape[0]
        return kv_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor):  # batch size 1: no-op safe
        return

    def reset(self):
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0

    # ----------------------------------------------------------- Encoding
    def get_svd(self, new_k_cache: torch.Tensor, layer_idx: int) -> None:
        # new_k_cache is PRE-RoPE [bsz, kv_heads, prefill, head_dim] OR [bsz, prefill, kv_heads*head_dim]
        if new_k_cache.shape[1] <= 32:
            k_cache = new_k_cache.transpose(1, 2).reshape(
                self.batch_size, -1, self.num_key_value_heads * self.head_dim
            )
        else:
            k_cache = new_k_cache

        if layer_idx == 0:
            self.U = torch.zeros(
                self.num_layers, self.batch_size, k_cache.shape[1], self.rank,
                device=self.device, dtype=self.dtype,
            )
            self.SV = torch.zeros(
                self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim,
                device=self.device, dtype=self.dtype,
            )

        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1, 2)
        self.U[layer_idx].copy_(u[:, :, : self.rank].to(self.dtype))
        self.SV[layer_idx].copy_(
            torch.matmul(torch.diag_embed(s[:, : self.rank]), v[:, : self.rank])
            .to(self.dtype)
            .view(self.batch_size, -1, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

    def register_k_landmark(self, k_landmark: torch.Tensor, k_landmark_idx: torch.Tensor, layer_idx: int) -> None:
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            self.k_landmark = [None] * self.num_layers  # type: ignore[list-item]
            self.k_landmark_idx = [None] * self.num_layers  # type: ignore[list-item]
        self.k_landmark[layer_idx] = k_landmark.contiguous()
        self.k_landmark_idx[layer_idx] = k_landmark_idx.contiguous()

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> None:
        incoming = new_v_cache.shape[-2]
        self.prefill = incoming
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

        self.chunks = max(incoming // self.chunk_size - self.local_chunk, 0)
        # Clamp the active budget for short prompts so top-k never over-asks.
        self.active_outlier_chunk = min(self.outlier_chunk, self.chunks)
        self.active_select_sets = min(self.select_sets, max(self.chunks - self.active_outlier_chunk, 0))
        self.active_sparse_budget = self.active_select_sets * self.chunk_size

        self.prefill_local = incoming - self.chunks * self.chunk_size
        self.k_cache_buffer[layer_idx][:, :, : self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local :])
        self.v_cache_buffer[layer_idx][:, :, : self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local :])

        if self.chunks == 0:
            self.sparse_start = self.prefill_local
            self.sparse_end = self.sparse_start
            empty_landmarks = key_states_roped.new_empty(self.batch_size, self.num_key_value_heads, 0, self.head_dim)
            empty_idx = torch.empty(
                self.batch_size, self.num_key_value_heads, 0, device=key_states_roped.device, dtype=torch.long
            )
            self.register_k_landmark(empty_landmarks, empty_idx, layer_idx)
            if layer_idx == self.num_layers - 1:
                self.kv_offset += incoming
            return

        key_states_roped_ctx = key_states_roped[:, :, : self.chunks * self.chunk_size].view(
            self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim
        )
        landmark_candidates = key_states_roped_ctx.mean(dim=-2)  # [bsz, kv_heads, chunks, head_dim]
        cos_sim = torch.nn.functional.cosine_similarity(
            landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1),
            key_states_roped_ctx,
            dim=-1,
        )  # [bsz, kv_heads, chunks, chunk_size]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.active_outlier_chunk, largest=False).indices

        outlier_chunk_k_cache = key_states_roped_ctx.gather(
            dim=2,
            index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim),
        ).view(self.batch_size, self.num_key_value_heads, self.active_outlier_chunk * self.chunk_size, self.head_dim)
        outlier_chunk_v_cache = (
            new_v_cache[:, :, : self.chunks * self.chunk_size]
            .view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
            .gather(
                dim=2,
                index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim),
            )
            .view(self.batch_size, self.num_key_value_heads, self.active_outlier_chunk * self.chunk_size, self.head_dim)
        )

        self.sparse_start = self.prefill_local + self.active_outlier_chunk * self.chunk_size
        self.sparse_end = self.sparse_start + self.active_sparse_budget

        self.k_cache_buffer[layer_idx][:, :, self.prefill_local : self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local : self.sparse_start].copy_(outlier_chunk_v_cache)

        all_idx = (
            torch.arange(self.chunks, device=key_states_roped.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.batch_size, self.num_key_value_heads, -1)
        )
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        self.register_k_landmark(
            landmark_candidates.gather(
                dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            ).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim),
            rest_idx,
            layer_idx,
        )

        if layer_idx == self.num_layers - 1:
            assert self.sparse_end <= self.k_cache_buffer.shape[-2]
            self.kv_offset += incoming

    # ----------------------------------------------------------- Decoding
    def get_retrieval_position_ids(self, layer_idx: int, query_states: torch.Tensor) -> torch.Tensor:
        self.incoming_q_len = query_states.shape[-2]  # 1
        k_landmark = self._layer(self.k_landmark, layer_idx)
        k_landmark_idx = self._layer(self.k_landmark_idx, layer_idx)
        selected_chunk_idx = self._layer(self.selected_chunk_idx, layer_idx)
        if k_landmark.shape[-2] == 0:
            self.last_selected_chunks = 0
            return torch.empty(
                self.batch_size, self.num_key_value_heads, 0, device=query_states.device, dtype=torch.long
            )
        query_by_group = query_states.view(
            -1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim
        )
        chunk_attn = (
            torch.einsum("bhgqd,bhdc->bhgqc", query_by_group, k_landmark.transpose(2, 3)).squeeze(2)
            / math.sqrt(self.head_dim)
        )
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_attn = chunk_attn.sum(dim=-2)
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2)
        select_sets = min(self.active_select_sets, chunk_attn.shape[-1])
        self.last_selected_chunks = int(select_sets)
        merged_results = torch.topk(chunk_attn, k=select_sets, dim=-1).indices
        selected_chunks = k_landmark_idx.gather(dim=-1, index=merged_results)

        selected_chunk_idx.zero_()
        selected_chunk_idx[:, :, :select_sets].copy_(selected_chunks, non_blocking=True)

        position_ids = (
            selected_chunks.unsqueeze(-1) * self.chunk_size
            + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ).view(self.batch_size, self.num_key_value_heads, -1)
        return position_ids

    def get_value_cache(self, layer_idx: int, position_ids: torch.Tensor) -> torch.Tensor:
        v_cache_cpu = self._layer(self.v_cache_cpu, layer_idx)
        v_cache_buffer = self._layer(self.v_cache_buffer, layer_idx)
        if position_ids.shape[-1] > 0:
            value_ = v_cache_cpu.gather(
                dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            )
            v_cache_buffer[:, :, self.sparse_start : self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
        return v_cache_buffer[:, :, : self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx: int, position_ids: torch.Tensor) -> torch.Tensor:
        u = self._layer(self.U, layer_idx)
        sv = self._layer(self.SV, layer_idx)
        k_cache_buffer = self._layer(self.k_cache_buffer, layer_idx)
        if position_ids.shape[-1] > 0:
            index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1))
            u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1)
            u_head = torch.gather(u_expand, 2, index_expanded)
            result = torch.einsum("bhrk,bhkd->bhrd", u_head, sv)
            result = self._rope_at_positions(result, position_ids)
            k_cache_buffer[:, :, self.sparse_start : self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
        return k_cache_buffer[:, :, : self.sparse_end + gen_offset]

    def update_kv_cache(self, new_k_cache: torch.Tensor, new_v_cache: torch.Tensor, layer_idx: int) -> None:
        incoming = new_k_cache.shape[-2]
        v_cache_buffer = self._layer(self.v_cache_buffer, layer_idx)
        k_cache_buffer = self._layer(self.k_cache_buffer, layer_idx)
        start = self.sparse_end + self.gen_offset
        v_cache_buffer[:, :, start : start + incoming].copy_(new_v_cache, non_blocking=True)
        k_cache_buffer[:, :, start : start + incoming].copy_(new_k_cache, non_blocking=True)
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming


def _shadowkv_sim_attention_forward(
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

    # Q/K/V projection — the only arch-specific part. Fused qkv_proj (Phi-3 /
    # Phi-4-mini: a single [Q|K|V] Linear, split by head counts) vs separate
    # q/k/v_proj (Llama / Qwen3), plus optional Qwen3 q_norm/k_norm.
    if getattr(self, "qkv_proj", None) is not None:
        qkv = self.qkv_proj(hidden_states)
        q_pos = self.config.num_attention_heads * self.head_dim
        kv_pos = self.config.num_key_value_heads * self.head_dim
        query_states = qkv[..., :q_pos].view(hidden_shape)
        key_states = qkv[..., q_pos : q_pos + kv_pos].view(hidden_shape)
        value_states = qkv[..., q_pos + kv_pos :].view(hidden_shape)
    else:
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
    if getattr(self, "q_norm", None) is not None:
        query_states = self.q_norm(query_states)
    if getattr(self, "k_norm", None) is not None:
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    cache: ShadowKVSimCache = past_key_value if past_key_value is not None else past_key_values
    assert cache is not None, "ShadowKV sim requires a ShadowKVSimCache via past_key_values."

    stats = self._shadowkv_stats
    q_len = query_states.shape[2]
    is_prefill = q_len > 1

    if is_prefill:
        # SVD on PRE-RoPE keys, then RoPE, then build the ShadowKV structures.
        cache.get_svd(key_states, self.layer_idx)
        query_states, key_states = _apply_rotary_partial(query_states, key_states, cos, sin)
        cache.prefill_kv_cache(value_states, self.layer_idx, key_states, query_states[:, :, -1:])
        # Dense attention over the full post-RoPE K/V (QUEST/ShadowKV are decode-only).
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        stats["prefill_calls"] += 1
    else:
        query_states, key_states = _apply_rotary_partial(query_states, key_states, cos, sin)
        cache.update_kv_cache(key_states, value_states, self.layer_idx)
        position_ids = cache.get_retrieval_position_ids(self.layer_idx, query_states)
        value_sel = cache.get_value_cache(self.layer_idx, position_ids)
        key_sel = cache.get_key_cache(self.layer_idx, position_ids)
        # Attend over the compact [local | outlier | selected | generated] buffer.
        key_rep = repeat_kv(key_sel, cache.num_key_value_groups)
        value_rep = repeat_kv(value_sel, cache.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_rep, value_rep, attn_mask=None, dropout_p=0.0, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2)
        stats["decode_calls"] += 1
        stats["last_selected_chunks"] = int(cache.last_selected_chunks)
        stats["last_seq_length"] = int(cache.kv_offset)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_shadowkv_sim(model: torch.nn.Module, cfg: ShadowKVSimConfig) -> dict[str, Any]:
    """Patch every attention layer of a stock HF model to run ShadowKV sim.

    The ShadowKVSimCache (per-sample, via ``past_key_values``) holds the SVD /
    landmark / buffer state; this patch runs the prefill build + decode selection
    in the attention forward, since the HF ``Cache.update`` interface never sees
    the query. Returns a shared stats dict used by the runner guardrail to prove
    the ShadowKV decode path actually engaged.
    """
    stats: dict[str, Any] = {
        "installed": 0,
        "prefill_calls": 0,
        "decode_calls": 0,
        "last_selected_chunks": None,
        "last_seq_length": None,
    }
    for layer in model.model.layers:
        attn = layer.self_attn
        attn._shadowkv_cfg = cfg
        attn._shadowkv_stats = stats
        attn.forward = types.MethodType(_shadowkv_sim_attention_forward, attn)
        stats["installed"] += 1
    return stats
