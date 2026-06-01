"""Pure-torch QUEST page-selection and sparse-attention oracle.

This module intentionally stays in ``kitty_sim`` and uses only PyTorch tensor
operations/Python control flow. It is a correctness oracle for the page16
QUEST-on-Kitty plan; it does not touch or depend on the real Triton kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class QuestConfig:
    """Configuration for the pure-torch QUEST oracle.

    Sparse selection applies only to the middle paged region. Sink and recent
    regions are kept outside the sparse budget when their keep flags are set.
    """

    page_size: int = 16
    token_budget: Optional[int] = None
    topk_pages: Optional[int] = None
    skip_layers: int = 0
    keep_sink: bool = True
    keep_recent: bool = True
    score_reduce: str = "max_gqa"
    sink_length: int = 32
    recent_length: int = 16

    def __post_init__(self) -> None:
        _validate_positive_int("page_size", self.page_size)
        _validate_non_negative_int("skip_layers", self.skip_layers)
        _validate_non_negative_int("sink_length", self.sink_length)
        _validate_non_negative_int("recent_length", self.recent_length)
        if self.token_budget is not None:
            _validate_non_negative_int("token_budget", self.token_budget)
        if self.topk_pages is not None:
            _validate_non_negative_int("topk_pages", self.topk_pages)
        if self.score_reduce != "max_gqa":
            raise ValueError("score_reduce must be 'max_gqa' for the current QUEST oracle")

    def resolve_topk_pages(self, page_count: int) -> int:
        """Resolve this config's sparse budget against a page count."""

        return resolve_topk_page_count(
            page_count,
            page_size=self.page_size,
            token_budget=self.token_budget,
            topk_pages=self.topk_pages,
        )


@dataclass(frozen=True)
class QuestPageBounds:
    """Min/max key bounds for logical pages."""

    page_min: torch.Tensor  # [batch, kv_heads, pages, head_dim]
    page_max: torch.Tensor  # [batch, kv_heads, pages, head_dim]
    page_starts: torch.Tensor  # [pages], absolute token start offsets
    page_ends: torch.Tensor  # [pages], absolute token end offsets, exclusive
    page_size: int

    @property
    def page_count(self) -> int:
        return int(self.page_starts.numel())


@dataclass(frozen=True)
class QuestSparseMetadata:
    """Debug/evidence metadata returned by ``quest_sparse_attention_page16``."""

    page_bounds: QuestPageBounds
    raw_query_scores: torch.Tensor  # [batch, query_heads, (query_len), pages]
    reduced_scores: torch.Tensor  # [batch, kv_heads, pages]
    selected_pages: torch.Tensor  # [batch, kv_heads, selected_pages]
    support_indices: tuple[tuple[torch.Tensor, ...], ...]  # [batch][kv_head] token indices
    sink_end: int
    recent_start: int


@dataclass(frozen=True)
class QuestSparseAttentionResult:
    output: torch.Tensor
    metadata: QuestSparseMetadata


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _check_kv_tensor(name: str, tensor: torch.Tensor) -> tuple[int, int, int, int]:
    if tensor.dim() != 4:
        raise ValueError(f"{name} must have shape [batch, heads, seq_len, head_dim]")
    return tuple(int(dim) for dim in tensor.shape)  # type: ignore[return-value]


def _normalize_query(query_states: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if query_states.dim() == 3:
        return query_states.unsqueeze(2), True
    if query_states.dim() == 4:
        return query_states, False
    raise ValueError("query_states must have shape [batch, heads, head_dim] or [batch, heads, query_len, head_dim]")


def _validate_gqa(query_heads: int, kv_heads: int) -> int:
    if kv_heads <= 0 or query_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError(f"query_heads ({query_heads}) must be a positive multiple of kv_heads ({kv_heads})")
    return query_heads // kv_heads


def build_page_minmax(
    key_states: torch.Tensor,
    *,
    page_size: int = 16,
    start: int = 0,
    end: Optional[int] = None,
) -> QuestPageBounds:
    """Build per-page min/max key bounds.

    Args:
        key_states: Key tensor shaped ``[batch, kv_heads, seq_len, head_dim]``.
        page_size: Logical page width. Defaults to QUEST+Kitty page16.
        start: Absolute token offset where page metadata begins.
        end: Exclusive absolute token offset. Defaults to the sequence length.

    Returns:
        A ``QuestPageBounds`` object. The final page may be partial; its bounds
        are computed only over real tokens, and ``page_starts/page_ends`` record
        the exact absolute token coverage.
    """

    _validate_positive_int("page_size", page_size)
    batch, kv_heads, seq_len, head_dim = _check_kv_tensor("key_states", key_states)
    if end is None:
        end = seq_len
    if start < 0 or end < start or end > seq_len:
        raise ValueError(f"invalid page range start={start}, end={end}, seq_len={seq_len}")

    token_count = end - start
    if token_count == 0:
        empty_bounds = key_states.new_empty((batch, kv_heads, 0, head_dim))
        empty_offsets = torch.empty((0,), dtype=torch.long, device=key_states.device)
        return QuestPageBounds(empty_bounds, empty_bounds.clone(), empty_offsets, empty_offsets.clone(), page_size)

    mins: list[torch.Tensor] = []
    maxs: list[torch.Tensor] = []
    starts: list[int] = []
    ends: list[int] = []
    for page_start in range(start, end, page_size):
        page_end = min(page_start + page_size, end)
        page = key_states[:, :, page_start:page_end, :]
        mins.append(page.amin(dim=2))
        maxs.append(page.amax(dim=2))
        starts.append(page_start)
        ends.append(page_end)

    return QuestPageBounds(
        page_min=torch.stack(mins, dim=2),
        page_max=torch.stack(maxs, dim=2),
        page_starts=torch.tensor(starts, dtype=torch.long, device=key_states.device),
        page_ends=torch.tensor(ends, dtype=torch.long, device=key_states.device),
        page_size=page_size,
    )


def score_pages_minmax_bound(
    query_states: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
) -> torch.Tensor:
    """Score pages using QUEST's query-aware min/max upper bound.

    For every query channel ``i`` and page channel bounds ``[min_i, max_i]``,
    the page contribution is ``max(q_i * min_i, q_i * max_i)``. Contributions
    are summed over the head dimension. With GQA, each query head scores the
    bounds of its corresponding KV head.

    Returns ``[batch, query_heads, pages]`` for 3-D queries and
    ``[batch, query_heads, query_len, pages]`` for 4-D queries.
    """

    query, squeezed = _normalize_query(query_states)
    if page_min.shape != page_max.shape or page_min.dim() != 4:
        raise ValueError("page_min and page_max must both have shape [batch, kv_heads, pages, head_dim]")

    batch, query_heads, _query_len, head_dim = (int(dim) for dim in query.shape)
    bounds_batch, kv_heads, _pages, bounds_dim = (int(dim) for dim in page_min.shape)
    if batch != bounds_batch or head_dim != bounds_dim:
        raise ValueError(
            "query_states and page bounds must agree on batch and head_dim: "
            f"query={tuple(query.shape)}, page_min={tuple(page_min.shape)}"
        )
    group_size = _validate_gqa(query_heads, kv_heads)

    kv_for_query = torch.arange(query_heads, device=query.device) // group_size
    min_for_query = page_min.index_select(1, kv_for_query)
    max_for_query = page_max.index_select(1, kv_for_query)
    q = query.unsqueeze(3)  # [B, Hq, Q, 1, D]
    lower = q * min_for_query.unsqueeze(2)
    upper = q * max_for_query.unsqueeze(2)
    scores = torch.maximum(lower, upper).sum(dim=-1)
    if squeezed:
        scores = scores.squeeze(2)
    return scores


def reduce_gqa_scores_to_kv_heads(
    query_page_scores: torch.Tensor,
    *,
    num_key_value_heads: int,
    reduce: str = "max_gqa",
) -> torch.Tensor:
    """Reduce query-head page scores to one shared page score per KV head.

    The current kernel plan processes all query heads in a KV group with one
    selected page set, so scores are reduced by max across query heads (and by
    max across query positions when a query length dimension is present).
    """

    if reduce != "max_gqa":
        raise ValueError("only max_gqa reduction is supported")
    if query_page_scores.dim() not in (3, 4):
        raise ValueError("query_page_scores must have shape [B,Hq,P] or [B,Hq,Q,P]")

    batch = int(query_page_scores.shape[0])
    query_heads = int(query_page_scores.shape[1])
    group_size = _validate_gqa(query_heads, num_key_value_heads)

    if query_page_scores.dim() == 3:
        pages = int(query_page_scores.shape[2])
        grouped = query_page_scores.reshape(batch, num_key_value_heads, group_size, pages)
        return grouped.amax(dim=2)

    query_len = int(query_page_scores.shape[2])
    pages = int(query_page_scores.shape[3])
    grouped = query_page_scores.reshape(batch, num_key_value_heads, group_size, query_len, pages)
    return grouped.amax(dim=(2, 3))


def resolve_topk_page_count(
    page_count: int,
    *,
    page_size: int = 16,
    token_budget: Optional[int] = None,
    topk_pages: Optional[int] = None,
) -> int:
    """Resolve token/page budgets to a clamped number of middle pages."""

    _validate_non_negative_int("page_count", page_count)
    _validate_positive_int("page_size", page_size)
    candidates: list[int] = []
    if token_budget is not None:
        _validate_non_negative_int("token_budget", token_budget)
        candidates.append(math.ceil(token_budget / page_size))
    if topk_pages is not None:
        _validate_non_negative_int("topk_pages", topk_pages)
        candidates.append(topk_pages)
    if not candidates:
        return page_count
    return min(page_count, min(candidates))


def select_topk_pages(
    page_scores: torch.Tensor,
    *,
    page_size: int = 16,
    token_budget: Optional[int] = None,
    topk_pages: Optional[int] = None,
) -> torch.Tensor:
    """Select deterministic top-k logical page indices.

    Ties are broken by lower logical page index. The returned indices are sorted
    in ascending logical-token order so QK and SV can consume the same support
    without changing token order.
    """

    if page_scores.dim() != 3:
        raise ValueError("page_scores must have shape [batch, kv_heads, pages]")
    batch, kv_heads, page_count = (int(dim) for dim in page_scores.shape)
    k = resolve_topk_page_count(page_count, page_size=page_size, token_budget=token_budget, topk_pages=topk_pages)
    if k == 0:
        return torch.empty((batch, kv_heads, 0), dtype=torch.long, device=page_scores.device)
    if k == page_count:
        return torch.arange(page_count, dtype=torch.long, device=page_scores.device).view(1, 1, -1).expand(batch, kv_heads, -1).clone()

    selected = torch.empty((batch, kv_heads, k), dtype=torch.long, device=page_scores.device)
    detached_scores = page_scores.detach().cpu()
    for b in range(batch):
        for h in range(kv_heads):
            order = sorted(range(page_count), key=lambda page: (-float(detached_scores[b, h, page]), page))
            logical_order = sorted(order[:k])
            selected[b, h] = torch.tensor(logical_order, dtype=torch.long, device=page_scores.device)
    return selected


def dense_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Dense GQA attention reference over the full KV sequence."""

    query, squeezed = _normalize_query(query_states)
    batch, query_heads, query_len, head_dim = (int(dim) for dim in query.shape)
    key_batch, kv_heads, key_len, key_dim = _check_kv_tensor("key_states", key_states)
    value_batch, value_heads, value_len, value_dim = _check_kv_tensor("value_states", value_states)
    if (batch, kv_heads, key_len) != (key_batch, value_heads, value_len) or value_batch != batch:
        raise ValueError("key_states and value_states must agree on batch, kv_heads, and seq_len")
    if key_dim != head_dim:
        raise ValueError("query and key head_dim must match")
    group_size = _validate_gqa(query_heads, kv_heads)
    if scale is None:
        scale = head_dim ** -0.5

    kv_for_query = torch.arange(query_heads, device=query.device) // group_size
    key_for_query = key_states.index_select(1, kv_for_query)
    value_for_query = value_states.index_select(1, kv_for_query)
    logits = torch.einsum("bhqd,bhtd->bhqt", query, key_for_query) * scale
    weights = torch.softmax(logits.float(), dim=-1)
    output = torch.einsum("bhqt,bhtd->bhqd", weights, value_for_query.float())
    output = output.to(value_states.dtype if value_states.dtype.is_floating_point else query_states.dtype)
    if squeezed:
        output = output.squeeze(2)
    return output


def _middle_ranges(
    seq_len: int,
    *,
    keep_sink: bool,
    sink_length: int,
    keep_recent: bool,
    recent_length: int,
) -> tuple[int, int]:
    sink_end = min(sink_length, seq_len) if keep_sink else 0
    recent_start = max(sink_end, seq_len - recent_length) if keep_recent else seq_len
    return sink_end, recent_start


def _build_support_indices(
    selected_pages: torch.Tensor,
    bounds: QuestPageBounds,
    *,
    seq_len: int,
    sink_end: int,
    recent_start: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    batch, kv_heads, _selected_count = (int(dim) for dim in selected_pages.shape)
    support_by_batch: list[tuple[torch.Tensor, ...]] = []
    for b in range(batch):
        support_by_head: list[torch.Tensor] = []
        for h in range(kv_heads):
            pieces: list[torch.Tensor] = []
            if sink_end > 0:
                pieces.append(torch.arange(0, sink_end, dtype=torch.long, device=selected_pages.device))
            for page_idx in selected_pages[b, h].tolist():
                start = int(bounds.page_starts[page_idx])
                end = int(bounds.page_ends[page_idx])
                if end > start:
                    pieces.append(torch.arange(start, end, dtype=torch.long, device=selected_pages.device))
            if recent_start < seq_len:
                pieces.append(torch.arange(recent_start, seq_len, dtype=torch.long, device=selected_pages.device))
            if pieces:
                support = torch.cat(pieces)
                # Regions are constructed to be non-overlapping, but unique keeps
                # the oracle robust for degenerate sink/recent settings.
                support = torch.unique(support, sorted=True)
            else:
                support = torch.empty((0,), dtype=torch.long, device=selected_pages.device)
            support_by_head.append(support)
        support_by_batch.append(tuple(support_by_head))
    return tuple(support_by_batch)


def quest_sparse_attention_page16(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    config: Optional[QuestConfig] = None,
    page_size: Optional[int] = None,
    token_budget: Optional[int] = None,
    topk_pages: Optional[int] = None,
    sink_length: Optional[int] = None,
    recent_length: Optional[int] = None,
    scale: Optional[float] = None,
    return_metadata: bool = False,
) -> torch.Tensor | QuestSparseAttentionResult:
    """Run pure-torch QUEST sparse attention over page16 logical pages.

    The selected sparse budget is applied only to the middle region between the
    always-kept sink and recent windows. When the selected middle-page budget is
    full, this oracle uses the same token set/order as dense attention.
    """

    cfg = config if config is not None else QuestConfig()
    effective_page_size = cfg.page_size if page_size is None else page_size
    effective_token_budget = cfg.token_budget if token_budget is None else token_budget
    effective_topk_pages = cfg.topk_pages if topk_pages is None else topk_pages
    effective_sink_length = cfg.sink_length if sink_length is None else sink_length
    effective_recent_length = cfg.recent_length if recent_length is None else recent_length
    _validate_positive_int("page_size", effective_page_size)
    _validate_non_negative_int("sink_length", effective_sink_length)
    _validate_non_negative_int("recent_length", effective_recent_length)

    query, squeezed = _normalize_query(query_states)
    batch, query_heads, query_len, head_dim = (int(dim) for dim in query.shape)
    key_batch, kv_heads, seq_len, key_dim = _check_kv_tensor("key_states", key_states)
    value_batch, value_heads, value_len, _value_dim = _check_kv_tensor("value_states", value_states)
    if batch != key_batch or batch != value_batch or kv_heads != value_heads or seq_len != value_len:
        raise ValueError("query/key/value batch and KV sequence dimensions must agree")
    if head_dim != key_dim:
        raise ValueError("query and key head_dim must match")
    group_size = _validate_gqa(query_heads, kv_heads)
    if scale is None:
        scale = head_dim ** -0.5

    sink_end, recent_start = _middle_ranges(
        seq_len,
        keep_sink=cfg.keep_sink,
        sink_length=effective_sink_length,
        keep_recent=cfg.keep_recent,
        recent_length=effective_recent_length,
    )
    bounds = build_page_minmax(key_states, page_size=effective_page_size, start=sink_end, end=recent_start)
    raw_scores = score_pages_minmax_bound(query, bounds.page_min, bounds.page_max)
    reduced_scores = reduce_gqa_scores_to_kv_heads(raw_scores, num_key_value_heads=kv_heads, reduce=cfg.score_reduce)
    selected_pages = select_topk_pages(
        reduced_scores,
        page_size=effective_page_size,
        token_budget=effective_token_budget,
        topk_pages=effective_topk_pages,
    )
    support_indices = _build_support_indices(selected_pages, bounds, seq_len=seq_len, sink_end=sink_end, recent_start=recent_start)

    output = value_states.new_empty((batch, query_heads, query_len, int(value_states.shape[-1])))
    for b in range(batch):
        for hq in range(query_heads):
            hkv = hq // group_size
            indices = support_indices[b][hkv]
            if indices.numel() == 0:
                output[b, hq].zero_()
                continue
            key_support = key_states[b, hkv].index_select(0, indices)
            value_support = value_states[b, hkv].index_select(0, indices)
            logits = torch.matmul(query[b, hq], key_support.transpose(0, 1)) * scale
            weights = torch.softmax(logits.float(), dim=-1)
            output[b, hq] = torch.matmul(weights, value_support.float()).to(output.dtype)

    if squeezed:
        output = output.squeeze(2)

    metadata = QuestSparseMetadata(
        page_bounds=bounds,
        raw_query_scores=raw_scores.squeeze(2) if raw_scores.dim() == 4 and raw_scores.shape[2] == 1 else raw_scores,
        reduced_scores=reduced_scores,
        selected_pages=selected_pages,
        support_indices=support_indices,
        sink_end=sink_end,
        recent_start=recent_start,
    )
    if return_metadata:
        return QuestSparseAttentionResult(output=output, metadata=metadata)
    return output


__all__ = [
    "QuestConfig",
    "QuestPageBounds",
    "QuestSparseAttentionResult",
    "QuestSparseMetadata",
    "build_page_minmax",
    "dense_attention",
    "quest_sparse_attention_page16",
    "reduce_gqa_scores_to_kv_heads",
    "resolve_topk_page_count",
    "score_pages_minmax_bound",
    "select_topk_pages",
]
