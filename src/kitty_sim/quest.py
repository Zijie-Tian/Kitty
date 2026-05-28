"""Compatibility surface for the pure-torch QUEST page16 oracle.

The full implementation lives in :mod:`kitty_sim.quest_sparse`.  This module
keeps the smaller API imported by ``kitty_sim.__init__`` while delegating to the
same tested implementation.
"""

from __future__ import annotations

from typing import Optional

import torch

from .quest_sparse import (
    QuestConfig,
    QuestPageBounds,
    QuestSparseAttentionResult,
    QuestSparseMetadata,
    build_page_minmax as _build_page_minmax,
    dense_attention,
    quest_sparse_attention_page16,
    reduce_gqa_scores_to_kv_heads,
    resolve_topk_page_count,
    score_pages_minmax_bound,
    select_topk_pages,
)


def build_page_bounds(
    key_states: torch.Tensor,
    page_size: int = 16,
    start_token: int = 0,
    end_token: Optional[int] = None,
) -> QuestPageBounds:
    """Return rich per-page min/max bounds including token offsets."""

    return _build_page_minmax(key_states, page_size=page_size, start=start_token, end=end_token)


def build_page_minmax(
    key_states: torch.Tensor,
    page_size: int = 16,
    start_token: int = 0,
    end_token: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(page_min, page_max)`` for logical pages.

    This tuple-returning wrapper preserves the compact API historically exposed
    by ``kitty_sim.quest``. Use ``build_page_bounds`` when page starts/ends are
    needed for partial final pages.
    """

    bounds = build_page_bounds(key_states, page_size=page_size, start_token=start_token, end_token=end_token)
    return bounds.page_min, bounds.page_max


def quest_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: QuestConfig | None = None,
    *,
    sink_length: int | None = None,
    recent_length: int | None = None,
    scaling: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run sparse QUEST attention and return ``(output, selected_pages)``."""

    cfg = config if config is not None else QuestConfig(
        sink_length=32 if sink_length is None else sink_length,
        recent_length=16 if recent_length is None else recent_length,
    )
    result = quest_sparse_attention_page16(
        query,
        key,
        value,
        config=cfg,
        sink_length=sink_length,
        recent_length=recent_length,
        scale=scaling,
        return_metadata=True,
    )
    assert isinstance(result, QuestSparseAttentionResult)
    return result.output, result.metadata.selected_pages


__all__ = [
    "QuestConfig",
    "QuestPageBounds",
    "QuestSparseAttentionResult",
    "QuestSparseMetadata",
    "build_page_bounds",
    "build_page_minmax",
    "dense_attention",
    "quest_sparse_attention",
    "quest_sparse_attention_page16",
    "reduce_gqa_scores_to_kv_heads",
    "resolve_topk_page_count",
    "score_pages_minmax_bound",
    "select_topk_pages",
]
