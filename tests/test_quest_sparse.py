import torch

from kitty_sim.quest_sparse import (
    QuestConfig,
    build_page_minmax,
    dense_attention,
    quest_sparse_attention_page16,
    reduce_gqa_scores_to_kv_heads,
    resolve_topk_page_count,
    score_pages_minmax_bound,
    select_topk_pages,
)


def test_page16_page_minmax_bounds_cover_all_tokens():
    key = torch.arange(1 * 2 * 35 * 3, dtype=torch.float32).reshape(1, 2, 35, 3)
    bounds = build_page_minmax(key)

    assert bounds.page_size == 16
    assert bounds.page_starts.tolist() == [0, 16, 32]
    assert bounds.page_ends.tolist() == [16, 32, 35]
    assert bounds.page_min.shape == (1, 2, 3, 3)
    assert bounds.page_max.shape == (1, 2, 3, 3)

    for page_idx, (start, end) in enumerate(zip(bounds.page_starts.tolist(), bounds.page_ends.tolist())):
        page = key[:, :, start:end, :]
        torch.testing.assert_close(bounds.page_min[:, :, page_idx, :], page.amin(dim=2))
        torch.testing.assert_close(bounds.page_max[:, :, page_idx, :], page.amax(dim=2))


def test_page16_minmax_bound_score_matches_manual_formula():
    query = torch.tensor([[[2.0, -3.0, 0.5]]])
    page_min = torch.tensor([[[[-1.0, -2.0, 4.0], [10.0, -1.0, -8.0]]]])
    page_max = torch.tensor([[[[3.0, 5.0, 7.0], [11.0, 2.0, -2.0]]]])

    scores = score_pages_minmax_bound(query, page_min, page_max)
    manual = torch.stack(
        [
            torch.maximum(query[0, 0] * page_min[0, 0, 0], query[0, 0] * page_max[0, 0, 0]).sum(),
            torch.maximum(query[0, 0] * page_min[0, 0, 1], query[0, 0] * page_max[0, 0, 1]).sum(),
        ]
    ).reshape(1, 1, 2)

    torch.testing.assert_close(scores, manual)


def test_page16_topk_full_budget_selects_all_pages_in_logical_order_and_ties_are_stable():
    scores = torch.tensor([[[3.0, 5.0, 5.0, 1.0]]])

    full = select_topk_pages(scores, topk_pages=99)
    assert full.tolist() == [[[0, 1, 2, 3]]]

    tied = select_topk_pages(scores, topk_pages=2)
    assert tied.tolist() == [[[1, 2]]]
    for _ in range(5):
        assert select_topk_pages(scores, topk_pages=2).tolist() == tied.tolist()

    assert resolve_topk_page_count(4, token_budget=17) == 2


def test_page16_quest_topk_all_matches_dense_oracle():
    torch.manual_seed(0)
    batch, query_heads, kv_heads, seq_len, head_dim = 1, 4, 2, 60, 8
    query = torch.randn(batch, query_heads, 1, head_dim)
    key = torch.randn(batch, kv_heads, seq_len, head_dim)
    value = torch.randn(batch, kv_heads, seq_len, head_dim)
    config = QuestConfig(page_size=16, sink_length=5, recent_length=7, topk_pages=3)

    sparse = quest_sparse_attention_page16(query, key, value, config=config)
    dense = dense_attention(query, key, value)

    torch.testing.assert_close(sparse, dense, rtol=1e-6, atol=1e-6)


def test_page16_quest_sink_and_recent_are_always_kept():
    torch.manual_seed(1)
    batch, query_heads, kv_heads, seq_len, head_dim = 1, 2, 1, 75, 8
    query = torch.randn(batch, query_heads, 1, head_dim)
    key = torch.randn(batch, kv_heads, seq_len, head_dim)
    value = torch.randn(batch, kv_heads, seq_len, head_dim)
    sink_length = 5
    recent_length = 6
    config = QuestConfig(page_size=16, sink_length=sink_length, recent_length=recent_length, topk_pages=0)

    result = quest_sparse_attention_page16(query, key, value, config=config, return_metadata=True)
    kept = list(range(sink_length)) + list(range(seq_len - recent_length, seq_len))
    assert result.metadata.selected_pages.tolist() == [[[]]]
    assert result.metadata.support_indices[0][0].tolist() == kept

    kept_idx = torch.tensor(kept, dtype=torch.long)
    manual = dense_attention(query, key.index_select(2, kept_idx), value.index_select(2, kept_idx))
    torch.testing.assert_close(result.output, manual, rtol=1e-6, atol=1e-6)


def test_page16_quest_gqa_reduction_shared_pages():
    # Four query heads share two KV heads. Each KV group reduces its two query
    # heads by max, producing one shared page-score vector per KV head.
    query_scores = torch.tensor(
        [
            [
                [1.0, 9.0, 0.0],
                [8.0, 2.0, 3.0],
                [5.0, 4.0, 0.0],
                [6.0, 1.0, 7.0],
            ]
        ]
    )

    reduced = reduce_gqa_scores_to_kv_heads(query_scores, num_key_value_heads=2)
    expected = torch.tensor([[[8.0, 9.0, 3.0], [6.0, 4.0, 7.0]]])
    torch.testing.assert_close(reduced, expected)

    selected = select_topk_pages(reduced, topk_pages=1)
    assert selected.tolist() == [[[1], [2]]]


def test_page16_budget_increase_is_monotonic_for_selected_pages():
    scores = torch.tensor([[[9.0, 1.0, 8.0, 7.0]]])

    top1 = set(select_topk_pages(scores, topk_pages=1)[0, 0].tolist())
    top3 = set(select_topk_pages(scores, topk_pages=3)[0, 0].tolist())
    budget_two_pages = set(select_topk_pages(scores, token_budget=32)[0, 0].tolist())

    assert top1 <= top3
    assert budget_two_pages <= top3
    assert budget_two_pages == {0, 2}
