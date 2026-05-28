import unittest
from types import SimpleNamespace

import torch

try:
    from kitty.kvcache import QuestConfig, get_kvcache_kitty
    from kitty.kvcache.kernels.kitty_attention import kitty_attention_forward
except Exception as exc:  # pragma: no cover
    QuestConfig = None
    get_kvcache_kitty = None
    kitty_attention_forward = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(num_hidden_layers=3, head_dim=16, num_key_value_heads=1)


def _skip_reason() -> str | None:
    if _IMPORT_ERROR is not None:
        return f"real Kitty runtime import failed: {_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return "CUDA is required for real Kitty QUEST sparse tests"
    return None


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class RealKittyQuestSparseTests(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def _cache(self, *, quest_config=None, length=96):
        assert get_kvcache_kitty is not None
        return get_kvcache_kitty(
            _tiny_config(),
            max_batch_size=1,
            max_length=length,
            page_size=16,
            promote_ratio=0.125,
            quest_config=quest_config,
        )

    def _fill_prefill(self, cache, seq_len=80, layer_idx=2):
        key = torch.randn(1, 1, seq_len, 16, dtype=torch.float16, device="cuda")
        value = torch.randn(1, 1, seq_len, 16, dtype=torch.float16, device="cuda")
        self.assertTrue(cache.update(key, value, layer_idx=layer_idx))
        cache.quantize_prefill(layer_idx=layer_idx)
        torch.cuda.synchronize()
        return cache.kv_cache[layer_idx]

    def test_forced_sparse_all_pages_does_not_take_dense_fallback_page16(self):
        assert QuestConfig is not None and kitty_attention_forward is not None
        dense_cache = self._cache()
        sparse_cache = self._cache(
            quest_config=QuestConfig(enabled=True, topk_pages=2, skip_layers=0, force_sparse_for_equivalence=True)
        )
        torch.manual_seed(123)
        key = torch.randn(1, 1, 80, 16, dtype=torch.float16, device="cuda")
        value = torch.randn(1, 1, 80, 16, dtype=torch.float16, device="cuda")
        for cache in (dense_cache, sparse_cache):
            self.assertTrue(cache.update(key, value, layer_idx=2))
            cache.quantize_prefill(layer_idx=2)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)

        dense, _ = kitty_attention_forward(module, query, dense_cache.kv_cache[2], scaling=16 ** -0.5)
        sparse, _ = kitty_attention_forward(module, query, sparse_cache.kv_cache[2], scaling=16 ** -0.5)
        torch.cuda.synchronize()

        layer = sparse_cache.kv_cache[2]
        self.assertEqual(layer.last_quest_path, "sparse_forced_all_pages")
        self.assertGreater(layer.last_sparse_qk_hits, 0)
        self.assertGreater(layer.last_sparse_sv_hits, 0)
        self.assertEqual(layer.last_selected_pages_shape, (1, 1, 2))
        torch.testing.assert_close(sparse, dense, atol=2e-2, rtol=2e-2)

    def test_page16_sparse_budget_finite_shape(self):
        assert QuestConfig is not None and kitty_attention_forward is not None
        cache = self._cache(quest_config=QuestConfig(enabled=True, topk_pages=1, skip_layers=0))
        self._fill_prefill(cache, seq_len=80, layer_idx=2)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)

        output, attn = kitty_attention_forward(module, query, cache.kv_cache[2], scaling=16 ** -0.5)
        torch.cuda.synchronize()

        self.assertIsNone(attn)
        self.assertEqual(output.shape, (1, 1, 1, 16))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(cache.kv_cache[2].last_quest_path, "sparse_reduced_budget")

    def test_page16_sparse_skip_first_layers(self):
        assert QuestConfig is not None and kitty_attention_forward is not None
        cache = self._cache(quest_config=QuestConfig(enabled=True, topk_pages=1, skip_layers=2))
        self._fill_prefill(cache, seq_len=80, layer_idx=1)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)

        output, _ = kitty_attention_forward(module, query, cache.kv_cache[1], scaling=16 ** -0.5)
        torch.cuda.synchronize()

        self.assertEqual(output.shape, (1, 1, 1, 16))
        self.assertEqual(cache.kv_cache[1].last_quest_path, "dense")
        self.assertEqual(cache.kv_cache[1].last_sparse_qk_hits, 0)


if __name__ == "__main__":
    unittest.main()
