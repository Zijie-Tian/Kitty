import os
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
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        return "requires CUDA_VISIBLE_DEVICES=0 for GPU0-only true QUEST+Kitty tests"
    if not torch.cuda.is_available():
        return "CUDA is required for real Kitty QUEST sparse tests"
    if torch.cuda.device_count() != 1:
        return "GPU0-only tests must see exactly one CUDA device"
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
        self.assertEqual(layer.last_quest_path, "triton_sparse_forced_all_pages")
        self.assertGreater(layer.last_sparse_qk_hits, 0)
        self.assertGreater(layer.last_sparse_sv_hits, 0)
        self.assertEqual(layer.last_selected_pages_shape, (1, 1, layer.last_shared_page_count))
        self.assertEqual(layer.last_sparse_qk_pages_loaded, layer.last_shared_page_count + max(0, layer.PageCount_K - layer.last_shared_page_count))
        self.assertEqual(layer.last_sparse_sv_pages_loaded, layer.last_shared_page_count + max(0, layer.PageCount_V - layer.last_shared_page_count))
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
        layer = cache.kv_cache[2]
        self.assertEqual(layer.last_quest_path, "triton_sparse_reduced_budget")
        self.assertEqual(layer.last_selected_pages_shape, (1, 1, 1))
        self.assertEqual(layer.last_selected_tokens, 16)
        self.assertEqual(layer.last_sparse_qk_pages_loaded, 1 + max(0, layer.PageCount_K - layer.last_shared_page_count))
        self.assertEqual(layer.last_sparse_sv_pages_loaded, 1 + max(0, layer.PageCount_V - layer.last_shared_page_count))
        self.assertLess(layer.last_sparse_qk_pages_loaded, layer.PageCount_K)

    def test_page16_sparse_skip_first_layers(self):
        assert QuestConfig is not None and kitty_attention_forward is not None
        cache = self._cache(quest_config=QuestConfig(enabled=True, topk_pages=1, skip_layers=2))
        self._fill_prefill(cache, seq_len=80, layer_idx=1)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)

        output, _ = kitty_attention_forward(module, query, cache.kv_cache[1], scaling=16 ** -0.5)
        torch.cuda.synchronize()

        self.assertEqual(output.shape, (1, 1, 1, 16))
        self.assertIn(cache.kv_cache[1].last_quest_path, {"dense", "dense_skip_layer"})
        self.assertEqual(cache.kv_cache[1].last_sparse_qk_hits, 0)


    def test_token_budget_and_topk_use_stricter_budget(self):
        assert QuestConfig is not None and kitty_attention_forward is not None
        cache = self._cache(quest_config=QuestConfig(enabled=True, topk_pages=2, token_budget=16, skip_layers=0))
        self._fill_prefill(cache, seq_len=96, layer_idx=2)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)

        output, _ = kitty_attention_forward(module, query, cache.kv_cache[2], scaling=16 ** -0.5)
        torch.cuda.synchronize()

        layer = cache.kv_cache[2]
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(layer.last_quest_path, "triton_sparse_reduced_budget")
        self.assertEqual(layer.last_effective_topk_pages, 1)
        self.assertEqual(layer.last_selected_tokens, 16)
        self.assertEqual(layer.last_selected_pages_shape, (1, 1, 1))

    def test_full_budget_without_force_records_dense_full_budget(self):
        assert QuestConfig is not None and kitty_attention_forward is not None
        cache = self._cache(quest_config=QuestConfig(enabled=True, topk_pages=999, skip_layers=0, force_sparse_for_equivalence=False))
        self._fill_prefill(cache, seq_len=80, layer_idx=2)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)

        output, _ = kitty_attention_forward(module, query, cache.kv_cache[2], scaling=16 ** -0.5)
        torch.cuda.synchronize()

        layer = cache.kv_cache[2]
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(layer.last_quest_path, "dense_full_budget")
        self.assertEqual(layer.last_sparse_qk_pages_loaded, 0)
        self.assertEqual(layer.last_sparse_sv_pages_loaded, 0)

    def test_python_sparse_debug_path_is_not_real_kernel_evidence(self):
        assert QuestConfig is not None and kitty_attention_forward is not None
        cache = self._cache(quest_config=QuestConfig(enabled=True, topk_pages=1, skip_layers=0, use_python_debug=True))
        self._fill_prefill(cache, seq_len=80, layer_idx=2)
        query = torch.randn(1, 1, 1, 16, dtype=torch.float16, device="cuda").contiguous()
        module = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)

        output, _ = kitty_attention_forward(module, query, cache.kv_cache[2], scaling=16 ** -0.5)
        torch.cuda.synchronize()

        layer = cache.kv_cache[2]
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(layer.last_quest_path, "python_sparse_debug")
        self.assertEqual(layer.last_sparse_qk_pages_loaded, 0)
        self.assertEqual(layer.last_sparse_sv_pages_loaded, 0)

    def test_negative_budget_values_raise_before_decode(self):
        assert QuestConfig is not None
        with self.assertRaises(ValueError):
            QuestConfig(enabled=True, topk_pages=-1)
        with self.assertRaises(ValueError):
            QuestConfig(enabled=True, token_budget=-1)


if __name__ == "__main__":
    unittest.main()
