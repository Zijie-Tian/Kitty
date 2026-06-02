import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig
from kitty_sim.cli.eval_longbench import build_parser
from kitty_sim.longbench.scorer import score_directory
from kitty_sim.longbench.templates import build_chat, format_longbench_prompt
from kitty_sim.longbench.runner import (
    build_variant,
    output_model_dir,
    default_prediction_dir,
    resolve_prediction_dir,
)
from kitty_sim.utils_quant import fake_quant_groupwise_lastdim
from latency_benchmarking.benchmark_kitty import build_parser as build_latency_parser


class DummyEncoding:
    def __init__(self, ids):
        self.input_ids = [ids]


class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    chat_template = "dummy"

    def __call__(self, text, truncation=False, return_tensors=None, add_special_tokens=True):
        return DummyEncoding(list(range(len(text.split()))))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        suffix = "<GEN>" if add_generation_prompt else ""
        return f"<CHAT>{messages[0]['content']}{suffix}"


class LongBenchTests(unittest.TestCase):
    def test_no_chat_dataset_is_not_wrapped(self):
        tok = DummyTokenizer()
        prompt = format_longbench_prompt(
            tok,
            "hello world",
            dataset="trec",
            model_family="qwen",
            max_model_len=100,
        )
        self.assertEqual(prompt, "hello world")

    def test_qwen_chat_template_wraps_non_exempt_dataset(self):
        tok = DummyTokenizer()
        self.assertEqual(build_chat(tok, "hello", "qwen"), "<CHAT>hello<GEN>")

    def test_variant_defaults_match_plan(self):
        kitty = build_variant(SimpleNamespace(variant="kitty"))
        pro = build_variant(SimpleNamespace(variant="kitty_pro"))
        self.assertEqual(kitty.promote_ratio, 0.125)
        self.assertEqual(pro.promote_ratio, 0.25)
        self.assertEqual(kitty.sink_length, 32)
        self.assertEqual(kitty.buffer_length, 128)
        self.assertEqual(kitty.group_size, 128)

    def test_quest_sim_variant_is_pure_torch_quest(self):
        variant = build_variant(
            SimpleNamespace(variant="quest_kitty_page16_sim", quest_token_budget=2048, quest_skip_layers=0)
        )
        self.assertTrue(variant.use_kitty)
        self.assertTrue(variant.sim_quest)
        self.assertFalse(variant.real_kernel)
        self.assertEqual(variant.name, "quest_kitty_page16_sim")
        # page16 fake-quant config the sim KittyKVCache is built from.
        self.assertEqual(variant.sink_length, 32)
        self.assertEqual(variant.buffer_length, 16)
        self.assertEqual(variant.group_size, 16)
        self.assertEqual(variant.page_size, 16)
        self.assertTrue(variant.quest_enabled)
        self.assertEqual(variant.quest_token_budget, 2048)
        self.assertTrue(variant.tag.endswith("_sim"))

    def test_quest_kernel_variant_is_real_kernel(self):
        variant = build_variant(
            SimpleNamespace(variant="quest_kitty_page16_kernel", quest_token_budget=2048, quest_skip_layers=0)
        )
        self.assertTrue(variant.real_kernel)
        self.assertFalse(variant.sim_quest)
        self.assertEqual(variant.page_size, 16)
        self.assertEqual(variant.quest_token_budget, 2048)
        self.assertTrue(variant.tag.endswith("_kernel"))

    def test_output_model_dir_uses_quest_sim_variant_tag(self):
        variant = build_variant(
            SimpleNamespace(variant="quest_kitty_page16_sim", quest_token_budget=2048, quest_skip_layers=0)
        )

        pred_dir = output_model_dir("longbench_out/pred", "qwen3-8b-gpu1-smoke2", variant)

        self.assertEqual(
            pred_dir.name,
            "qwen3-8b-gpu1-smoke2-quest_kitty_page16_sim_p16_pr0p125_qb2048_qsl0_sim",
        )

    def test_longbench_cli_accepts_quest_sim_and_kernel_variants(self):
        for v in ("quest_kitty_page16_sim", "quest_kitty_page16_kernel"):
            args = build_parser().parse_args(["Qwen/Qwen3-8B", "--variant", v])
            self.assertEqual(args.variant, v)

    def test_longbench_cli_rejects_removed_page16_proxy_variant(self):
        for v in ("kitty_page16", "quest_proxy_kitty_page16"):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["Qwen/Qwen3-8B", "--variant", v])

    def test_latency_parser_accepts_page16_smoke_plumbing(self):
        args = build_latency_parser().parse_args(
            [
                "--cache_implementation",
                "0",
                "--page_size",
                "16",
                "--warmup_runs",
                "1",
                "--repeat_runs",
                "1",
                "--batch_size",
                "1",
                "--promote_ratio",
                "0.125",
                "--quest-enabled",
                "--quest-token-budget",
                "2048",
                "--quest-skip-layers",
                "2",
            ]
        )

        self.assertEqual(args.page_size, 16)
        self.assertEqual(args.promote_ratio, 0.125)
        self.assertTrue(args.quest_enabled)
        self.assertEqual(args.quest_token_budget, 2048)
        self.assertEqual(args.quest_skip_layers, 2)

    def test_flat_prediction_dir_does_not_append_variant_subdir(self):
        variant = build_variant(SimpleNamespace(variant="quest_kitty_page16_sim"))
        self.assertEqual(
            resolve_prediction_dir(
                "longbench_out/llama31-8b-instruct-quest-kitty-sim/pred",
                "ignored",
                variant,
                flat_output_dir=True,
            ),
            Path("longbench_out/llama31-8b-instruct-quest-kitty-sim/pred"),
        )
        self.assertEqual(
            resolve_prediction_dir("longbench_out/root", "model-tag", variant).name,
            f"model-tag-{variant.tag}",
        )

    def test_default_prediction_dir_uses_normalized_smoke_and_full_layout(self):
        variant = build_variant(SimpleNamespace(variant="quest_kitty_page16_sim"))
        self.assertEqual(
            default_prediction_dir("meta-llama/Llama-3.1-8B-Instruct", None, variant, max_samples=1),
            Path("longbench_out/smoke/llama31-8b-instruct-quest-kitty-sim/pred"),
        )
        self.assertEqual(
            default_prediction_dir("meta-llama/Llama-3.1-8B-Instruct", None, variant, max_samples=-1),
            Path("longbench_out/llama31-8b-instruct-quest-kitty-sim/pred"),
        )

    def test_kitty_cache_accepts_short_prefill_without_assertion(self):
        cache = KittyKVCache(
            KittyKVCacheConfig(
                sink_length=32,
                buffer_length=128,
                group_size=128,
                kbits=2,
                vbits=2,
                promote_ratio=0.125,
                promote_bit=4,
                channel_selection=1,
            )
        )
        key = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        value = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        out_k, out_v = cache.update(key, value, layer_idx=0)
        self.assertEqual(out_k.shape, key.shape)
        self.assertEqual(out_v.shape, value.shape)

    def test_kitty_cache_accepts_llama32_head_dim_smaller_than_group_size(self):
        cache = KittyKVCache(
            KittyKVCacheConfig(
                sink_length=32,
                buffer_length=128,
                group_size=128,
                kbits=2,
                vbits=2,
                promote_ratio=0.125,
                promote_bit=4,
                channel_selection=1,
            )
        )
        key = torch.randn(1, 4, 200, 64, dtype=torch.float16)
        value = torch.randn(1, 4, 200, 64, dtype=torch.float16)
        out_k, out_v = cache.update(key, value, layer_idx=0)
        self.assertEqual(out_k.shape, key.shape)
        self.assertEqual(out_v.shape, value.shape)

    def test_fake_quant_allows_smaller_final_group(self):
        data = torch.randn(1, 2, 3, 64, dtype=torch.float16)
        quantized = fake_quant_groupwise_lastdim(data, group_size=128, bit=2)
        self.assertEqual(quantized.shape, data.shape)

    def test_strict_scorer_rejects_incomplete_manifest(self):
        with TemporaryDirectory() as tmp:
            pred_dir = Path(tmp) / "model"
            pred_dir.mkdir()
            jsonl = pred_dir / "passage_count.jsonl"
            jsonl.write_text(
                json.dumps({"pred": "1", "answers": ["1"], "all_classes": [], "length": 10}) + "\n",
                encoding="utf-8",
            )
            jsonl.with_suffix(".manifest.json").write_text(
                json.dumps({"expected_samples": 2, "written_samples": 1, "failed_sample_ids": []}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                score_directory(pred_dir, strict_complete=True)
            self.assertTrue((pred_dir / "result.partial.json").exists())

    def test_scorer_writes_result_for_complete_manifest(self):
        with TemporaryDirectory() as tmp:
            pred_dir = Path(tmp) / "model"
            pred_dir.mkdir()
            jsonl = pred_dir / "passage_count.jsonl"
            rows = [
                {"pred": "1", "answers": ["1"], "all_classes": [], "length": 10},
                {"pred": "2", "answers": ["2"], "all_classes": [], "length": 10},
            ]
            jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            jsonl.with_suffix(".manifest.json").write_text(
                json.dumps({"expected_samples": 2, "written_samples": 2, "failed_sample_ids": []}),
                encoding="utf-8",
            )
            scores = score_directory(pred_dir, strict_complete=True)
            self.assertEqual(scores["passage_count"], 100.0)
            self.assertTrue((pred_dir / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
