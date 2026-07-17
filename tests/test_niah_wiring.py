"""0-GPU tests for the RULER-NIAH eval path: data, prompts, scoring, hashes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kitty_sim.longbench.runner import build_variant
from kitty_sim.niah.data import load_niah_records, resolve_niah_file
from kitty_sim.niah.runner import (
    build_niah_prompt,
    niah_run_config_hash,
)
from kitty_sim.niah.scorer import (
    plot_depth_heatmap,
    score_pred_dir,
    string_match_all_score,
)


def _write_niah_file(root: Path, task: str, seq_len: int, records: list[dict]) -> Path:
    path = root / str(seq_len) / task / "validation.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _record(value: str = "1234567", tpa: int = 100, length: int = 4000) -> dict:
    return {
        "index": 0,
        "input": f"haystack text. One of the special magic numbers for x is: {value}. "
                 "What is the special magic number for x mentioned in the provided text?",
        "outputs": [value],
        "length": length,
        "answer_prefix": " The special magic number for x mentioned in the provided text is",
        "token_position_answer": tpa,
    }


class _StubTokenizer:
    bos_token = "<s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return f"<s>[U]{messages[0]['content']}[/U][A]"


class TestNiahData(unittest.TestCase):
    def test_load_ok_and_missing_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_niah_file(root, "niah_single_1", 4096, [_record()])
            records = load_niah_records("niah_single_1", 4096, data_root=root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["outputs"], ["1234567"])

            bad = _record()
            bad.pop("answer_prefix")
            _write_niah_file(root, "niah_single_2", 4096, [bad])
            with self.assertRaisesRegex(ValueError, "answer_prefix"):
                load_niah_records("niah_single_2", 4096, data_root=root)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_niah_records("niah_single_1", 8192, data_root=tmp)

    def test_resolve_path_layout(self):
        path = resolve_niah_file("niah_single_3", 32768, data_root="/x")
        self.assertEqual(str(path), "/x/32768/niah_single_3/validation.jsonl")


class TestNiahPrompt(unittest.TestCase):
    def test_llama3_family_is_raw_plus_answer_prefix(self):
        record = _record()
        prompt = build_niah_prompt(_StubTokenizer(), record, "llama3")
        self.assertEqual(prompt, record["input"] + record["answer_prefix"])

    def test_chat_family_appends_prefix_after_assistant_header(self):
        record = _record()
        prompt = build_niah_prompt(_StubTokenizer(), record, "llama3.2")
        self.assertEqual(
            prompt, f"<s>[U]{record['input']}[/U][A]{record['answer_prefix']}"
        )


class TestNiahScorer(unittest.TestCase):
    def test_string_match_all(self):
        self.assertEqual(string_match_all_score("the answer is 42.", ["42"]), 1.0)
        self.assertEqual(string_match_all_score("no idea", ["42"]), 0.0)
        self.assertEqual(
            string_match_all_score(
                "found 1111111 and 3333333", ["1111111", "2222222", "3333333", "4444444"]
            ),
            0.5,
        )
        with self.assertRaises(ValueError):
            string_match_all_score("x", [])

    def test_score_dir_and_heatmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp)
            rows = [
                # depth 0.025 -> bin 0 (hit), depth 0.975 -> bin 9 (miss)
                {"pred": "is 111", "outputs": ["111"], "token_position_answer": 100,
                 "length": 4000},
                {"pred": "wrong", "outputs": ["222"], "token_position_answer": 3900,
                 "length": 4000},
            ]
            with (pred_dir / "niah_single_1__4096.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            result = score_pred_dir(pred_dir, n_depth_bins=10)
            self.assertEqual(result["scores"]["niah_single_1"]["4096"], 50.0)
            self.assertEqual(result["per_len_mean"]["4096"], 50.0)
            matrix = result["depth_matrix_pooled"]["acc"]
            self.assertEqual(matrix[0][0], 100.0)
            self.assertEqual(matrix[9][0], 0.0)
            self.assertIsNone(matrix[5][0])
            png = plot_depth_heatmap(
                result, pred_dir / "hm.png", title="t", task="niah_single_1"
            )
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 0)

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                score_pred_dir(Path(tmp))


def _variant_args(**overrides):
    ns = SimpleNamespace(
        variant="fp16",
        kbits=2,
        vbits=2,
        promote_bit=4,
        promote_ratio=None,
        promote_ratio_config=None,
        sink_length=32,
        buffer_length=128,
        group_size=128,
        channel_selection=0,
        k_quant_mode="per_token",
        quest_kernel=False,
        quest_token_budget=None,
        quest_skip_layers=0,
        shadowkv_budget=None,
        shadowkv_rank=None,
        shadowkv_chunk_size=None,
        model="meta-llama/Llama-3.2-1B-Instruct",
        model_path=None,
        model_tag="llama32-1b-instruct",
        model_family="llama3",
        data_root=None,
        max_samples=-1,
        max_model_len=32768,
        max_new_tokens=128,
        torch_dtype="float16",
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


class TestNiahRunConfigHash(unittest.TestCase):
    def test_hash_changes_with_task_len_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_niah_file(root, "niah_single_1", 4096, [_record("111")])
            _write_niah_file(root, "niah_single_1", 8192, [_record("111")])
            _write_niah_file(root, "niah_single_2", 4096, [_record("222")])
            args = _variant_args(data_root=str(root))
            variant = build_variant(args)
            h_base = niah_run_config_hash(args, variant, "niah_single_1", 4096)
            self.assertEqual(
                h_base, niah_run_config_hash(args, variant, "niah_single_1", 4096)
            )
            self.assertNotEqual(
                h_base, niah_run_config_hash(args, variant, "niah_single_1", 8192)
            )
            self.assertNotEqual(
                h_base, niah_run_config_hash(args, variant, "niah_single_2", 4096)
            )
            args2 = _variant_args(data_root=str(root), max_samples=1)
            self.assertNotEqual(
                h_base, niah_run_config_hash(args2, variant, "niah_single_1", 4096)
            )


class TestEvalNiahCli(unittest.TestCase):
    def test_parser_defaults_and_vbits_env(self):
        import os
        from unittest import mock

        from kitty_sim.cli.eval_niah import build_parser, finalize_args

        with mock.patch.dict(os.environ, {"VBITS": ""}, clear=False):
            args = finalize_args(
                build_parser().parse_args(["m", "--variant", "fp16"])
            )
        self.assertEqual(args.vbits, 2)
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.max_model_len, 32768)
        self.assertFalse(args.quest_kernel)
        with mock.patch.dict(os.environ, {"VBITS": "4"}, clear=False):
            args = finalize_args(
                build_parser().parse_args(["m", "--variant", "fp16"])
            )
        self.assertEqual(args.vbits, 4)


if __name__ == "__main__":
    unittest.main()
