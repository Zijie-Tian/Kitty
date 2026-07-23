"""Deterministic CPU contracts for cache-aware perplexity evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import textwrap
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig
from kitty_sim.longbench.runner import VariantConfig, _stable_json_hash
from kitty_sim.ppl.data import build_window_plan, load_wikitext_documents
from kitty_sim.ppl.runner import (
    PPL_PROTOCOL,
    PPLWindow,
    _empty_engagement,
    _merge_engagement,
    evaluate_window_streaming,
    ppl_run_config_payload,
)
from kitty_sim.ppl.scorer import compare_results, score_pred_dir


class _Tokenizer:
    is_fast = True

    def __call__(
        self, text: str, *, add_special_tokens: bool, **_: object
    ) -> dict[str, list[int]]:
        if add_special_tokens:
            raise AssertionError("PPL must not add tokenizer special tokens")
        return {"input_ids": [(ord(char) % 47) + 1 for char in text]}

class _ParquetColumn:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def to_pylist(self) -> list[object]:
        return self.values


class _ParquetTable:
    column_names = ["page"]

    def __init__(self, values: list[object]) -> None:
        self.values = values

    def __getitem__(self, name: str) -> _ParquetColumn:
        if name != "page":
            raise KeyError(name)
        return _ParquetColumn(self.values)


class _FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length


class _FakeCausalLM:
    def __init__(self, *, prefill_tokens: int, vocab_size: int = 11) -> None:
        self.device = torch.device("cpu")
        self.prefill_tokens = prefill_tokens
        self.vocab_size = vocab_size
        self.last_returned_cache: _FakeCache | None = None
        self.prefill_calls = 0
        self.decode_calls = 0
        self.seen_past: list[_FakeCache | None] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: _FakeCache | None,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int | None = None,
    ) -> SimpleNamespace:
        self.seen_past.append(past_key_values)
        query_length = int(input_ids.shape[-1])
        if past_key_values is None:
            self.prefill_calls += 1
            if query_length != self.prefill_tokens:
                raise AssertionError("expected one dense prefill")
            next_length = query_length
        else:
            self.decode_calls += 1
            if query_length != 1:
                raise AssertionError("scoring must use one-token decode calls")
            if past_key_values is not self.last_returned_cache:
                raise AssertionError("runner did not hand off the returned cache object")
            next_length = past_key_values.length + query_length
        if attention_mask.shape[-1] != next_length:
            raise AssertionError("attention mask/cache length mismatch")

        current = int(input_ids[0, -1])
        logits = torch.zeros((1, 1, self.vocab_size), dtype=torch.float32)
        logits[0, 0, (current + 1) % self.vocab_size] = 2.0
        next_cache = _FakeCache(next_length)
        self.last_returned_cache = next_cache
        return SimpleNamespace(logits=logits, past_key_values=next_cache)

    __call__ = forward


class TestPPLData(unittest.TestCase):
    def test_document_loading_and_windows_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "test.parquet"
            data_path.write_bytes(b"fixture")
            table = _ParquetTable(
                [" = Long Article = \nabcdefghijklmnopqrstuvwxyz", "short"]
            )
            with mock.patch("pyarrow.parquet.read_table", return_value=table):
                documents = load_wikitext_documents(data_path)
            tokenizer = _Tokenizer()
            first = build_window_plan(
                documents,
                tokenizer,
                prefill_tokens=5,
                score_tokens=3,
                max_samples=2,
            )
            second = build_window_plan(
                documents,
                tokenizer,
                prefill_tokens=5,
                score_tokens=3,
                max_samples=2,
            )

        self.assertEqual(first, second)
        self.assertEqual(first.window_tokens, 9)
        self.assertEqual(first.selected_windows, 2)
        self.assertEqual(first.skipped_short_documents, 1)
        self.assertEqual([window.start_token for window in first.windows], [0, 9])
        self.assertEqual(len(first.selected_token_sha256), 64)
        self.assertEqual(len(first.windows[0].token_ids), 9)

    def test_window_hash_changes_with_targets(self) -> None:
        tokenizer = _Tokenizer()
        docs_a = [
            SimpleNamespace(source_index=0, document_id="doc", text="abcdefghijklm")
        ]
        docs_b = [
            SimpleNamespace(source_index=0, document_id="doc", text="xbcdefghijklm")
        ]
        plan_a = build_window_plan(docs_a, tokenizer, prefill_tokens=4, score_tokens=2)
        plan_b = build_window_plan(docs_b, tokenizer, prefill_tokens=4, score_tokens=2)
        self.assertNotEqual(plan_a.selected_token_sha256, plan_b.selected_token_sha256)


class TestStreamingPPL(unittest.TestCase):
    def setUp(self) -> None:
        self.variant = VariantConfig(name="fp16", use_kitty=False)
        self.prefill_tokens = 4
        self.score_tokens = 3
        self.token_ids = tuple(index % 11 for index in range(8))
        self.window = PPLWindow(
            sample_idx=0,
            source_index=0,
            document_id="doc",
            window_index=0,
            start_token=0,
            token_ids=self.token_ids,
        )

    def test_alignment_scores_only_next_tokens_after_prefill(self) -> None:
        model = _FakeCausalLM(prefill_tokens=self.prefill_tokens)
        result = evaluate_window_streaming(
            model=model,
            window=self.window,
            variant=self.variant,
            prefill_tokens=self.prefill_tokens,
            score_tokens=self.score_tokens,
        )

        expected = 0.0
        for token_index in range(
            self.prefill_tokens, self.prefill_tokens + self.score_tokens
        ):
            logits = torch.zeros((1, 11), dtype=torch.float32)
            logits[0, (self.token_ids[token_index] + 1) % 11] = 2.0
            target = torch.tensor([self.token_ids[token_index + 1]])
            expected += float(F.cross_entropy(logits, target, reduction="sum"))
        self.assertAlmostEqual(result.nll_sum, expected, places=6)
        self.assertEqual(result.scored_tokens, self.score_tokens)
        self.assertEqual(result.prefill_cache_length, self.prefill_tokens)
        self.assertEqual(
            result.final_cache_length, self.prefill_tokens + self.score_tokens
        )
        self.assertTrue(result.cache_length_monotonic)
        self.assertEqual(model.prefill_calls, 1)
        self.assertEqual(model.decode_calls, self.score_tokens)
        self.assertIsNone(model.seen_past[0])
        self.assertTrue(all(cache is not None for cache in model.seen_past[1:]))

    def test_each_document_window_starts_with_a_fresh_cache(self) -> None:
        model = _FakeCausalLM(prefill_tokens=self.prefill_tokens)
        first = evaluate_window_streaming(
            model=model,
            window=self.window,
            variant=self.variant,
            prefill_tokens=self.prefill_tokens,
            score_tokens=self.score_tokens,
        )
        second_window = PPLWindow(
            **{**self.window.__dict__, "sample_idx": 1, "document_id": "other"}
        )
        second = evaluate_window_streaming(
            model=model,
            window=second_window,
            variant=self.variant,
            prefill_tokens=self.prefill_tokens,
            score_tokens=self.score_tokens,
        )
        fresh = evaluate_window_streaming(
            model=_FakeCausalLM(prefill_tokens=self.prefill_tokens),
            window=second_window,
            variant=self.variant,
            prefill_tokens=self.prefill_tokens,
            score_tokens=self.score_tokens,
        )
        self.assertEqual(model.prefill_calls, 2)
        self.assertIsNone(model.seen_past[0])
        self.assertIsNone(model.seen_past[self.score_tokens + 1])
        self.assertEqual(first.nll_sum, second.nll_sum)
        self.assertEqual(second.nll_sum, fresh.nll_sum)


class TestPPLCacheEngagement(unittest.TestCase):
    def test_kitty_cache_counters_increment_and_reset(self) -> None:
        config = KittyKVCacheConfig(
            sink_length=1,
            buffer_length=2,
            group_size=2,
            kbits=2,
            vbits=2,
            promote_ratio=0.5,
            promote_bit=4,
            channel_selection=1,
        )
        cache = KittyKVCache(config)
        torch.manual_seed(7)
        key = torch.randn(1, 2, 8, 4, dtype=torch.float16)
        value = torch.randn(1, 2, 8, 4, dtype=torch.float16)
        cache.update(key, value, 0)
        self.assertEqual(cache.prefill_updates, 1)
        self.assertGreater(cache.k_quant_calls, 0)
        self.assertGreater(cache.k_quantized_tokens, 0)
        self.assertGreater(cache.v_quant_calls, 0)
        self.assertGreater(cache.v_quantized_tokens, 0)
        self.assertIsNotNone(cache.last_k_quant_mode)
        cache.update(key[:, :, :1], value[:, :, :1], 0)
        self.assertEqual(cache.decode_updates, 1)

        cache.reset()
        self.assertEqual(cache.prefill_updates, 0)
        self.assertEqual(cache.decode_updates, 0)
        self.assertEqual(cache.k_quant_calls, 0)
        self.assertEqual(cache.k_quantized_tokens, 0)
        self.assertEqual(cache.v_quant_calls, 0)
        self.assertEqual(cache.v_quantized_tokens, 0)
        self.assertIsNone(cache.last_k_quant_mode)

    def test_prompt_mean_evidence_survives_manifest_aggregation(self) -> None:
        aggregate = _empty_engagement()
        _merge_engagement(
            aggregate,
            {"k_prompt_mean_layers": 16},
            decode_steps=1,
        )
        self.assertEqual(aggregate["k_prompt_mean_layers"], 16)


class TestPPLFingerprints(unittest.TestCase):
    def _args(self, model_dir: Path, data_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            model=str(model_dir),
            model_path=str(model_dir),
            model_tag="fixture-model",
            model_family="llama3",
            local_files_only=True,
            data_path=str(data_path),
            text_field="page",
            prefill_tokens=4,
            score_tokens=2,
            max_samples=2,
            max_model_len=64,
            torch_dtype="float16",
        )

    def test_methods_share_comparison_hash_but_not_run_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            data_path = root / "test.parquet"
            data_path.write_bytes(b"corpus-v1")
            args = self._args(model_dir, data_path)
            documents = [
                SimpleNamespace(
                    source_index=0,
                    document_id="doc",
                    title=None,
                    text="abcdefghijklmnopqrstuvwxyz",
                )
            ]
            config = SimpleNamespace(
                max_position_embeddings=64,
                hidden_size=16,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_hidden_layers=2,
            )
            with (
                mock.patch(
                    "kitty_sim.ppl.runner.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
                mock.patch(
                    "kitty_sim.ppl.runner.AutoConfig.from_pretrained",
                    return_value=config,
                ),
                mock.patch(
                    "kitty_sim.ppl.runner.load_wikitext_documents",
                    return_value=documents,
                ),
            ):
                fp16 = ppl_run_config_payload(
                    args, VariantConfig(name="fp16", use_kitty=False)
                )
                kitty = ppl_run_config_payload(
                    args, VariantConfig(name="kitty", use_kitty=True)
                )
                self.assertEqual(
                    fp16["comparison_config_hash"], kitty["comparison_config_hash"]
                )
                self.assertNotEqual(_stable_json_hash(fp16), _stable_json_hash(kitty))
                data_path.write_bytes(b"corpus-v2")
                changed = ppl_run_config_payload(
                    args, VariantConfig(name="fp16", use_kitty=False)
                )
                self.assertNotEqual(
                    fp16["comparison_config_hash"], changed["comparison_config_hash"]
                )


class TestPPLScorer(unittest.TestCase):
    def _write_arm(self, root: Path, method: str, nlls: list[float]) -> Path:
        pred_dir = root / method / "pred"
        pred_dir.mkdir(parents=True)
        run_config = {
            "corpus": "wikitext-2-raw-v1",
            "split": "test",
            "model_slug": "fixture-model",
            "comparison_config_hash": "same-target",
            "expected_samples": len(nlls),
            "expected_scored_tokens": len(nlls) * 2,
            "window_tokens": 6,
            "prefill_tokens": 3,
            "score_tokens": 2,
        }
        rows = [
            {
                "sample_idx": index,
                "input_tokens": 6,
                "scored_tokens": 2,
                "nll_sum": nll,
                "cache_length_monotonic": True,
                "prefill_cache_length": 3,
                "final_cache_length": 5,
            }
            for index, nll in enumerate(nlls)
        ]
        jsonl_path = pred_dir / "wikitext2.jsonl"
        jsonl_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        variant = VariantConfig(name=method, use_kitty=False)
        manifest = {
            "status": "ok",
            "benchmark": "ppl",
            "protocol": PPL_PROTOCOL,
            "expected_samples": len(rows),
            "written_samples": len(rows),
            "method_slug": method,
            "variant": asdict(variant),
            "engagement": {},
            "comparison_config_hash": "same-target",
            "run_config_hash": _stable_json_hash(run_config),
            "run_config": run_config,
            "jsonl_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        }
        jsonl_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return pred_dir

    def test_aggregates_nll_before_exponentiation_and_compares_to_fp16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp16 = score_pred_dir(self._write_arm(root, "fp16", [2.0, 4.0]))
            quant = score_pred_dir(self._write_arm(root, "qlutattn", [3.0, 5.0]))
            self.assertAlmostEqual(fp16["avg_nll"], 1.5)
            self.assertAlmostEqual(fp16["token_ppl"], math.exp(1.5))
            comparison = compare_results([quant, fp16])
            quant_row = next(
                row for row in comparison["rows"] if row["method"] == "qlutattn"
            )
            self.assertAlmostEqual(quant_row["delta_nll_vs_fp16"], 0.5)
            self.assertAlmostEqual(quant_row["ppl_ratio_vs_fp16"], math.exp(0.5))

    def test_rejects_checksum_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = self._write_arm(Path(tmp), "fp16", [2.0])
            with (pred_dir / "wikitext2.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                score_pred_dir(pred_dir)

    def test_qlutattn_requires_persisted_prompt_mean_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = self._write_arm(Path(tmp), "qlutattn", [2.0])
            manifest_path = pred_dir / "wikitext2.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["variant"]["v_tile_channels"] = 64
            manifest["variant"]["use_kitty"] = True
            manifest["engagement"] = {
                "decode_steps": 2,
                "k_quantized_tokens": 8,
                "v_quantized_tokens": 8,
                "v_tile_blocks": 1,
                "last_v_quant_mode": "tile16_rescued",
                "last_v_tile_channels": 64,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prompt-mean"):
                score_pred_dir(pred_dir)

            manifest["engagement"]["k_prompt_mean_layers"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = score_pred_dir(pred_dir)
            self.assertEqual(result["engagement"]["k_prompt_mean_layers"], 1)

    def test_rejects_mismatched_comparison_target(self) -> None:
        baseline = {
            "method_slug": "fp16",
            "comparison_config_hash": "a",
            "avg_nll": 1.0,
            "token_ppl": math.e,
        }
        quant = {
            "method_slug": "kitty",
            "comparison_config_hash": "b",
            "avg_nll": 1.1,
            "token_ppl": math.exp(1.1),
        }
        with self.assertRaisesRegex(RuntimeError, "comparison_config_hash"):
            compare_results([baseline, quant])


class TestPPLMultiModelShellScheduling(unittest.TestCase):
    def test_partitions_gpus_and_launches_all_models_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            barrier_dir = root / "barrier"
            barrier_dir.mkdir()
            invocation_log = root / "invocations.tsv"
            data_path = root / "wikitext.parquet"
            data_path.write_bytes(b"fixture")

            model_paths = {
                "KITTY_LLAMA32_1B_PATH": root / "llama1b",
                "KITTY_LLAMA32_3B_PATH": root / "llama3b",
                "KITTY_MINICPM5_1B_PATH": root / "minicpm",
            }
            mask_paths = {
                "KITTY_LLAMA32_1B_QLUTATTN_MASK": root / "llama1b-mask.pt",
                "KITTY_LLAMA32_3B_QLUTATTN_MASK": root / "llama3b-mask.pt",
                "KITTY_MINICPM5_1B_QLUTATTN_MASK": root / "minicpm-mask.pt",
            }
            for path in model_paths.values():
                path.mkdir()
            for path in mask_paths.values():
                path.write_bytes(b"mask")

            fake_runner = root / "fake-run-ppl.sh"
            fake_runner.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -Eeuo pipefail

                    value() {
                      local wanted="$1"
                      shift
                      while [[ $# -gt 0 ]]; do
                        if [[ "$1" == "${wanted}" ]]; then
                          printf '%s' "$2"
                          return 0
                        fi
                        shift
                      done
                      return 1
                    }

                    tag="$(value --model-tag "$@")"
                    gpus="$(value --gpus "$@")"
                    variants="$(value --variants "$@")"
                    touch "${PPL_TEST_BARRIER}/${tag}.started"
                    shopt -s nullglob
                    ready=0
                    for _ in {1..200}; do
                      started=("${PPL_TEST_BARRIER}"/*.started)
                      if [[ "${#started[@]}" -eq "${PPL_EXPECTED_MODELS}" ]]; then
                        ready=1
                        break
                      fi
                      sleep 0.01
                    done
                    [[ "${ready}" == "1" ]]
                    printf '%s\\t%s\\t%s\\t%s\\n' \
                      "${tag}" "${gpus}" "${variants}" "${QLUT_CB_MASK}" \
                      >> "${PPL_TEST_LOG}"
                    """
                ),
                encoding="utf-8",
            )
            fake_runner.chmod(0o755)

            env = os.environ.copy()
            env.update({name: str(path) for name, path in model_paths.items()})
            env.update({name: str(path) for name, path in mask_paths.items()})
            env.update(
                {
                    "PPL_DATA_PATH": str(data_path),
                    "PPL_MODEL_RUNNER_TEST_ONLY": str(fake_runner),
                    "PPL_TESTING": "1",
                    "PPL_TEST_BARRIER": str(barrier_dir),
                    "PPL_TEST_LOG": str(invocation_log),
                    "PPL_EXPECTED_MODELS": "3",
                }
            )
            command = [
                "bash",
                str(Path(__file__).parents[1] / "scripts" / "run_ppl_models.sh"),
                "--models",
                "llama32-1b,llama32-3b,minicpm5-1b",
                "--variants",
                "fp16,shadowkv,kivi,qlutattn",
                "--gpus",
                "0,1,2,3,4,5",
                "--max-samples",
                "1",
                "--out-root",
                str(root / "out"),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )

            rows = {}
            for line in invocation_log.read_text(encoding="utf-8").splitlines():
                tag, gpus, variants, mask = line.split("\t")
                rows[tag] = (gpus, variants, mask)
            expected_variants = "fp16,shadowkv,kivi,qlutattn"
            self.assertEqual(
                rows,
                {
                    "llama32-1b-instruct": (
                        "0,3",
                        expected_variants,
                        str(mask_paths["KITTY_LLAMA32_1B_QLUTATTN_MASK"]),
                    ),
                    "llama32-3b-instruct": (
                        "1,4",
                        expected_variants,
                        str(mask_paths["KITTY_LLAMA32_3B_QLUTATTN_MASK"]),
                    ),
                    "minicpm5-1b": (
                        "2,5",
                        expected_variants,
                        str(mask_paths["KITTY_MINICPM5_1B_QLUTATTN_MASK"]),
                    ),
                },
            )
            self.assertIn("all model matrices completed successfully", completed.stdout)

    def test_termination_reaps_model_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "wikitext.parquet"
            data_path.write_bytes(b"fixture")
            model_path = root / "llama1b"
            model_path.mkdir()
            child_pid_path = root / "child.pid"

            fake_runner = root / "fake-run-ppl.sh"
            fake_runner.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -Eeuo pipefail
                    sleep 60 &
                    child_pid=$!
                    printf '%s\n' "${child_pid}" > "${PPL_TEST_CHILD_PID}"
                    cleanup() {
                      trap - INT TERM
                      kill "${child_pid}" 2>/dev/null || true
                      wait "${child_pid}" 2>/dev/null || true
                      exit 143
                    }
                    trap cleanup INT TERM
                    wait "${child_pid}"
                    """
                ),
                encoding="utf-8",
            )
            fake_runner.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "KITTY_LLAMA32_1B_PATH": str(model_path),
                    "PPL_DATA_PATH": str(data_path),
                    "PPL_MODEL_RUNNER_TEST_ONLY": str(fake_runner),
                    "PPL_TESTING": "1",
                    "PPL_TEST_CHILD_PID": str(child_pid_path),
                }
            )
            command = [
                "bash",
                str(Path(__file__).parents[1] / "scripts" / "run_ppl_models.sh"),
                "--models",
                "llama32-1b",
                "--variants",
                "fp16,kivi",
                "--gpus",
                "0",
                "--max-samples",
                "1",
                "--out-root",
                str(root / "out"),
            ]
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            child_pid: int | None = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not child_pid_path.exists():
                    self.assertIsNone(process.poll(), "launcher exited before fake runner started")
                    time.sleep(0.01)
                self.assertTrue(child_pid_path.exists(), "fake runner did not publish its child PID")
                child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())

                process.terminate()
                self.assertNotEqual(process.wait(timeout=10), 0)
                for _ in range(100):
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.01)
                else:
                    self.fail(f"model runner child {child_pid} survived launcher termination")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, 9)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
