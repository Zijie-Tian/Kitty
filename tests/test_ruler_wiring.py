"""Deterministic 0-GPU contract tests for the clean-cutover RULER path."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from kitty_sim.longbench.runner import build_variant, method_layout_slug
from kitty_sim.longbench.templates import infer_model_family
from kitty_sim.ruler import runner as ruler_runner
from kitty_sim.ruler.data import (
    load_ruler_records,
    resolve_ruler_file,
    resolve_ruler_manifest,
)
from kitty_sim.ruler.runner import (
    build_ruler_prompt,
    generate_ruler_pair,
    pair_name,
    resolve_ruler_preflight,
    ruler_run_config_hash,
    select_task_shard,
    tokenizer_config_hashes,
    tokenizer_identity_sha256,
)
from kitty_sim.ruler.scorer import (
    score_pred_dir,
    string_match_all,
    string_match_all_score,
    string_match_part,
    string_match_part_score,
)
from kitty_sim.ruler.tasks import (
    DEFAULT_TASKS,
    DEFAULT_SEQ_LENS,
    NIAH_TASKS,
    TASK_SPECS,
    TASK_VERSION,
    get_task_spec,
)


_EXPECTED_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)

_EXPECTED_CAPS = {
    **{task: 128 for task in _EXPECTED_TASKS[:8]},
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}


def _record(
    *,
    task: str = "niah_single_1",
    index: int = 0,
    value: str = "1234567",
    gen_prefix: str = " Answer:",
) -> dict:
    record = {
        "index": index,
        "input": f"haystack with the value {value}",
        "outputs": [value],
        "length": 3800,
        "max_length": 3840,
        "gen_prefix": gen_prefix,
    }
    if task in NIAH_TASKS:
        record["token_position_answer"] = 960
    return record


def _jsonl_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_data_pair(
    root: Path,
    task: str,
    seq_len: int,
    records: list[dict],
    *,
    tokenizer_hashes: dict[str, str] | None = None,
) -> Path:
    path = root / str(seq_len) / task / "validation.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tokenizer_hashes = tokenizer_hashes or {"tokenizer.json": "0" * 64}
    manifest = {
        "schema_version": 1,
        "task": task,
        "task_version": get_task_spec(task).version,
        "nominal_length": seq_len,
        "generated_length": max(
            int(record.get("max_length", seq_len)) for record in records
        ),
        "max_new_tokens": get_task_spec(task).max_new_tokens,
        "count": len(records),
        "seed": 42,
        "model_family": "llama3",
        "model_tag": "fixture-model",
        "model_path": "fixture",
        "tokenizer": {
            "algorithm": "ruler-tokenizer-config-v2",
            "requested_path": "fixture",
            "resolved_path": "fixture",
            "requested_use_fast": True,
            "class": "fixture.Tokenizer",
            "is_fast": True,
            "vocab_size": 1,
            "model_max_length": 4096,
        },
        "tokenizer_config_hashes": tokenizer_hashes,
        "tokenizer_identity_sha256": tokenizer_identity_sha256(
            tokenizer_hashes,
            requested_use_fast=True,
            tokenizer_class="fixture.Tokenizer",
            is_fast=True,
        ),
        "source_hashes": {},
        "generator_hashes": {"fixture.py": "0" * 64},
        "generation_config_sha256": "0" * 64,
        "jsonl_sha256": _jsonl_sha256(path),
    }
    path.with_name("validation.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return path


def _write_prediction_pair(
    pred_dir: Path,
    task: str,
    seq_len: int,
    rows: list[dict],
    *,
    status: str = "ok",
    expected_samples: int | None = None,
) -> Path:
    path = pred_dir / f"{task}__{seq_len}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    expected = len(rows) if expected_samples is None else expected_samples
    run_config = {
        "benchmark": "ruler",
        "task": task,
        "seq_len": seq_len,
        "expected_samples": expected,
    }
    run_config_hash = hashlib.sha256(
        json.dumps(
            run_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "status": status,
                "benchmark": "ruler",
                "task": task,
                "seq_len": seq_len,
                "expected_samples": expected,
                "written_samples": len(rows),
                "run_config_hash": run_config_hash,
                "run_config": run_config,
                "jsonl_sha256": _jsonl_sha256(path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _variant_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
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
        model="fixture-model",
        model_path=None,
        model_tag="fixture-model",
        model_family="llama3",
        data_root=None,
        tasks="niah_single_1",
        seq_lens="4096",
        max_samples=-1,
        max_model_len=32768,
        torch_dtype="float16",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class _StubChatTokenizer:
    bos_token = "<s>"

    def __init__(self) -> None:
        self.messages = None
        self.tokenize = None
        self.add_generation_prompt = None
        self.date_string = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=None,
    ):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.date_string = date_string
        assert not tokenize and add_generation_prompt
        return f"<s>[U]{messages[0]['content']}[/U][A]"


class _TokenBatch(dict):
    def __init__(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class _GenerationTokenizer(_StubChatTokenizer):
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, _prompt, **_kwargs):
        return _TokenBatch()

    def decode(self, _tokens, skip_special_tokens=True):
        assert skip_special_tokens
        return "generated answer"


class _FakeModel:
    device = torch.device("cpu")

    def __init__(self, on_generate=None) -> None:
        self.generate_calls = 0
        self.last_generate_kwargs = None
        self.on_generate = on_generate

    def generate(self, input_ids, **kwargs):
        self.generate_calls += 1
        self.last_generate_kwargs = kwargs
        if self.on_generate is not None:
            self.on_generate(kwargs)
        suffix = torch.tensor([[9]], dtype=input_ids.dtype)
        return torch.cat((input_ids, suffix), dim=1)

class _FakeShadowCache:
    def get_seq_length(self):
        return 3


class TestRulerRegistry(unittest.TestCase):
    def test_exact_registry_caps_metrics_depth_and_versions(self):
        self.assertEqual(tuple(DEFAULT_TASKS), _EXPECTED_TASKS)
        self.assertEqual(tuple(TASK_SPECS), _EXPECTED_TASKS)
        self.assertEqual(set(NIAH_TASKS), set(_EXPECTED_TASKS[:8]))

        for task in _EXPECTED_TASKS:
            spec = get_task_spec(task)
            self.assertEqual(spec.name, task)
            self.assertEqual(spec.max_new_tokens, _EXPECTED_CAPS[task])
            self.assertEqual(
                spec.metric,
                "string_match_part" if task.startswith("qa_") else "string_match_all",
            )
            self.assertEqual(spec.depth_eligible, task in NIAH_TASKS)
            self.assertTrue(spec.version)

    def test_pair_name_is_unambiguous(self):
        self.assertEqual(pair_name("niah_multiquery", 32768), "niah_multiquery__32768")

    def test_task_shards_are_disjoint_complete_and_stable(self):
        tasks = _EXPECTED_TASKS[:8]
        shards = tuple(select_task_shard(tasks, index, 3) for index in range(3))

        self.assertEqual(shards[0], tasks[0::3])
        self.assertEqual(shards[1], tasks[1::3])
        self.assertEqual(shards[2], tasks[2::3])
        flattened = tuple(task for shard in shards for task in shard)
        self.assertCountEqual(flattened, tasks)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(select_task_shard(tasks, None, None), tasks)

    def test_invalid_or_empty_task_shards_are_rejected(self):
        tasks = _EXPECTED_TASKS[:2]
        invalid = (
            (0, None, "set together"),
            (None, 2, "set together"),
            (0, 0, "must be positive"),
            (-1, 2, "must be in"),
            (2, 2, "must be in"),
            (2, 3, "is empty"),
        )
        for shard_index, shard_count, message in invalid:
            with self.subTest(shard_index=shard_index, shard_count=shard_count):
                with self.assertRaisesRegex(ValueError, message):
                    select_task_shard(tasks, shard_index, shard_count)

    def test_data_prep_imports_registry_and_uses_its_generation_cap(self):
        script_path = Path(__file__).parents[1] / "scripts" / "prepare_ruler_data.py"
        module_spec = importlib.util.spec_from_file_location(
            "_kitty_prepare_ruler_data_test", script_path
        )
        assert module_spec is not None and module_spec.loader is not None
        prep = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(prep)

        self.assertIs(prep.DEFAULT_TASKS, DEFAULT_TASKS)
        self.assertIs(prep.DEFAULT_SEQ_LENS, DEFAULT_SEQ_LENS)
        self.assertIs(prep.NIAH_TASKS, NIAH_TASKS)
        self.assertEqual(prep.TASK_VERSION, TASK_VERSION)

        canonical_cap = get_task_spec("vt").max_new_tokens
        matching_backend = {
            "vt_utils": SimpleNamespace(
                CONFIG={
                    "variable_tracking": {
                        "tokens_to_generate": canonical_cap,
                    }
                }
            )
        }
        self.assertEqual(
            prep._task_max_new_tokens("vt", matching_backend), canonical_cap
        )

        mismatched_backend = {
            "vt_utils": SimpleNamespace(
                CONFIG={
                    "variable_tracking": {
                        "tokens_to_generate": canonical_cap + 1,
                    }
                }
            )
        }
        with self.assertRaisesRegex(
            prep.PrepError, rf"canonical registry cap {canonical_cap}"
        ):
            prep._task_max_new_tokens("vt", mismatched_backend)


class TestRulerShellScheduling(unittest.TestCase):
    def test_full_rerun_replaces_worker_evidence_when_gpu_count_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_python = root / "fake-python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import json
                    import os
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    if args and args[0] == "-c":
                        os.execv(sys.executable, [sys.executable, *args])

                    def value(flag):
                        return args[args.index(flag) + 1]

                    module = args[1] if len(args) > 1 and args[0] == "-m" else None
                    if module == "kitty_sim.cli.preflight_ruler":
                        print(json.dumps({{
                            "method_slug": "fp16",
                            "model_slug": "fixture",
                            "preflight_hash": "preflight-hash",
                            "canonical_variant": "fp16",
                            "tasks": ["niah_single_1", "vt", "qa_1"],
                            "pairs": {{"niah_single_1__4096": {{}}}},
                        }}))
                    elif module == "kitty_sim.cli.eval_ruler":
                        report_path = Path(value("--report-json"))
                        report_path.write_text(json.dumps({{
                            "status": "ok",
                            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                            "task_shard": {{
                                "index": int(value("--task-shard-index")),
                                "count": int(value("--task-shard-count")),
                            }},
                        }}), encoding="utf-8")
                        print("fake worker ok")
                    elif module == "kitty_sim.cli.score_ruler":
                        print("fake scorer ok")
                    else:
                        raise SystemExit(f"unexpected fake-python invocation: {{args}}")
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            out_root = root / "out"
            command = [
                "bash",
                str(Path(__file__).parents[1] / "scripts" / "run_ruler.sh"),
                "--model",
                "fixture",
                "--model-path",
                str(root / "model"),
                "--model-tag",
                "fixture",
                "--model-family",
                "llama3.2",
                "--variants",
                "fp16",
                "--tasks",
                "niah_single_1,vt,qa_1",
                "--lengths",
                "4096",
                "--data-root",
                str(root / "data"),
                "--out-root",
                str(out_root),
                "--max-samples",
                "-1",
                "--max-model-len",
                "4096",
            ]
            env = os.environ.copy()
            env["PYTHON_BIN"] = str(fake_python)

            subprocess.run(
                [*command, "--gpus", "0,1,2"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            logs_dir = out_root / "fixture_fp16" / "logs"
            (logs_dir / "report.json").write_text("legacy", encoding="utf-8")
            (logs_dir / "run.log").write_text("legacy", encoding="utf-8")
            (logs_dir / "keep.txt").write_text("keep", encoding="utf-8")

            subprocess.run(
                [*command, "--gpus", "0,1"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )

            self.assertEqual(
                {path.name for path in logs_dir.glob("worker-*.report.json")},
                {
                    "worker-0-of-2-gpu0.report.json",
                    "worker-1-of-2-gpu1.report.json",
                },
            )
            self.assertEqual(
                {path.name for path in logs_dir.glob("worker-*.run.log")},
                {
                    "worker-0-of-2-gpu0.run.log",
                    "worker-1-of-2-gpu1.run.log",
                },
            )
            self.assertFalse((logs_dir / "report.json").exists())
            self.assertFalse((logs_dir / "run.log").exists())
            self.assertEqual((logs_dir / "keep.txt").read_text(encoding="utf-8"), "keep")


class TestRulerData(unittest.TestCase):
    def test_canonical_schema_loads_for_niah_and_non_niah(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            niah = _record(task="niah_single_1")
            vt = _record(task="vt")
            _write_data_pair(root, "niah_single_1", 4096, [niah])
            _write_data_pair(root, "vt", 4096, [vt])

            self.assertEqual(load_ruler_records("niah_single_1", 4096, root), [niah])
            self.assertEqual(load_ruler_records("vt", 4096, root), [vt])
            self.assertNotIn("token_position_answer", vt)

    def test_old_answer_prefix_is_rejected_even_with_gen_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = _record()
            legacy["answer_prefix"] = legacy["gen_prefix"]
            _write_data_pair(root, "niah_single_1", 4096, [legacy])
            with self.assertRaisesRegex(ValueError, "answer_prefix"):
                load_ruler_records("niah_single_1", 4096, root)

    def test_missing_canonical_field_and_niah_depth_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_length = _record()
            missing_length.pop("max_length")
            _write_data_pair(root, "niah_single_1", 4096, [missing_length])
            with self.assertRaisesRegex(ValueError, "max_length"):
                load_ruler_records("niah_single_1", 4096, root)

            missing_depth = _record(task="niah_single_2")
            missing_depth.pop("token_position_answer")
            _write_data_pair(root, "niah_single_2", 4096, [missing_depth])
            with self.assertRaisesRegex(ValueError, "token_position_answer"):
                load_ruler_records("niah_single_2", 4096, root)

            empty_outputs = _record(task="vt")
            empty_outputs["outputs"] = []
            _write_data_pair(root, "vt", 4096, [empty_outputs])
            with self.assertRaisesRegex(ValueError, "outputs"):
                load_ruler_records("vt", 4096, root)

    def test_resolve_paths_use_the_ruler_layout(self):
        data = resolve_ruler_file("qa_2", 16384, "/ruler-data")
        manifest = resolve_ruler_manifest("qa_2", 16384, "/ruler-data")
        self.assertEqual(str(data), "/ruler-data/16384/qa_2/validation.jsonl")
        self.assertEqual(
            str(manifest), "/ruler-data/16384/qa_2/validation.manifest.json"
        )


class TestRulerPrompt(unittest.TestCase):
    def test_raw_prompt_appends_gen_prefix_once(self):
        record = _record(gen_prefix=" <GEN-PREFIX>")
        prompt = build_ruler_prompt(_StubChatTokenizer(), record, "llama3")
        self.assertEqual(prompt, record["input"] + record["gen_prefix"])
        self.assertEqual(prompt.count(record["gen_prefix"]), 1)

    def test_chat_prompt_keeps_prefix_outside_user_message_and_appends_once(self):
        tokenizer = _StubChatTokenizer()
        record = _record(gen_prefix=" <GEN-PREFIX>")
        prompt = build_ruler_prompt(tokenizer, record, "llama3.2")
        self.assertEqual(
            prompt, f"<s>[U]{record['input']}[/U][A]{record['gen_prefix']}"
        )
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": record["input"]}])
        self.assertEqual(prompt.count(record["gen_prefix"]), 1)
        self.assertEqual(tokenizer.date_string, "26 Jul 2024")

    def test_minicpm_inference_uses_chat_template_generation_prompt(self):
        tokenizer = _StubChatTokenizer()
        record = _record(gen_prefix=" <GEN-PREFIX>")
        family = infer_model_family(
            "openbmb/MiniCPM5-1B", "/models/MiniCPM5-1B"
        )
        prompt = build_ruler_prompt(tokenizer, record, family)
        self.assertEqual(family, "minicpm")
        self.assertEqual(
            prompt, f"<s>[U]{record['input']}[/U][A]{record['gen_prefix']}"
        )
        self.assertFalse(tokenizer.tokenize)
        self.assertIs(tokenizer.add_generation_prompt, True)
        self.assertEqual(tokenizer.date_string, "26 Jul 2024")


class TestRulerMetricsAndAggregation(unittest.TestCase):
    def test_all_and_part_match_single_and_multi_reference_boundaries(self):
        self.assertEqual(string_match_all_score("Found ALPHA", ["alpha"]), 1.0)
        self.assertEqual(string_match_all_score("nothing", ["alpha"]), 0.0)
        self.assertAlmostEqual(
            string_match_all_score("alpha and GAMMA", ["ALPHA", "beta", "gamma"]),
            2.0 / 3.0,
        )
        self.assertEqual(string_match_part_score("answer beta", ["alpha", "BETA"]), 1.0)
        self.assertEqual(string_match_part_score("answer none", ["alpha", "beta"]), 0.0)
        with self.assertRaises(ValueError):
            string_match_all_score("anything", [])
        with self.assertRaises(ValueError):
            string_match_part_score("anything", [])

        predictions = ["has one", "has neither", "has B and C"]
        references = [["one", "missing"], ["two"], ["B", "C"]]
        self.assertEqual(string_match_all(predictions, references), 50.0)
        self.assertEqual(string_match_part(predictions, references), 66.67)

    def test_full_13_task_aggregation_and_depth_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp)
            successful = set(_EXPECTED_TASKS[:7])
            for task in _EXPECTED_TASKS:
                gold = f"gold-for-{task}"
                row = {
                    "pred": f"the answer is {gold}" if task in successful else "wrong",
                    "outputs": [gold],
                    "length": 4000,
                    # Deliberately attach depth to every task. The registry, not
                    # incidental row shape, must decide which tasks enter heatmaps.
                    "token_position_answer": 1000,
                }
                _write_prediction_pair(pred_dir, task, 4096, [row])

            result = score_pred_dir(
                pred_dir, tasks=_EXPECTED_TASKS, seq_lens=[4096], n_depth_bins=4
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["tasks"], list(_EXPECTED_TASKS))
            self.assertEqual(result["seq_lens"], [4096])
            self.assertEqual(result["per_len_mean"]["4096"], 53.85)
            self.assertEqual(result["overall_mean"], 53.85)
            self.assertEqual(
                result["metrics"],
                {
                    task: (
                        "string_match_part"
                        if task.startswith("qa_")
                        else "string_match_all"
                    )
                    for task in _EXPECTED_TASKS
                },
            )
            for task in _EXPECTED_TASKS:
                self.assertEqual(result["counts"][task]["4096"], 1)

            self.assertEqual(set(result["depth_matrix_by_task"]), set(NIAH_TASKS))
            pooled_count = sum(
                sum(row) for row in result["depth_matrix_pooled"]["n"]
            )
            self.assertEqual(pooled_count, len(NIAH_TASKS))

    def test_missing_pair_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp)
            for task in _EXPECTED_TASKS[:-1]:
                _write_prediction_pair(
                    pred_dir,
                    task,
                    4096,
                    [{"pred": "gold", "outputs": ["gold"], "length": 4000}],
                )

            result = score_pred_dir(pred_dir, tasks=_EXPECTED_TASKS, seq_lens=[4096])
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("qa_2__4096", result["diagnostics"]["missing_pairs"])
            self.assertIsNone(result["scores"]["qa_2"]["4096"])
            self.assertEqual(result["counts"]["qa_2"]["4096"], 0)

    def test_sidecar_is_required_and_output_checksum_is_verified(self):
        row = {"pred": "gold", "outputs": ["gold"], "length": 4000}
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp)
            path = _write_prediction_pair(
                pred_dir, "niah_single_1", 4096, [row]
            )
            path.with_suffix(".manifest.json").unlink()
            result = score_pred_dir(
                pred_dir, tasks=["niah_single_1"], seq_lens=[4096]
            )
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(
                result["diagnostics"]["unverified_pairs"],
                ["niah_single_1__4096"],
            )

        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp)
            path = _write_prediction_pair(
                pred_dir, "niah_single_1", 4096, [row]
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            result = score_pred_dir(
                pred_dir, tasks=["niah_single_1"], seq_lens=[4096]
            )
            self.assertEqual(result["status"], "incomplete")
            self.assertIn(
                "jsonl_sha256_mismatch",
                result["diagnostics"]["partial_pairs"][0]["reasons"],
            )


class TestRulerRunConfigHash(unittest.TestCase):
    def test_hash_is_sensitive_to_every_generation_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            config_path = model_dir / "config.json"
            config_path.write_text('{"hidden_size": 128}', encoding="utf-8")
            (model_dir / "tokenizer.json").write_text(
                '{"model": "fixture"}', encoding="utf-8"
            )
            tokenizer_hashes = tokenizer_config_hashes(model_dir)

            base_records = [
                _record(task="niah_single_1", index=0, value="111"),
                _record(task="niah_single_1", index=1, value="222"),
            ]
            _write_data_pair(
                root,
                "niah_single_1",
                4096,
                base_records,
                tokenizer_hashes=tokenizer_hashes,
            )
            _write_data_pair(
                root,
                "niah_single_1",
                8192,
                base_records,
                tokenizer_hashes=tokenizer_hashes,
            )
            task2_records = [
                _record(task="niah_single_2", index=0, value="111"),
                _record(task="niah_single_2", index=1, value="222"),
            ]
            _write_data_pair(
                root,
                "niah_single_2",
                4096,
                task2_records,
                tokenizer_hashes=tokenizer_hashes,
            )

            args = _variant_args(data_root=str(root), model_path=str(model_dir))
            variant = build_variant(args)
            base_hash = ruler_run_config_hash(args, variant, "niah_single_1", 4096)
            self.assertEqual(
                base_hash,
                ruler_run_config_hash(args, variant, "niah_single_1", 4096),
            )

            comparisons = {
                "task": ruler_run_config_hash(args, variant, "niah_single_2", 4096),
                "length": ruler_run_config_hash(args, variant, "niah_single_1", 8192),
                "sample limit": ruler_run_config_hash(
                    _variant_args(
                        data_root=str(root), model_path=str(model_dir), max_samples=1
                    ),
                    variant,
                    "niah_single_1",
                    4096,
                ),
                "dtype": ruler_run_config_hash(
                    _variant_args(
                        data_root=str(root),
                        model_path=str(model_dir),
                        torch_dtype="float32",
                    ),
                    variant,
                    "niah_single_1",
                    4096,
                ),
                "variant": ruler_run_config_hash(
                    args,
                    build_variant(
                        _variant_args(
                            data_root=str(root),
                            model_path=str(model_dir),
                            variant="custom",
                        )
                    ),
                    "niah_single_1",
                    4096,
                ),
            }
            for dimension, changed_hash in comparisons.items():
                with self.subTest(dimension=dimension):
                    self.assertNotEqual(base_hash, changed_hash)

            changed_data = [dict(record) for record in base_records]
            changed_data[0]["input"] += " changed"
            _write_data_pair(
                root,
                "niah_single_1",
                4096,
                changed_data,
                tokenizer_hashes=tokenizer_hashes,
            )
            self.assertNotEqual(
                base_hash,
                ruler_run_config_hash(args, variant, "niah_single_1", 4096),
            )

            changed_prefix = [dict(record) for record in base_records]
            changed_prefix[0]["gen_prefix"] = " Different prefix:"
            _write_data_pair(
                root,
                "niah_single_1",
                4096,
                changed_prefix,
                tokenizer_hashes=tokenizer_hashes,
            )
            self.assertNotEqual(
                base_hash,
                ruler_run_config_hash(args, variant, "niah_single_1", 4096),
            )
            _write_data_pair(
                root,
                "niah_single_1",
                4096,
                base_records,
                tokenizer_hashes=tokenizer_hashes,
            )

            config_path.write_text('{"hidden_size": 256}', encoding="utf-8")
            self.assertNotEqual(
                base_hash,
                ruler_run_config_hash(args, variant, "niah_single_1", 4096),
            )
            config_path.write_text('{"hidden_size": 128}', encoding="utf-8")

            spec = get_task_spec("niah_single_1")
            changed_specs = {
                "generation cap": replace(
                    spec, max_new_tokens=spec.max_new_tokens + 1
                ),
                "task version": replace(spec, version=spec.version + "-changed"),
            }
            for dimension, changed_spec in changed_specs.items():
                with self.subTest(dimension=dimension):
                    self.assertNotEqual(
                        base_hash,
                        ruler_run_config_hash(
                            args,
                            variant,
                            "niah_single_1",
                            4096,
                            task_spec=changed_spec,
                        ),
                    )

    def test_preflight_uses_build_variant_canonical_name_and_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text(
                '{"model": "fixture"}', encoding="utf-8"
            )
            tokenizer_hashes = tokenizer_config_hashes(model_dir)
            _write_data_pair(
                root,
                "niah_single_1",
                4096,
                [_record()],
                tokenizer_hashes=tokenizer_hashes,
            )
            args = _variant_args(
                variant="llamacpp_q40",
                data_root=str(root),
                model_path=str(model_dir),
            )
            expected_variant = build_variant(args)
            resolved = resolve_ruler_preflight(args)
            self.assertEqual(resolved["canonical_variant"], expected_variant.name)
            self.assertEqual(resolved["method_slug"], method_layout_slug(expected_variant))
            self.assertEqual(
                resolved["variant_semantic_hash"],
                ruler_runner.variant_semantic_hash(expected_variant),
            )
            self.assertEqual(list(resolved["pairs"]), ["niah_single_1__4096"])
            pair = resolved["pairs"]["niah_single_1__4096"]
            self.assertEqual(pair["expected_rows"], 1)
            self.assertEqual(
                pair["run_config_hash"],
                ruler_run_config_hash(
                    args, expected_variant, "niah_single_1", 4096
                ),
            )

    def test_preflight_rejects_tokenizer_loader_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text(
                '{"model": "fixture"}', encoding="utf-8"
            )
            tokenizer_hashes = tokenizer_config_hashes(model_dir)
            data_path = _write_data_pair(
                root,
                "niah_single_1",
                4096,
                [_record()],
                tokenizer_hashes=tokenizer_hashes,
            )
            manifest_path = data_path.with_name("validation.manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tokenizer"]["requested_use_fast"] = False
            manifest["tokenizer_identity_sha256"] = tokenizer_identity_sha256(
                tokenizer_hashes,
                requested_use_fast=False,
                tokenizer_class=manifest["tokenizer"]["class"],
                is_fast=manifest["tokenizer"]["is_fast"],
            )
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            args = _variant_args(data_root=str(root), model_path=str(model_dir))
            with self.assertRaisesRegex(ValueError, "tokenizer policy mismatch"):
                resolve_ruler_preflight(args)

    def test_runtime_tokenizer_class_and_fast_mode_must_match_preflight(self):
        tokenizer = _StubChatTokenizer()
        tokenizer_class = (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        )
        metadata = {
            "requested_use_fast": True,
            "class": tokenizer_class,
            "is_fast": False,
        }
        preflight = {
            "pairs": {
                "niah_single_1__4096": {
                    "run_config": {"data_tokenizer": metadata}
                }
            }
        }
        ruler_runner._validate_runtime_tokenizer(tokenizer, preflight, "llama3")
        metadata["class"] = "different.Tokenizer"
        with self.assertRaisesRegex(RuntimeError, "runtime tokenizer mismatch"):
            ruler_runner._validate_runtime_tokenizer(tokenizer, preflight, "llama3")


class TestRulerManifestResume(unittest.TestCase):
    @staticmethod
    def _run_config(expected_samples: int) -> dict:
        spec = get_task_spec("niah_single_1")
        return {
            "schema_version": 2,
            "benchmark": "ruler",
            "task": "niah_single_1",
            "seq_len": 4096,
            "task_spec": asdict(spec),
            "max_new_tokens": spec.max_new_tokens,
            "expected_samples": expected_samples,
            "model_config_sha256": "0" * 64,
            "data_sha256": "1" * 64,
            "data_manifest_sha256": "2" * 64,
            "tokenizer_identity_sha256": "3" * 64,
            "torch_dtype": "float16",
        }

    @staticmethod
    def _run_hash(run_config: dict) -> str:
        encoded = json.dumps(
            run_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _generate(
        self,
        out_path: Path,
        records: list[dict],
        model: _FakeModel,
        run_config: dict,
        *,
        variant=None,
        kitty_stats: dict | None = None,
    ) -> dict:
        spec = get_task_spec("niah_single_1")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            return generate_ruler_pair(
                model=model,
                tokenizer=_GenerationTokenizer(),
                task="niah_single_1",
                seq_len=4096,
                records=records,
                task_spec=spec,
                max_model_len=4096,
                max_new_tokens=spec.max_new_tokens,
                out_path=out_path,
                variant=variant or build_variant(_variant_args()),
                model_family="llama3",
                model_name="fixture-model",
                run_config=run_config,
                expected_run_config_hash=self._run_hash(run_config),
                kitty_stats=kitty_stats,
            )

    @staticmethod
    def _write_existing(
        out_path: Path,
        rows: int,
        *,
        run_config: dict,
        manifest_hash: str | None,
        status: str = "ok",
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n" * rows, encoding="utf-8")
        if manifest_hash is not None:
            out_path.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "expected_samples": run_config["expected_samples"],
                        "written_samples": rows,
                        "run_config_hash": manifest_hash,
                        "run_config": run_config,
                        "jsonl_sha256": _jsonl_sha256(out_path),
                    }
                ),
                encoding="utf-8",
            )

    def test_completed_matching_manifest_is_reused_without_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "niah_single_1__4096.jsonl"
            records = [_record(index=0), _record(index=1)]
            run_config = self._run_config(len(records))
            expected_hash = self._run_hash(run_config)
            self._write_existing(
                out_path,
                len(records),
                run_config=run_config,
                manifest_hash=expected_hash,
                status="ok",
            )
            model = _FakeModel()
            manifest = self._generate(out_path, records, model, run_config)
            self.assertEqual(model.generate_calls, 0)
            self.assertEqual(manifest["run_config_hash"], expected_hash)
            self.assertEqual(manifest["written_samples"], len(records))

    def test_completed_output_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "niah_single_1__4096.jsonl"
            records = [_record()]
            run_config = self._run_config(len(records))
            self._write_existing(
                out_path,
                len(records),
                run_config=run_config,
                manifest_hash=self._run_hash(run_config),
                status="ok",
            )
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                self._generate(out_path, records, _FakeModel(), run_config)

    def test_partial_matching_manifest_resumes_only_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "niah_single_1__4096.jsonl"
            records = [_record(index=0), _record(index=1)]
            run_config = self._run_config(len(records))
            self._write_existing(
                out_path,
                1,
                run_config=run_config,
                manifest_hash=self._run_hash(run_config),
                status="partial",
            )
            model = _FakeModel()
            manifest = self._generate(out_path, records, model, run_config)
            self.assertEqual(model.generate_calls, 1)
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["written_samples"], 2)
            self.assertEqual(manifest["written_this_run"], 1)
            self.assertEqual(len(out_path.read_text(encoding="utf-8").splitlines()), 2)

    def test_stale_manifest_is_rejected_for_complete_and_partial_outputs(self):
        records = [_record(index=0), _record(index=1)]
        run_config = self._run_config(len(records))
        with tempfile.TemporaryDirectory() as tmp:
            for rows, status in ((2, "ok"), (1, "partial")):
                with self.subTest(rows=rows):
                    out_path = Path(tmp) / f"case-{rows}" / "niah_single_1__4096.jsonl"
                    self._write_existing(
                        out_path,
                        rows,
                        run_config=run_config,
                        manifest_hash="stale-hash",
                        status=status,
                    )
                    with self.assertRaisesRegex(RuntimeError, "mismatch"):
                        self._generate(out_path, records, _FakeModel(), run_config)

    def test_missing_manifest_is_rejected_for_complete_and_partial_outputs(self):
        records = [_record(index=0), _record(index=1)]
        run_config = self._run_config(len(records))
        with tempfile.TemporaryDirectory() as tmp:
            for rows in (2, 1):
                with self.subTest(rows=rows):
                    out_path = Path(tmp) / f"case-{rows}" / "niah_single_1__4096.jsonl"
                    self._write_existing(
                        out_path,
                        rows,
                        run_config=run_config,
                        manifest_hash=None,
                    )
                    with self.assertRaisesRegex(RuntimeError, "without .*manifest"):
                        self._generate(out_path, records, _FakeModel(), run_config)

    def test_kitty_generation_records_cache_engagement(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "niah_single_1__4096.jsonl"
            records = [_record()]
            run_config = self._run_config(len(records))
            variant = build_variant(_variant_args(variant="kitty"))
            cache = _FakeShadowCache()
            with mock.patch.object(
                ruler_runner, "_cache_factory", return_value=cache
            ):
                manifest = self._generate(
                    out_path,
                    records,
                    _FakeModel(),
                    run_config,
                    variant=variant,
                )
            engagement = manifest["engagement"]
            self.assertIs(engagement["kitty_cache_checked"], True)
            self.assertIs(engagement["kitty_cache_engaged"], True)
            self.assertEqual(engagement["kitty_cache_seq_length"], 3)

    def test_shadowkv_generation_uses_hook_cache_and_records_engagement(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "niah_single_1__4096.jsonl"
            records = [_record()]
            run_config = self._run_config(len(records))
            stats = {
                "installed": 1,
                "prefill_calls": 0,
                "decode_calls": 0,
                "last_selected_chunks": None,
                "last_seq_length": None,
            }
            cache = _FakeShadowCache()

            def engage_shadowkv(kwargs):
                self.assertIs(kwargs["past_key_values"], cache)
                self.assertIs(kwargs["disable_compile"], True)
                self.assertIsNone(kwargs["temperature"])
                stats["prefill_calls"] += 1
                stats["decode_calls"] += 1
                stats["last_selected_chunks"] = 7
                stats["last_seq_length"] = 3

            model = _FakeModel(on_generate=engage_shadowkv)
            variant = build_variant(_variant_args(variant="shadowkv"))
            with mock.patch.object(
                ruler_runner, "_shadowkv_cache", return_value=cache
            ) as cache_factory:
                manifest = self._generate(
                    out_path,
                    records,
                    model,
                    run_config,
                    variant=variant,
                    kitty_stats=stats,
                )

            cache_factory.assert_called_once_with(
                variant,
                model,
                3,
                get_task_spec("niah_single_1").max_new_tokens,
            )
            self.assertEqual(manifest["status"], "ok")
            engagement = manifest["engagement"]
            self.assertIs(engagement["kitty_cache_checked"], True)
            self.assertIs(engagement["kitty_cache_engaged"], True)
            self.assertEqual(engagement["kitty_cache_seq_length"], 3)
            self.assertEqual(engagement["shadowkv_installed"], 1)
            self.assertEqual(engagement["shadowkv_prefill_calls"], 1)
            self.assertEqual(engagement["shadowkv_decode_calls"], 1)
            self.assertEqual(
                engagement["shadowkv_decode_calls_at_pair_start"], 0
            )
            self.assertEqual(engagement["shadowkv_decode_calls_delta"], 1)
            self.assertEqual(engagement["shadowkv_last_selected_chunks"], 7)
            self.assertEqual(engagement["shadowkv_last_seq_length"], 3)

    def test_shadowkv_without_decode_hook_engagement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "niah_single_1__4096.jsonl"
            records = [_record()]
            run_config = self._run_config(len(records))
            stats = {
                "installed": 1,
                "prefill_calls": 0,
                "decode_calls": 0,
                "last_selected_chunks": None,
                "last_seq_length": None,
            }
            model = _FakeModel()
            variant = build_variant(_variant_args(variant="shadowkv"))
            with mock.patch.object(
                ruler_runner, "_shadowkv_cache", return_value=_FakeShadowCache()
            ):
                with self.assertRaisesRegex(RuntimeError, "did not engage"):
                    self._generate(
                        out_path,
                        records,
                        model,
                        run_config,
                        variant=variant,
                        kitty_stats=stats,
                    )
            self.assertEqual(model.generate_calls, 1)
            self.assertEqual(stats["decode_calls"], 0)

    def test_shadowkv_without_installed_hook_stats_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [_record()]
            run_config = self._run_config(len(records))
            model = _FakeModel()
            variant = build_variant(_variant_args(variant="shadowkv"))
            with self.assertRaisesRegex(RuntimeError, "install_shadowkv_sim hook"):
                self._generate(
                    Path(tmp) / "niah_single_1__4096.jsonl",
                    records,
                    model,
                    run_config,
                    variant=variant,
                )
            self.assertEqual(model.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
