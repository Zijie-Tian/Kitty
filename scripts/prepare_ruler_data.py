#!/usr/bin/env python3
"""Prepare canonical, offline-first NVIDIA RULER data for Kitty.

The coordinator starts one fresh Python process for every task/length pair.  A
worker loads only the vendored generator module it needs, normalizes its output,
and atomically installs the JSONL plus its provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import types
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kitty_sim.longbench.templates import use_fast_tokenizer
from kitty_sim.ruler.tasks import (
    DEFAULT_SEQ_LENS,
    DEFAULT_TASKS,
    NIAH_TASKS,
    TASK_VERSION,
    get_task_spec,
    resolve_task_names,
)

RULER_VENDOR_DIR = (
    REPO_ROOT
    / "third_party"
    / "lm-evaluation-harness"
    / "lm_eval"
    / "tasks"
    / "ruler"
)

SCHEMA_VERSION = 2
GENERATOR_VERSION = "kitty-ruler-prep-v2"
TOKENIZER_HASH_ALGORITHM = "ruler-tokenizer-config-v2"
FWE_GENERATOR_SLACK = 32

ESSAY_NIAH_TASKS = frozenset(
    {
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multivalue",
        "niah_multiquery",
    }
)

SOURCE_FILENAMES = (
    "PaulGrahamEssays.json",
    "squad.json",
    "hotpotqa.json",
)
SOURCE_URLS = {
    "squad.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "hotpotqa.json": (
        "https://s3.amazonaws.com/hotpotqa/"
        "hotpot_dev_distractor_v1.json"
    ),
}
PAUL_GRAHAM_DATASET = "baber/paul_graham_essays"

TOKENIZER_CONFIG_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
)

NIAH_ARGUMENTS: dict[str, dict[str, Any]] = {
    "niah_single_1": {
        "type_haystack": "repeat",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
    },
    "niah_single_2": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
    },
    "niah_single_3": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "uuids",
    },
    "niah_multikey_1": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 4,
    },
    "niah_multikey_2": {
        "type_haystack": "needle",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
    },
    "niah_multikey_3": {
        "type_haystack": "needle",
        "type_needle_k": "uuids",
        "type_needle_v": "uuids",
    },
    "niah_multivalue": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_v": 4,
    },
    "niah_multiquery": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_q": 4,
    },
}


class PrepError(RuntimeError):
    """A user-actionable data preparation error."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise PrepError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise PrepError(f"{name} must be an integer; got {value!r}") from exc


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical NVIDIA RULER data from Kitty's vendored "
            "lm-evaluation-harness generators. Sources and tokenizers are "
            "offline-only by default."
        )
    )
    parser.add_argument(
        "--model-path",
        default=_first_env("MODEL_PATH", "KITTY_LLAMA32_1B_PATH"),
        help="Model directory or Hub id used for provenance; also the tokenizer fallback.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=_first_env("TOKENIZER_PATH"),
        help="Tokenizer directory or Hub id (default: --model-path).",
    )
    parser.add_argument(
        "--model-family",
        default=_first_env("MODEL_FAMILY") or "llama3",
        help="Kitty prompt-template family recorded in the manifest (default: llama3).",
    )
    parser.add_argument(
        "--model-tag",
        default=_first_env("MODEL_TAG", "MODEL_SLUG"),
        help="Human-readable model tag recorded for provenance.",
    )
    parser.add_argument(
        "--data-root",
        default=_first_env("RULER_DATA_ROOT", "DATA_ROOT"),
        help="Output root (default: ~/data/ruler/<model-family>).",
    )
    parser.add_argument(
        "--source-root",
        default=_first_env("RULER_SOURCE_ROOT"),
        help="Directory containing PaulGrahamEssays.json, squad.json, and hotpotqa.json.",
    )
    parser.add_argument(
        "--tasks",
        default=_first_env("TASKS") or "all",
        help="'all' or a comma-separated subset of the 13 canonical tasks.",
    )
    parser.add_argument(
        "--lengths",
        "--lens",
        dest="lengths",
        default=_first_env("LENGTHS", "LENS") or ",".join(map(str, DEFAULT_SEQ_LENS)),
        help="Comma-separated nominal context lengths (default: 4096,8192,16384,32768).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=_env_int("NUM_SAMPLES", 100),
        help="Records per task/length pair (default: 100).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=_env_int("MARGIN", 256),
        help="Tokens reserved below each nominal context length (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("SEED", 42),
        help="Python and NumPy seed reset independently for every pair (default: 42).",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("LOCAL_FILES_ONLY", True),
        help="Forbid tokenizer Hub access (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=_env_bool("FORCE", False),
        help="Replace an existing pair whose generation hash differs.",
    )
    parser.add_argument(
        "--download-sources",
        action="store_true",
        default=_env_bool("DOWNLOAD_SOURCES", False),
        help="Explicitly allow downloading missing source JSON files.",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _parse_tasks(value: str) -> tuple[str, ...]:
    try:
        return resolve_task_names(value)
    except ValueError as exc:
        raise PrepError(str(exc)) from exc


def _parse_lengths(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise PrepError("--lengths/--lens must be a non-empty CSV")
    try:
        lengths = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise PrepError(f"invalid --lengths/--lens CSV: {value!r}") from exc
    if any(length <= 0 for length in lengths):
        raise PrepError("all nominal lengths must be positive")
    if len(set(lengths)) != len(lengths):
        raise PrepError("--lengths/--lens contains duplicates")
    return lengths


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.tasks = _parse_tasks(args.tasks)
    args.lengths = _parse_lengths(args.lengths)
    if args.num_samples <= 0:
        raise PrepError("--num-samples must be positive")
    if args.margin < 0:
        raise PrepError("--margin must be non-negative")
    if any(length <= args.margin for length in args.lengths):
        raise PrepError("every nominal length must be greater than --margin")

    tokenizer_path = args.tokenizer_path or args.model_path
    model_path = args.model_path or tokenizer_path
    if not tokenizer_path:
        raise PrepError("set --tokenizer-path or --model-path")
    args.tokenizer_path = str(Path(tokenizer_path).expanduser()) if Path(tokenizer_path).expanduser().exists() else tokenizer_path
    args.model_path = str(Path(model_path).expanduser()) if Path(model_path).expanduser().exists() else model_path
    args.model_family = args.model_family.strip().lower()
    if not args.model_family:
        raise PrepError("--model-family must not be empty")
    args.model_tag = args.model_tag or Path(str(model_path).rstrip("/")).name
    if not args.model_tag:
        raise PrepError("--model-tag could not be inferred; pass it explicitly")

    args.data_root = Path(
        args.data_root or (Path.home() / "data" / "ruler" / args.model_family)
    ).expanduser()
    args.source_root = Path(
        args.source_root or (Path.home() / "data" / "ruler_sources")
    ).expanduser()
    return args


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _required_sources(tasks: Sequence[str]) -> tuple[str, ...]:
    needed: set[str] = set()
    if any(task in ESSAY_NIAH_TASKS for task in tasks):
        needed.add("PaulGrahamEssays.json")
    if "qa_1" in tasks:
        needed.add("squad.json")
    if "qa_2" in tasks:
        needed.add("hotpotqa.json")
    return tuple(name for name in SOURCE_FILENAMES if name in needed)


def _validate_source_json(name: str, path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrepError(f"invalid source JSON {path}: {exc}") from exc
    if name == "PaulGrahamEssays.json":
        valid = isinstance(value, dict) and isinstance(value.get("text"), str) and bool(value["text"].strip())
    elif name == "squad.json":
        valid = isinstance(value, dict) and isinstance(value.get("data"), list) and bool(value["data"])
    elif name == "hotpotqa.json":
        valid = isinstance(value, list) and bool(value)
    else:
        valid = False
    if not valid:
        raise PrepError(f"source {path} has the wrong structure for {name}")


def _atomic_install_temp(temp_path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, target)


def _download_http_json(name: str, target: Path) -> None:
    url = SOURCE_URLS[name]
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "kitty-ruler-prep/1"})
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            with urllib.request.urlopen(request, timeout=180) as response:
                shutil.copyfileobj(response, temp_handle, length=1024 * 1024)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        _validate_source_json(name, temp_path)
        _atomic_install_temp(temp_path, target)
        temp_path = None
    except Exception as exc:
        raise PrepError(f"failed to download {name} from {url}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _download_paul_graham(target: Path) -> None:
    try:
        import datasets
    except ImportError as exc:
        raise PrepError(
            "downloading PaulGrahamEssays.json requires the existing 'datasets' "
            "dependency; install nothing in the prep wrapper"
        ) from exc
    try:
        dataset = datasets.load_dataset(PAUL_GRAHAM_DATASET, split="train")
        texts = dataset["text"]
        text = " ".join(item for item in texts if isinstance(item, str) and item.strip())
    except Exception as exc:
        raise PrepError(
            f"failed to download {PAUL_GRAHAM_DATASET} for PaulGrahamEssays.json: {exc}"
        ) from exc
    if not text:
        raise PrepError(f"downloaded {PAUL_GRAHAM_DATASET} contains no essay text")

    payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8") + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            temp_handle.write(payload)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        _validate_source_json(target.name, temp_path)
        _atomic_install_temp(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _ensure_sources(
    source_root: Path, tasks: Sequence[str], download_sources: bool
) -> None:
    needed = _required_sources(tasks)
    if not needed:
        return
    source_root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in needed if not (source_root / name).is_file()]
    if missing and not download_sources:
        paths = ", ".join(str(source_root / name) for name in missing)
        raise PrepError(
            f"missing local RULER source file(s): {paths}. No network access was "
            "attempted; provide them under --source-root or explicitly pass "
            "--download-sources."
        )
    for name in missing:
        target = source_root / name
        print(f"[prepare-ruler] downloading source {name} -> {target}", flush=True)
        if name == "PaulGrahamEssays.json":
            _download_paul_graham(target)
        else:
            _download_http_json(name, target)
    for name in needed:
        _validate_source_json(name, source_root / name)


def _package_stub(name: str, path: Path) -> types.ModuleType:
    package = types.ModuleType(name)
    package.__file__ = str(path / "__init__.py")
    package.__package__ = name
    package.__path__ = [str(path)]
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = package
    return package


def _install_lm_eval_stubs() -> None:
    lm_eval_dir = RULER_VENDOR_DIR.parents[1]
    tasks_dir = RULER_VENDOR_DIR.parent
    lm_eval_package = _package_stub("lm_eval", lm_eval_dir)
    tasks_package = _package_stub("lm_eval.tasks", tasks_dir)
    ruler_package = _package_stub("lm_eval.tasks.ruler", RULER_VENDOR_DIR)
    lm_eval_package.tasks = tasks_package
    tasks_package.ruler = ruler_package


def _load_vendor_module(short_name: str) -> types.ModuleType:
    full_name = f"lm_eval.tasks.ruler.{short_name}"
    existing = sys.modules.get(full_name)
    if existing is not None:
        return existing
    path = RULER_VENDOR_DIR / f"{short_name}.py"
    if not path.is_file():
        raise PrepError(f"vendored RULER module is missing: {path}")
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise PrepError(f"cannot load vendored RULER module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
        setattr(sys.modules["lm_eval.tasks.ruler"], short_name, module)
    except Exception:
        sys.modules.pop(full_name, None)
        raise
    return module


def _block_nltk_downloads() -> None:
    try:
        import nltk
    except ImportError as exc:
        raise PrepError("NIAH generation requires the existing 'nltk' dependency") from exc
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError as exc:
        raise PrepError(
            "NIAH generation is offline and requires the NLTK punkt_tab resource "
            "to already exist in NLTK_DATA; no download was attempted"
        ) from exc

    def blocked_download(*_args: Any, **_kwargs: Any) -> bool:
        raise PrepError("vendored NIAH attempted an implicit NLTK download")

    nltk.download = blocked_download


def _load_task_backend(task: str) -> dict[str, types.ModuleType]:
    _install_lm_eval_stubs()
    _load_vendor_module("common_utils")
    if task in NIAH_TASKS:
        _block_nltk_downloads()
        prepare_niah = _load_vendor_module("prepare_niah")
        niah_utils = _load_vendor_module("niah_utils")
        return {"prepare_niah": prepare_niah, "niah_utils": niah_utils}
    module_name = {
        "vt": "vt_utils",
        "cwe": "cwe_utils",
        "fwe": "fwe_utils",
        "qa_1": "qa_utils",
        "qa_2": "qa_utils",
    }[task]
    return {module_name: _load_vendor_module(module_name)}


def _task_max_new_tokens(task: str, backend: dict[str, types.ModuleType]) -> int:
    """Return the registry cap after asserting the vendored generator agrees."""

    spec = get_task_spec(task)
    if task in NIAH_TASKS:
        parameter = inspect.signature(
            backend["prepare_niah"].generate_samples
        ).parameters["tokens_to_generate"]
        vendor_cap = int(parameter.default)
    elif task == "vt":
        vendor_cap = int(
            backend["vt_utils"].CONFIG["variable_tracking"]["tokens_to_generate"]
        )
    elif task == "cwe":
        vendor_cap = int(backend["cwe_utils"].CONFIG["tokens_to_generate"])
    elif task == "fwe":
        vendor_cap = int(backend["fwe_utils"].CONFIG["tokens_to_generate"])
    elif task in {"qa_1", "qa_2"}:
        vendor_cap = int(backend["qa_utils"].CONFIG["tokens_to_generate"])
    else:
        raise PrepError(f"no vendored generator for task {task}")

    if vendor_cap != spec.max_new_tokens:
        raise PrepError(
            f"{task}: vendored generation cap {vendor_cap} does not match "
            f"canonical registry cap {spec.max_new_tokens}"
        )
    return spec.max_new_tokens


def _task_generator_files(task: str) -> tuple[Path, ...]:
    common = RULER_VENDOR_DIR / "common_utils.py"
    if task in NIAH_TASKS:
        files = (
            RULER_VENDOR_DIR / "prepare_niah.py",
            RULER_VENDOR_DIR / "niah_utils.py",
        )
    else:
        name = {
            "vt": "vt_utils.py",
            "cwe": "cwe_utils.py",
            "fwe": "fwe_utils.py",
            "qa_1": "qa_utils.py",
            "qa_2": "qa_utils.py",
        }[task]
        files = (common, RULER_VENDOR_DIR / name)
    return (SCRIPT_PATH, *files)


def _generator_hashes(task: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in _task_generator_files(task):
        if not path.is_file():
            raise PrepError(f"generator source is missing: {path}")
        try:
            name = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            name = str(path)
        hashes[name] = _sha256_file(path)
    return dict(sorted(hashes.items()))


def _source_hashes(task: str, source_root: Path) -> dict[str, str]:
    names: tuple[str, ...]
    if task in ESSAY_NIAH_TASKS:
        names = ("PaulGrahamEssays.json",)
    elif task == "qa_1":
        names = ("squad.json",)
    elif task == "qa_2":
        names = ("hotpotqa.json",)
    else:
        names = ()
    hashes: dict[str, str] = {}
    for name in names:
        path = source_root / name
        if not path.is_file():
            raise PrepError(
                f"missing local source {path}; no network access was attempted"
            )
        hashes[name] = _sha256_file(path)
    return hashes


def _cached_tokenizer_file(
    model_id: str, filename: str, local_files_only: bool
) -> Path | None:
    try:
        from transformers.utils.hub import cached_file
    except ImportError as exc:
        raise PrepError("tokenizer loading requires the existing 'transformers' dependency") from exc
    kwargs = {
        "local_files_only": local_files_only,
        "_raise_exceptions_for_gated_repo": False,
        "_raise_exceptions_for_missing_entries": False,
        "_raise_exceptions_for_connection_errors": False,
    }
    try:
        resolved = cached_file(model_id, filename, **kwargs)
    except TypeError:
        resolved = cached_file(model_id, filename, local_files_only=local_files_only)
    except OSError:
        resolved = None
    return Path(resolved) if resolved else None


def _tokenizer_config_hashes(
    tokenizer_path: str, local_files_only: bool
) -> tuple[dict[str, str], str]:
    local_dir = Path(tokenizer_path).expanduser()
    files: dict[str, Path] = {}
    if local_dir.is_dir():
        for filename in TOKENIZER_CONFIG_FILENAMES:
            candidate = local_dir / filename
            if candidate.is_file():
                files[filename] = candidate
        resolved_path = str(local_dir.resolve())
    else:
        for filename in TOKENIZER_CONFIG_FILENAMES:
            candidate = _cached_tokenizer_file(
                tokenizer_path, filename, local_files_only=local_files_only
            )
            if candidate is not None and candidate.is_file():
                files[filename] = candidate
        resolved_path = tokenizer_path
    if not files:
        mode = "local cache" if local_files_only else "model repository"
        raise PrepError(
            f"no tokenizer config artifacts found for {tokenizer_path!r} in the {mode}; "
            f"expected at least one of {', '.join(TOKENIZER_CONFIG_FILENAMES)}"
        )
    hashes = {name: _sha256_file(files[name]) for name in sorted(files)}
    return hashes, resolved_path


def _tokenizer_identity_sha256(
    config_hashes: dict[str, str], tokenizer_metadata: dict[str, Any]
) -> str:
    payload = {
        "algorithm": TOKENIZER_HASH_ALGORITHM,
        "config_hashes": config_hashes,
        "requested_use_fast": tokenizer_metadata["requested_use_fast"],
        "class": tokenizer_metadata["class"],
        "is_fast": tokenizer_metadata["is_fast"],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _load_tokenizer(
    tokenizer_path: str, local_files_only: bool, model_family: str
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PrepError("data generation requires the existing 'transformers' dependency") from exc
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=use_fast_tokenizer(model_family),
            local_files_only=local_files_only,
        )
    except Exception as exc:
        mode = "local files only" if local_files_only else "Hub access allowed"
        raise PrepError(f"failed to load tokenizer {tokenizer_path!r} ({mode}): {exc}") from exc


def _tokenizer_metadata(
    tokenizer: Any,
    requested_path: str,
    resolved_path: str,
    model_family: str,
) -> dict[str, Any]:
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max_length, (str, int, float, bool, type(None))):
        model_max_length = str(model_max_length)
    return {
        "algorithm": TOKENIZER_HASH_ALGORITHM,
        "requested_path": requested_path,
        "resolved_path": resolved_path,
        "requested_use_fast": use_fast_tokenizer(model_family),
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(len(tokenizer)),
        "model_max_length": model_max_length,
    }


def _seed_pair(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError as exc:
        raise PrepError("vendored RULER generation requires the existing 'numpy' dependency") from exc
    np.random.seed(seed % (2**32))


def _load_essay_haystack(source_root: Path) -> list[str]:
    path = source_root / "PaulGrahamEssays.json"
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    text = value.get("text") if isinstance(value, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise PrepError(f"{path} must contain a non-empty string field 'text'")
    return re.sub(r"\s+", " ", text).split(" ")


def _generate_niah(
    task: str,
    backend: dict[str, types.ModuleType],
    tokenizer: Any,
    source_root: Path,
    generated_length: int,
    num_samples: int,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prepare_niah = backend["prepare_niah"]
    niah_utils = backend["niah_utils"]
    task_args = dict(NIAH_ARGUMENTS[task])
    haystack_type = task_args["type_haystack"]
    if haystack_type == "essay":
        haystack = _load_essay_haystack(source_root)
    else:
        haystack = prepare_niah.get_haystack(type_haystack=haystack_type)
    return prepare_niah.generate_samples(
        haystack,
        TOKENIZER=tokenizer,
        max_seq_length=generated_length,
        template=niah_utils.TEMPLATE,
        num_samples=num_samples,
        tokens_to_generate=max_new_tokens,
        random_seed=seed,
        **task_args,
    )


def _generate_vt(
    module: types.ModuleType,
    tokenizer: Any,
    generated_length: int,
    num_samples: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    icl_example = module.sys_vartrack_w_noise_random(
        tokenizer=tokenizer,
        num_samples=1,
        max_seq_length=500,
        incremental=5,
        tokens_to_generate=max_new_tokens,
    )[0]
    return module.sys_vartrack_w_noise_random(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_seq_length=generated_length,
        icl_example=icl_example,
        tokens_to_generate=max_new_tokens,
    )


def _load_local_qa(
    module: types.ModuleType, task: str, source_root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    source_name = "squad.json" if task == "qa_1" else "hotpotqa.json"
    source_path = source_root / source_name

    def local_json(path: str) -> Any:
        candidate = Path(path)
        if candidate.resolve() != source_path.resolve():
            raise PrepError(f"QA generator attempted to access undeclared source {path}")
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    module.download_json = local_json
    if task == "qa_1":
        module.read_squad.cache_clear()
        return module.read_squad(str(source_path))
    module.read_hotpotqa.cache_clear()
    return module.read_hotpotqa(str(source_path))


def _generate_raw_records(
    task: str,
    backend: dict[str, types.ModuleType],
    tokenizer: Any,
    source_root: Path,
    generated_length: int,
    num_samples: int,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if task in NIAH_TASKS:
        return _generate_niah(
            task,
            backend,
            tokenizer,
            source_root,
            generated_length,
            num_samples,
            seed,
            max_new_tokens,
        )
    if task == "vt":
        return _generate_vt(
            backend["vt_utils"],
            tokenizer,
            generated_length,
            num_samples,
            max_new_tokens,
        )
    if task == "cwe":
        module = backend["cwe_utils"]
        module.WORDS.sort()
        module.RNG.seed(seed)
        module.RNG.shuffle(module.WORDS)
        return module.sys_word_pair_random(
            num_samples=num_samples,
            max_seq_length=generated_length,
            tokenizer=tokenizer,
            tokens_to_generate=max_new_tokens,
        )
    if task == "fwe":
        module = backend["fwe_utils"]
        module.SEED = seed
        fwe_length = generated_length - FWE_GENERATOR_SLACK
        if fwe_length <= max_new_tokens:
            raise PrepError(
                f"fwe generation target {fwe_length} must exceed cap "
                f"{max_new_tokens}"
            )
        return module.sys_kwext(
            tokenizer=tokenizer,
            max_seq_length=fwe_length,
            num_samples=num_samples,
            tokens_to_generate=max_new_tokens,
        )
    if task in {"qa_1", "qa_2"}:
        module = backend["qa_utils"]
        module.SEED = seed
        qas, docs = _load_local_qa(module, task, source_root)
        if len(qas) < num_samples:
            raise PrepError(
                f"{task} source has only {len(qas)} usable questions, fewer than "
                f"--num-samples={num_samples}"
            )
        return module.generate_samples(
            tokenizer=tokenizer,
            docs=docs,
            qas=qas,
            max_seq_length=generated_length,
            num_samples=num_samples,
            tokens_to_generate=max_new_tokens,
        )
    raise PrepError(f"no generator adapter for task {task}")


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise PrepError("tokenizer returned a batched encoding for one string")
        input_ids = input_ids[0]
    return len(input_ids)


def _canonical_prefix(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PrepError("generator record has an empty or non-string gen_prefix")
    return " " + value.strip()


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_records(
    raw_records: Iterable[dict[str, Any]],
    task: str,
    tokenizer: Any,
    generated_length: int,
    num_samples: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    records = list(raw_records)
    if len(records) < num_samples:
        raise PrepError(
            f"{task} generator returned {len(records)} records, expected at least {num_samples}"
        )
    records = records[:num_samples]
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise PrepError(f"{task} record {index} is not a JSON object")
        input_text = raw.get("input")
        outputs = raw.get("outputs")
        length = raw.get("length")
        if not isinstance(input_text, str) or not input_text:
            raise PrepError(f"{task} record {index} has an empty or non-string input")
        if (
            not isinstance(outputs, list)
            or not outputs
            or any(not isinstance(output, str) or not output for output in outputs)
        ):
            raise PrepError(f"{task} record {index} has invalid outputs")
        if not _is_plain_int(length) or length < 0:
            raise PrepError(f"{task} record {index} has invalid generator length {length!r}")
        if length > generated_length:
            raise PrepError(
                f"{task} record {index} length {length} exceeds generated target "
                f"{generated_length}"
            )
        gen_prefix = _canonical_prefix(raw.get("gen_prefix"))
        measured_length = _token_count(tokenizer, input_text + gen_prefix) + max_new_tokens
        if measured_length > generated_length:
            raise PrepError(
                f"{task} record {index} prompt+cap measures {measured_length} tokens, "
                f"above generated target {generated_length}"
            )
        record: dict[str, Any] = {
            "index": index,
            "input": input_text,
            "outputs": list(outputs),
            "length": int(length),
            "max_length": generated_length,
            "gen_prefix": gen_prefix,
        }
        if task in NIAH_TASKS:
            answer_offset = input_text.find(outputs[0])
            if answer_offset < 0:
                raise PrepError(
                    f"{task} record {index} first gold {outputs[0]!r} is absent from input"
                )
            record["token_position_answer"] = _token_count(
                tokenizer, input_text[:answer_offset]
            )
        normalized.append(record)
    if len(normalized) != num_samples:
        raise PrepError(
            f"{task} normalization produced {len(normalized)} records, expected {num_samples}"
        )
    return normalized


def _jsonl_line(record: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise PrepError(f"existing JSONL contains a blank row: {path}")
            count += 1
    return count


def _matching_existing_pair(
    jsonl_path: Path,
    manifest_path: Path,
    generation_config_sha256: str,
    expected_count: int,
) -> bool:
    if not jsonl_path.exists() and not manifest_path.exists():
        return False
    if not jsonl_path.is_file() or not manifest_path.is_file():
        raise PrepError(
            f"incomplete existing pair at {jsonl_path.parent}: both validation.jsonl "
            "and validation.manifest.json are required"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrepError(f"invalid existing manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PrepError(f"existing manifest is not a JSON object: {manifest_path}")
    if manifest.get("generation_config_sha256") != generation_config_sha256:
        return False
    expected_jsonl_sha = manifest.get("jsonl_sha256")
    if not isinstance(expected_jsonl_sha, str) or _sha256_file(jsonl_path) != expected_jsonl_sha:
        raise PrepError(f"existing JSONL checksum does not match {manifest_path}")
    if manifest.get("count") != expected_count or _count_jsonl_rows(jsonl_path) != expected_count:
        raise PrepError(
            f"existing pair count does not match requested count {expected_count}: "
            f"{jsonl_path}"
        )
    return True


def _atomic_write_pair(
    jsonl_path: Path,
    manifest_path: Path,
    records: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    pair_dir = jsonl_path.parent
    pair_dir.mkdir(parents=True, exist_ok=True)
    jsonl_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=pair_dir,
            prefix=".validation.jsonl.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            jsonl_temp = Path(handle.name)
            for record in records:
                line = _jsonl_line(record)
                handle.write(line)
                digest.update(line)
            handle.flush()
            os.fsync(handle.fileno())
        jsonl_sha256 = digest.hexdigest()
        manifest_data = json.dumps(
            {**manifest, "jsonl_sha256": jsonl_sha256},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=pair_dir,
            prefix=".validation.manifest.json.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            manifest_temp = Path(handle.name)
            handle.write(manifest_data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(jsonl_temp, jsonl_path)
        jsonl_temp = None
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
        return jsonl_sha256
    finally:
        if jsonl_temp is not None:
            jsonl_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def _worker(args: argparse.Namespace) -> int:
    if len(args.tasks) != 1 or len(args.lengths) != 1:
        raise PrepError("internal worker requires exactly one task and one length")
    task = args.tasks[0]
    nominal_length = args.lengths[0]
    generated_length = nominal_length - args.margin
    _seed_pair(args.seed)

    backend = _load_task_backend(task)
    max_new_tokens = _task_max_new_tokens(task, backend)
    if generated_length <= max_new_tokens:
        raise PrepError(
            f"{task}/{nominal_length}: generated length {generated_length} must exceed "
            f"generation cap {max_new_tokens}"
        )

    source_hashes = _source_hashes(task, args.source_root)
    generator_hashes = _generator_hashes(task)
    tokenizer_config_hashes, resolved_tokenizer_path = _tokenizer_config_hashes(
        args.tokenizer_path, args.local_files_only
    )
    tokenizer = _load_tokenizer(
        args.tokenizer_path, args.local_files_only, args.model_family
    )
    tokenizer_metadata = _tokenizer_metadata(
        tokenizer,
        args.tokenizer_path,
        resolved_tokenizer_path,
        args.model_family,
    )
    tokenizer_identity_sha256 = _tokenizer_identity_sha256(
        tokenizer_config_hashes, tokenizer_metadata
    )
    generation_config = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "task": task,
        "task_version": TASK_VERSION,
        "nominal_length": nominal_length,
        "generated_length": generated_length,
        "margin": args.margin,
        "max_new_tokens": max_new_tokens,
        "count": args.num_samples,
        "seed": args.seed,
        "model_family": args.model_family,
        "model_path": args.model_path,
        "model_tag": args.model_tag,
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "tokenizer_config_hashes": tokenizer_config_hashes,
        "source_hashes": source_hashes,
        "generator_hashes": generator_hashes,
    }
    generation_config_sha256 = _sha256_bytes(
        _canonical_json_bytes(generation_config)
    )

    pair_dir = args.data_root / str(nominal_length) / task
    jsonl_path = pair_dir / "validation.jsonl"
    manifest_path = pair_dir / "validation.manifest.json"
    try:
        matches = _matching_existing_pair(
            jsonl_path,
            manifest_path,
            generation_config_sha256,
            args.num_samples,
        )
    except PrepError:
        if not args.force:
            raise
        matches = False
    if matches:
        print(
            f"[skip] task={task} nominal={nominal_length} hash={generation_config_sha256}",
            flush=True,
        )
        return 0
    if (jsonl_path.exists() or manifest_path.exists()) and not args.force:
        raise PrepError(
            f"existing pair has a different generation hash: {pair_dir}; "
            "refusing to overwrite without --force"
        )

    raw_records = _generate_raw_records(
        task,
        backend,
        tokenizer,
        args.source_root,
        generated_length,
        args.num_samples,
        args.seed,
        max_new_tokens,
    )
    records = _normalize_records(
        raw_records,
        task,
        tokenizer,
        generated_length,
        args.num_samples,
        max_new_tokens,
    )
    manifest = {
        **generation_config,
        "generation_config_sha256": generation_config_sha256,
        "model_path": args.model_path,
        "model_tag": args.model_tag,
        "tokenizer": tokenizer_metadata,
    }
    jsonl_sha256 = _atomic_write_pair(
        jsonl_path, manifest_path, records, manifest
    )
    print(
        f"[write] task={task} nominal={nominal_length} generated={generated_length} "
        f"count={len(records)} sha256={jsonl_sha256} -> {jsonl_path}",
        flush=True,
    )
    return 0


def _worker_command(args: argparse.Namespace, task: str, length: int) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--_worker",
        "--model-path",
        args.model_path,
        "--tokenizer-path",
        args.tokenizer_path,
        "--model-family",
        args.model_family,
        "--model-tag",
        args.model_tag,
        "--data-root",
        str(args.data_root),
        "--source-root",
        str(args.source_root),
        "--tasks",
        task,
        "--lengths",
        str(length),
        "--num-samples",
        str(args.num_samples),
        "--margin",
        str(args.margin),
        "--seed",
        str(args.seed),
        "--local-files-only" if args.local_files_only else "--no-local-files-only",
    ]
    if args.force:
        command.append("--force")
    return command


def _coordinator(args: argparse.Namespace) -> int:
    if not RULER_VENDOR_DIR.is_dir():
        raise PrepError(
            f"vendored lm-evaluation-harness RULER directory is missing: {RULER_VENDOR_DIR}"
        )
    _ensure_sources(args.source_root, args.tasks, args.download_sources)
    args.data_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[prepare-ruler] model={args.model_tag} family={args.model_family} "
        f"tokenizer={args.tokenizer_path}",
        flush=True,
    )
    print(
        f"[prepare-ruler] out={args.data_root} sources={args.source_root} "
        f"tasks={','.join(args.tasks)} lengths={','.join(map(str, args.lengths))} "
        f"count={args.num_samples} margin={args.margin} seed={args.seed}",
        flush=True,
    )
    worker_env = os.environ.copy()
    worker_env["PYTHONHASHSEED"] = str(args.seed % (2**32))
    worker_env["TOKENIZERS_PARALLELISM"] = "false"
    if args.local_files_only:
        worker_env["HF_HUB_OFFLINE"] = "1"
        worker_env["TRANSFORMERS_OFFLINE"] = "1"
        worker_env["HF_DATASETS_OFFLINE"] = "1"
    for length in args.lengths:
        for task in args.tasks:
            command = _worker_command(args, task, length)
            completed = subprocess.run(command, env=worker_env, check=False)
            if completed.returncode != 0:
                raise PrepError(
                    f"worker failed for task={task}, nominal_length={length} "
                    f"with exit code {completed.returncode}"
                )
    print("[prepare-ruler] done", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _normalize_args(_build_parser().parse_args(argv))
        return _worker(args) if args._worker else _coordinator(args)
    except PrepError as exc:
        print(f"[prepare-ruler] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
