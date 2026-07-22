"""Strict loader for canonical NVIDIA RULER JSONL data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .tasks import TaskSpec, get_task_spec

_BASE_FIELDS = frozenset(
    {"index", "input", "outputs", "length", "max_length", "gen_prefix"}
)
_DEPTH_FIELD = "token_position_answer"


def default_ruler_data_root() -> Path:
    env_root = os.environ.get("RULER_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / "data" / "ruler" / "llama3"


def resolve_ruler_file(
    task: str,
    seq_len: int,
    data_root: str | os.PathLike[str] | None = None,
) -> Path:
    get_task_spec(task)
    nominal_length = int(seq_len)
    if nominal_length <= 0:
        raise ValueError(f"RULER sequence length must be positive, got {seq_len!r}")
    root = Path(data_root).expanduser() if data_root else default_ruler_data_root()
    return root / str(nominal_length) / task / "validation.jsonl"


def resolve_ruler_manifest(
    task: str,
    seq_len: int,
    data_root: str | os.PathLike[str] | None = None,
) -> Path:
    return resolve_ruler_file(task, seq_len, data_root).with_name(
        "validation.manifest.json"
    )


def _record_error(path: Path, line_no: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line_no}: {message}")


def _validate_record(
    record: Any,
    *,
    spec: TaskSpec,
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise _record_error(path, line_no, "record must be a JSON object")
    if "answer_prefix" in record:
        raise _record_error(
            path,
            line_no,
            "legacy answer_prefix is not accepted; regenerate canonical RULER data "
            "with gen_prefix",
        )

    expected_fields = _BASE_FIELDS | ({_DEPTH_FIELD} if spec.depth_eligible else set())
    actual_fields = set(record)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing required fields {missing}")
        if unexpected:
            details.append(f"unexpected fields {unexpected}")
        raise _record_error(
            path,
            line_no,
            "; ".join(details) + "; regenerate canonical RULER data",
        )

    index = record["index"]
    if type(index) is not int or index < 0:
        raise _record_error(path, line_no, "index must be a non-negative integer")
    if not isinstance(record["input"], str):
        raise _record_error(path, line_no, "input must be a string")
    outputs = record["outputs"]
    if (
        not isinstance(outputs, list)
        or not outputs
        or any(not isinstance(output, str) or not output for output in outputs)
    ):
        raise _record_error(path, line_no, "outputs must be a non-empty list of non-empty strings")
    length = record["length"]
    if type(length) is not int or length <= 0:
        raise _record_error(path, line_no, "length must be a positive integer")
    max_length = record["max_length"]
    if type(max_length) is not int or max_length <= 0:
        raise _record_error(path, line_no, "max_length must be a positive integer")
    if not isinstance(record["gen_prefix"], str):
        raise _record_error(path, line_no, "gen_prefix must be a string")

    if spec.depth_eligible:
        position = record[_DEPTH_FIELD]
        if type(position) is not int or not 0 <= position <= length:
            raise _record_error(
                path,
                line_no,
                f"token_position_answer must be an integer in [0, length={length}]",
            )
    return record


def load_ruler_records(
    task: str,
    seq_len: int,
    data_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load and validate one ``<length>/<task>/validation.jsonl`` pair."""

    spec = get_task_spec(task)
    path = resolve_ruler_file(task, seq_len, data_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"RULER data file not found: {path}. Generate it with "
            "scripts/prepare_ruler_data.sh (set RULER_DATA_ROOT / --data-root)."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise _record_error(path, line_no, "blank JSONL records are not allowed")
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _record_error(path, line_no, f"invalid JSON: {exc.msg}") from exc
            records.append(
                _validate_record(raw_record, spec=spec, path=path, line_no=line_no)
            )
    if not records:
        raise ValueError(f"RULER data file is empty: {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ruler_data_manifest(
    task: str,
    seq_len: int,
    data_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load a pair's provenance manifest and verify it names the exact JSONL."""

    spec = get_task_spec(task)
    data_path = resolve_ruler_file(task, seq_len, data_root)
    manifest_path = resolve_ruler_manifest(task, seq_len, data_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"RULER data manifest not found: {manifest_path}. Regenerate this pair "
            "with scripts/prepare_ruler_data.sh."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid RULER data manifest {manifest_path}: {exc.msg}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"RULER data manifest must be a JSON object: {manifest_path}")

    expected = {
        "task": task,
        "task_version": spec.version,
        "nominal_length": int(seq_len),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"RULER data manifest {manifest_path} has {field}={manifest.get(field)!r}; "
                f"expected {value!r}"
            )
    declared_sha = manifest.get("jsonl_sha256")
    actual_sha = _sha256(data_path)
    if declared_sha != actual_sha:
        raise ValueError(
            f"RULER data manifest {manifest_path} jsonl_sha256 mismatch: "
            f"declared={declared_sha!r}, actual={actual_sha!r}"
        )
    return manifest
