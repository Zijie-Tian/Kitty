"""Local JSONL LongBench dataset loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset


def default_data_root() -> Path:
    env_root = os.environ.get("LONGBENCH_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path("data") / "LongBench"


def resolve_dataset_file(dataset_name: str, data_root: str | os.PathLike[str] | None = None) -> Path:
    root = Path(data_root).expanduser() if data_root is not None else default_data_root()
    return root / "data" / f"{dataset_name}.jsonl"


def load_longbench_dataset(
    dataset_name: str,
    *,
    split: str = "test",
    data_root: str | os.PathLike[str] | None = None,
) -> Dataset:
    """Load a LongBench test split from ``<data_root>/data/<dataset>.jsonl``."""
    if split != "test":
        raise ValueError(f"LongBench only provides test split, got {split!r}")

    jsonl_path = resolve_dataset_file(dataset_name, data_root)
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}. "
            "Set LONGBENCH_DATA_ROOT or pass --data-root so it points to a "
            "LongBench directory containing data/<dataset>.jsonl."
        )

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)
