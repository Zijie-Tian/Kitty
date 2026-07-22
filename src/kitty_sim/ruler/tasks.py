"""Canonical NVIDIA RULER synthetic-task registry."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

MetricName = Literal["string_match_all", "string_match_part"]
TASK_VERSION = "ruler-synthetic-v1"


@dataclass(frozen=True)
class TaskSpec:
    """Generation and scoring contract for one public RULER task."""

    name: str
    max_new_tokens: int
    metric: MetricName
    depth_eligible: bool
    version: str


def _spec(
    name: str,
    max_new_tokens: int,
    metric: MetricName = "string_match_all",
    *,
    depth_eligible: bool = False,
) -> TaskSpec:
    return TaskSpec(
        name=name,
        max_new_tokens=max_new_tokens,
        metric=metric,
        depth_eligible=depth_eligible,
        version=TASK_VERSION,
    )


_TASK_SPECS = {
    "niah_single_1": _spec("niah_single_1", 128, depth_eligible=True),
    "niah_single_2": _spec("niah_single_2", 128, depth_eligible=True),
    "niah_single_3": _spec("niah_single_3", 128, depth_eligible=True),
    "niah_multikey_1": _spec("niah_multikey_1", 128, depth_eligible=True),
    "niah_multikey_2": _spec("niah_multikey_2", 128, depth_eligible=True),
    "niah_multikey_3": _spec("niah_multikey_3", 128, depth_eligible=True),
    "niah_multivalue": _spec("niah_multivalue", 128, depth_eligible=True),
    "niah_multiquery": _spec("niah_multiquery", 128, depth_eligible=True),
    "vt": _spec("vt", 30),
    "cwe": _spec("cwe", 120),
    "fwe": _spec("fwe", 50),
    "qa_1": _spec("qa_1", 32, "string_match_part"),
    "qa_2": _spec("qa_2", 32, "string_match_part"),
}

TASK_SPECS: Mapping[str, TaskSpec] = MappingProxyType(_TASK_SPECS)
DEFAULT_TASKS = tuple(TASK_SPECS)
NIAH_TASKS = tuple(name for name, spec in TASK_SPECS.items() if spec.depth_eligible)
DEFAULT_SEQ_LENS = (4096, 8192, 16384, 32768)


def get_task_spec(name: str) -> TaskSpec:
    """Resolve a public task name or fail with the canonical choices."""

    try:
        return TASK_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(DEFAULT_TASKS)
        raise ValueError(f"Unknown RULER task {name!r}; expected one of: {choices}") from exc


def resolve_task_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize ``all`` or a CSV/sequence without changing registry order."""

    if value is None:
        return DEFAULT_TASKS
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() == "all":
            return DEFAULT_TASKS
        names = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in value if str(part).strip())
    if not names:
        raise ValueError("RULER task selection must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"RULER task selection contains duplicates: {names}")
    for name in names:
        get_task_spec(name)
    return names
