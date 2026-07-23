"""Deterministic document-level WikiText windows for cache-aware PPL."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_NAME = "wikitext-2-raw-v1"
CORPUS_SPLIT = "test"
DEFAULT_TEXT_FIELD = "page"
DETOKENIZER_VERSION = "lm-eval-wikitext-v2"
TOKENIZATION_POLICY = "no-special-tokens-v1"
WINDOW_POLICY = "document-local-nonoverlap-v1"


@dataclass(frozen=True)
class PPLDocument:
    source_index: int
    document_id: str
    text: str
    text_sha256: str


@dataclass(frozen=True)
class PPLWindow:
    sample_idx: int
    document_id: str
    source_index: int
    window_index: int
    start_token: int
    token_ids: tuple[int, ...]

    @property
    def input_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class PPLWindowPlan:
    windows: tuple[PPLWindow, ...]
    total_documents: int
    eligible_documents: int
    skipped_short_documents: int
    available_windows: int
    selected_windows: int
    window_tokens: int
    selected_token_sha256: str


def wikitext_detokenize(text: str) -> str:
    """Match lm-eval's WikiText v2 detokenization contract."""

    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", "/'[0-9]/", text)
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" . ", ". ")
    text = text.replace(" ! ", "! ")
    text = text.replace(" ? ", "? ")
    text = text.replace(" , ", ", ")
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r'"\s*([^\"]*?)\s*"', r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    text = text.replace("= = = =", "====")
    text = text.replace("= = =", "===")
    text = text.replace("= =", "==")
    text = text.replace(" " + chr(176) + " ", chr(176))
    text = text.replace(" \n", "\n")
    text = text.replace("\n ", "\n")
    text = text.replace(" N ", " 1 ")
    return text.replace(" 's", "'s")


def load_wikitext_documents(
    data_path: str | Path,
    *,
    text_field: str = DEFAULT_TEXT_FIELD,
) -> list[PPLDocument]:
    """Load one local document-level WikiText parquet with strict schema checks."""

    path = Path(data_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PPL corpus parquet does not exist: {path}")

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if text_field not in table.column_names:
        raise ValueError(
            f"PPL corpus {path} lacks text field {text_field!r}; "
            f"columns={table.column_names}"
        )

    documents: list[PPLDocument] = []
    for source_index, value in enumerate(table[text_field].to_pylist()):
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"PPL corpus {path} row {source_index} field {text_field!r} "
                f"must be a string, got {type(value).__name__}"
            )
        if not value.strip():
            continue
        text_sha256 = hashlib.sha256(value.encode("utf-8")).hexdigest()
        documents.append(
            PPLDocument(
                source_index=source_index,
                document_id=f"doc-{source_index:06d}-{text_sha256[:12]}",
                text=value,
                text_sha256=text_sha256,
            )
        )
    if not documents:
        raise ValueError(f"PPL corpus {path} has no non-empty {text_field!r} documents")
    return documents


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        wikitext_detokenize(text),
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("PPL tokenizer unexpectedly returned a batched encoding")
        token_ids = token_ids[0]
    result = [int(token_id) for token_id in token_ids]
    if any(token_id < 0 or token_id >= 2**32 for token_id in result):
        raise ValueError("PPL tokenizer emitted an out-of-range token id")
    return result


def _selected_token_sha256(windows: list[PPLWindow]) -> str:
    digest = hashlib.sha256()
    for window in windows:
        digest.update(window.document_id.encode("utf-8"))
        digest.update(window.window_index.to_bytes(8, "little", signed=False))
        digest.update(window.start_token.to_bytes(8, "little", signed=False))
        digest.update(len(window.token_ids).to_bytes(8, "little", signed=False))
        for token_id in window.token_ids:
            digest.update(token_id.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def build_window_plan(
    documents: list[PPLDocument],
    tokenizer: Any,
    *,
    prefill_tokens: int,
    score_tokens: int,
    max_samples: int = -1,
) -> PPLWindowPlan:
    """Create deterministic, non-overlapping windows within each document."""

    if prefill_tokens <= 1:
        raise ValueError(f"prefill_tokens must be > 1, got {prefill_tokens}")
    if score_tokens <= 0:
        raise ValueError(f"score_tokens must be positive, got {score_tokens}")
    window_tokens = prefill_tokens + score_tokens + 1
    limit = max_samples if max_samples > 0 else None

    selected: list[PPLWindow] = []
    available_windows = 0
    eligible_documents = 0
    for document in documents:
        token_ids = _token_ids(tokenizer, document.text)
        document_windows = max(0, len(token_ids) // window_tokens)
        if document_windows:
            eligible_documents += 1
        for window_index in range(document_windows):
            start = window_index * window_tokens
            available_windows += 1
            if limit is not None and len(selected) >= limit:
                continue
            selected.append(
                PPLWindow(
                    sample_idx=len(selected),
                    document_id=document.document_id,
                    source_index=document.source_index,
                    window_index=window_index,
                    start_token=start,
                    token_ids=tuple(token_ids[start : start + window_tokens]),
                )
            )

    if not selected:
        raise RuntimeError(
            "PPL corpus produced no eligible windows: "
            f"documents={len(documents)}, window_tokens={window_tokens}, "
            f"max_samples={max_samples}"
        )
    return PPLWindowPlan(
        windows=tuple(selected),
        total_documents=len(documents),
        eligible_documents=eligible_documents,
        skipped_short_documents=len(documents) - eligible_documents,
        available_windows=available_windows,
        selected_windows=len(selected),
        window_tokens=window_tokens,
        selected_token_sha256=_selected_token_sha256(selected),
    )
