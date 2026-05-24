"""Dependency-light LongBench metrics.

This mirrors the metric semantics used by the LUTAttn LongBench script while
avoiding new required packages. Chinese QA/ROUGE use character-level tokenization
when jieba is unavailable; code similarity uses difflib instead of fuzzywuzzy.
"""

from __future__ import annotations

import difflib
import re
import string
from collections import Counter
from typing import Any, Iterable


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def normalize_zh_answer(s: str) -> str:
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in s.lower() if ch not in all_punctuation and not ch.isspace())


def _zh_tokens(text: str) -> list[str]:
    try:
        import jieba  # type: ignore

        return [normalize_zh_answer(tok) for tok in jieba.cut(text, cut_all=False) if normalize_zh_answer(tok)]
    except Exception:
        return [ch for ch in normalize_zh_answer(text) if ch]


def count_score(prediction: str, ground_truth: str, **_: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(ground_truth) for number in numbers) / len(numbers)


def retrieval_score(prediction: str, ground_truth: str, **_: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(ground_truth_id) for number in numbers) / len(numbers)


def retrieval_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(ground_truth_id) for number in numbers) / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **_: Any) -> float:
    chosen = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            chosen = line
            break
    return difflib.SequenceMatcher(None, chosen, ground_truth).ratio()


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    all_classes = kwargs["all_classes"]
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [term for term in matches if not (term in ground_truth and term != ground_truth)]
    if matches:
        return (1.0 / len(matches)) if ground_truth in matches else 0.0
    best_match = max(all_classes, key=lambda value: difflib.SequenceMatcher(None, value, prediction).ratio())
    return float(best_match == ground_truth)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l_f(tokens_pred: list[str], tokens_gold: list[str]) -> float:
    lcs = _lcs_len(tokens_pred, tokens_gold)
    if lcs == 0 or not tokens_pred or not tokens_gold:
        return 0.0
    precision = lcs / len(tokens_pred)
    recall = lcs / len(tokens_gold)
    return 2 * precision * recall / (precision + recall)


def rouge_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return _rouge_l_f(prediction.split(), ground_truth.split())


def rouge_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return _rouge_l_f(_zh_tokens(prediction), _zh_tokens(ground_truth))


def f1_score(prediction: Iterable[str], ground_truth: Iterable[str], **_: Any) -> float:
    pred = list(prediction)
    gold = list(ground_truth)
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred) if pred else 0.0
    recall = num_same / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return f1_score(_zh_tokens(prediction), _zh_tokens(ground_truth))


DATASET_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
