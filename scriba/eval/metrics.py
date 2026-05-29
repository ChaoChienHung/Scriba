from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _levenshtein_distance(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    pred_chars = list(pred)
    ref_chars = list(ref)
    if len(ref_chars) == 0:
        return 0.0 if len(pred_chars) == 0 else 1.0
    return _levenshtein_distance(pred_chars, ref_chars) / len(ref_chars)


def wer(pred: str, ref: str) -> float:
    pred_words = pred.split()
    ref_words = ref.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _levenshtein_distance(pred_words, ref_words) / len(ref_words)


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if len(values_list) == 0:
        return 0.0
    return float(sum(values_list) / len(values_list))


@dataclass(frozen=True)
class Metrics:
    cer: float
    wer: float


def compute_ocr_metrics(processor: Any, eval_pred: Any) -> dict[str, float]:
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    label_ids = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)

    pred_str = processor.batch_decode(preds, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cers = [cer(p.strip(), r.strip()) for p, r in zip(pred_str, label_str)]
    wers = [wer(p.strip(), r.strip()) for p, r in zip(pred_str, label_str)]
    return {
        "cer": mean(cers),
        "wer": mean(wers),
    }

