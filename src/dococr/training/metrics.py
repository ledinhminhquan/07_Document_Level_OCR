"""Post-OCR correction metrics — CER/WER and, crucially, **error reduction**.

The headline for a corrector is not just CER but how much it *reduces* the OCR
error: ``CER_after`` vs ``CER_before`` (the raw OCR). A corrector must not make
things worse, so we also report the fraction of examples improved vs degraded.
"""

from __future__ import annotations

from typing import Dict, List

from ..models.text_utils import levenshtein, normalize_ws


def char_cer(pred: str, ref: str) -> float:
    ref = normalize_ws(ref)
    return levenshtein(list(normalize_ws(pred)), list(ref)) / max(1, len(ref))


def word_wer(pred: str, ref: str) -> float:
    r = normalize_ws(ref).split()
    return levenshtein(normalize_ws(pred).split(), r) / max(1, len(r))


def corpus_cer(preds: List[str], refs: List[str]) -> float:
    edits = total = 0
    for p, r in zip(preds, refs):
        r = normalize_ws(r)
        edits += levenshtein(list(normalize_ws(p)), list(r))
        total += len(r)
    return round(edits / max(1, total), 4)


def corpus_wer(preds: List[str], refs: List[str]) -> float:
    edits = total = 0
    for p, r in zip(preds, refs):
        rw = normalize_ws(r).split()
        edits += levenshtein(normalize_ws(p).split(), rw)
        total += len(rw)
    return round(edits / max(1, total), 4)


def exact_match(preds: List[str], refs: List[str]) -> float:
    if not refs:
        return 0.0
    return round(sum(normalize_ws(p) == normalize_ws(r) for p, r in zip(preds, refs)) / len(refs), 4)


def reduction_metrics(noisy: List[str], preds: List[str], refs: List[str]) -> Dict:
    """Compare CER before (noisy vs ref) and after (pred vs ref) correction."""
    improved = degraded = same = 0
    for nz, p, r in zip(noisy, preds, refs):
        before, after = char_cer(nz, r), char_cer(p, r)
        if after < before - 1e-6:
            improved += 1
        elif after > before + 1e-6:
            degraded += 1
        else:
            same += 1
    n = max(1, len(refs))
    before_cer = corpus_cer(noisy, refs)
    after_cer = corpus_cer(preds, refs)
    rel = (before_cer - after_cer) / before_cer if before_cer > 0 else 0.0
    return {"cer_before": before_cer, "cer_after": after_cer,
            "cer_reduction_abs": round(before_cer - after_cer, 4),
            "cer_reduction_rel": round(rel, 4),
            "pct_improved": round(improved / n, 4), "pct_degraded": round(degraded / n, 4),
            "pct_unchanged": round(same / n, 4)}


def compute_all(noisy: List[str], preds: List[str], refs: List[str]) -> Dict:
    out = {"n": len(refs), "cer": corpus_cer(preds, refs), "wer": corpus_wer(preds, refs),
           "exact_match": exact_match(preds, refs)}
    out.update(reduction_metrics(noisy, preds, refs))
    return out


__all__ = ["char_cer", "word_wer", "corpus_cer", "corpus_wer", "exact_match",
           "reduction_metrics", "compute_all"]
