"""Text utilities — whitespace normalization + Levenshtein (shared by metrics,
the baseline corrector, and the agent's correction-acceptance gate)."""

from __future__ import annotations

import re


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def levenshtein(a, b) -> int:
    """Edit distance over sequences (chars or word-tokens)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def edit_ratio(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return levenshtein(a, b) / m if m else 0.0


__all__ = ["normalize_ws", "levenshtein", "edit_ratio"]
