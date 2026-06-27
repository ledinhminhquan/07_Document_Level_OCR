"""Baseline correctors the neural model must beat.

* ``IdentityCorrector`` — returns the raw OCR unchanged. This is the canonical
  post-OCR baseline: the neural model must *reduce* CER below "no correction".
* ``DictionaryCorrector`` — optional lexicon/edit-distance correction
  (``pyspellchecker`` if installed), a stronger non-neural reference.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..logging_utils import get_logger
from .text_utils import normalize_ws

logger = get_logger(__name__)


class IdentityCorrector:
    name = "identity"
    version = "identity-1.0"

    def correct(self, text: str) -> str:
        return text

    def correct_batch(self, texts: List[str]) -> List[str]:
        return list(texts)


class DictionaryCorrector:
    """Per-token spell correction over a frequency lexicon (optional dependency)."""
    name = "dictionary"
    version = "dictionary-1.0"

    def __init__(self):
        from spellchecker import SpellChecker  # lazy; raises if unavailable
        self._sp = SpellChecker(distance=1)

    def _fix_token(self, tok: str) -> str:
        core = re.sub(r"[^A-Za-z]", "", tok)
        if len(core) < 3 or core.lower() in self._sp:
            return tok
        cand = self._sp.correction(core.lower())
        if not cand or cand == core.lower():
            return tok
        fixed = cand.capitalize() if core[:1].isupper() else cand
        return tok.replace(core, fixed)

    def correct(self, text: str) -> str:
        return normalize_ws(" ".join(self._fix_token(t) for t in text.split()))

    def correct_batch(self, texts: List[str]) -> List[str]:
        return [self.correct(t) for t in texts]


def load_baseline(prefer: str = "identity") -> "IdentityCorrector | DictionaryCorrector":
    if prefer == "dictionary":
        try:
            return DictionaryCorrector()
        except Exception as exc:
            logger.info("DictionaryCorrector unavailable (%s); using identity baseline", exc)
    return IdentityCorrector()


__all__ = ["IdentityCorrector", "DictionaryCorrector", "load_baseline"]
