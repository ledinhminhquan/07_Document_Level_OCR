"""Synthetic OCR-noise generator (the PRIMARY training data for the corrector).

Corrupts clean English text with a realistic OCR error model — character
confusions (``rn↔m``, ``cl↔d``, ``O↔0``, ``l↔1``, ``e↔c`` …), insertions,
deletions, word merge/split (spacing), and case flips — to build ``(noisy, clean)``
pairs. Deterministic given a seed; zero heavy dependencies. The real PleIAs corpus
is mixed in separately for realism (see ``data/dataset.py``).
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

from ..config import DataConfig
from .corpus import load_corpus, random_paragraph

# clean char -> OCR-confusable alternatives
_SUBS: Dict[str, List[str]] = {
    "o": ["0", "c", "e"], "O": ["0", "Q", "D"], "0": ["o", "O"], "l": ["1", "i", "|", "I"],
    "1": ["l", "I"], "I": ["l", "1", "i"], "i": ["l", "j", "1"], "s": ["5"], "S": ["5", "8"],
    "B": ["8"], "8": ["B", "3"], "g": ["9", "q"], "q": ["g"], "e": ["c", "o"], "c": ["e", "o", "("],
    "a": ["o", "e", "@"], "n": ["u", "h"], "u": ["n", "v"], "h": ["b", "n"], "b": ["h", "6"],
    "v": ["u", "y"], "r": ["t", "f"], "t": ["f", "r", "+"], "f": ["t"], "D": ["O"], "G": ["6"],
    "Z": ["2"], "5": ["s", "S"], "6": ["b", "G"], "9": ["g", "q"], ".": [",", ";"], ",": [".", "'"],
    "-": ["~", "="], ":": [";", "."], "m": ["nn"], "w": ["vv"],
}
# clean substring -> noisy substring (both merge and split directions)
_MULTI = [("m", "rn"), ("rn", "m"), ("cl", "d"), ("d", "cl"), ("w", "vv"), ("vv", "w"),
          ("h", "li"), ("ll", "ll"), ("ri", "n"), ("nn", "m"), ("tt", "n")]
_MULTI_LHS = sorted({a for a, _ in _MULTI}, key=len, reverse=True)
_MULTI_MAP: Dict[str, List[str]] = {}
for _a, _b in _MULTI:
    _MULTI_MAP.setdefault(_a, []).append(_b)

_RAND_CHARS = "abcdefghijklmnopqrstuvwxyz.,'"


def corrupt(text: str, rng: random.Random, rate: float = 0.08) -> str:
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        # word merge: drop a space (joins two words)
        if ch == " " and rng.random() < rate * 1.2:
            i += 1
            continue
        if rng.random() >= rate:
            out.append(ch)
            i += 1
            continue
        # corruption — pick an operation
        # 1) multi-char confusion (longest match first)
        applied = False
        for lhs in _MULTI_LHS:
            if text.startswith(lhs, i) and rng.random() < 0.5:
                out.append(rng.choice(_MULTI_MAP[lhs]))
                i += len(lhs)
                applied = True
                break
        if applied:
            continue
        op = rng.random()
        if op < 0.45 and ch in _SUBS:                 # substitution
            out.append(rng.choice(_SUBS[ch]))
            i += 1
        elif op < 0.60:                               # substitution with a random char
            out.append(rng.choice(_RAND_CHARS) if ch.isalnum() else ch)
            i += 1
        elif op < 0.75:                               # deletion
            i += 1
        elif op < 0.88:                               # insertion (spurious char)
            out.append(rng.choice(_RAND_CHARS))
            out.append(ch)
            i += 1
        elif op < 0.95 and ch.isalpha():              # word split (spurious space)
            out.append(ch)
            out.append(" ")
            i += 1
        else:                                         # case flip
            out.append(ch.swapcase())
            i += 1
    return "".join(out)


class OCRNoiseGenerator:
    """Deterministic generator of synthetic ``(noisy, clean)`` post-OCR pairs."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.corpus = load_corpus(getattr(cfg, "corpus_file", None))

    def _clean(self, rng: random.Random) -> str:
        if self.corpus:
            return rng.choice(self.corpus)
        text = random_paragraph(rng, 1, 4)
        return text[: self.cfg.max_chars]

    def example(self, index: int, base_seed: int = 0) -> Dict:
        rng = random.Random((self.cfg.seed + base_seed) * 1_000_003 + index)
        clean = self._clean(rng)
        noisy = corrupt(clean, rng, self.cfg.char_error_rate)
        return {"noisy": noisy, "clean": clean}

    def iter_examples(self, n: int, seed: Optional[int] = None) -> Iterator[Dict]:
        base = 0 if seed is None else seed
        for i in range(n):
            ex = self.example(i, base_seed=base)
            if ex["noisy"] and ex["clean"]:
                yield ex

    def generate(self, n: int, seed: Optional[int] = None) -> List[Dict]:
        return list(self.iter_examples(n, seed))


__all__ = ["OCRNoiseGenerator", "corrupt"]
