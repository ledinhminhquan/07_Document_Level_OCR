"""Clean English text source for the synthetic post-OCR corpus.

A built-in sentence bank (varied vocabulary, numbers, punctuation, capitalization)
composed into short paragraphs — the *clean* side of the (noisy, clean) training
pairs. An external corpus file can be supplied to broaden coverage.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

SENTENCES: List[str] = [
    "The annual report was published on March 3, 2021, after a long review.",
    "Researchers measured a 12% increase in efficiency across all departments.",
    "Please return the signed form to the office before the 15th of next month.",
    "The committee approved a budget of $4.5 million for the new facility.",
    "In 1984, the company opened its first branch in the northern district.",
    "Each chapter ends with a short summary and a set of practice questions.",
    "The patient was advised to rest and to drink plenty of fluids daily.",
    "According to the survey, most customers preferred the redesigned interface.",
    "The train departs at 9:45 in the morning and arrives by noon.",
    "Our findings suggest that early intervention reduces long-term costs.",
    "The manuscript describes a method for cleaning noisy historical documents.",
    "She carefully copied the figures from the ledger into the spreadsheet.",
    "The museum displays artifacts collected over more than two centuries.",
    "A balanced diet and regular exercise contribute to better health.",
    "The software update fixed several bugs and improved overall performance.",
    "The contract specifies a delivery date of no later than December 31st.",
    "Visitors are kindly asked to keep their belongings with them at all times.",
    "The lecture covered the history, theory, and practice of the discipline.",
    "Heavy rainfall caused minor flooding in the low-lying areas of the city.",
    "The author thanks the reviewers for their thoughtful and detailed comments.",
    "Sales grew steadily throughout the quarter, exceeding the original forecast.",
    "The library extended its opening hours during the examination period.",
    "Engineers tested the bridge under a range of load and weather conditions.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Members of the board met on Tuesday to discuss the proposed merger.",
    "The article argues that public investment can stimulate local economies.",
    "A clear and concise abstract helps readers decide whether to continue.",
    "The garden was full of roses, tulips, and a variety of fragrant herbs.",
    "Students submitted their assignments through the online portal by Friday.",
    "The instrument records temperature, humidity, and atmospheric pressure.",
    "He kept a careful record of every transaction in a small leather notebook.",
    "The festival attracted thousands of visitors from across the region.",
    "The new policy takes effect immediately and applies to all employees.",
    "Volunteers spent the weekend cleaning the beach and planting young trees.",
    "The map shows the main roads, rivers, and settlements of the province.",
    "A good night of sleep improves memory, mood, and concentration.",
    "The factory reduced its energy consumption by nearly a third last year.",
    "The novel is set in a quiet coastal town during the early twentieth century.",
    "Their proposal was praised for its clarity, feasibility, and low cost.",
    "The teacher explained the concept using a simple and memorable example.",
]


def random_paragraph(rng: random.Random, min_sents: int = 1, max_sents: int = 4) -> str:
    k = rng.randint(min_sents, max_sents)
    return " ".join(rng.choice(SENTENCES) for _ in range(k))


def load_corpus(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines or None


__all__ = ["SENTENCES", "random_paragraph", "load_corpus"]
