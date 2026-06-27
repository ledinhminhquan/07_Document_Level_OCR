"""Robustness analysis: CER reduction across increasing OCR-noise severity.

Generates noisy text at several error rates and measures how well the corrector
recovers at each level — quantifying robustness to the noisy, degraded scans real
documents produce.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..models.corrector import load_corrector
from ..models.baseline import IdentityCorrector
from ..data.corpus import random_paragraph
from ..data.ocr_noise import corrupt
from ..training import metrics as M

logger = get_logger(__name__)

_LEVELS = [0.04, 0.08, 0.12, 0.16, 0.20]


def robustness_report(cfg: AppConfig, limit: int = 80, save: bool = True) -> Dict:
    rng = random.Random(99)
    clean = [random_paragraph(rng, 1, 3) for _ in range(limit)]
    corrector = load_corrector(cfg.model, prefer="neural")
    trained = not isinstance(corrector, IdentityCorrector)

    by_level: Dict[str, Dict] = {}
    for rate in _LEVELS:
        noisy = [corrupt(c, random.Random(hash((rate, i)) & 0xFFFF), rate) for i, c in enumerate(clean)]
        try:
            preds = corrector.correct_batch(noisy)
        except Exception:
            preds = [corrector.correct(t) for t in noisy]
        m = M.reduction_metrics(noisy, preds, clean)
        by_level[f"{rate:.2f}"] = {"cer_before": m["cer_before"], "cer_after": m["cer_after"],
                                   "cer_reduction_rel": m["cer_reduction_rel"], "pct_degraded": m["pct_degraded"]}

    result = {"corrector": getattr(corrector, "name", "identity"), "trained_model": trained,
              "n_per_level": limit, "by_noise_level": by_level}
    if save:
        d = run_dir() / "robustness"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"robustness-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["robustness_report"]
