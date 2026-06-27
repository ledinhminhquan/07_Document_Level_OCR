"""Error analysis for the post-OCR corrector.

Classifies each example as improved / degraded / unchanged (the safety gate),
reports the full reduction metrics, and collects example corrections — including
the worst regressions (where the corrector made things worse).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..models.baseline import IdentityCorrector
from ..models.corrector import load_corrector
from ..data.dataset import load_or_build
from ..training import metrics as M

logger = get_logger(__name__)


def error_analysis(cfg: AppConfig, limit: Optional[int] = None, save: bool = True) -> Dict:
    splits = load_or_build(cfg)
    rows = splits.get("test", [])
    if limit:
        rows = rows[:limit]
    if not rows:
        return {"n": 0, "note": "empty test split"}
    noisy = [r["noisy"] for r in rows]
    clean = [r["clean"] for r in rows]
    corrector = load_corrector(cfg.model, prefer="neural")
    trained = not isinstance(corrector, IdentityCorrector)
    try:
        preds = corrector.correct_batch(noisy)
    except Exception:
        preds = [corrector.correct(t) for t in noisy]

    examples, regressions = [], []
    for nz, p, r in zip(noisy, preds, clean):
        before, after = M.char_cer(nz, r), M.char_cer(p, r)
        rec = {"noisy": nz[:120], "pred": p[:120], "clean": r[:120],
               "cer_before": round(before, 3), "cer_after": round(after, 3)}
        if after > before + 1e-6 and len(regressions) < 10:
            regressions.append(rec)
        elif after < before - 1e-6 and len(examples) < 10:
            examples.append(rec)

    result = {
        "corrector": getattr(corrector, "name", "identity"), "trained_model": trained, "n": len(rows),
        "metrics": M.compute_all(noisy, preds, clean),
        "improved_examples": examples, "regression_examples": regressions,
    }
    if save:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("error analysis: reduction=%s, degraded=%s", result["metrics"]["cer_reduction_rel"],
                    result["metrics"]["pct_degraded"])
    return result


__all__ = ["error_analysis"]
