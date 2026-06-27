"""Evaluate the post-OCR corrector vs baselines, with CER/WER-reduction metrics.

Reports CER before (raw OCR) vs after (corrected), the % improvement, and the
safety gate (% sentences improved vs degraded) on the synthetic test slice and a
real PleIAs slice. Runs baseline-only (no torch) when no model is trained.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..models.baseline import IdentityCorrector, load_baseline
from ..models.corrector import load_corrector
from ..data.dataset import load_or_build
from . import metrics as M

logger = get_logger(__name__)


def _score(corrector, rows: List[Dict]) -> Dict:
    noisy = [r["noisy"] for r in rows]
    clean = [r["clean"] for r in rows]
    try:
        preds = corrector.correct_batch(noisy)
    except Exception as exc:
        logger.warning("correct_batch failed (%s); per-item", exc)
        preds = [corrector.correct(n) for n in noisy]
    return M.compute_all(noisy, preds, clean)


def evaluate(cfg: AppConfig, which: str = "test", limit: Optional[int] = None, save: bool = True) -> Dict:
    splits = load_or_build(cfg)
    rows = splits.get(which, [])
    if limit:
        rows = rows[:limit]
    real_rows = splits.get("real", [])
    if limit:
        real_rows = real_rows[:limit]

    corrector = load_corrector(cfg.model, prefer="neural")
    trained = not isinstance(corrector, IdentityCorrector)
    identity = IdentityCorrector()

    result: Dict = {
        "which": which, "n": len(rows), "corrector": getattr(corrector, "name", "identity"),
        "trained_model": trained,
        "model": _score(corrector, rows) if rows else {},
        "baseline_identity": _score(identity, rows) if rows else {},
    }
    # optional dictionary baseline (SymSpell)
    dic = load_baseline("dictionary")
    if dic.name == "dictionary" and rows:
        try:
            result["baseline_dictionary"] = _score(dic, rows)
        except Exception as exc:
            logger.info("dictionary baseline skipped (%s)", exc)
    if real_rows and trained:
        result["real"] = _score(corrector, real_rows)
        result["real_baseline"] = _score(identity, real_rows)

    m = result["model"]
    result["summary"] = {
        "corrector": result["corrector"], "trained_model": trained,
        "cer_before": m.get("cer_before"), "cer_after": m.get("cer_after"),
        "cer_reduction_rel": m.get("cer_reduction_rel"),
        "pct_improved": m.get("pct_improved"), "pct_degraded": m.get("pct_degraded"),
        "beats_identity": (m.get("cer_after", 1) < m.get("cer_before", 1)) if trained else None,
    }
    if save:
        out = run_dir() / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"eval-{which}-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Eval [%s] saved: corrector=%s cer_after=%s reduction=%s", which,
                    result["corrector"], m.get("cer_after"), m.get("cer_reduction_rel"))
    return result


__all__ = ["evaluate"]
