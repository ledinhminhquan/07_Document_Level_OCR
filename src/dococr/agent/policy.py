"""Decision-point logic for the document-OCR agent (pure, testable, no model deps).

Four explicit decision points act on intermediate outputs:
* **D1** page-quality / preprocess routing,
* **D2** born-digital vs scanned routing (skip OCR when a text layer exists),
* **D3** OCR-confidence gate (flag low-confidence regions),
* **D4** correction acceptance (accept a correction only within an edit budget).
"""

from __future__ import annotations

from typing import Dict

from ..config import AgentConfig
from ..models.text_utils import edit_ratio


# ── D1 ───────────────────────────────────────────────────────────────────────
def quality_route(metrics: Dict[str, float], cfg: AgentConfig, reproc_count: int) -> str:
    q = metrics.get("quality", 1.0)
    if q >= cfg.quality_min:
        return "ok"
    if reproc_count < 1:
        return "reprocess"
    return "degraded"


# ── D3 ───────────────────────────────────────────────────────────────────────
def ocr_gate(conf: float, cfg: AgentConfig, degraded: bool = False) -> Dict:
    bar = cfg.ocr_confidence_min * (0.6 if degraded else 1.0)
    return {"accept": conf >= bar, "bar": round(bar, 3)}


# ── D4 ───────────────────────────────────────────────────────────────────────
def accept_correction(raw: str, corrected: str, conf: float, cfg: AgentConfig) -> Dict:
    """Accept the corrected text only if it is a *bounded* edit and confident enough."""
    if not corrected or corrected == raw:
        return {"accept": False, "reason": "no change", "edit_ratio": 0.0}
    er = edit_ratio(raw, corrected)
    if er > cfg.correct_max_edit_ratio:
        return {"accept": False, "reason": f"edit ratio {er:.2f} > budget", "edit_ratio": round(er, 3)}
    if conf < cfg.correct_min_conf:
        return {"accept": False, "reason": f"low confidence {conf:.2f}", "edit_ratio": round(er, 3)}
    return {"accept": True, "reason": "within budget", "edit_ratio": round(er, 3)}


__all__ = ["quality_route", "ocr_gate", "accept_correction"]
