"""Lightweight hyperparameter search for the post-OCR corrector (LR grid)."""

from __future__ import annotations

import copy
import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_DEFAULT_GRID: List[float] = [3.0e-4, 5.0e-4, 1.0e-3]


def tune_corrector(cfg: AppConfig, n_trials: int = 3, limit: int = 4000,
                   epochs: int = 1, grid: Optional[List[float]] = None) -> Dict:
    from .train_corrector import train_corrector

    lrs = (grid or _DEFAULT_GRID)[:n_trials]
    trials, best = [], None
    for lr in lrs:
        tcfg = copy.deepcopy(cfg)
        tcfg.model.learning_rate = lr
        tcfg.model.num_train_epochs = epochs
        tcfg.model.output_subdir = f"postocr_tune/lr_{lr:.0e}"
        tcfg.model.eval_steps = max(100, tcfg.model.eval_steps // 2)
        tcfg.model.save_steps = tcfg.model.eval_steps
        try:
            res = train_corrector(tcfg, limit=limit, resume=False)
            cer = res["metrics"].get("eval_cer", 1.0)
        except Exception as exc:
            logger.warning("trial lr=%s failed: %s", lr, exc)
            cer = 1.0
        rec = {"learning_rate": lr, "eval_cer": cer}
        trials.append(rec)
        if best is None or cer < best["eval_cer"]:
            best = rec
        logger.info("trial lr=%.0e -> CER=%.4f", lr, cer)

    out = {"best": best, "trials": trials}
    d = run_dir() / "tune"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"tune-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (d / "best.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["tune_corrector"]
