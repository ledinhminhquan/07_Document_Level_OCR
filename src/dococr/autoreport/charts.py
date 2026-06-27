"""Matplotlib charts for the report/slides. Returns saved PNG paths; degrades to
``None`` when matplotlib is unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def cer_reduction_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not eval_art or "model" not in eval_art:
        return None
    try:
        plt = _mpl()
        m = eval_art["model"]
        before, after = m.get("cer_before", 0), m.get("cer_after", 0)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.bar(["raw OCR", "corrected"], [before, after], color=["#9aa7b4", "#2b6cb0"])
        ax.set_ylabel("CER (lower is better)")
        ax.set_title("Post-OCR correction: CER before vs after")
        for i, v in enumerate([before, after]):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("cer_reduction_chart skipped (%s)", exc)
        return None


def safety_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not eval_art or "model" not in eval_art:
        return None
    try:
        plt = _mpl()
        m = eval_art["model"]
        vals = [m.get("pct_improved", 0), m.get("pct_unchanged", 0), m.get("pct_degraded", 0)]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.bar(["improved", "unchanged", "degraded"], vals, color=["#2f855a", "#a0aec0", "#c53030"])
        ax.set_title("Correction safety (fraction of examples)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("safety_chart skipped (%s)", exc)
        return None


def robustness_chart(rob_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not rob_art or "by_noise_level" not in rob_art:
        return None
    try:
        plt = _mpl()
        bl = rob_art["by_noise_level"]
        levels = list(bl.keys())
        before = [bl[k]["cer_before"] for k in levels]
        after = [bl[k]["cer_after"] for k in levels]
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.plot(levels, before, "o-", label="raw OCR", color="#9aa7b4")
        ax.plot(levels, after, "o-", label="corrected", color="#2b6cb0")
        ax.set_xlabel("noise level (char error rate)"); ax.set_ylabel("CER")
        ax.set_title("Robustness to OCR-noise severity"); ax.legend()
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("robustness_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: List[Tuple[str, Path]] = []
    for name, fn, key in [("cer_reduction", cer_reduction_chart, "eval"), ("safety", safety_chart, "eval"),
                          ("robustness", robustness_chart, "robustness")]:
        p = fn(arts.get(key) or {}, out_dir / f"{name}.png")
        if p:
            charts.append((name, p))
    return charts


__all__ = ["cer_reduction_chart", "safety_chart", "robustness_chart", "build_all"]
