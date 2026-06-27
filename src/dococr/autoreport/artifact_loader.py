"""Collect generated artifacts into one dict for the report + slides generators.
Everything is optional; missing pieces resolve to ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig, run_dir
from ..models.model_registry import read_metadata, resolve_latest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_artifacts(cfg: AppConfig) -> Dict[str, Any]:
    rd = run_dir()
    arts = {
        "eval": _load_json(rd / "eval" / "latest.json"),
        "error_analysis": _load_json(rd / "error_analysis" / "latest.json"),
        "benchmark": _load_json(rd / "benchmark" / "latest.json"),
        "robustness": _load_json(rd / "robustness" / "latest.json"),
        "tune": _load_json(rd / "tune" / "best.json"),
        "monitoring": _load_json(rd / "monitoring" / "latest.json"),
    }
    latest = resolve_latest(cfg.model.output_dir)
    arts["model_meta"] = read_metadata(latest) if latest else {}
    return arts


def read_doc(name: str) -> str:
    p = repo_root() / "docs" / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


__all__ = ["load_artifacts", "read_doc", "repo_root"]
