"""Monitoring & drift report from production job logs (JSONL).

Aggregates job outcomes (status mix, flag-rate, mean confidence, latency) and
computes a drift signal by comparing a recent window against an earlier baseline
window — the operational early-warning (rising flag-rate / falling confidence
signals a shift in incoming document quality).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _read_logs(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _window_stats(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0}
    statuses: Dict[str, int] = {}
    flags, confs, lats, blocks = [], [], [], []
    for r in rows:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
        nb = r.get("n_blocks", 0) or 0
        blocks.append(nb)
        flags.append((r.get("n_flagged", 0) or 0) / max(1, nb))
        m = r.get("metrics", {}) or {}
        if isinstance(m.get("mean_confidence"), (int, float)):
            confs.append(m["mean_confidence"])
        if isinstance(m.get("latency_ms"), (int, float)):
            lats.append(m["latency_ms"])
    return {"n": len(rows), "statuses": statuses,
            "flag_rate": round(mean(flags), 4) if flags else 0.0,
            "mean_confidence": round(mean(confs), 4) if confs else None,
            "mean_latency_ms": round(mean(lats), 1) if lats else None,
            "mean_blocks": round(mean(blocks), 1) if blocks else 0.0}


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None, save: bool = True) -> Dict:
    path = Path(log_path) if log_path else cfg.serving.job_log_path
    rows = _read_logs(path)
    overall = _window_stats(rows)
    drift = {}
    if len(rows) >= 6:
        half = len(rows) // 2
        base, recent = _window_stats(rows[:half]), _window_stats(rows[half:])

        def delta(k):
            a, b = base.get(k), recent.get(k)
            return round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        drift = {"baseline_window": base, "recent_window": recent,
                 "delta_flag_rate": delta("flag_rate"), "delta_mean_confidence": delta("mean_confidence"),
                 "alert": bool((delta("flag_rate") or 0) > 0.1 or (delta("mean_confidence") or 0) < -0.1)}
    result = {"log_path": str(path), "n_jobs": len(rows), "overall": overall, "drift": drift,
              "note": "no job logs found yet" if not rows else ""}
    if save:
        d = run_dir() / "monitoring"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"monitor-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["monitoring_report"]
