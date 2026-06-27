"""Leaf modules: evaluate, error analysis, robustness, monitoring, grading."""

from __future__ import annotations

from pathlib import Path

from dococr.analysis.error_analysis import error_analysis
from dococr.analysis.robustness import robustness_report
from dococr.grading.checklist import build_checklist
from dococr.monitoring.drift_report import monitoring_report
from dococr.training.evaluate import evaluate


def test_evaluate_structure(cfg):
    res = evaluate(cfg, which="test", limit=40, save=False)
    assert "model" in res and "cer" in res["model"]
    assert "cer_before" in res["model"]
    assert "summary" in res


def test_error_analysis_structure(cfg):
    res = error_analysis(cfg, limit=40, save=False)
    assert "metrics" in res
    assert "pct_degraded" in res["metrics"]


def test_robustness_structure(cfg):
    res = robustness_report(cfg, limit=20, save=False)
    assert res["by_noise_level"]
    assert all("cer_before" in v for v in res["by_noise_level"].values())


def test_monitoring_handles_empty(cfg):
    res = monitoring_report(cfg, log_path="/nonexistent/jobs.jsonl", save=False)
    assert res["n_jobs"] == 0


def test_grade_repo():
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["FAIL"] == 0, [i for i in res["items"] if i["status"] == "FAIL"]
    assert res["ok"] is True
