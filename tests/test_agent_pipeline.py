"""End-to-end document agent (stub OCR + identity corrector, fully offline)."""

from __future__ import annotations

from dococr.agent.doc_agent import DocumentAgent
from dococr.agent.policy import accept_correction, ocr_gate, quality_route
from dococr.config import AppConfig


def _agent():
    return DocumentAgent(AppConfig(), load_model=False)   # identity corrector + stub OCR


def test_born_digital_path(sample_doc):
    job = _agent().process(text=sample_doc, filename="report.txt", save=False)
    sd = job.to_dict()
    assert sd["status"] in ("completed", "needs_review")
    assert sd["n_blocks"] >= 3
    assert {d["id"] for d in sd["decisions"]} == {"D1", "D2", "D3", "D4"}
    assert all(t["ok"] for t in sd["trace"])
    assert "##" in sd["markdown"]            # at least one heading


def test_scanned_path_with_stub(sample_page):
    job = _agent().process(path=sample_page, filename="scan.png", save=False)
    sd = job.to_dict()
    # D2 routes to scanned; OCR is the stub (no tesseract) -> 0 blocks but pipeline runs
    assert any(d["id"] == "D2" and d["branch"] == "scanned" for d in sd["decisions"])
    assert all(t["ok"] for t in sd["trace"])


def test_decision_helpers():
    cfg = AppConfig().agent
    assert quality_route({"quality": 0.9}, cfg, 0) == "ok"
    assert quality_route({"quality": 0.1}, cfg, 1) == "degraded"
    assert ocr_gate(0.9, cfg)["accept"] is True
    assert ocr_gate(0.1, cfg)["accept"] is False
    # correction acceptance: within budget accepts, huge rewrite rejected
    assert accept_correction("teh cat", "the cat", 0.9, cfg)["accept"] is True
    assert accept_correction("hello world", "totally different sentence entirely", 0.9, cfg)["accept"] is False
    assert accept_correction("same", "same", 0.9, cfg)["accept"] is False
