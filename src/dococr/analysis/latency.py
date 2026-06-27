"""Latency benchmark: post-OCR correction throughput + agent per-document latency."""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _percentiles(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "mean": round(float(a.mean()), 2)}


def benchmark(cfg: AppConfig, n: int = 30, warmup: int = 3, save: bool = True) -> Dict:
    from ..data.ocr_noise import OCRNoiseGenerator
    from ..models.corrector import load_corrector

    gen = OCRNoiseGenerator(cfg.data)
    texts = [gen.example(i)["noisy"] for i in range(n + warmup)]
    corrector = load_corrector(cfg.model, prefer="neural")
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    for t in texts[:warmup]:
        corrector.correct(t)
    lat: List[float] = []
    for t in texts[warmup:]:
        t0 = time.perf_counter()
        corrector.correct(t)
        lat.append((time.perf_counter() - t0) * 1000.0)

    out = {"device": device, "corrector": getattr(corrector, "name", "identity"),
           "correct_ms": _percentiles(lat), "throughput_per_s": round(1000.0 / max(0.05, np.mean(lat)), 2), "n": n}

    # agent end-to-end on a born-digital sample doc
    try:
        from ..agent.doc_agent import DocumentAgent
        from ..data.samples import SAMPLE_DOC
        agent = DocumentAgent(cfg, load_model=True)
        agent.process(text=SAMPLE_DOC, save=False)
        t0 = time.perf_counter()
        agent.process(text=SAMPLE_DOC, save=False)
        out["agent_doc_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as exc:
        logger.info("agent latency skipped (%s)", exc)

    if save:
        d = run_dir() / "benchmark"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"benchmark-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["benchmark"]
