"""The document-OCR agent — a deterministic FSM that turns a document into
structured text.

    ingest (D2 born-digital/scanned) -> layout (D1 quality, D3 OCR-confidence)
        -> post-OCR correct (D4 acceptance, optional LLM) -> assemble (text + markdown + JSON)

Runs fully offline (stub OCR + identity corrector) and upgrades automatically
when a real OCR engine + a fine-tuned corrector are present. Every step is timed
and traced; a manifest.json captures the full record.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from ..config import AppConfig, ensure_dirs, output_dir
from ..logging_utils import JsonlLogger, get_logger, utc_stamp
from ..models.corrector import load_corrector
from ..models.ocr_engine import load_ocr_engine
from . import tools
from .llm_orchestrator import LLMBrain
from .state import Decision, JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()[:48] or "document"


class DocumentAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True,
                 ocr_engine: Optional[str] = None):
        self.cfg = cfg or AppConfig()
        self.corrector = load_corrector(self.cfg.model, prefer="neural" if load_model else "identity")
        self.brain = LLMBrain(self.cfg.agent)
        self._ocr = None
        self._ocr_backend = ocr_engine
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.job_log_path) if self.cfg.serving.log_jobs else None

    def _ocr_engine(self):
        if self._ocr is None:
            self._ocr = load_ocr_engine(self.cfg.ocr, engine=self._ocr_backend)
        return self._ocr

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def process(self, *, path: Optional[str] = None, image=None, text: Optional[str] = None,
                filename: Optional[str] = None, out_dir: Optional[str] = None, save: bool = True) -> JobState:
        fname = filename or (Path(path).name if path else "document")
        job = JobState(filename=fname)
        t0 = time.perf_counter()

        job = self._step(job, "ingest", lambda: tools.tool_ingest(job, self.cfg, path=path, image=image, text=text),
                         summary="ingest + route (D2)")
        if not job._pages:
            job.review_reasons.append("could not load/parse the document")
            return self._finish(job, JobStatus.FAILED, out_dir, save)

        ocr = self._ocr_engine() if any(not p.born_digital for p in job._pages) else _NoOCR()
        job = self._step(job, "layout", lambda: tools.tool_layout(job, self.cfg, ocr_engine=ocr),
                         summary="preprocess + OCR + layout (D1, D3)")
        job = self._step(job, "correct", lambda: tools.tool_correct(job, self.cfg, corrector=self.corrector, brain=self.brain),
                         summary="post-OCR correction (D4)")
        job = self._step(job, "assemble", lambda: tools.tool_assemble(job), summary="assemble text + markdown")

        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        flagged = sum(b.flagged for b in job.blocks)
        if not job.blocks or not job.full_text.strip():
            from ..models.ocr_engine import StubEngine
            if isinstance(ocr, (StubEngine, _NoOCR)) and not any(p.born_digital for p in job._pages):
                job.review_reasons.append("no OCR engine available (stub) — install tesseract for real output")
            status = JobStatus.COMPLETED if isinstance(ocr, (StubEngine, _NoOCR)) else JobStatus.NEEDS_REVIEW
        else:
            status = JobStatus.NEEDS_REVIEW if flagged else JobStatus.COMPLETED
        return self._finish(job, status, out_dir, save)

    def _finish(self, job: JobState, status: JobStatus, out_dir: Optional[str], save: bool) -> JobState:
        job.status = status
        job.model_versions.setdefault("model_version", self.cfg.serving.model_version)
        if save:
            try:
                odir = Path(out_dir or (output_dir() / f"{_slug(job.filename)}-{utc_stamp()}"))
                odir.mkdir(parents=True, exist_ok=True)
                (odir / "document.txt").write_text(job.full_text, encoding="utf-8")
                (odir / "document.md").write_text(job.markdown, encoding="utf-8")
                (odir / "manifest.json").write_text(json.dumps(job.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
                job.outputs.update({"text": str(odir / "document.txt"), "markdown": str(odir / "document.md"),
                                    "manifest": str(odir / "manifest.json")})
            except Exception as exc:
                logger.warning("output write failed: %s", exc)
        if self._log is not None:
            try:
                self._log.log("job", filename=job.filename, status=status.value, n_pages=job.n_pages,
                              n_blocks=len(job.blocks), n_flagged=sum(b.flagged for b in job.blocks),
                              metrics=job.metrics)
            except Exception:
                pass
        return job


class _NoOCR:
    """Sentinel OCR engine used when every page is born-digital (OCR never called)."""
    name = "none"

    def recognize(self, image):
        from ..models.ocr_engine import OcrResult
        return OcrResult(words=[], engine="none")


_AGENT: Optional[DocumentAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> DocumentAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = DocumentAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["DocumentAgent", "get_agent"]
