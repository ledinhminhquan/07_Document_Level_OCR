"""FastAPI service for the Document-Level OCR system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /ocr``      – upload an image/PDF -> structured text (markdown + blocks)
* ``POST /correct``  – post-OCR correct raw text (no image) -> corrected text
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import (BlockView, CorrectRequest, CorrectResponse, DocResponse, HealthResponse)

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)


def _to_response(job) -> DocResponse:
    sd = job.to_dict()
    return DocResponse(
        filename=sd["filename"], status=sd["status"], n_pages=sd["n_pages"], n_blocks=sd["n_blocks"],
        n_flagged=sd["n_flagged"], mean_confidence=sd["mean_confidence"], full_text=sd["full_text"],
        markdown=sd["markdown"], decisions=sd["decisions"], metrics=sd["metrics"], model_versions=sd["model_versions"],
        blocks=[BlockView(page=b["page"], reading_index=b["reading_index"], kind=b["kind"],
                          text=b["text"], conf=b["conf"], flagged=b["flagged"]) for b in sd["blocks"]])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", corrector=getattr(agent.corrector, "name", "identity"),
                          ocr_engine=getattr(agent._ocr_engine(), "name", "stub"), version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "corrector": getattr(agent.corrector, "version", "identity"),
            "ocr_engine": getattr(agent._ocr_engine(), "name", "stub")}


@app.post("/correct", response_model=CorrectResponse)
def correct(req: CorrectRequest) -> CorrectResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="empty text")
    job = get_agent().process(text=req.text, filename="pasted", save=False)
    return CorrectResponse(corrected=job.full_text, n_blocks=len(job.blocks),
                           full_text=job.full_text, markdown=job.markdown)


def _multipart_available() -> bool:
    for mod in ("multipart", "python_multipart"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


if _multipart_available():
    import tempfile

    from fastapi import File, UploadFile
    from PIL import Image

    @app.post("/ocr", response_model=DocResponse)
    async def ocr(file: UploadFile = File(...)) -> DocResponse:
        data = await file.read()
        name = file.filename or "document"
        if name.lower().endswith(".pdf"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(data)
            tmp.close()
            try:
                job = get_agent().process(path=tmp.name, filename=name, save=False)
            finally:
                try:
                    Path(tmp.name).unlink()
                except OSError:
                    pass
        else:
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"invalid image: {exc}")
            job = get_agent().process(image=img, filename=name, save=False)
        return _to_response(job)
else:
    logger.warning("python-multipart not installed; POST /ocr is disabled (use /correct)")


__all__ = ["app"]
