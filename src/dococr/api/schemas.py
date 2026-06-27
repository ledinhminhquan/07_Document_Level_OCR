"""Pydantic request/response schemas for the Document-Level OCR API."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class BlockView(BaseModel):
    page: int
    reading_index: int
    kind: str
    text: str
    conf: float
    flagged: bool


class DocResponse(BaseModel):
    filename: str
    status: str
    n_pages: int
    n_blocks: int
    n_flagged: int
    mean_confidence: float
    full_text: str
    markdown: str
    blocks: List[BlockView]
    decisions: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    model_versions: Dict[str, str]


class CorrectRequest(BaseModel):
    text: str


class CorrectResponse(BaseModel):
    corrected: str
    n_blocks: int
    full_text: str
    markdown: str


class HealthResponse(BaseModel):
    status: str
    corrector: str
    ocr_engine: str
    version: str


__all__ = ["BlockView", "DocResponse", "CorrectRequest", "CorrectResponse", "HealthResponse"]
