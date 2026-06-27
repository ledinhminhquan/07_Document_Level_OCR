"""Shared state types for the document-OCR agent (FSM context + audit records)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class JobStatus(str, Enum):
    PENDING = "pending"
    INGESTED = "ingested"
    OCR_DONE = "ocr_done"
    CORRECTED = "corrected"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class BlockResult:
    page: int
    reading_index: int
    kind: str = "paragraph"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    raw_text: str = ""             # OCR / extracted text before correction
    text: str = ""                 # final (post-corrected) text
    conf: float = 1.0
    source: str = "ocr"            # ocr | born_digital
    corrected: bool = False
    correction_source: str = ""    # neural | rule | llm | ""
    flagged: bool = False
    flag_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"page": self.page, "reading_index": self.reading_index, "kind": self.kind,
                "bbox": list(self.bbox), "raw_text": self.raw_text, "text": self.text,
                "conf": round(self.conf, 4), "source": self.source, "corrected": self.corrected,
                "correction_source": self.correction_source, "flagged": self.flagged,
                "flag_reason": self.flag_reason}


@dataclass
class JobState:
    filename: str = "document"
    status: JobStatus = JobStatus.PENDING
    n_pages: int = 0
    n_born_digital: int = 0
    blocks: List[BlockResult] = field(default_factory=list)
    full_text: str = ""
    markdown: str = ""
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    review_reasons: List[str] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)
    _pages: List[Any] = field(default_factory=list, repr=False)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def to_dict(self) -> Dict[str, Any]:
        n_flagged = sum(b.flagged for b in self.blocks)
        return {
            "filename": self.filename, "status": self.status.value,
            "n_pages": self.n_pages, "n_born_digital": self.n_born_digital,
            "n_blocks": len(self.blocks), "n_flagged": n_flagged,
            "mean_confidence": round(sum(b.conf for b in self.blocks) / len(self.blocks), 4) if self.blocks else 0.0,
            "full_text": self.full_text, "markdown": self.markdown,
            "blocks": [b.to_dict() for b in self.blocks],
            "decisions": [d.to_dict() for d in self.decisions],
            "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics, "outputs": self.outputs,
            "review_reasons": self.review_reasons, "model_versions": self.model_versions,
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "BlockResult", "JobState"]
