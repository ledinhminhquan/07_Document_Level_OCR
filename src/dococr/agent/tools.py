"""Agent tools — each operates on the JobState and returns it.

Tools assume neither a GPU nor a trained model: the OCR engine can be the stub
and the corrector can be the identity baseline, so the whole pipeline runs offline
for tests/CI. The orchestrator wraps each call with timing/trace.
"""

from __future__ import annotations

from typing import List, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..ocr import layout as L
from ..ocr.preprocess import preprocess_image
from . import policy
from .state import BlockResult, Decision, JobState

logger = get_logger(__name__)


def tool_ingest(job: JobState, cfg: AppConfig, *, path: Optional[str] = None,
                image=None, text: Optional[str] = None) -> JobState:
    if text is not None:
        import re
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] or [text.strip()]
        dblocks = [L.Block(text=p, bbox=(0, 100 + i * 100, 600, 60)) for i, p in enumerate(paras)]
        job._pages = [L.PageInput(index=0, born_digital=True, digital_blocks=dblocks,
                                  width=600, height=len(paras) * 100 + 200)]
    elif image is not None:
        img = image if hasattr(image, "size") else __import__("PIL.Image", fromlist=["Image"]).fromarray(image)
        job._pages = [L.PageInput(index=0, born_digital=False, image=img.convert("RGB"),
                                  width=img.width, height=img.height)]
    else:
        job._pages = L.ingest(path, cfg)
    job.n_pages = len(job._pages)
    job.n_born_digital = sum(p.born_digital for p in job._pages)
    branch = "born_digital" if job.n_born_digital == job.n_pages and job.n_pages else (
        "scanned" if job.n_born_digital == 0 else "mixed")
    job.add_decision(Decision("D2", "born_digital_routing", branch,
                              detail=f"{job.n_born_digital}/{job.n_pages} pages born-digital"))
    return job


def tool_layout(job: JobState, cfg: AppConfig, *, ocr_engine) -> JobState:
    min_quality = 1.0
    scanned = 0
    blocks: List[BlockResult] = []
    for page in job._pages:
        if page.born_digital:
            pblocks = list(page.digital_blocks)
            L.reading_order(pblocks, page.width, cfg.layout)
            L.classify_blocks(pblocks, page.width, page.height, cfg.layout)
            for b in pblocks:
                blocks.append(BlockResult(page=page.index, reading_index=b.reading_index, kind=b.kind,
                                          bbox=b.bbox, raw_text=b.text, text=b.text, conf=1.0,
                                          source="born_digital"))
        else:
            scanned += 1
            proc, qmetrics = preprocess_image(page.image, cfg.preprocess)
            min_quality = min(min_quality, qmetrics.get("quality", 1.0))
            ocr = ocr_engine.recognize(proc)
            pblocks = L.words_to_blocks(ocr, cfg.layout)
            pblocks = L.reading_order(pblocks, ocr.width, cfg.layout)
            L.classify_blocks(pblocks, ocr.width, ocr.height, cfg.layout)
            for b in pblocks:
                blocks.append(BlockResult(page=page.index, reading_index=b.reading_index, kind=b.kind,
                                          bbox=b.bbox, raw_text=b.text, text=b.text, conf=b.conf, source="ocr"))
    blocks.sort(key=lambda b: (b.page, b.reading_index))
    job.blocks = blocks
    job.model_versions["ocr_engine"] = getattr(ocr_engine, "name", "?")

    # D1 — page-quality routing (over scanned pages)
    route = policy.quality_route({"quality": min_quality}, cfg.agent, 0) if scanned else "ok"
    job.metrics["min_page_quality"] = round(min_quality, 4) if scanned else None
    job.add_decision(Decision("D1", "page_quality_routing", route,
                              score=round(min_quality, 4) if scanned else None,
                              detail=f"{scanned} scanned page(s); min quality={min_quality:.3f}" if scanned else "born-digital only"))
    degraded = route == "degraded"

    # D3 — OCR-confidence gate (flag low-confidence OCR blocks)
    n_flag = 0
    for b in job.blocks:
        if b.source == "ocr":
            gate = policy.ocr_gate(b.conf, cfg.agent, degraded)
            if not gate["accept"]:
                b.flagged = True
                b.flag_reason = f"low OCR confidence {b.conf:.2f} < {gate['bar']}"
                n_flag += 1
    job.add_decision(Decision("D3", "ocr_confidence_gate", "all_accept" if n_flag == 0 else "some_flagged",
                              score=round(sum(b.conf for b in job.blocks) / len(job.blocks), 4) if job.blocks else 0.0,
                              detail=f"flagged={n_flag}/{len(job.blocks)}"))
    return job


def tool_correct(job: JobState, cfg: AppConfig, *, corrector, brain=None) -> JobState:
    blocks = [b for b in job.blocks if b.raw_text.strip()]
    if not blocks:
        job.add_decision(Decision("D4", "correction_acceptance", "none", detail="no text blocks"))
        return job
    texts = [b.raw_text for b in blocks]
    is_neural = hasattr(corrector, "correct_batch_with_conf")
    if is_neural:
        cands, confs = corrector.correct_batch_with_conf(texts)
        job.model_versions["corrector"] = getattr(corrector, "version", "neural")
    else:
        cands, confs = corrector.correct_batch(texts), [1.0] * len(texts)
        job.model_versions["corrector"] = getattr(corrector, "version", "identity")

    n_acc = n_llm = 0
    for b, cand, conf in zip(blocks, cands, confs):
        dec = policy.accept_correction(b.raw_text, cand, conf, cfg.agent)
        if dec["accept"]:
            b.text, b.corrected, b.correction_source = cand, True, ("neural" if is_neural else "rule")
            n_acc += 1
        else:
            b.text = b.raw_text
        if b.flagged and brain is not None and brain.available():
            adv = brain.correct_region(b.raw_text)
            if adv is not None:
                b.text, b.corrected, b.correction_source = adv["text"], True, "llm"
                n_llm += 1
    job.add_decision(Decision("D4", "correction_acceptance",
                              "llm" if n_llm else ("corrected" if n_acc else "kept_raw"),
                              detail=f"accepted={n_acc}/{len(blocks)}, llm={n_llm}", llm_used=n_llm > 0))
    job.metrics["corrections"] = {"accepted": n_acc, "llm": n_llm, "n_blocks": len(blocks)}
    return job


def tool_assemble(job: JobState) -> JobState:
    lblocks = [L.Block(text=b.text, bbox=b.bbox, kind=b.kind, conf=b.conf, reading_index=b.reading_index)
               for b in job.blocks]
    out = L.assemble(lblocks)
    job.full_text = out["text"]
    job.markdown = out["markdown"]
    job.metrics["n_chars"] = len(job.full_text)
    return job


__all__ = ["tool_ingest", "tool_layout", "tool_correct", "tool_assemble"]
