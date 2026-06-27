"""Layout analysis, reading order and document structure.

Ingests page images / PDFs (born-digital PDFs are read directly via PyMuPDF — no
OCR needed; scanned pages are rasterized for OCR), groups OCR words into typed
blocks, orders them (XY-cut, multi-column aware), and assembles a structured
document (plain text + Markdown + JSON blocks). Heavy imports are lazy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, LayoutConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Block:
    text: str
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    kind: str = "paragraph"              # heading | paragraph | list | header_footer | blank
    conf: float = 1.0
    reading_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "bbox": list(self.bbox), "kind": self.kind,
                "conf": round(self.conf, 4), "reading_index": self.reading_index}


@dataclass
class PageInput:
    index: int
    born_digital: bool
    image: Any = None                    # PIL image for scanned pages
    digital_blocks: List[Block] = field(default_factory=list)
    width: int = 0
    height: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion (image / PDF; born-digital detection)
# ─────────────────────────────────────────────────────────────────────────────
def ingest(path: str, cfg: AppConfig) -> List[PageInput]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return _ingest_pdf(p, cfg)
    from PIL import Image
    img = Image.open(p).convert("RGB")
    return [PageInput(index=0, born_digital=False, image=img, width=img.width, height=img.height)]


def _ingest_pdf(path: Path, cfg: AppConfig) -> List[PageInput]:
    import fitz  # PyMuPDF, lazy
    doc = fitz.open(str(path))
    pages: List[PageInput] = []
    for i, page in enumerate(doc):
        if i >= cfg.serving.max_pages:
            break
        text = page.get_text().strip()
        if len(text) >= cfg.layout.born_digital_min_chars:
            blocks = []
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
                if btext.strip():
                    blocks.append(Block(text=btext.strip(),
                                        bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0))))
            pages.append(PageInput(index=i, born_digital=True, digital_blocks=blocks,
                                   width=int(page.rect.width), height=int(page.rect.height)))
        else:
            pix = page.get_pixmap(dpi=cfg.ocr.dpi)
            from PIL import Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(PageInput(index=i, born_digital=False, image=img, width=pix.width, height=pix.height))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Words -> blocks -> reading order -> classification
# ─────────────────────────────────────────────────────────────────────────────
def words_to_blocks(ocr_result, cfg: LayoutConfig) -> List[Block]:
    groups: Dict[int, list] = {}
    for w in ocr_result.words:
        if w.text.strip():
            groups.setdefault(w.block, []).append(w)
    blocks: List[Block] = []
    for _, ws in groups.items():
        xs = [w.bbox[0] for w in ws]
        ys = [w.bbox[1] for w in ws]
        x2 = [w.bbox[0] + w.bbox[2] for w in ws]
        y2 = [w.bbox[1] + w.bbox[3] for w in ws]
        x, y = min(xs), min(ys)
        bbox = (x, y, max(x2) - x, max(y2) - y)
        if bbox[2] * bbox[3] < cfg.min_region_area:
            continue
        line_groups: Dict[int, list] = {}
        for w in ws:
            line_groups.setdefault(w.line, []).append(w)
        text = "\n".join(" ".join(t.text for t in line_groups[k]) for k in sorted(line_groups))
        conf = sum(w.conf for w in ws) / len(ws)
        blocks.append(Block(text=text, bbox=bbox, conf=conf))
    return blocks


def reading_order(blocks: List[Block], page_w: int, cfg: LayoutConfig) -> List[Block]:
    if not blocks:
        return blocks
    if cfg.reading_order == "xycut" and page_w > 0:
        # 2-column detection: a vertical gap near the page centre with blocks on both sides
        mid = page_w / 2
        centers = [(b.bbox[0] + b.bbox[2] / 2) for b in blocks]
        left = [c for c in centers if c < mid * 0.85]
        right = [c for c in centers if c > mid * 1.15]
        if left and right and len(blocks) >= 4:
            def key(b):
                cx = b.bbox[0] + b.bbox[2] / 2
                return (0 if cx < mid else 1, b.bbox[1], b.bbox[0])
            ordered = sorted(blocks, key=key)
        else:
            ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    else:
        ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    for idx, b in enumerate(ordered):
        b.reading_index = idx
    return ordered


_LIST_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


def classify_block(b: Block, page_w: int, page_h: int) -> str:
    t = b.text.strip()
    if not t:
        return "blank"
    first = t.split("\n", 1)[0]
    y = b.bbox[1]
    if page_h and (y < page_h * 0.06 or y > page_h * 0.94) and len(t) <= 80:
        return "header_footer"
    if _LIST_RE.match(first):
        return "list"
    if len(t) <= 70 and "\n" not in t and not first.endswith((".", ",", ";")) and (first.isupper() or first.istitle()):
        return "heading"
    return "paragraph"


def classify_blocks(blocks: List[Block], page_w: int, page_h: int, cfg: LayoutConfig) -> List[Block]:
    if cfg.classify_blocks:
        for b in blocks:
            b.kind = classify_block(b, page_w, page_h)
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Structured output
# ─────────────────────────────────────────────────────────────────────────────
def to_markdown(blocks: List[Block]) -> str:
    out: List[str] = []
    for b in blocks:
        t = b.text.strip()
        if not t or b.kind in ("blank",):
            continue
        if b.kind == "heading":
            out.append(f"## {t}")
        elif b.kind == "list":
            out.append("\n".join(f"- {re.sub(_LIST_RE, '', ln).strip()}" for ln in t.split("\n") if ln.strip()))
        elif b.kind == "header_footer":
            out.append(f"<sub>{t}</sub>")
        else:
            out.append(t.replace("\n", " "))
    return "\n\n".join(out)


def assemble(blocks: List[Block]) -> Dict[str, Any]:
    text = "\n\n".join(b.text.strip() for b in blocks if b.text.strip() and b.kind != "blank")
    return {"text": text, "markdown": to_markdown(blocks),
            "blocks": [b.to_dict() for b in blocks if b.kind != "blank"]}


__all__ = ["Block", "PageInput", "ingest", "words_to_blocks", "reading_order",
           "classify_block", "classify_blocks", "to_markdown", "assemble"]
