"""OCR engines (the document front-end): page image -> words + boxes + confidence.

``TesseractEngine`` (default) returns per-word boxes, confidence and the
block/paragraph/line hierarchy via ``image_to_data``. ``StubEngine`` is the
no-binary fallback (empty result) so the agent + tests run offline. docTR /
EasyOCR are optional upgrades. All heavy imports are lazy; ``load_ocr_engine``
never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import OcrConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Word:
    text: str
    conf: float                 # 0..1
    bbox: tuple                 # (x, y, w, h)
    block: int = 0
    par: int = 0
    line: int = 0


@dataclass
class OcrResult:
    words: List[Word] = field(default_factory=list)
    engine: str = "stub"
    width: int = 0
    height: int = 0

    @property
    def mean_conf(self) -> float:
        cs = [w.conf for w in self.words if w.text.strip()]
        return round(sum(cs) / len(cs), 4) if cs else 0.0

    def lines_text(self) -> List[str]:
        groups: dict = {}
        for w in self.words:
            if w.text.strip():
                groups.setdefault((w.block, w.par, w.line), []).append(w.text)
        return [" ".join(v) for _, v in sorted(groups.items())]

    @property
    def full_text(self) -> str:
        return "\n".join(self.lines_text())


class StubEngine:
    name = "stub"

    def __init__(self, cfg: Optional[OcrConfig] = None):
        self.cfg = cfg

    def recognize(self, image) -> OcrResult:
        w, h = (image.size if hasattr(image, "size") else (0, 0))
        return OcrResult(words=[], engine="stub", width=w, height=h)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, cfg: OcrConfig):
        import pytesseract  # lazy; raises if unavailable
        self._pt = pytesseract
        self.cfg = cfg
        self._pt.get_tesseract_version()

    def recognize(self, image) -> OcrResult:
        from PIL import Image
        img = image if hasattr(image, "size") else Image.fromarray(image)
        cfg_str = f"--psm {self.cfg.psm}"
        data = self._pt.image_to_data(img, lang=self.cfg.lang, config=cfg_str,
                                      output_type=self._pt.Output.DICT)
        words: List[Word] = []
        for i in range(len(data["text"])):
            txt = data["text"][i]
            conf = float(data["conf"][i])
            if not txt.strip() or conf < 0:
                continue
            words.append(Word(text=txt, conf=conf / 100.0,
                              bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                              block=data["block_num"][i], par=data["par_num"][i], line=data["line_num"][i]))
        return OcrResult(words=words, engine="tesseract", width=img.width, height=img.height)


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, cfg: OcrConfig):
        import easyocr  # lazy
        import numpy as np
        self._np = np
        self._reader = easyocr.Reader([cfg.lang[:2] if cfg.lang != "eng" else "en"], gpu=False)
        self.cfg = cfg

    def recognize(self, image) -> OcrResult:
        import numpy as np
        arr = np.asarray(image)
        res = self._reader.readtext(arr)
        words: List[Word] = []
        for li, (box, text, conf) in enumerate(res):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            words.append(Word(text=text, conf=float(conf),
                              bbox=(x, y, int(max(xs) - x), int(max(ys) - y)), block=0, par=0, line=li))
        h, w = arr.shape[:2]
        return OcrResult(words=words, engine="easyocr", width=w, height=h)


_ENGINES = {"tesseract": TesseractEngine, "easyocr": EasyOcrEngine, "stub": StubEngine}


def load_ocr_engine(cfg: OcrConfig, engine: Optional[str] = None):
    requested = engine or cfg.engine
    if requested == "stub":
        return StubEngine(cfg)
    order: List[str] = []
    if requested and requested != "auto":
        order.append(requested)
    for e in ("tesseract", "easyocr"):
        if e not in order:
            order.append(e)
    for name in order:
        cls = _ENGINES.get(name)
        if cls is None or name == "stub":
            continue
        try:
            inst = cls(cfg)
            logger.info("OCR engine: %s", inst.name)
            return inst
        except Exception as exc:
            logger.info("OCR engine %s unavailable (%s); trying next", name, exc)
    logger.info("No OCR engine available; using stub (no recognition).")
    return StubEngine(cfg)


__all__ = ["Word", "OcrResult", "TesseractEngine", "EasyOcrEngine", "StubEngine", "load_ocr_engine"]
