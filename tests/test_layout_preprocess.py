"""Image preprocessing, OCR-engine fallback, layout and reading order."""

from __future__ import annotations

import numpy as np

from dococr.config import AppConfig, LayoutConfig, PreprocessConfig
from dococr.models.ocr_engine import StubEngine, load_ocr_engine
from dococr.ocr import layout as L
from dococr.ocr.preprocess import binarize, preprocess_image, quality_metrics, to_gray


def test_preprocess_quality(sample_page):
    from PIL import Image
    img = Image.open(sample_page)
    proc, m = preprocess_image(img, PreprocessConfig())
    assert proc.mode == "RGB"
    assert 0.0 <= m["quality"] <= 1.0
    b = binarize(to_gray(img), "otsu")
    assert set(np.unique(b)).issubset({0, 255})


def test_ocr_engine_falls_back_to_stub():
    eng = load_ocr_engine(AppConfig().ocr, engine="stub")
    assert isinstance(eng, StubEngine)
    from PIL import Image
    res = eng.recognize(Image.new("RGB", (50, 20)))
    assert res.words == []


def test_layout_reading_order_and_classify():
    cfg = LayoutConfig()
    blocks = [
        L.Block(text="CONCLUSION", bbox=(10, 300, 200, 40)),
        L.Block(text="Introduction", bbox=(10, 120, 200, 40)),
        L.Block(text="This is a paragraph of body text that continues for a while.", bbox=(10, 170, 400, 60)),
    ]
    ordered = L.reading_order(blocks, 600, cfg)
    assert [b.reading_index for b in ordered] == [0, 1, 2]
    assert ordered[0].bbox[1] <= ordered[1].bbox[1]    # top-to-bottom
    L.classify_blocks(ordered, 600, 800, cfg)
    kinds = {b.text: b.kind for b in ordered}
    assert kinds["Introduction"] == "heading"
    assert kinds["This is a paragraph of body text that continues for a while."] == "paragraph"


def test_assemble_markdown():
    blocks = [L.Block(text="Title", bbox=(0, 100, 200, 40), kind="heading"),
              L.Block(text="Body text here.", bbox=(0, 160, 400, 40), kind="paragraph")]
    out = L.assemble(blocks)
    assert "## Title" in out["markdown"]
    assert "Body text here." in out["text"]
    assert len(out["blocks"]) == 2
