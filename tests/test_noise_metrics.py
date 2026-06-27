"""OCR-noise generator + metrics (CER/WER reduction)."""

from __future__ import annotations

import random

from dococr.config import AppConfig
from dococr.data.ocr_noise import OCRNoiseGenerator, corrupt
from dococr.training import metrics as M


def test_corrupt_changes_text():
    clean = "The company reported steady growth in the third quarter."
    noisy = corrupt(clean, random.Random(0), 0.12)
    assert noisy != clean
    assert M.char_cer(noisy, clean) > 0


def test_generator_deterministic(cfg):
    a = OCRNoiseGenerator(cfg.data).example(5)
    b = OCRNoiseGenerator(cfg.data).example(5)
    assert a == b
    assert a["noisy"] and a["clean"]


def test_identity_baseline_has_measurable_error(cfg):
    ex = OCRNoiseGenerator(cfg.data).generate(400, seed=0)
    noisy = [e["noisy"] for e in ex]
    clean = [e["clean"] for e in ex]
    assert 0.02 < M.corpus_cer(noisy, clean) < 0.3      # a real gap to close


def test_reduction_metrics():
    noisy = ["teh cornpany", "hte third quarrter"]
    preds = ["the company", "the third quarter"]         # a good corrector
    refs = ["the company", "the third quarter"]
    r = M.reduction_metrics(noisy, preds, refs)
    assert r["cer_after"] < r["cer_before"]
    assert r["cer_reduction_rel"] > 0
    assert r["pct_improved"] > 0
    # a corrector that does nothing => zero reduction
    r2 = M.reduction_metrics(noisy, noisy, refs)
    assert r2["cer_reduction_rel"] == 0.0
