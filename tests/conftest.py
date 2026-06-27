"""Shared pytest fixtures. Tests are CPU-only and never download models/data:
they use the synthetic generator + the stub OCR engine + the identity corrector.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLE_DATA = REPO / "sample_data"


@pytest.fixture(autouse=True, scope="session")
def _artifacts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("dococr_artifacts")
    os.environ["DOCOCR_ARTIFACTS_DIR"] = str(d)
    os.environ.setdefault("DOCOCR_LOG_LEVEL", "WARNING")
    yield


@pytest.fixture
def cfg():
    from dococr.config import AppConfig
    c = AppConfig()
    c.data.use_real = False                 # offline: synthetic only
    c.data.synthetic_train_size = 800
    c.data.synthetic_val_size = 200
    c.data.synthetic_test_size = 200
    return c


@pytest.fixture
def sample_page():
    return str(SAMPLE_DATA / "sample_page.png")


@pytest.fixture
def sample_doc():
    return (SAMPLE_DATA / "sample_doc.txt").read_text(encoding="utf-8")
