"""Data preparation entrypoint.

Builds + caches the synthetic post-OCR corpus (the primary data is generated, not
downloaded), writes sample documents, and best-effort prefetches the real PleIAs
corpus. Network steps degrade gracefully so ``data`` works fully offline.
"""

from __future__ import annotations

from typing import Dict

from ..config import AppConfig, data_dir
from ..logging_utils import get_logger
from .dataset import build_corpus, corpus_dir, load_real_split
from .samples import write_samples

logger = get_logger(__name__)


def prepare_corpus(cfg: AppConfig) -> Dict:
    splits = build_corpus(cfg, save=True)
    samples = write_samples(data_dir() / "samples")
    return {"task": "postocr_corpus", "dir": str(corpus_dir()),
            "counts": {k: len(v) for k, v in splits.items()},
            "samples": [p for p, _ in samples]}


def prefetch_real(cfg: AppConfig) -> Dict:
    rows = load_real_split(cfg.data, "train", limit=100)
    return {"task": "real_data", "dataset": cfg.data.real_dataset,
            "config": cfg.data.real_config, "sampled_rows": len(rows)}


def download_all(cfg: AppConfig) -> Dict:
    return {"corpus": prepare_corpus(cfg), "real": prefetch_real(cfg)}


def download_task(task: str, cfg: AppConfig) -> Dict:
    if task == "corpus":
        return prepare_corpus(cfg)
    if task == "real":
        return prefetch_real(cfg)
    raise ValueError(f"Unknown data task: {task}")


__all__ = ["prepare_corpus", "prefetch_real", "download_all", "download_task"]
