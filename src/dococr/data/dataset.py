"""Build, cache and load the post-OCR correction splits.

Primary source = the synthetic OCR-noise generator (``ocr_noise``). Produces
leakage-free ``train / val / test`` splits, cached as JSONL under
``data_dir/postocr_corpus``, with the real PleIAs corpus (CC0) mixed into train
and supplying a real eval slice. ``datasets`` is imported lazily.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, DataConfig, data_dir
from ..logging_utils import get_logger
from .ocr_noise import OCRNoiseGenerator

logger = get_logger(__name__)

_SPLITS = ("train", "val", "test", "real")


def corpus_dir() -> Path:
    return data_dir() / "postocr_corpus"


def corpus_signature(dc: DataConfig) -> Dict[str, Any]:
    return {"source": "synthetic+real" if dc.use_real else "synthetic",
            "real_dataset": dc.real_dataset if dc.use_real else None,
            "train": dc.synthetic_train_size, "val": dc.synthetic_val_size,
            "test": dc.synthetic_test_size, "char_error_rate": dc.char_error_rate, "seed": dc.seed}


def _write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_real_split(dc: DataConfig, split: str = "train", limit: Optional[int] = None) -> List[Dict]:
    """Load the real PleIAs (text=noisy, corrected_text=clean) pairs. [] on failure."""
    if not dc.use_real:
        return []
    try:
        from datasets import load_dataset  # lazy
        ds = load_dataset(dc.real_dataset, dc.real_config, split=split)
        rows: List[Dict] = []
        for i, ex in enumerate(ds):
            if limit and i >= limit:
                break
            noisy = str(ex.get(dc.real_text_col, "")).strip()
            clean = str(ex.get(dc.real_target_col, "")).strip()
            if noisy and clean and len(clean) <= 2000:
                rows.append({"noisy": noisy[:1500], "clean": clean[:1500], "src": "real"})
        logger.info("Loaded %d real rows from %s[%s]", len(rows), dc.real_dataset, dc.real_config)
        return rows
    except Exception as exc:
        logger.warning("Could not load real dataset %s (%s); synthetic only.", dc.real_dataset, exc)
        return []


def build_corpus(cfg: AppConfig, save: bool = True) -> Dict[str, List[Dict]]:
    dc = cfg.data
    gen = OCRNoiseGenerator(dc)
    total = dc.synthetic_train_size + dc.synthetic_val_size + dc.synthetic_test_size
    pool = [{**e, "src": "synth"} for e in gen.generate(total, seed=0)]
    rng = random.Random(dc.seed)
    rng.shuffle(pool)

    real = load_real_split(dc, "train", limit=20000)
    n_tr, n_va = dc.synthetic_train_size, dc.synthetic_val_size
    train = pool[:n_tr] + real
    rng.shuffle(train)
    val = pool[n_tr:n_tr + n_va]
    test = pool[n_tr + n_va:]

    train_clean = {r["clean"] for r in train}
    val = [r for r in val if r["clean"] not in train_clean]
    test = [r for r in test if r["clean"] not in train_clean]
    real_eval = [r for r in load_real_split(dc, "train", limit=21000)[20000:] if r["clean"] not in train_clean][:1500]

    splits = {"train": train, "val": val, "test": test, "real": real_eval}
    logger.info("Corpus built: %s", {k: len(v) for k, v in splits.items()})
    if save:
        for name, rows in splits.items():
            _write_jsonl(rows, corpus_dir() / f"{name}.jsonl")
        (corpus_dir() / "signature.json").write_text(json.dumps(corpus_signature(dc), indent=2), encoding="utf-8")
    return splits


def load_or_build(cfg: AppConfig, force: bool = False) -> Dict[str, List[Dict]]:
    cdir = corpus_dir()
    if not force and all((cdir / f"{s}.jsonl").exists() for s in ("train", "val", "test")):
        try:
            sig = json.loads((cdir / "signature.json").read_text(encoding="utf-8"))
            if sig.get("seed") == cfg.data.seed and sig.get("train") == cfg.data.synthetic_train_size:
                out = {s: _read_jsonl(cdir / f"{s}.jsonl") for s in ("train", "val", "test")}
                out["real"] = _read_jsonl(cdir / "real.jsonl") if (cdir / "real.jsonl").exists() else []
                return out
        except Exception:
            pass
    return build_corpus(cfg)


def to_hf_datasets(splits: Dict[str, List[Dict]]):
    from datasets import Dataset, DatasetDict  # lazy
    out = {}
    for name, rows in splits.items():
        if rows:
            out[name] = Dataset.from_list([{"noisy": r["noisy"], "clean": r["clean"]} for r in rows])
    return DatasetDict(out)


__all__ = ["build_corpus", "load_or_build", "to_hf_datasets", "load_real_split",
           "corpus_signature", "corpus_dir"]
