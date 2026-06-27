"""Typed configuration + YAML loader for the Document-Level OCR system.

Single source of truth for the post-OCR correction corpus, the trainable
corrector, the OCR engine, layout analysis, agent thresholds and serving. Paths
come from environment variables so nothing is hard-coded (required by the assignment).

Environment overrides
---------------------
* ``DOCOCR_ARTIFACTS_DIR`` – base for data/models/runs (Drive on Colab)
* ``DOCOCR_DATA_DIR``      – datasets / generated corpus cache
* ``DOCOCR_MODEL_DIR``     – trained models
* ``DOCOCR_RUN_DIR``       – eval/benchmark/analysis JSON
* ``DOCOCR_OUTPUT_DIR``    – extracted documents (text/markdown/json)
* ``HF_HOME``              – HuggingFace cache
* ``DOCOCR_LLM_API_KEY``   – optional key for the LLM correction-review brain
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("DOCOCR_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("DOCOCR_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("DOCOCR_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def run_dir() -> Path:
    return Path(_env("DOCOCR_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


def output_dir() -> Path:
    return Path(_env("DOCOCR_OUTPUT_DIR", str(artifacts_dir() / "outputs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Post-OCR correction corpus (see docs/data_description.md).

    PRIMARY = a reproducible SYNTHETIC OCR-noise generator (``data/ocr_noise.py``)
    that corrupts clean English text with a realistic OCR confusion model. The
    real PleIAs corpus is mixed in and supplies a real eval slice.
    """
    # real (text=noisy OCR, corrected_text=gold) dataset; CC0
    real_dataset: str = "PleIAs/Post-OCR-Correction"
    real_config: str = "english"
    real_text_col: str = "text"
    real_target_col: str = "corrected_text"
    use_real: bool = True
    # optional ICDAR-2019 benchmark (manual corpus; loader degrades gracefully)
    icdar_dataset: str = "FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction"
    # synthetic generator
    synthetic_train_size: int = 60000
    synthetic_val_size: int = 4000
    synthetic_test_size: int = 4000
    char_error_rate: float = 0.08          # mean fraction of characters corrupted
    max_chars: int = 360                   # window length of a training example
    seed: int = 42


@dataclass
class ModelConfig:
    """Trainable post-OCR corrector (ByT5/T5 seq2seq)."""
    base_model: str = "google/byt5-small"            # char-level => robust to OCR noise
    base_model_fallback: str = "google-t5/t5-small"  # small-GPU fallback (T4)
    task_prefix: str = "correct: "
    max_source_length: int = 384
    max_target_length: int = 384
    num_train_epochs: int = 3
    learning_rate: float = 5.0e-4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 8   # effective batch 256 on H100
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    label_smoothing_factor: float = 0.1
    max_grad_norm: float = 1.0
    dropout_rate: float = 0.1
    early_stopping_patience: int = 4
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    generation_num_beams: int = 1
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = False
    group_by_length: bool = True
    seed: int = 42
    output_subdir: str = "postocr_corrector"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir


@dataclass
class OcrConfig:
    """OCR engine for the document front-end (reads page images -> text + boxes)."""
    engine: str = "auto"                   # "auto"|"tesseract"|"doctr"|"surya"|"stub"
    lang: str = "eng"
    dpi: int = 200                         # rasterise scanned PDFs at 200-300 DPI
    min_word_conf: float = 0.0             # keep all words; conf used for gating
    psm: int = 3                           # tesseract page segmentation mode (auto)


@dataclass
class LayoutConfig:
    """Layout analysis, reading order & structure."""
    detect_layout: bool = True
    reading_order: str = "xycut"           # "xycut" | "topdown"
    min_region_area: int = 200
    born_digital_min_chars: int = 40       # PDFs with >= this many extractable chars => born-digital
    classify_blocks: bool = True


@dataclass
class PreprocessConfig:
    deskew: bool = True
    denoise: bool = True
    binarize: str = "adaptive"             # "adaptive" | "otsu" | "none"
    max_skew_deg: float = 15.0


@dataclass
class AgentConfig:
    """Agent thresholds (decision points) + optional LLM correction-review brain."""
    # D1 — page-quality routing
    quality_min: float = 0.35
    # D2 — born-digital vs scanned (uses LayoutConfig.born_digital_min_chars)
    # D3 — OCR-confidence gate (per region)
    ocr_confidence_min: float = 0.55
    max_ocr_attempts: int = 2
    # D4 — correction acceptance: accept the corrected text only within an edit budget
    correct_max_edit_ratio: float = 0.35   # reject a "correction" that rewrites too much
    correct_min_conf: float = 0.5
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "DOCOCR_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Document-Level OCR API"
    api_version: str = "1.0.0"
    log_jobs: bool = True
    job_log_subdir: str = "job_logs"
    max_file_mb: int = 25
    max_pages: int = 30

    @property
    def job_log_path(self) -> Path:
        return run_dir() / self.job_log_subdir / "jobs.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Document-Level OCR System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "ocr": OcrConfig, "layout": LayoutConfig,
             "preprocess": PreprocessConfig, "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(),
            "runs": run_dir(), "outputs": output_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "ModelConfig", "OcrConfig", "LayoutConfig", "PreprocessConfig",
           "AgentConfig", "ServingConfig", "AppConfig", "load_config", "save_config", "ensure_dirs",
           "artifacts_dir", "data_dir", "model_dir", "run_dir", "output_dir"]
