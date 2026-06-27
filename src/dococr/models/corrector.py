"""Neural post-OCR corrector (the TRAINED core).

Wraps a fine-tuned ByT5/T5 seq2seq checkpoint behind a ``correct`` /
``correct_batch`` interface, with a per-example confidence (length-normalized
sequence probability) used by the agent's D4 acceptance gate. Heavy imports are
lazy; ``load_corrector`` degrades to the identity baseline when transformers or a
checkpoint is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import ModelConfig
from ..logging_utils import get_logger
from .baseline import IdentityCorrector
from .text_utils import normalize_ws

logger = get_logger(__name__)


class NeuralCorrector:
    name = "neural"

    def __init__(self, model, tokenizer, *, prefix: str, device: str,
                 max_source_length: int, max_target_length: int, version: str, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.device = device
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.version = version
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_path: str, mc: ModelConfig, device: Optional[str] = None) -> "NeuralCorrector":
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(dev).eval()
        logger.info("Loaded neural corrector from %s (device=%s)", model_path, dev)
        return cls(model, tok, prefix=mc.task_prefix, device=dev,
                   max_source_length=mc.max_source_length, max_target_length=mc.max_target_length,
                   version=_read_version(model_path), model_id=str(model_path))

    def _generate(self, texts: List[str], with_conf: bool) -> Tuple[List[str], List[float]]:
        import torch
        enc = self.tokenizer([self.prefix + t for t in texts], return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_source_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        kw = dict(max_new_tokens=self.max_target_length, num_beams=1, do_sample=False)
        if with_conf:
            kw.update(output_scores=True, return_dict_in_generate=True)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        seqs = out.sequences if with_conf else out
        confs = self._confidence(out, seqs) if with_conf else [1.0] * len(texts)
        decoded = [normalize_ws(s) for s in self.tokenizer.batch_decode(seqs, skip_special_tokens=True)]
        return decoded, confs

    def _confidence(self, out, seqs) -> List[float]:
        try:
            trans = self.model.compute_transition_scores(seqs, out.scores, normalize_logits=True)
            confs = []
            for row in trans:
                vals = [v.item() for v in row if math.isfinite(v.item())]
                confs.append(float(math.exp(sum(vals) / len(vals))) if vals else 0.0)
            return confs
        except Exception:
            return [1.0] * len(seqs)

    def correct(self, text: str) -> str:
        return self._generate([text], with_conf=False)[0][0]

    def correct_with_conf(self, text: str) -> Tuple[str, float]:
        o, c = self._generate([text], with_conf=True)
        return o[0], c[0]

    def correct_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        res: List[str] = []
        for i in range(0, len(texts), batch_size):
            res.extend(self._generate(texts[i:i + batch_size], with_conf=False)[0])
        return res

    def correct_batch_with_conf(self, texts: List[str], batch_size: int = 32) -> Tuple[List[str], List[float]]:
        outs, confs = [], []
        for i in range(0, len(texts), batch_size):
            o, c = self._generate(texts[i:i + batch_size], with_conf=True)
            outs.extend(o)
            confs.extend(c)
        return outs, confs


def _read_version(model_path: str) -> str:
    meta = Path(model_path) / "model_meta.json"
    if meta.exists():
        try:
            import json
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "neural-1.0")
        except Exception:
            pass
    return "neural-1.0"


def default_model_path(mc: ModelConfig) -> Path:
    latest = mc.output_dir / "latest"
    return latest if latest.exists() else mc.output_dir


def load_corrector(mc: ModelConfig, *, prefer: str = "neural", device: Optional[str] = None):
    if prefer == "identity":
        return IdentityCorrector()
    path = default_model_path(mc)
    if not Path(path).exists():
        logger.info("No fine-tuned corrector at %s; using identity baseline.", path)
        return IdentityCorrector()
    try:
        return NeuralCorrector.from_pretrained(str(path), mc, device=device)
    except Exception as exc:
        logger.warning("Could not load neural corrector (%s); using identity baseline.", exc)
        return IdentityCorrector()


__all__ = ["NeuralCorrector", "IdentityCorrector", "load_corrector", "default_model_path"]
