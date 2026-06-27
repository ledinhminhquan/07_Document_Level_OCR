"""Optional LLM reasoning brain for the agent (anthropic), with rule fallback.

Consulted only at the D4 correction hook for low-confidence regions. Disabled by
default; validates its own output and on any problem the caller keeps the
rule/neural result. Default deployment makes zero paid API calls.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

from ..config import AgentConfig
from ..logging_utils import get_logger
from ..models.text_utils import normalize_ws

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        return bool(self.cfg.llm_fallback_enabled and os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic
            key = os.environ.get(self.cfg.llm_api_key_env)
            self._client = anthropic.Anthropic(api_key=key) if key else None
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def correct_region(self, raw_text: str) -> Optional[Dict]:
        if not self.available():
            return None
        client = self._get_client()
        if client is None:
            return None
        prompt = (
            "You correct OCR errors in a block of text from a scanned document. "
            "Fix obvious OCR mistakes (rn->m, 0<->O, 1<->l, merged/split words) but DO NOT "
            "rewrite, summarize, or change meaning. Preserve line breaks.\n\n"
            f"OCR text:\n{raw_text}\n\n"
            'Reply with ONLY JSON: {"text": "<corrected block>", "confidence": <0..1>}.'
        )
        try:
            msg = client.messages.create(model=self.cfg.llm_model, max_tokens=600, temperature=0.0,
                                         messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") for b in msg.content)
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                data = json.loads(m.group(0))
                if isinstance(data.get("text"), str) and data["text"].strip():
                    return {"text": normalize_ws(data["text"]), "confidence": float(data.get("confidence", 0.7))}
        except Exception as exc:
            logger.info("LLM region correction failed (%s)", exc)
        return None


__all__ = ["LLMBrain"]
