"""Gradio UI for the Document-Level OCR system.

Upload a page image / PDF (or paste raw OCR text) -> structured Markdown + the
agent's decision log + a per-block table. ``gradio`` is imported lazily.
"""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_ui(cfg: Optional[AppConfig] = None):
    import gradio as gr  # lazy
    from ..agent.doc_agent import DocumentAgent

    cfg = cfg or AppConfig()
    agent = DocumentAgent(cfg, load_model=True)

    def _render(job):
        sd = job.to_dict()
        rows = [[b["page"], b["reading_index"], b["kind"], b["text"][:80],
                 round(b["conf"], 2), "⚠️" if b["flagged"] else "✓"] for b in sd["blocks"]]
        dec = "\n".join(f"{d['id']} {d['name']}: {d['branch']} — {d['detail']}" for d in sd["decisions"])
        dec += (f"\n\nstatus={sd['status']} | pages={sd['n_pages']} | blocks={sd['n_blocks']} | "
                f"flagged={sd['n_flagged']} | ocr={sd['model_versions'].get('ocr_engine')}")
        return sd["markdown"], rows, dec

    def from_image(image):
        if image is None:
            return "", [], "Provide an image."
        return _render(agent.process(image=image, filename="upload", save=False))

    def from_text(text):
        if not text.strip():
            return "", [], "Provide OCR text."
        return _render(agent.process(text=text, filename="pasted", save=False))

    with gr.Blocks(title=cfg.project_title) as demo:
        gr.Markdown(f"# 📄 {cfg.project_title}\nUpload a page image, or paste raw OCR text — the agent "
                    "lays out, OCRs, post-corrects and assembles structured Markdown.")
        with gr.Tab("Image / page"):
            img = gr.Image(type="pil", label="Page image", sources=["upload", "clipboard"])
            btn1 = gr.Button("Extract", variant="primary")
        with gr.Tab("Paste OCR text"):
            txt = gr.Textbox(label="Raw OCR text", lines=6,
                             value="The cornpany reported steacly growth in the thlrd quarter of 2O21.")
            btn2 = gr.Button("Correct", variant="primary")
        md = gr.Markdown(label="Structured output")
        table = gr.Dataframe(headers=["page", "#", "kind", "text", "conf", "ok"], label="Blocks", wrap=True)
        dec = gr.Textbox(label="Agent decision log", lines=6)
        btn1.click(from_image, [img], [md, table, dec])
        btn2.click(from_text, [txt], [md, table, dec])
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False) -> None:
    build_ui().launch(server_name=server_name, server_port=server_port, share=share)


__all__ = ["build_ui", "launch"]
