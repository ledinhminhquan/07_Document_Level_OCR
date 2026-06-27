# Deploying to a Hugging Face Space (Gradio)

The Gradio UI (`app/gradio_app.py`) runs the full agent: upload a page image / PDF
(or paste raw OCR text) → layout → OCR → post-correct → structured Markdown + a
per-block table + decision log.

## Option A — Gradio SDK Space (simplest)

1. Create a Space → SDK **Gradio**.
2. Add at the Space repo root:
   - `app.py`:
     ```python
     from dococr.api.ui import build_ui
     demo = build_ui()
     ```
   - `requirements.txt` (copy `requirements_colab.txt` **plus** `torch`),
   - `packages.txt` (apt deps): `tesseract-ocr`, `tesseract-ocr-eng`, `poppler-utils`, `libgl1`,
   - the `src/` folder (so `dococr` imports), or `pip install git+https://github.com/<you>/07_Document_Level_OCR`.
3. Hardware: a **T4/A10 GPU** makes the corrector real-time; CPU works for the demo.
4. Push your trained corrector to the Hub and set `DOCOCR_MODEL_DIR`, or bake it into the Space.

## Option B — Docker Space (REST API + UI)

1. Create a Space → SDK **Docker**; push this repo (the `Dockerfile` has the apt deps).
2. The image serves `dococr.api.app_combined:app` on port **7860** (REST `/ocr` + `/correct` + Gradio at `/ui`).

## Notes
- Pre-download `google/byt5-small` (+ your fine-tuned corrector) into the image/cache for fast cold starts.
- Without Tesseract installed, the image OCR path falls back to a stub; `/correct` (text → corrected) still works.
- The corrector runs locally — no document data leaves the host (good for PII-sensitive documents).
- Avoid shipping non-commercial OCR backends (Surya weights, Nougat) in a commercial Space; keep Tesseract/docTR/Paddle (permissive).
