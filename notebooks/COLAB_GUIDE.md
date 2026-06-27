# ☁️ Colab Training Guide — Document-Level OCR

Train the **ByT5 post-OCR corrector** on Colab (Pro/Pro+), then test it and collect the
deliverables. The notebook auto-adapts to **H100 / A100 / L4 / T4** and **resumes** after a disconnect.

---

## 0. What you need
- Google **Colab** (Pro+ for H100/A100; T4/L4 also work — the profile downshifts to `t5-small`).
- (Recommended) a **public GitHub repo** with this project, or upload the folder to Drive.

## 1. Get the project onto Colab
- **Option A (GitHub):** push this folder to `https://github.com/<you>/07_Document_Level_OCR`.
- **Option B (Drive):** upload `07_Document_Level_OCR/` to `MyDrive/07_Document_Level_OCR/`.

## 2. Drive layout (artifacts persist here → training survives disconnects)
Auto-created by the notebook:
```
MyDrive/dococr/
├── data/        # cached synthetic post-OCR corpus + samples
├── models/      # postocr_corrector/latest
├── runs/        # eval / error-analysis / robustness / benchmark JSON
├── outputs/     # extracted documents
├── submission/  # report.pdf + slides.pptx + bundle.zip
└── hf_cache/    # HuggingFace model cache
```

## 3. Configure & run
1. Open `notebooks/DocOCR_Colab_Training_H100_AUTOPILOT.ipynb` in Colab.
2. `Runtime → Change runtime type → GPU` (H100 if available).
3. **Cell 0 (Controls):** set `GIT_REPO_URL` (or use Drive); `BASE_MODEL=auto`;
   `USE_REAL=True` (mix in the real PleIAs corpus + a real eval slice); `TRAIN_SIZE`/`EPOCHS`.
4. `Runtime → Run all` → installs Colab-safe deps (never touches torch), installs Tesseract,
   auto-profiles the GPU, builds the corpus, and runs **autopilot** (cell 10): train → evaluate →
   analysis → `report.pdf` + `slides.pptx`.
5. **Disconnected?** Re-run **cell 10** — it resumes from the last checkpoint on Drive.

## 4. Verify it worked
- **Cell 11b / 12** — `evaluate` should show the corrector **reducing CER** vs the identity baseline
  (raw OCR), with a low **% degraded** (the safety gate), on both the synthetic and the real PleIAs slice.
- **Cell 13a** — the model fixes tricky OCR text (`cornpany`→`company`, `2O21`→`2021`, `l2%`→`12%`).
- **Cell 13b** — the agent OCRs a rendered page and emits structured Markdown (headings, lists, paragraphs).
- **Cell 14** — find `report.pdf` + `slides.pptx` in `…/submission/`.

## 5. Use the model later
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("…/models/postocr_corrector/latest")
m = AutoModelForSeq2SeqLM.from_pretrained("…/models/postocr_corrector/latest")
ids = tok("correct: The cornpany rep0rted gr0wth.", return_tensors="pt").input_ids
print(tok.decode(m.generate(ids, max_new_tokens=128)[0], skip_special_tokens=True))
```
or simply: `dococr correct --text "..."` / `dococr ocr --file scan.png`.

## Troubleshooting
- **OOM** → lower `TRAIN_SIZE`; the profile enables gradient checkpointing on A100/L4/T4 and switches
  to `google-t5/t5-small` on T4.
- **No Tesseract** → cell 3 installs it; without it the image OCR path uses a stub (text `correct` still works).
- **Corrector makes text worse** → the agent's D4 edit-budget gate rejects over-large rewrites at inference;
  watch the **% degraded** metric during eval.
