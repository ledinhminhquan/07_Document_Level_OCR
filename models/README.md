# models/

Trained checkpoints live here (git-ignored — only this README is committed).

```
models/postocr_corrector/
├── byt5-small-YYYYMMDD-HHMMSS/   # fine-tuned corrector (weights + tokenizer + model_meta.json)
└── latest/                       # pointer to the most recent version
```

- Train: `dococr --config configs/train.yaml train`
- The agent / API resolve `latest` automatically; with no checkpoint they fall back to
  the **identity** baseline (raw OCR), so the system always runs.
- Override the location with `DOCOCR_MODEL_DIR`; on Colab point it at Google Drive so
  checkpoints survive disconnects (resume-safe via `get_last_checkpoint`).

Pretrained models / engines (downloaded to `HF_HOME` or system, never committed):
`google/byt5-small` (+ `google-t5/t5-small` fallback), the **Tesseract** OCR engine
(`tesseract-ocr-eng`), and optional `python-doctr` / `easyocr` / `symspellpy`.
