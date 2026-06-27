# data/

This project's **primary training data is generated, not downloaded**.

- The synthetic **post-OCR** corpus is built by `src/dococr/data/ocr_noise.py` — it
  corrupts clean English text with a realistic OCR error model (`rn↔m`, `0↔O`, `1↔l`,
  merge/split, …) to make `(noisy, clean)` pairs. Cached under
  `${DOCOCR_DATA_DIR}/postocr_corpus/{train,val,test,real}.jsonl`.
- Build it with: `dococr data --task corpus` (auto-built on first `train` / `evaluate`).
- Preview pairs: `dococr synth --n 6`.

Large artifacts (the cached corpus, downloaded models, generated outputs) are
**git-ignored** — only this README is committed. All paths come from environment
variables (`DOCOCR_DATA_DIR`, `DOCOCR_ARTIFACTS_DIR`, `HF_HOME`).

## Real datasets (VERIFIED on HF)
- `PleIAs/Post-OCR-Correction` config **`english`** — CC0-1.0, 31.3K rows, columns
  `text` (noisy OCR) + `corrected_text` (gold). Mixed into training + a real eval slice.
- `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` — the canonical
  benchmark (optional; loader degrades gracefully if unavailable).
