<!-- DESIGN BRIEF — P07 Document-Level OCR. Single source of truth (verified research).
     Package src/dococr/ mirrors the P02-P06 template. -->

# P07 — Document-Level OCR — Design Brief

> Single source of truth for the P07 build. The **trainable core** is a character-level **ByT5-small post-OCR correction seq2seq** model (synthetic OCR-noise generator = primary data, real PleIAs/ICDAR mixed in). The OCR front-end, layout analysis, and reading-order are **pretrained / algorithmic** (not trained here). A **deterministic FSM agent** (optional LLM brain) orchestrates *document-image → structured text*. All model/dataset ids below were confirmed live on the HF Hub / PyPI on 2026-06-26 and are marked **VERIFIED**; copy ids exactly.

---

## 1. Problem & business value

**Problem.** Real-world documents arrive as page images and scanned/born-digital PDFs. A flat OCR pass produces an *unordered, error-riddled* text dump: wrong reading order on multi-column pages, no block typing (title vs. table vs. footer), and character-level OCR garble (`rn→m`, `0↔O`, `1↔l`, merged/split words). P07 turns *document images → clean, structured, reading-order-correct text (Markdown + JSON)*. Its one **trained** differentiator is a **post-OCR corrector** that measurably lowers CER/WER on top of any OCR engine.

**Why it matters (business value).** Clean structured text is the upstream feed for search, RAG, analytics, compliance, and data entry. Three levers: (1) **accuracy** — fewer downstream errors from garbled OCR; (2) **structure** — Markdown/JSON with typed blocks + reading order unlocks RAG/table extraction a flat dump can't; (3) **engine-agnostic uplift** — the corrector improves *any* OCR backend, so it compounds with cheap CPU OCR (Tesseract) instead of requiring an expensive engine.

### Success metrics

**Business**
- **Manual-review reduction** — % pages auto-accepted (`needs_review=false`) at the target confidence gate; goal: minimize human-in-the-loop touches.
- **Structure fidelity** — % blocks correctly typed + ordered (eval on DocLayNet/FUNSD/SROIE), so downstream RAG/table consumers get usable structure.
- **Cost per page** — born-digital pages skip OCR entirely (router D2), so blended cost is dominated by the scanned fraction; CPU-only default (Tesseract + ByT5/t5-small) keeps unit cost low.

**Technical (headline = CER/WER reduction, the trained-model KPI)**
- **CER reduction (primary):** `CER_before = CER(raw_OCR, gold)`, `CER_after = CER(corrected, gold)`; report **% improvement = (CER_before − CER_after) / CER_before × 100**, measured with `jiwer`.
- **WER reduction:** same formula at word level.
- **Safety gate — % sentences improved vs. degraded vs. unchanged.** A corrector that "fixes" 40% but breaks 30% is near-useless; require **degraded ≪ improved**. This is the go/no-go gate.
- **Must beat two baselines** on CER reduction *and* keep `% degraded` low: **identity (no-op)** and **SymSpell dictionary** corrector.
- **Layout/order quality:** region-detection mAP + reading-order correctness on `pierreguillou/DocLayNet-base`.
- **Latency targets (scanned A4 @ 300 DPI, CPU):** preprocess ~150 ms · layout ~300 ms · OCR ~0.6–1.2 s/region (dominant) · correction ~80 ms/region (small model) / ~300 ms (large) · LLM brain only on flagged regions ~0.8–2 s. **Born-digital page (D2 skip): ~200 ms end-to-end.**

---

## 2. VERIFIED stack table

Every dataset & model id below is confirmed live (HF Hub / PyPI, 2026-06-26). **Prefer Apache-2.0 / MIT / CC0 ids for any commercial framing**; non-commercial (cc-by-nc-sa / OpenRAIL) ids are flagged and have permissive swaps.

### 2a. OCR engines (front-end, pretrained/algorithmic)

| Role | Id (pip / HF) | License | VERIFIED facts |
|---|---|---|---|
| **OCR default (CPU, boxes+conf+structure)** | `pytesseract` (PyPI) + system `tesseract` 5.x | Apache-2.0 (both) | **VERIFIED** PyPI. `image_to_data(img, output_type=Output.DICT)` → keys `level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text`. Per-word boxes + `conf` (0–100, −1=non-text) + block/par/line/word hierarchy. CPU-only, no download. Filter `conf>0`, group by `(block_num, par_num, line_num)`. |
| **OCR upgrade (best Apache accuracy)** | `python-doctr` (PyPI, current 1.0.1) | Apache-2.0 (code + weights) | **VERIFIED** PyPI. DBNet detector + CRNN/ViTSTR recognizer. `result.export()` → Page→Block→Line→Word tree; `Word.geometry`=`((xmin,ymin),(xmax,ymax))` relative, `Word.confidence`; `export_as_xml()`=hOCR. NOTE: there is **no** `mindee/doctr` HF repo (404); `Felix92/doctr-*` are dummy weights — do **not** use. Weights auto-download from project release assets. |
| **OCR upgrade (best doc structure: layout+order+tables)** | `surya-ocr` (PyPI 0.20.0) + `datalab-to/surya-ocr-2` | Code Apache-2.0; **weights modified AI-Pubs OpenRAIL-M (NC over $5M rev/funding)** | **VERIFIED** PyPI + HF (686M params, qwen3_5 arch, 382K dl). Native **layout + reading order + table recognition**, 90+ langs. Needs vLLM (NVIDIA GPU) or llama.cpp (CPU/Apple) backend; Python ≥3.10. License caveat = main catch. |
| OCR alt (layout-rich, pure-pip permissive) | `paddleocr` (PyPI) + PaddlePaddle ≥3.0 | Apache-2.0 | **VERIFIED** PyPI. PP-OCRv5 → `rec_texts`/`rec_scores`/`rec_boxes`; **PP-StructureV3** adds layout+tables+formulas+Markdown/JSON + multi-column reading order. Heavier (Paddle runtime). Use as all-in-one baseline. |
| OCR recognizer (printed, behind a detector) | `microsoft/trocr-base-printed` (also `-large-printed`, `-small-printed`) | MIT (repo-level) | **VERIFIED** (8.6M / 8.8M / 2.1M dl). VisionEncoderDecoder, line/crop recognizer **only — no detector, no boxes**. Pair with docTR/Tesseract detector for hard printed lines. |
| OCR recognizer (handwriting) | `microsoft/trocr-base-handwritten` | MIT | **VERIFIED** (30.2M dl). Routed to when a region is classed handwritten. |
| Pure-Python fallback (no system binary) | `easyocr` (PyPI) | Apache-2.0 | Returns `(bbox, text, confidence)` per box, PyTorch-only, zero system deps. Use when a `tesseract` binary can't be installed. |
| (Optional) full-page VLM, math/markdown | `facebook/nougat-base` | **cc-by-nc-4.0 (NC)** | **VERIFIED** (2.9M dl, 349M). Whole-page image→markdown for scientific PDFs; optional D-bypass. NC license — flag. |

### 2b. Post-OCR correction — the TRAINABLE core

| Role | Id | License | VERIFIED facts |
|---|---|---|---|
| **Base model (PRIMARY trainable)** | `google/byt5-small` | Apache-2.0 | **VERIFIED** (30.1M dl, ~300M params, ~1.2 GB fp32). `AutoModelForSeq2SeqLM`, t5 arch, **byte/char-level — no SentencePiece**; `AutoTokenizer` = ByT5Tokenizer (384 base vocab). Multilingual incl. `en`. Char-level I/O matches char-level OCR noise; ByT5 paper (arXiv:2105.13626) shows byte models markedly more noise-robust. |
| Ablation / T4 fallback base | `google-t5/t5-small` | Apache-2.0 | **VERIFIED** (192.7M dl, 60.5M params). Subword; needs `correct: ` prefix too. Faster, lighter — recommended T4 path & ablation row. |
| (Optional) off-the-shelf corrector comparator | `oliverguhr/spelling-correction-english-base` | **MIT** | **VERIFIED** (701K dl, 139M, BART). Permissive baseline to compare the trained ByT5 against. (Other grammar checkpoints — `pszemraj/flan-t5-large-grammar-synthesis`, `grammar-synthesis-small`, `vennify/t5-base-grammar-correction` — are **cc-by-nc-sa, NC**; use only as research comparators, not the shipped commercial model.) |
| Non-neural baseline | `symspellpy` (PyPI) | MIT | Symmetric-Delete; `lookup_compound` does compound-aware word split+merge (addresses merge/split OCR errors). Classic dictionary baseline the seq2seq must beat. |

### 2c. Real post-OCR datasets

| Role | Id | License | VERIFIED facts |
|---|---|---|---|
| **Primary train/eval (real pairs)** | `PleIAs/Post-OCR-Correction`, config `english` | **CC0-1.0** | **VERIFIED** (50.4K rows total; **english=31.3K**, ~2.1 GB; 19K dl). `load_dataset("PleIAs/Post-OCR-Correction","english")`. Cols: `index_id,id,date,edition,page,file_name,word_count,text,corrected_text,…`. **`text`=raw OCR, `corrected_text`=gold** (real US newspaper scans). Configs also `french(16.5K)/german(672)/italian(1.9K)`, all `train` split only — carve own eval. Caveat: `corrected_text` is model-generated → **silver, not perfect gold**. |
| **Held-out real benchmark** | `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` | not declared on card; origin Zenodo 3515403 (CC-BY-NC-SA in source) | **VERIFIED (exists)**. 22M OCRed chars, 10 langs incl. **`eng`**. **Viewer DISABLED (501)** — raw ICDAR format (`[OCR_toInput]`/`[OCR_aligned]`/`[ GS_aligned]` blocks + per-token offsets); **manual download + custom parser required**. Field-standard gold → use as the trustworthy eval anchor for literature-comparable numbers. |
| Synthetic clean source (PRIMARY data feedstock) | `Salesforce/wikitext`, config `wikitext-103-raw-v1` (or lighter `wikitext-2-raw-v1`) | CC-BY-SA-3.0 / GFDL | **VERIFIED** (33M dl). Single `text` col, English. Use **`-raw-v1`** (keeps real casing/punctuation; non-raw lowercases + `<unk>`-masks). Feed the synthetic noise generator. Alt: `wikimedia/wikipedia` 20231101.en. |
| Design reference only (NOT for English training) | `jeanflop/post-ocr-correction` | Apache-2.0 | **VERIFIED** (4.7M rows; cols `input,output`). **French only** — use as a corruption-design reference, not English data. |

### 2d. Layout datasets & detectors (pretrained/algorithmic front-end)

| Role | Id | License | VERIFIED facts |
|---|---|---|---|
| **Layout detector (default)** | `juliozhao/DocLayout-YOLO-DocStructBench` | **Apache-2.0** | **VERIFIED** (56 likes, arXiv 2410.12628). YOLOv10-based; `pip install doclayout-yolo`. Best speed/accuracy general detector; 11 DocLayNet classes. Recommended default. |
| Layout train/eval (viewer works) | `pierreguillou/DocLayNet-base` | other (research) | **VERIFIED** (12.8K dl, arXiv 2206.01062). `load_dataset(...)` direct. train 6.9K/val 648/test 499. Cols `image, bboxes_block, bboxes_line, texts, categories` (11-class: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title), `page_no, doc_category`. **Use this one.** |
| Layout full set (heavy) | `docling-project/DocLayNet` (`ds4sd/DocLayNet` redirects here) | other | **VERIFIED** (20.7K dl). 80,863 pages, COCO format, no viewer (500). Manual COCO+PNG download. |
| Table detect + structure | `microsoft/table-transformer-detection` (+ `-structure-recognition`) | **MIT** | **VERIFIED** (73.1M dl, 28.8M params, DETR, arXiv 2110.00061). `AutoModelForObjectDetection`. Detect Table regions → cell structure → HTML/Markdown table. |
| Form region/entity classify | `nielsr/funsd-layoutlmv3` | (FUNSD) | **VERIFIED** (138.3K dl). train 149/test 50; cols `tokens, bboxes, ner_tags` (O,B/I-HEADER/QUESTION/ANSWER), `image`. Form understanding eval (P04 dataset). |
| Receipts (boxes+entities) | `darentang/sroie` | (SROIE) | **VERIFIED** (973 rows; `words, bboxes, ner_tags, image_path`). Structured-doc assembly eval. |
| Token-level layout classifier (optional) | `microsoft/layoutlmv3-base` | **cc-by-nc-sa-4.0 (NC)** | **VERIFIED** (164.5M dl, 125.3M). Fine-tune for region/entity classification. NC — flag; not for commercial ship. |
| Page-type router (optional) | `microsoft/dit-base-finetuned-rvlcdip` | (research) | **VERIFIED** (3.3M dl, BEiT). `AutoModelForImageClassification`, 16 RVL-CDIP classes. Assists D2 / doc-type routing. |

### 2e. Libraries

| Lib | License | Role |
|---|---|---|
| `pymupdf` (`fitz`) | AGPL/commercial | Born-digital text+bbox extract (`page.get_text("dict", sort=True)`), rasterize (`get_pixmap(dpi=200-300)`), text-layer detection (D2). |
| `pdf2image` (+ `poppler-utils`) | MIT (+ poppler GPL) | `convert_from_path(pdf, dpi=300)` scanned rasterization. |
| `pillow`, `opencv-python` | HPND / Apache-2.0 | Image I/O; deskew/denoise/binarize (preprocess D1). |
| `jiwer` | Apache-2.0 | **CER/WER** via RapidFuzz C++ edit distance: `jiwer.cer`, `jiwer.wer`, `jiwer.process_characters` (alignment). The metric backbone. |
| `symspellpy` | MIT | Dictionary baseline (Symmetric-Delete, `lookup_compound`). |
| `rapidfuzz` | MIT | Edit-distance for D4 acceptance proxy. |
| `transformers, datasets, accelerate, sentencepiece` | Apache-2.0 | Training/inference stack. |
| `anthropic` | MIT | Optional LLM brain (D4/D1), advisory + validated. |
| `fastapi, uvicorn, gradio` | MIT / Apache-2.0 | API + UI. |

Install: `pip install transformers datasets jiwer symspellpy sentencepiece accelerate pymupdf pdf2image pillow opencv-python rapidfuzz python-doctr pytesseract gradio fastapi uvicorn anthropic` + system `tesseract-ocr tesseract-ocr-eng poppler-utils libgl1`.

---

## 3. System pipeline

```
 image / single-page or multi-page PDF
            │
            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ INGEST / LOAD            load_page()  (pymupdf / pdf2image)  │
 │   per page: raster image + (PDF) embedded text layer        │
 └───────────────┬─────────────────────────────────────────────┘
                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ PREPROCESS  assess_quality() → D1 plan                      │
 │   deskew · denoise · binarize(otsu/adaptive) · upscale      │◄── LLM (opt, validated)
 └───────────────┬─────────────────────────────────────────────┘
                 ▼
        ┌────────────────────────────────────────────┐
        │ ROUTE   has_text_layer()  ── D2            │
        │  coverage≥0.80 & chars≥200 ?               │
        └──────┬───────────────────────────────┬─────┘
       BORN-DIGITAL (skip OCR)            SCANNED / MIXED
               │                                │
               │ get_text("dict",sort)          ▼
               │  text + vector bboxes  ┌──────────────────────────────┐
               │                        │ LAYOUT + READING ORDER       │
               │                        │  DocLayout-YOLO → regions+   │
               │                        │  types; XY-cut + column      │
               │                        │  clustering → reading_index  │
               │                        └───────────────┬──────────────┘
               │                                        ▼  (per region ∥)
               │                        ┌──────────────────────────────┐
               │                        │ OCR REGIONS  ocr_region()    │
               │                        │  Tesseract/docTR(/TrOCR)     │
               │                        │  D3 conf gate: ≥85 accept ·  │
               │                        │  60–85 re-OCR · <60 → FLAG   │
               │                        │  Table→Table-Transformer     │
               │                        └───────────────┬──────────────┘
               └──────────────┬─────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │ POST-OCR CORRECT  (THE TRAINED ByT5 MODEL)             │
        │  correct_text() per region/line · D4 acceptance:       │◄── LLM brain
        │   accept iff edit_ratio≤0.25 AND error reduced;        │    (flagged/
        │   else keep RAW (deterministic fallback)               │    rejected only,
        └───────────────┬─────────────────────────────────────────┘    validated)
                        ▼
        ┌─────────────────────────────────────────────────────────┐
        │ ASSEMBLE  sort by reading_index →                      │
        │   Markdown (class→md) · structured JSON (blocks[]) ·    │
        │   plain text (linear read).  confidence + needs_review │
        └───────────────┬─────────────────────────────────────────┘
                        ▼  DONE  → {text, markdown, blocks[], confidence, meta}
```

---

## 4. Trainable model plan — ByT5 post-OCR corrector

The **only model trained in P07**. Char-level seq2seq: noisy OCR text → clean text. Base = `google/byt5-small` (Apache-2.0), ablation = `google-t5/t5-small`.

### 4.1 Synthetic OCR-noise generator (PRIMARY data)

**Why primary:** unlimited volume, perfectly clean gold, tunable difficulty matched to the real PleIAs/ICDAR error rate. Clean feedstock = `Salesforce/wikitext` `wikitext-103-raw-v1` → sentence/line-segment → keep 5–60-word spans → drop headers (`= Title =`)/empties → corrupt each into a `(noisy, clean)` pair.

**Confusion model** (apply per-char/per-word with tunable global `error_rate`; sample noise rate from a band so the model sees light *and* heavy degradation; **vary the noise every epoch** via `set_transform`/per-epoch seed):

- **(a) Char substitution — OCR confusion sets** (bidirectional, visual-similarity weighted):
  `rn↔m  cl↔d  vv↔w  nn↔m  ii↔n  li↔h  0↔O↔o↔Q↔D  1↔l↔I↔i↔|  5↔S↔s  8↔B  6↔b↔G  2↔Z↔z  9↔g↔q  c↔e↔o  t↔f↔l  u↔n↔ii  h↔b  .↔,  :↔;  '↔\`  -↔~  !↔l/1  %↔°`
- **(b) Insertion** — stray punctuation/marks (`. , ' " - ~ ° |`), doubled letters, random speckle (dirt/bleed-through).
- **(c) Deletion** — drop a char (faded/broken glyph); occasionally drop a whole short word.
- **(d) Spacing — merge & split** (high-impact): delete space → merge (`the cat`→`thecat`); insert space mid-word → split (`correction`→`correc tion`); collapse/expand space runs.
- **(e) Case errors** — random upper/lower flips (`The`→`THe`); models all-caps headline mis-segmentation (`TUEBPAY`).
- **(f) (optional) Diacritic/ligature** — `fi`/`fl` ligature damage, accent drop.

**Realistic mix:** ~60% substitution · ~15% spacing · ~10% insertion · ~10% deletion · ~5% case (tune to measured PleIAs profile). **Calibrate global `error_rate` so synthetic CER ≈ real PleIAs/ICDAR CER (measure both with `jiwer`).** Seed it; ship as a reusable, reproducible function. **Curriculum:** start low error rate, increase over epochs; final fine-tune + eval on real data.

### 4.2 Real-data mix & anti-overfitting

- **Blend ~60% real PleIAs / ~40% synthetic.** Real `text`→`corrected_text` captures genuine scanner artifacts the synthesizer won't invent; synthetic gives volume + clean gold.
- **Hold a clean eval set untouched by the noise transform** (real PleIAs pairs only) so eval CER reflects real OCR, not the synthesizer. Use **ICDAR-2019 `eng`** as the literature-comparable held-out anchor.
- **Dedup** exact + near-dup (MinHash/normalized) across configs *before* the eval split (PleIAs repeats headers/boilerplate) so train/eval don't leak.
- `label_smoothing=0.1` + `weight_decay=0.01` + **early stopping on `cer_reduction`** (patience 4) + `load_best_model_at_end`.

### 4.3 Input/target format

```python
PREFIX = "correct: "            # T5-style task prefix
MAX_SOURCE = 320               # BYTES not tokens (256–384 band; 320 default)
MAX_TARGET = 320
# CHUNK long PleIAs pages to ≤300 bytes on whitespace BEFORE training —
# rows are full pages; skipping this truncates the tail of every long row (top gotcha).

def preprocess(batch, tokenizer):
    inputs = [PREFIX + t for t in batch["text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE, truncation=True)   # no padding here
    labels = tokenizer(text_target=batch["corrected_text"], max_length=MAX_TARGET, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# DataCollatorForSeq2Seq pads per-batch (label_pad_token_id=-100, pad_to_multiple_of=8) — no hand -100 masking.
```

### 4.4 Metrics, baselines, safety gate

`compute_metrics` returns `cer, wer, input_cer, cer_reduction = input_cer − corrected_cer, exact_match`. Set `metric_for_best_model="cer_reduction"`, `greater_is_better=True`. Pass `eval_accumulation_steps=8` so generated preds don't pile in VRAM.

**Baselines to beat:** **Identity/no-op** (floor; headline number = improvement vs. identity) and **SymSpell** (`symspellpy.lookup_compound`, strong on isolated misspellings, weak on garbled/contextual — the gap the seq2seq fills).

**Report table:** rows = {Identity, SymSpell, ByT5-small (synth), ByT5-small (synth+real FT), *ablation:* t5-small} × cols = {CER_before, CER_after, CER %↓, WER %↓, % improved, % degraded}, on **both** PleIAs-english test **and** ICDAR-2019-eng. **Success = ByT5 beats both baselines on CER reduction AND keeps `% degraded` low** (safety gate).

### 4.5 H100 `Seq2SeqTrainer` config dict

```python
config = {
    "model_id": "google/byt5-small",
    "prefix": "correct: ",
    "max_source_length": 320, "max_target_length": 320,
    "dataset_id": "PleIAs/Post-OCR-Correction",
    "dataset_configs": ["english", "french", "italian", "german"],  # mix real langs
    "eval_fraction": 0.03,          # carve held-out eval (no native eval split)
    "chunk_bytes": 300,             # window long pages on whitespace
    "synthetic_real_mix": (0.40, 0.60),  # synth / real blend
    "args": dict(
        output_dir="byt5-postocr",
        bf16=True, tf32=True, fp16=False,                 # H100 loves bf16+tf32
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,                    # effective batch = 256
        group_by_length=True,                             # big speedup, variable-length bytes
        learning_rate=5e-4,                               # ByT5 likes a touch higher than t5
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, label_smoothing_factor=0.1,
        num_train_epochs=5, max_grad_norm=1.0,
        optim="adamw_torch_fused",                        # fused AdamW on H100
        gradient_checkpointing=False,                     # off on H100 (VRAM plentiful)
        predict_with_generate=True,
        generation_max_length=320, generation_num_beams=1,  # greedy eval; beam=4 final test only
        eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=500, save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer_reduction", greater_is_better=True,
        eval_accumulation_steps=8,
        logging_steps=50, dataloader_num_workers=4,
        report_to="none", seed=42,
    ),
    "early_stopping_patience": 4,   # EarlyStoppingCallback on cer_reduction
}
# Trainer: Seq2SeqTrainer(model, args, train_ds, eval_ds, data_collator,
#   tokenizer, compute_metrics, callbacks=[EarlyStoppingCallback(patience=4)])
# Resume-safe: trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))
```

### 4.6 GPU-profile table (effective batch held at 256)

| GPU | VRAM | Model | Precision | per-dev bs | grad-accum | grad-ckpt | Peak VRAM | ~Time / 5 epochs | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **H100 80GB** | 80 | byt5-small | bf16+tf32 | 32 | 8 | off | ~22–30 GB | ~30–50 min | `adamw_torch_fused`; greedy eval; fastest |
| A100 40GB | 40 | byt5-small | bf16+tf32 | 16 | 16 | off | ~24–34 GB | ~45–90 min | bs 12 / accum 22 if OOM |
| L4 24GB | 24 | byt5-small | bf16 | 8 | 32 | on | ~18–22 GB | ~1.5–2.5 h | `optim="adamw_torch"` |
| T4 16GB | 16 | byt5-small | fp16 | 4 | 64 | on | ~13–15 GB | ~4–6 h | tight; drop max_len→256 |
| **T4 16GB (recommended)** | 16 | **t5-small** | fp16 | 16 | 16 | optional | ~9–12 GB | ~50–90 min | subword far lighter; keep `correct: ` prefix |

T4: no bf16 (Turing) → `fp16=True, bf16=False, tf32=False`; if NaNs, drop `label_smoothing_factor` + LR→`3e-4`. On T4 **prefer `t5-small`** (ByT5 byte blowup is painfully slow there). `predict_with_generate` eval is the slow part — keep `eval_steps` modest, `num_beams=1` until final test (then 4).

---

## 5. Agent architecture

**Deterministic FSM** — every transition is decided by an explicit rule on measurable signals (DPI, text-layer presence, OCR confidence, edit distance). The **LLM brain (`anthropic`) is strictly advisory**: consulted only at LLM-eligible decision points, its proposal **re-validated by the same deterministic rule**, and on any failure (timeout, API error, schema violation, rule rejection) the FSM **falls back to the deterministic branch**. Fully runnable with `ANTHROPIC_API_KEY` unset.

**States:** `LOAD → PREPROCESS → ROUTE → LAYOUT → OCR → CORRECT → ASSEMBLE → DONE` (+ `FLAG` sink for low-confidence regions).

### 5.1 Tool contracts (typed; FSM owns control flow; tools never raise past orchestrator)

```python
load_page(path: str) -> Page          # {page_id, image HxWx3, dpi, source:'pdf'|'image', pdf_text_layer:str|None}
assess_quality(page) -> Quality       # {skew_deg, blur_var (Laplacian), noise, est_dpi, is_dark_bg}
preprocess(page, plan) -> Page        # plan={deskew,denoise,binarize:'otsu'|'adaptive'|None,upscale}; idempotent
has_text_layer(page) -> TextLayer     # {present, coverage 0..1, char_count, text}   → drives D2
detect_layout(image) -> list[Region]  # Region{id,bbox,type:'title|paragraph|list|table|figure|caption|header|footer',column,order}
ocr_region(image, region, engine='tesseract') -> OCRResult   # {region_id,text,conf 0..100,word_confs[],engine}
correct_text(text, context, mode='model'|'llm') -> Correction # {text_out,edits,edit_ratio,source:'model'|'llm'|'none'}
assemble(regions, texts) -> Document  # {text, markdown, blocks[], confidence}
llm_brain(prompt, schema, timeout=8.0) -> dict|None           # anthropic→parsed JSON|None; caller RE-VALIDATES
```

### 5.2 Decision points (D1–D4)

| DP | State | Signal(s) | Deterministic rule | LLM-eligible? |
|---|---|---|---|---|
| **D1** preprocess routing | PREPROCESS | `skew_deg, blur_var, est_dpi, is_dark_bg` | `deskew if |skew|>1.0°`; `denoise if blur_var<120`; `binarize='adaptive' if is_dark_bg else 'otsu'`; `upscale=2.0 if est_dpi<200`. If `est_dpi<120 and blur_var<40` → mark page `degraded` (lowers D3 threshold, forces flag-on-fail). | Optional (LLM picks a plan from the menu; validated = must be subset of allowed ops) |
| **D2** born-digital vs scanned | ROUTE | `has_text_layer()` | `if present and coverage≥0.80 and char_count≥200` → **SCANNED=False**: take embedded text, **skip OCR & layout-from-pixels**. Else SCANNED=True → full path. Mixed `0.2<coverage<0.8` → hybrid (OCR only image-only regions). | No (objective signal) |
| **D3** OCR-confidence gate | OCR (per region) | `OCRResult.conf, word_confs` | `conf≥85` accept · `60≤conf<85` re-OCR (PSM change / engine swap tesseract↔docTR / TrOCR crop; keep higher) · `<60` after retry → **FLAG** (kept, `needs_review=true`). Thresholds drop to 75/50 on `degraded` pages. | No (deterministic gate) |
| **D4** correction acceptance | CORRECT (per region) | `edits, edit_ratio, conf` | Run trained-model correction. **Accept iff `edit_ratio ≤ 0.25` AND error reduced** (proxy: ≥ as many dictionary words / fewer non-words than input). `>0.25` → reject, keep raw (likely hallucination). If region was **flagged (D3) or model-rejected** → escalate to LLM brain with neighbour context. | **Yes** — LLM proposes; **same D4 rule re-checked locally**; reject/timeout/error → model output, then raw. |

### 5.3 LLM-brain safety contract

```
D4 per region:
  cand_model = correct_text(raw, ctx, mode='model')
  if accept(cand_model):                       # edit_ratio<=0.25 AND error_reduced
        use cand_model
  elif region.flagged or near-budget:
        cand_llm = llm_brain(prompt(raw,ctx), schema={text:str, edits:int})   # advisory
        if cand_llm and accept_llm(cand_llm):  # SAME rule re-checked locally
              use cand_llm.text
        else: use raw                          # timeout/bad JSON/over budget/no reduction → fallback
  else: use raw
```

### 5.4 Worked example — 2-column scanned page (tilted, title spanning both columns, 1 figure+caption)

```
LOAD       load_page("p12.png") → source=image, dpi≈220, pdf_text_layer=None
PREPROCESS assess → skew=1.8, blur=210, est_dpi=220, dark=False
           D1: |1.8|>1.0 → deskew; blur 210>120 → no denoise; light bg → binarize='otsu'; dpi>200 → no upscale
ROUTE      has_text_layer → present=False → D2: SCANNED=True → full OCR path
LAYOUT     5 regions: R0 title(span,order0) · R1,R2 left col(order1,2) · R3 right col(order3) · R4 figure / R5 caption(order4,5)
           reading order = title → left col top→bottom → right col top→bottom  (XY-cut + column clustering)
OCR (∥)    R0 conf93 accept · R1 conf88 accept · R2 conf71 → D3 re-OCR (PSM6→docTR) →86 accept ·
           R3 conf54 → re-OCR→58(<60) → FLAG (needs_review=true, keep text) · R4 figure no-OCR · R5 conf90 accept
CORRECT    R1 "the resuits were significant"→"the results were significant" edits2 ratio0.07 → D4 ACCEPT (model)
           R2 model ratio0.31>0.25 → D4 REJECT, keep raw
           R3 (flagged) → LLM brain w/ R2/R4 context → JSON{text,edits4}; local re-check ratio0.11 & err↓ → ACCEPT (llm)
                          [if API failed → keep raw, needs_review stays true]
ASSEMBLE   "# <title>\n\n<left ¶s>\n\n<right ¶s>\n\n![figure](fig_r4.png)\n*<caption>*"
           blocks[] carry bbox+type+conf+source+needs_review; doc confidence = area-weighted mean (~0.82), R3 surfaced.
```

---

## 6. Deployment

**Response object** (lossless, every block traceable to a bbox, `needs_review` never silently dropped, `confidence` always present, `meta.*_version` pinned, markdown=human view / `blocks[]`=machine view):

```jsonc
// POST /ocr -> 200
{ "doc_id":"a1b2c3",
  "text":"Plain reading-order text …",
  "markdown":"# Title\n\nLeft column…\n\nRight column…\n\n![](fig_0.png)\n*Caption*",
  "blocks":[ { "id":"r1","page":12,"type":"paragraph","bbox":[x,y,w,h],"column":0,"order":1,
               "text":"the results were significant","ocr_conf":88.0,
               "correction_source":"model","edit_ratio":0.07,"needs_review":false } ],
  "confidence":0.82,
  "meta":{ "pages":1,"scanned":true,"engine":"tesseract+trocr",
           "model_version":"byt5-postocr@1.3.0","pipeline_version":"ocr-agent@2.0.1","latency_ms":4120 } }
```

**FastAPI** — `POST /ocr` (multipart `file` + `?engine=&correct=true&llm=false`) · `POST /ocr/batch` (list/zip, async) · `POST /ocr/stream` (SSE, per-page blocks for long PDFs) · `GET /healthz` · `GET /version` (pinned model+pipeline). Heavy jobs → enqueue (Celery/RQ+Redis), return `job_id`, poll `GET /jobs/{id}`.

**Gradio** — `gr.File` (image/PDF) → tabs: **Markdown** (`gr.Markdown`) · **Overlay** (bbox+conf boxes, flagged in red) · **JSON** (`gr.JSON`) · **Raw vs Corrected** diff. Toggles: engine, correct on/off, use-LLM-brain. **Import the same backend functions as FastAPI — don't duplicate.**

**CLI batch** —
```bash
ocr-agent run ./scans --glob "*.pdf,*.png" --out ./out \
    --engine tesseract --correct --no-llm --workers 8 --format md,json --manifest manifest.csv
```
Walks folder, worker pool over pages → `out/<name>.{md,json}` + `manifest.csv` (file, pages, mean_conf, n_flagged, latency). `--resume` skips done outputs (idempotent).

**Docker** — base `python:3.11-slim`; `apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils libgl1`. Pip layer cached separately from code; **pre-download HF weights at build** (`snapshot_download`) so cold start ≠ first-request stall. `uvicorn app:app --host 0.0.0.0 --port 7860 --workers $WEB_CONCURRENCY`. Optional CUDA tag for TrOCR/large-model GPU path; CPU image stays small using `t5-small`/`byt5-small`.

**HF Space** — Gradio SDK (or Docker SDK to keep Tesseract system deps). `app.py` + `requirements.txt` + `packages.txt` (`tesseract-ocr`, `poppler-utils`). CPU-basic runs the small corrector; bump to GPU tier for TrOCR / large correction. Weights pulled from verified ids on startup.

**Latency** — scanned A4 @300 DPI CPU: preprocess ~150 ms · layout ~300 ms · OCR ~0.6–1.2 s/region (dominant) · correction ~80 ms (small) / ~300 ms (large) per region · LLM brain only on flagged ~0.8–2 s. **Born-digital page skips OCR (D2) → ~200 ms.**

**Scalability — two parallelism axes:** **page parallelism** (PDFs fan pages across a process pool / queue workers — CPU-bound OCR ⇒ processes not threads); **region parallelism** (OCR regions concurrently; correction **batched across regions in one model forward pass**). GPU-batch TrOCR/ByT5, keep Tesseract on CPU workers (heterogeneous pool). Autoscale on queue depth; bound `MAX_PAGES`/request for tail latency; **cache by file hash** (idempotent re-submits return instantly).

**Versioning** — pin three things independently, surface all in `meta`: `pipeline_version` (FSM + thresholds), `model_version` (corrector checkpoint, e.g. `byt5-postocr@<rev>`), `engine` (tesseract 5.x / docTR / TrOCR rev). **Pin HF revisions by commit SHA, not `main`.** D1–D4 thresholds live in a versioned `config.yaml` so a threshold change is a traceable release.

```
 client: Gradio UI | CLI batch | HTTP/SDK ──► FastAPI(uvicorn) ──► Redis job queue + worker pool
                                                  │ same code import        │ page∥ + region∥
                                                  ▼                         ▼
                            OCR-AGENT FSM: LOAD→PREP(D1)→ROUTE(D2)→LAYOUT→OCR(D3)→CORRECT(D4+LLM)→ASSEMBLE
                                 ▼ Tesseract/docTR (CPU)   ▼ TrOCR (GPU)   ▼ ByT5 corrector ── anthropic (LLM brain, validated)
                            weights pinned by HF commit SHA · meta.*_version on every response
            Packaging: Docker (tesseract+poppler+weights baked) → HF Space / k8s autoscale on queue depth
```

---

## 7. Risks, limitations, ethics

- **Corrector making things worse (top risk).** A seq2seq corrector can *introduce* errors on text it doesn't understand. Mitigations: the **D4 acceptance gate** (`edit_ratio≤0.25` AND error-reduction proxy; else keep raw), the **identity baseline** as the floor the model must beat, and the **`% degraded` safety metric** as an explicit go/no-go gate. Never accept a correction blindly.
- **Hallucinated corrections.** ByT5 (and especially the LLM brain) can fluently rewrite garbled text into plausible-but-wrong content. Mitigations: char-level model (stays near the input), `edit_ratio` budget, **LLM brain advisory-only + re-validated by the same rule + deterministic fallback**, and `correction_source` recorded per block for audit.
- **OCR of private/copyright documents.** The system ingests arbitrary uploads. Surface data-handling expectations; don't persist content beyond the request unless configured; respect copyright for ingested material. PleIAs is **CC0** (public-domain newspapers — safe); flag that ICDAR's source is **CC-BY-NC-SA** and wikitext is **CC-BY-SA** (attribution). Non-commercial model weights (`layoutlmv3-base`, `nougat-base`, `surya` weights, grammar checkpoints) must **not** ship in a commercial build — use the permissive swaps (Tesseract/docTR Apache-2.0, `oliverguhr/spelling-correction-english-base` MIT, DocLayout-YOLO/Table-Transformer Apache/MIT).
- **Layout failures on complex pages.** XY-cut + column clustering can mis-order dense magazines, nested tables, rotated text, overlapping regions. Mitigations: pre-strip headers/footers before XY-cut; column clustering fallback when whitespace valleys are weak; optional learned reading-order (Surya / PP-StructureV3) as an upgrade; **flag low-structure-confidence pages for review** rather than emitting silently wrong order.
- **Language scope.** Trained corrector + datasets are **English-primary** (real multilingual configs mixed in but English is the eval anchor). Tesseract/docTR are configured for English; do not claim multilingual correction quality without per-language eval. State the scope explicitly.
- **Silver-gold caveat.** PleIAs `corrected_text` is model-generated → **silver**; ICDAR-2019 `eng` is the trustworthy gold eval anchor. Report numbers on both and treat PleIAs gold accordingly.

---

## 8. Repo module map

Package layout mirrors P02–P06 (`src/dococr/`); heavy deps (torch/transformers/datasets/pymupdf/reportlab/pptx/matplotlib) imported **lazily inside functions** so package import + CPU tests run on core deps only.

```
src/dococr/
  config.py          dataclasses + YAML loader; paths from env (*_ARTIFACTS_DIR/_MODEL_DIR/_RUN_DIR, HF_HOME);
                     D1–D4 thresholds + model/engine/pipeline versions live here; unknown-key tolerant.
  cli.py             single argparse entrypoint → all commands (run, train, eval, benchmark, serve, autopilot);
                     console_script `ocr-agent = dococr.cli:main`. Logs to stderr (stdout stays pipeable JSON).
  logging_utils.py   stderr logging, ToolTrace audit helpers.

  data/              dataset loaders + the SYNTHETIC NOISE GENERATOR (the primary data engine):
                       pleias_loader.py  → load_dataset("PleIAs/Post-OCR-Correction","english"); chunk pages ≤300 bytes.
                       icdar_parser.py   → manual parse of ICDAR-2019 eng ([OCR_toInput]/[GS_aligned]); held-out gold.
                       wikitext_source.py→ clean feedstock (wikitext-103-raw-v1).
                       noise_generator.py→ seeded confusion-set corruption (sub/ins/del/space/case), per-epoch transform,
                                           error_rate calibrated to real CER; real/synthetic blend (0.4/0.6).
                       layout_data.py    → pierreguillou/DocLayNet-base, funsd, sroie loaders.

  ocr/               front-end engines + layout + reading order (PRETRAINED/ALGORITHMIC, not trained):
                       engines.py        → Tesseract (image_to_data), docTR, TrOCR, easyocr adapters → uniform OCRResult.
                       router.py         → has_text_layer / born-digital vs scanned (D2); pymupdf + pdf2image ingest.
                       preprocess.py     → assess_quality + deskew/denoise/binarize/upscale (D1).
                       layout.py         → DocLayout-YOLO regions + types; Table-Transformer for table cells.
                       reading_order.py  → XY-cut + column clustering → reading_index.
                       assemble.py       → blocks → Markdown / structured JSON / plain text.

  models/            corrector_byt5.py  → load/save/generate for byt5-small (+ t5-small ablation); correct_text();
                       baselines.py     → identity + symspellpy dictionary baselines.

  training/          trainer.py         → Seq2SeqTrainer wiring (config dict §4.5), GPU auto-profile (H100/A100/L4/T4),
                                          resume via get_last_checkpoint, EarlyStopping on cer_reduction.
                       metrics.py       → jiwer CER/WER + cer_reduction + % improved/degraded safety metric.
                       data_module.py   → preprocess/collator, dedup + clean held-out eval split.

  agent/             fsm.py             → LOAD→PREPROCESS→ROUTE→LAYOUT→OCR→CORRECT→ASSEMBLE (+FLAG); D1–D4 rules.
                       tools.py         → typed tool contracts (§5.1), uniform run()->dict, never raise past orchestrator.
                       llm_brain.py     → anthropic advisory call, schema-validated, same-rule re-check + deterministic fallback.

  api/               app.py             → FastAPI (/ocr, /batch, /stream, /healthz, /version) + job queue hooks.
                       gradio_app.py    → Gradio UI (Markdown/Overlay/JSON/diff tabs); imports api backend fns.
                       schemas.py       → pydantic response model (blocks[], confidence, meta.*_version).

  analysis/          error analysis (CER/WER by error type, substitution/insertion/deletion breakdown via jiwer),
                     baseline-vs-model + ablation tables, layout/reading-order eval.
  autoreport/        one-button report.pdf + slides.pptx + grading checklist + zip (lazy reportlab/pptx/matplotlib).
  monitoring/        latency/confidence/needs_review-rate metrics; drift on CER over time.
  automation/        autopilot: download→synth-data→train→eval→benchmark→error-analysis→report (one command).
  grading/           self-grade checklist against the assignment rubric.
```

Supporting (mirrors P02): `configs/*.yaml` (D1–D4 thresholds, model revs), `data/` (download scripts, no large data), `models/` (gitignored), `tests/` (CPU-only via graceful fallbacks), `docs/` (14 md files incl. `DESIGN_BRIEF.md`), `notebooks/` (H100 autopilot `.ipynb` + `COLAB_GUIDE.md`), `app/` (gradio), `deploy/` (HF Space: `app.py`+`requirements.txt`+`packages.txt`), `sample_data/`, `Dockerfile` + `docker-compose.yml` + `Makefile` + `pyproject.toml` + `requirements.txt` + `requirements_colab.txt` + `.github/workflows/ci.yml` + `README.md`.

**Build target dir:** `D:\NLP Industry Projects\P07-Document-OCR\` (per the P02 exemplar pattern).
