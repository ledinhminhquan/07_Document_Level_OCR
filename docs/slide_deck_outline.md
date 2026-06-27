# P07 — Document-Level OCR System: Slide-Deck Outline

**Author:** Le Dinh Minh Quan (23127460) · **Course:** NLP in Industry — Final Assignment
**Project:** Full-page document image/PDF → clean, structured, reading-order-correct text (Markdown + JSON blocks + plain text)

This outline maps the presentation to both the written report and the assignment's required slide list. Target length: **13 slides** (within the 10–15 range). Each slide lists a title, 3–6 concise bullets, and a note on the visual/diagram to render.

---

## Slide 1 — Title & Project Info

- **Document-Level OCR System** — full-page image/PDF to clean, structured text
- Author: **Le Dinh Minh Quan**, student **23127460**
- Course: NLP in Industry — Final Assignment
- One-line pitch: *layout + reading order give STRUCTURE; a trainable post-OCR corrector measurably lowers CER/WER on top of any OCR engine*
- Package: `src/dococr/` (mirrors the P02–P06 template)

> **Visual:** Title card with a clean before/after teaser — a noisy scanned page on the left, structured Markdown blocks on the right.

---

## Slide 2 — Business Problem & Motivation

- A **flat OCR pass** produces an unordered, error-riddled text dump
- Three concrete failures it leaves behind:
  - **Wrong reading order** on multi-column pages
  - **No block typing** (title / paragraph / list / table)
  - **Char-level OCR garble** — `rn→m`, `0↔O`, `1↔l`, merged/split words
- Downstream NLP (search, RAG, extraction) inherits all of these errors
- Manual review of OCR output is slow and expensive at document scale
- **The job:** restore STRUCTURE *and* measurably reduce character/word error

> **Visual:** Split panel — raw OCR text dump (red error highlights) vs. the same page correctly structured and ordered.

---

## Slide 3 — Proposed NLP Solution

- **OCR front-end** (pretrained/algorithmic): Tesseract via `pytesseract` → per-word boxes, confidence, block/line hierarchy
- **Layout + reading order** (algorithmic): block detection, heuristic classification, XY-cut ordering → STRUCTURE
- **Post-OCR corrector** (the **trained** differentiator): a seq2seq NLP model that fixes char-level OCR errors on top of *any* OCR engine
- **Agent** (mandatory agentic component): a deterministic FSM with 4 decision points that routes pages and gates corrections
- Key framing: the OCR + layout are pretrained/algorithmic; **the trained ML core is the corrector**

> **Visual:** Four labeled lanes (OCR → Layout → Corrector → Agent), with the Corrector lane highlighted as "the trained model."

---

## Slide 4 — System Architecture Diagram

- Pipeline: **ingest → layout (preprocess + OCR + reading-order + classify) → correct → assemble**
- **Ingest** (PyMuPDF): born-digital pages read directly; scanned pages rasterized at DPI; images → scanned path
- **Layout**: group words into blocks → classify (heading / paragraph / list / header_footer / blank) → XY-cut order
- **Correct**: ByT5 post-OCR corrector, gated by the agent
- **Assemble**: blocks in reading order → Markdown + JSON blocks (type, bbox, text, reading_index) + plain text
- Every step timed/traced (`ToolTrace`); every decision recorded (`Decision`) → `manifest.json`

> **Visual:** End-to-end block diagram with the 4 agent decision points (D1–D4) drawn as diamonds on the flow.

---

## Slide 5 — Data Overview

- **PRIMARY (synthetic):** reproducible OCR-noise generator `src/dococr/data/ocr_noise.py`
  - Confusion sets `rn↔m`, `cl↔d`, `O↔0`, `l↔1`, `e↔c`, `S↔5`, `g↔q`, plus insert/delete, merge/split, case flips
  - Tunable `char_error_rate` (default **0.08**); default sizes **train 60,000 / val 4,000 / test 4,000**
- **REAL mix:** `PleIAs/Post-OCR-Correction` config `english` (CC0-1.0, **31.3K rows**; `text` = noisy OCR, `corrected_text` = gold)
- **Optional benchmark:** `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` (viewer disabled → loader degrades gracefully)
- Splits are **leakage-free** (dedup by clean text); a real eval slice is held out

> **Visual:** Table comparing Synthetic vs. PleIAs (license, size, columns, role) + a sample noisy→gold pair.

---

## Slide 6 — Model & Evaluation Results

- **Model:** `google/byt5-small` (Apache-2.0, ~300M, byte-level → robust to char-level OCR noise; no SentencePiece). Prefix `"correct: "`. T4 fallback: `google-t5/t5-small`
- **Baseline to beat:** **identity** (raw OCR, no correction); optional SymSpell dictionary corrector
- **Validated identity baseline** on synthetic test: CER ≈ **0.088**, WER ≈ **0.49**, ExactMatch ≈ **0.0005** → a clear, measurable job
- **Headline metric = ERROR REDUCTION:** `% reduction = (CER_before − CER_after) / CER_before` (same for WER)
- The trained ByT5 reduces CER **substantially** below identity
- **Safety gate:** report **% improved vs. degraded vs. unchanged** — require degraded ≪ improved (a corrector that fixes 40% but breaks 30% is useless)

> **Visual:** Bar chart of CER/WER before vs. after + a stacked "improved / unchanged / degraded" safety-gate bar.

---

## Slide 7 — Training Setup

- HF `Seq2SeqTrainer`, `predict_with_generate`; **bf16 + tf32** on H100/A100, **fp16** on T4
- LR **5e-4** (ByT5) / **3e-4** (t5-small); effective batch **256** (per-device 32 × grad-accum 8 on H100)
- Cosine schedule, warmup 0.05, weight decay 0.01, label smoothing 0.1
- Early stopping (patience 4 on CER, lower is better), `load_best_model_at_end`, `group_by_length`, resume via `get_last_checkpoint`
- **Anti-overfitting:** diverse confusion model + real PleIAs mix + early stopping + weight decay
- GPU profiles: H100 bf16 bs32/accum8; A100-40 bf16 bs16/accum16; L4 bf16 bs8/accum32; T4 fp16 bs4/accum64 (+ switch base to t5-small). Rough ByT5-small: **~3–6h on one H100**

> **Visual:** Table of GPU profiles (GPU, precision, batch, grad-accum) + training-curve sketch (CER ↓ over steps).

---

## Slide 8 — Agentic AI Component (FSM + D1–D4)

- Deterministic **FSM** with an optional LLM brain; **4 decision points**:
  - **D1 — page quality / preprocess routing:** Laplacian-variance blur + ink ratio + contrast → ok / reprocess / degraded
  - **D2 — born-digital vs. scanned:** PDFs with a text layer **skip OCR entirely**; scanned → OCR
  - **D3 — OCR-confidence gate:** per-region confidence below **0.55** → flag region for human review
  - **D4 — correction acceptance:** accept the corrector's output **only if it's a BOUNDED edit (edit ratio ≤ 0.35)** and confident enough → a corrector cannot hallucinate/rewrite a region away
- Optional LLM brain (`anthropic`) for flagged regions, validates + falls back; **OFF by default → 0 paid API, runs on CPU**

> **Visual:** FSM state diagram with D1–D4 as labeled diamonds; spotlight the **D4 edit-budget gate** (≤ 0.35) blocking an over-edit.

---

## Slide 9 — Deployment Overview

- **FastAPI** (`api/main.py`): `GET /healthz /readyz /version`; `POST /ocr` (image/PDF → `full_text`, `markdown`, `blocks[]`, `decisions`, `metrics`); `POST /correct` (raw text → corrected). Models loaded once at startup
- **Gradio UI** (`api/ui.py`): upload image/PDF or paste OCR text → structured Markdown + per-block table + decision log; combined app mounts UI at `/ui`
- **CLI** (`dococr`): data, synth, train, tune, evaluate, ocr, correct, demo-agent, serve, benchmark, error-analysis, robustness, monitor, generate-report, generate-slides, autopilot, grade
- **Docker:** `python:3.11-slim` + `tesseract-ocr` + `tesseract-ocr-eng` + `poppler-utils` + `libgl1`; docker-compose; HF Space (Gradio)
- **Latency:** born-digital page **~200ms** (D2 skip); scanned ~OCR-dominated (~0.6–1.2s/region); correction **~80ms/region**

> **Visual:** Deployment topology (CLI / FastAPI / Gradio / Docker) with the **D2 "born-digital skip"** fast path called out.

---

## Slide 10 — Live Demo (Validated Offline)

- Agent runs **end-to-end** offline: ingest → layout → correct → assemble
- **All 4 decisions fire** on a single document
- A born-digital doc produces **7 typed blocks** + correct Markdown (`##` headings, `-` lists)
- Graceful degradation: **stub OCR** engine + identity corrector + numpy/PIL fallbacks when cv2/torch absent
- `report.pdf` + `slides.pptx` generate from the CLI

> **Visual:** Screenshot of the Gradio UI — uploaded page on the left, structured Markdown + per-block table + decision log on the right.

---

## Slide 11 — Ethics, Privacy & Risks

- **Corrector makes things WORSE** (the main risk): guarded by the **D4 edit-budget gate** + the **%-degraded** metric; keep raw OCR if the edit is too large
- **Hallucinated corrections:** bounded edit ratio (≤ 0.35) prevents rewriting a region away
- **Privacy / PII:** documents may contain names, IDs, addresses, financial data → minimize retention/logging of raw images & text; on-prem / no-retention option; TTL cleanup
- **Rights/consent:** OCR of private or copyright documents needs proper rights
- **Sim-to-real gap:** synthetic noise may not match a specific engine → also report **real PleIAs CER** and mix real data
- **Scope:** English-first (PleIAs english + English synthetic); complex pages (dense tables, rotated scans) → D1 degraded path + flag

> **Visual:** Risk matrix (risk × mitigation) with the corrector-makes-worse row highlighted.

---

## Slide 12 — Robustness & Success Metrics

- **Robustness report** across increasing `char_error_rate` levels — noise severity vs. CER reduction
- OOD scans (skew / blur) handled by preprocessing; graceful degradation everywhere
- **Business metrics:** manual-review reduction (% pages auto-accepted at the confidence gate); structure fidelity (% blocks correctly typed + ordered); cost per page (born-digital skips OCR; CPU-able default)
- **Technical metrics:** CER/WER **reduction** (headline), % improved vs. degraded (safety gate), ExactMatch, layout/reading-order correctness, latency per page

> **Visual:** Line chart — CER reduction vs. `char_error_rate`, plus a compact business/technical metrics scorecard.

---

## Slide 13 — Key Takeaways & Future Work

- **Takeaway 1:** Document-level structure (layout + XY-cut reading order) + a **trainable** post-OCR corrector beats a flat OCR pass on both structure and error rate
- **Takeaway 2:** The **D4 edit budget** + **%-degraded** safety gate make the corrector *trustworthy*, not just accurate
- **Takeaway 3:** Born-digital D2 skip + CPU-able default keep cost and latency low; everything degrades gracefully offline
- **Future work:** upgrade OCR front-end (python-doctr, PaddleOCR / PP-StructureV3, Surya — flag Surya's non-commercial license)
- **Future work:** layout/reading-order eval on DocLayNet / FUNSD / PubLayNet; multilingual beyond English-first

> **Visual:** Three-icon takeaways row + a short "next steps" roadmap strip.

---

### Required-slide coverage map

| Required slide | Covered by |
| --- | --- |
| Title & info (Le Dinh Minh Quan, 23127460) | Slide 1 |
| Business Problem & Motivation | Slide 2 |
| Proposed NLP Solution (OCR + layout + ByT5 corrector + agent) | Slide 3 |
| System Architecture Diagram | Slide 4 |
| Data Overview (synthetic + PleIAs CC0) | Slide 5 |
| Model & Evaluation Results (CER/WER, safety gate, baselines) | Slides 6–7 |
| Agentic AI Component (FSM + D1–D4, esp. D4 edit budget) | Slide 8 |
| Deployment Overview (FastAPI/Gradio/CLI/Docker, D2 skip) | Slides 9–10 |
| Ethics, Privacy & Risks | Slide 11 |
| Key Takeaways & Future Work | Slides 12–13 |
