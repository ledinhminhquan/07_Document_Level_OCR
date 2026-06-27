# P07 Document-Level OCR — System Architecture

**Project:** Document-Level OCR System — full-page document image/PDF → clean, structured, reading-order-correct text (Markdown + JSON blocks + plain text).
**Author:** Le Dinh Minh Quan (student 23127460) · NLP in Industry final assignment
**Package:** `src/dococr/` (mirrors the P02–P06 template)

---

## 1. Design thesis

A flat OCR pass produces an unordered, error-riddled text dump: wrong reading order on multi-column pages, no block typing (title / paragraph / list / table), and char-level OCR garble (`rn`→`m`, `0`↔`O`, `1`↔`l`, merged/split words). P07 attacks both failures:

- **Structure** comes from a *layout + reading-order* stage (algorithmic / pretrained).
- **Accuracy** comes from a *trainable post-OCR corrector* — an NLP seq2seq model that measurably lowers CER/WER on top of **any** OCR engine.

The OCR front-end and layout logic are pretrained or algorithmic. The **trained differentiator is the post-OCR corrector** (`google/byt5-small`, char/byte-level, robust to char-level noise). Everything is built so the pipeline still runs when heavy dependencies are missing.

---

## 2. End-to-end component diagram

```
                            ┌───────────────────────────────────────────────────────────┐
   image / PDF / raw text   │                        CLIENTS                              │
   ───────────────────────► │  FastAPI (api/main.py)   Gradio UI (api/ui.py)   CLI (cli.py)│
                            │        \________________   __|___________   _____/           │
                            └─────────────────────────\─/──────────────\─/─────────────────┘
                                                       │   shared via api/dependencies.py
                                                       ▼
                            ┌───────────────────────────────────────────────────────────┐
                            │              DocAgent  (agent/doc_agent.py)                 │
                            │  deterministic FSM + optional LLM brain · 4 decision points │
                            │                                                             │
                            │  ingest ─► layout ─► correct ─► assemble  (+ audit trace)   │
                            └───┬──────────┬──────────────┬───────────────┬──────────────┘
                                │          │              │               │
                 ┌──────────────▼───┐  ┌───▼───────────┐  │        ┌──────▼───────────┐
                 │ INGEST           │  │ LAYOUT        │  │        │ ASSEMBLE         │
                 │ ocr/preprocess.py│  │ ocr/layout.py │  │        │ blocks→reading   │
                 │ fitz (PyMuPDF):  │  │ blocks, class │  │        │ order →          │
                 │  D2 born-digital │  │ XY-cut order  │  │        │ Markdown + JSON  │
                 │  vs scanned      │  │ D1 quality    │  │        │ + plain text     │
                 └───────┬──────────┘  └───┬───────────┘  │        └──────────────────┘
            born-digital │   scanned       │              │
            (text layer) │   (raster@DPI)  ▼              ▼
                         │           ┌──────────────┐  ┌────────────────────────────┐
                         │           │ OCR ENGINE   │  │ CORRECTOR                  │
                         │           │ models/      │  │ models/corrector.py        │
                         │           │ ocr_engine.py│  │  ByT5  "correct: " seq2seq │
                         │           │ Tesseract /  │  │  D4 bounded-edit gate      │
                         │           │ doctr / …    │  │  baseline.py (identity /   │
                         │           │ STUB fallback│  │  SymSpell) fallback        │
                         │           │ D3 conf gate │  └─────────────▲──────────────┘
                         │           └──────────────┘                │
                         └──── skip OCR entirely ───────────────────►│ (text only)
                                                                     │
                            ┌────────────────────────────────────────┴───────────────┐
                            │   MODEL REGISTRY  models/model_registry.py              │
                            │   model_meta.json + latest pointer (repo@revision)      │
                            │   training/  ──writes──►  registry  ──loads──►  agent   │
                            └─────────────────────────────────────────────────────────┘
```

**Decision points (FSM):** D1 page quality/preprocess routing · D2 born-digital vs scanned · D3 OCR-confidence gate (human-review flag) · D4 correction acceptance (bounded edit). Every step is timed and traced (`ToolTrace`), every decision recorded (`Decision`), emitted as `manifest.json`.

---

## 3. Data flow

```
ingest ──► layout (preprocess → OCR → reading-order → classify) ──► correct ──► assemble
```

1. **Ingest.** PDFs open with PyMuPDF (`fitz`). Each page is tested for a text layer: pages with `>= threshold` characters are **born-digital** (read directly, no OCR — D2); pages below threshold are **scanned** and rasterized at DPI. Image files (png/jpg) always take the scanned path.
2. **Layout.**
   - **D1 quality** — scanned pages get a quality score (Laplacian-variance blur + ink ratio + contrast) → `ok` / `reprocess` / `degraded` (degraded lowers the D3 bar).
   - **OCR** — scanned regions go through the OCR engine (default Tesseract `image_to_data` → per-word boxes, conf 0–100, block/par/line hierarchy). **D3** flags any region below confidence `0.55` for human review.
   - **Reading order** — XY-cut: detect a vertical gap near page centre → 2-column sort (column-then-top), else top-down.
   - **Classify** — heuristic block typing: heading / paragraph / list / header_footer / blank.
3. **Correct.** Each text region is corrected by the ByT5 seq2seq model (prefix `"correct: "`). **D4** accepts the output only if it is a *bounded* edit (edit ratio `<= 0.35`) and confident enough; otherwise the raw OCR text is kept (anti-hallucination).
4. **Assemble.** Blocks are emitted in reading order as **Markdown** (`##` headings, `-` lists) + **JSON** blocks (`type`, `bbox`, `text`, `reading_index`) + **plain text**.

---

## 4. Repository module map (`src/dococr/`)

| Module | Responsibility |
|---|---|
| `config.py` | Central config: paths, thresholds (text-layer, D3 `0.55`, D4 `0.35`), Drive/env wiring. |
| `cli.py` | Console-script `dococr`: `data, synth, train, tune, evaluate, ocr, correct, demo-agent, serve, benchmark, error-analysis, robustness, monitor, generate-report, generate-slides, autopilot, grade`. |
| `logging_utils.py` | Structured logging. |
| `data/` | `corpus.py`, `ocr_noise.py` (synthetic OCR-noise generator), `dataset.py`, `samples.py`, `download_dataset.py`. |
| `models/` | `text_utils.py`, `corrector.py` (ByT5 seq2seq), `baseline.py` (identity / SymSpell), `ocr_engine.py` (Tesseract/doctr/…/stub), `model_registry.py` (meta + latest pointer). |
| `ocr/` | `preprocess.py` (ingest, born-digital vs scanned, quality), `layout.py` (blocks, classification, XY-cut reading order). |
| `training/` | `train_corrector.py` (Seq2SeqTrainer), `evaluate.py`, `tune.py`, `metrics.py` (CER/WER reduction, safety gate). |
| `agent/` | `state.py`, `policy.py` (D1–D4), `tools.py` (`ToolTrace`/`Decision`), `llm_orchestrator.py` (optional LLM brain), `doc_agent.py` (FSM). |
| `api/` | `schemas.py`, `dependencies.py` (shared agent/model singletons), `main.py` (FastAPI), `ui.py` (Gradio), `app_combined.py` (mounts UI at `/ui`). |
| `analysis/` | `latency.py`, `error_analysis.py`, `robustness.py`. |
| `autoreport/`, `monitoring/`, `automation/`, `grading/` | report.pdf + slides.pptx, monitoring, autopilot, grade. |

Repo top level also has `configs/ data/ models/ tests/ docs/ notebooks/ app/ deploy/ Dockerfile docker-compose.yml Makefile pyproject.toml requirements*.txt README.md`.

---

## 5. Config, env-vars, and artifact wiring (Drive on Colab)

`config.py` is the single place that resolves where artifacts live. On Colab, model/checkpoint paths point at mounted **Drive** so training survives runtime resets and `resume via get_last_checkpoint` works across sessions; locally the same keys resolve to repo-relative `models/` and `data/`. Environment variables override config (data sizes, thresholds, GPU profile, optional LLM key). The optional LLM brain (`anthropic`) is **OFF by default**, so the default deployment makes **zero paid API calls** and runs on CPU.

---

## 6. Lazy-import & graceful degradation

Heavy dependencies are imported lazily inside the functions that need them, never at module import time, so the agent and the test-suite run on a bare interpreter. Optional vs. always-on:

| Component | Optional (heavy) dep | Fallback when absent |
|---|---|---|
| Image ops | `cv2` | `numpy` / `PIL` array paths |
| Corrector | `torch`, `transformers` | identity corrector (raw OCR passes through) |
| OCR | `pytesseract` (default), `doctr`/PaddleOCR/Surya/EasyOCR/TrOCR (upgrades) | **STUB engine** (empty result) |
| PDF | `fitz` (PyMuPDF) | images / pdf2image path |

**Consequences of this design:**

- With no OCR binary, the stub engine returns empty results so the agent still executes ingest → layout → correct → assemble (mirrors P05/P06).
- With no `torch`/`transformers`, the corrector degrades to **identity**, and D4 simply keeps raw text — the structured-output pipeline is unaffected.
- With no `cv2`, preprocessing/quality scoring fall back to numpy/PIL.

This was **validated offline**: the agent runs end-to-end with all 4 decisions firing, a born-digital doc produces 7 typed blocks with correct Markdown (`##` headings, `-` lists), and report.pdf + slides.pptx generate.

---

## 7. Born-digital vs scanned router (D2)

The router is the single biggest performance lever. PyMuPDF reads each page's text layer; if the character count meets the threshold the page is **born-digital** and OCR is **skipped entirely** (no rasterization, no engine call) — blocks come straight from PyMuPDF's text blocks. Pages below threshold (and all image inputs) are **scanned**: rasterized at DPI and sent through the OCR engine. When every page is born-digital, OCR is never invoked at all.

Latency impact: a born-digital page is ~**200 ms** (D2 skip); a scanned page is OCR-dominated (~0.6–1.2 s/region) plus correction (~80 ms/region for the small model).

---

## 8. Model registry: connecting training to the agent

`training/train_corrector.py` produces a fine-tuned ByT5 corrector and writes `model_meta.json` plus a **latest pointer** (`repo@revision`) via `models/model_registry.py`. At load time the agent / API resolve the corrector through the registry's latest pointer rather than a hard-coded path, so:

- **Versioning** is explicit — every corrector is identified by `repo@revision` in `model_meta.json`.
- **Rollout/rollback** is a pointer move, not a code change.
- The **agent, API, UI, and CLI all load the same registered model**, guaranteeing identical behaviour across surfaces.

```
training/  ──train──►  model_meta.json + latest pointer  ──resolve/load──►  DocAgent corrector
```

---

## 9. How API / UI / CLI share the agent

All three surfaces are thin wrappers over the **same `DocAgent`**, constructed once via `api/dependencies.py` (models loaded a single time at startup):

- **FastAPI** (`api/main.py`): `GET /healthz /readyz /version`; `POST /ocr` (image/PDF → `{full_text, markdown, blocks[], decisions, metrics}`); `POST /correct` (raw text → corrected, no image). Per-region status routes low-confidence regions to human review (D3).
- **Gradio UI** (`api/ui.py`): upload image/PDF **or** paste OCR text → structured Markdown + per-block table + decision log. `app_combined.py` mounts the UI at `/ui`.
- **CLI** (`cli.py`, console-script `dococr`): exposes `ocr`, `correct`, `demo-agent`, `serve`, etc. over the identical agent.

Because the agent is shared, a document processed via the API, the UI, or the CLI runs the exact same FSM, decision points, registry-resolved corrector, and audit trace.

---

## 10. Deployment surface

- **Docker:** `python:3.11-slim` + `tesseract-ocr` + `tesseract-ocr-eng` + `poppler-utils` + `libgl1`; `docker-compose`; HF Space (Gradio).
- **Scalability:** page/region parallelism + GPU batching for the corrector.
- **Versioning:** model registry (`model_meta.json` + latest pointer, `repo@revision`).

---

## 11. Trainable core (reference)

- **Model:** `google/byt5-small` (Apache-2.0, ~300M, char/byte-level — no SentencePiece). T4 fallback `google-t5/t5-small` (Apache-2.0, 60.5M). Prefix `"correct: "`.
- **Data:** synthetic OCR-noise generator (`data/ocr_noise.py`, default `char_error_rate=0.08`; train 60000 / val 4000 / test 4000) mixed with real **`PleIAs/Post-OCR-Correction`** config `english` (CC0-1.0, 31.3K rows, `text` + `corrected_text`). Optional benchmark `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` (loader degrades gracefully — viewer disabled).
- **Baseline to beat:** identity (raw OCR) — validated CER ~0.088, WER ~0.49, exact-match ~0.0005 on the synthetic test set; the trained ByT5 reduces CER substantially.
- **Headline metric:** % error reduction `(CER_before − CER_after)/CER_before` (same for WER), plus the **safety gate** (% improved vs degraded vs unchanged — degraded must be ≪ improved) and ExactMatch.

---

## 12. Risk controls baked into the architecture

- **Corrector making things worse / hallucinating** → D4 bounded-edit gate (`<= 0.35`) + the %-degraded safety metric.
- **PII privacy** → minimize retention/logging of raw images & text, on-prem/no-retention option, TTL cleanup.
- **Complex/rotated/dense pages** → D1 `degraded` path + flag; preprocessing handles skew/blur.
- **Sim-to-real gap** → report real PleIAs CER, mix real data into training.
- **Missing dependencies** → graceful degradation everywhere (stub OCR, identity corrector, numpy/PIL fallbacks).
