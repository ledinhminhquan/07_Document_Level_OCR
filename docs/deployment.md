# Deployment — P07 Document-Level OCR

This document describes how the Document-Level OCR system is packaged, served, and operated. The system turns a full-page document image or PDF into clean, structured, reading-order-correct text — delivered as Markdown, a typed JSON block list, and plain text — and exposes the trained **post-OCR corrector** (`google/byt5-small`) both inside the full pipeline and as a standalone text endpoint.

- **Author:** Le Dinh Minh Quan (student 23127460)
- **Course:** NLP in Industry — final assignment
- **Package:** `src/dococr/`

---

## 1. Delivery formats

The same inference pipeline is shipped behind five interfaces so it can be consumed by services, humans, scripts, and managed hosting alike.

| Format | Entry point | Purpose |
|--------|-------------|---------|
| **FastAPI service** | `src/dococr/api/main.py` | Programmatic HTTP access — full document OCR and standalone text correction |
| **Gradio UI** | `src/dococr/api/ui.py` | Interactive browser app — upload a document or paste OCR text, see structured output |
| **Combined app** | `src/dococr/api/app_combined.py` | FastAPI service with the Gradio UI mounted at `/ui` |
| **CLI** | `src/dococr/cli.py` (console-script `dococr`) | Batch and operational tasks (training, evaluation, OCR, serving, reporting) |
| **Docker / Compose** | `Dockerfile`, `docker-compose.yml` | Reproducible container with the OCR system binaries bundled |
| **Hugging Face Space** | Gradio app | Public managed demo |

Models are **loaded once at process startup** and reused across requests, so per-request cost is inference only — not model loading.

---

## 2. FastAPI service

### Endpoints

| Method | Path | Input | Output |
|--------|------|-------|--------|
| `GET` | `/healthz` | — | Liveness probe |
| `GET` | `/readyz` | — | Readiness probe (models loaded) |
| `GET` | `/version` | — | Model/version metadata |
| `POST` | `/ocr` | Uploaded image or PDF | `{full_text, markdown, blocks[], decisions, metrics}` |
| `POST` | `/correct` | Raw text (no image) | Corrected text |

`/ocr` runs the complete agentic pipeline (ingest → layout → correct → assemble). `/correct` exposes only the trained seq2seq corrector for callers that already have OCR text from another engine and just want the error-reduction step — the corrector improves CER/WER on top of **any** OCR front-end.

The service marks **per-region status**: regions whose OCR confidence falls below the gate are flagged in the response so a downstream system can route them to human review rather than trusting them silently.

### `/ocr` response shape

```json
{
  "full_text": "Plain reading-order text of the whole document",
  "markdown": "## Heading\n\nParagraph text\n\n- list item\n- list item",
  "blocks": [
    {
      "type": "heading",
      "bbox": [x0, y0, x1, y1],
      "text": "Heading text",
      "reading_index": 0
    }
  ],
  "decisions": [
    { "point": "D2", "choice": "born_digital_skip_ocr" }
  ],
  "metrics": { }
}
```

- **`blocks[]`** carries the structure: each block has a `type` (`heading` / `paragraph` / `list` / `header_footer` / `blank`), its bounding box, the text, and a `reading_index` produced by the XY-cut reading-order pass (multi-column aware).
- **`decisions`** is the agent audit trail — the four decision points (D1–D4) that fired for this document, so an operator can see *why* the pipeline behaved as it did.

---

## 3. Inference pipeline

Inputs are **images** (PNG/JPG) or **PDFs**; outputs are always the `{full_text, markdown, blocks[], decisions, metrics}` bundle. The pipeline is a deterministic finite-state machine with four decision points:

```
ingest → layout (preprocess → OCR → reading-order → classify) → correct → assemble
```

1. **Ingest** — PDFs are opened with PyMuPDF (`fitz`). A page with a sufficient text layer is **born-digital** and read directly; a page below the threshold is rasterized for OCR. Image inputs always take the scanned path.
2. **Layout** — words are grouped into regions/blocks (Tesseract block/par/line hierarchy, or PyMuPDF blocks for born-digital), classified heuristically, and ordered by **XY-cut** (detects a vertical centre gap → two columns sorted column-then-top; otherwise top-down).
3. **Correct** — each region's text passes through the trained corrector with the `"correct: "` prefix.
4. **Assemble** — blocks are emitted in reading order as Markdown (`##` headings, `-` lists), the JSON block list, and plain text.

### Agent decision points

| Point | Decision | Effect |
|-------|----------|--------|
| **D1** | Page-quality / preprocess routing | Blur + ink-ratio + contrast score → `ok` / `reprocess` / `degraded` (lowers the D3 bar) |
| **D2** | Born-digital vs scanned | A PDF text layer **skips OCR entirely** — the main speed win |
| **D3** | OCR-confidence gate | Per-region confidence below **0.55** → flag region for human review |
| **D4** | Correction acceptance | Accept the corrector's output only if it is a **bounded edit (edit ratio ≤ 0.35)** and confident enough, so it cannot hallucinate a region away; otherwise keep raw OCR |

Every step is timed and traced (`ToolTrace`), every decision is recorded (`Decision`), and the full run is captured in `manifest.json`. The optional LLM brain (Anthropic) for flagged regions is **off by default**, so the default deployment makes **zero paid API calls and runs on CPU**.

---

## 4. User interaction

- **HTTP clients** `POST /ocr` with a multipart file (image or PDF) and receive the structured JSON, or `POST /correct` with raw text to use the corrector alone. Health, readiness, and version are plain `GET` calls suitable for orchestration probes.
- **Gradio UI** (`api/ui.py`) lets a user **upload an image/PDF OR paste OCR text**, and renders the structured Markdown, a per-block table, and the decision log. Mounted at `/ui` in the combined app.
- **CLI** (`dococr`) covers the full lifecycle: `data`, `synth`, `train`, `tune`, `evaluate`, `ocr`, `correct`, `demo-agent`, `serve`, `benchmark`, `error-analysis`, `robustness`, `monitor`, `generate-report`, `generate-slides`, `autopilot`, `grade`. Use `dococr ocr` for single/batch document processing and `dococr serve` to launch the API.

---

## 5. Latency

| Path | Typical latency | Why |
|------|-----------------|-----|
| Born-digital page | **~200 ms** | D2 skips OCR entirely — text is read straight from the PDF layer |
| Scanned page | **OCR-dominated, ~0.6–1.2 s / region** | The OCR engine is the bottleneck on rasterized pages |
| Post-OCR correction | **~80 ms / region** | Small seq2seq model running per region |

The single biggest lever is **D2**: documents that arrive as born-digital PDFs avoid the OCR cost altogether, which is also why per-page cost stays low.

---

## 6. Scalability

- **Page- and region-level parallelism** — pages and the regions within them are independent units of work and can be processed concurrently.
- **GPU batching** — when a GPU is present, correction requests across regions are batched through the seq2seq model.
- **Models loaded once** — the corrector and OCR front-end are initialized at startup and shared across all requests, so throughput scales without repeated load cost.
- **CPU-able default** — the default stack (Tesseract + ByT5 corrector, LLM off) runs without a GPU, so horizontal scaling is just adding stateless replicas behind a load balancer.

---

## 7. Model versioning

Versioning is handled by a **model registry** (`src/dococr/models/model_registry.py`):

- Each trained corrector is recorded with a `model_meta.json` plus a **`latest` pointer**, so deployments can pin a specific build or always follow the newest.
- Pretrained components are referenced by **`repo@revision`** (e.g. `google/byt5-small` at a fixed revision), making the exact weights reproducible.
- `GET /version` surfaces the active model metadata so a running service can be audited against the registry.

---

## 8. Docker

The container is built on **`python:3.11-slim`** with the OCR system dependencies installed, since the OCR front-end needs native binaries that are not pip-installable:

| System package | Reason |
|----------------|--------|
| `tesseract-ocr` + `tesseract-ocr-eng` | Default OCR engine (via pytesseract) + English language data |
| `poppler-utils` | PDF rasterization support |
| `libgl1` | Native dependency for image processing (OpenCV) |

`docker-compose.yml` orchestrates the service, and the Hugging Face **Space** ships the Gradio UI for a public managed demo.

```bash
# Build and run the API + UI locally
docker compose up --build
# or directly
docker build -t dococr .
docker run -p 8000:8000 dococr
```

---

## 9. Deployment challenges and limitations

- **OCR engine installation.** The default engine relies on native binaries (`tesseract-ocr-eng`, `poppler-utils`, `libgl1`) that must be present in the runtime image — this is the main reason a slim-Python base alone is insufficient and the container bundles them explicitly. For environments without an OCR binary, a **stub engine** (empty result) keeps the agent and tests runnable, and fully born-digital documents skip OCR entirely.
- **Complex layouts.** Dense tables, figures, and rotated/skewed scans can defeat the heuristic block classifier and XY-cut reading order. The **D1 degraded path** flags such pages and lowers the confidence bar rather than emitting confidently-wrong structure; low-confidence regions are routed to human review via the **D3 gate (threshold 0.55)**.
- **VRAM.** GPU batching of the corrector is bounded by available VRAM; profiles scale per device (large batches on H100/A100, gradient-accumulated smaller batches on L4/T4, with the base model switchable to `t5-small` on T4). The CPU-only default avoids the constraint at the cost of throughput.
- **Corrector safety.** A post-OCR corrector that rewrites or hallucinates is the principal risk. The **D4 bounded-edit gate (edit ratio ≤ 0.35)** plus the reported %-degraded safety metric ensure the corrector only applies bounded, confident edits and otherwise falls back to raw OCR.
- **Privacy.** Documents may contain PII (names, IDs, addresses, financial data). Deployments should minimize retention and logging of raw images and text, offer an on-prem / no-retention mode, and apply TTL cleanup.

---

## 10. Validated end-to-end

Offline (no OCR binary, LLM off) the system runs the full pipeline end-to-end — ingest → layout → correct → assemble — with **all four decisions firing**, and a born-digital document produces **7 typed blocks with correct Markdown** (`##` headings, `-` lists). The reporting commands (`generate-report`, `generate-slides`) produce `report.pdf` and `slides.pptx`.
