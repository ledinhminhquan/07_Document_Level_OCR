# P07 Document-Level OCR — Agent Architecture

**Project:** Document-Level OCR System — full-page document image/PDF → clean, structured, reading-order-correct text (Markdown + JSON blocks + plain text).
**Author:** Le Dinh Minh Quan (student 23127460) · NLP in Industry final assignment
**Package:** `src/dococr/`

---

## 1. Why an agent, not a flat OCR pass

A single flat OCR pass over a full page produces an **unordered, error-riddled text dump**:

- **Wrong reading order** on multi-column pages (text from column 2 interleaved into column 1).
- **No block typing** — title, paragraph, list, table, and header/footer are all flattened into one blob.
- **Char-level OCR garble** — `rn→m`, `0↔O`, `1↔l`, merged/split words.

P07 fixes both halves of the problem with a **deterministic finite-state machine (FSM)**:

1. **Structure** comes from layout detection + reading-order recovery (XY-cut) + block classification.
2. **Quality** comes from a **trainable post-OCR corrector** (`google/byt5-small`, the ML core) that measurably lowers CER/WER on top of *any* OCR engine.

The OCR front-end (Tesseract / docTR / PaddleOCR / Surya) and the layout algorithms are **pretrained / algorithmic**; the corrector is the **trained differentiator**. The agent is what wires these together, makes 4 explicit routing decisions, and guards against the corrector hallucinating.

---

## 2. The FSM at a glance

```
                          ┌─────────────────────────────────────────────┐
   image / PDF  ─────────▶│  tool_ingest                                │
                          │  PyMuPDF opens PDF; rasterize or read text   │
                          └───────────────────┬─────────────────────────┘
                                              │
                                       ┌──────▼───────┐
                                       │  D2  route   │  born-digital vs scanned
                                       │ text layer?  │  (per page)
                                       └──┬────────┬──┘
                            born-digital  │        │  scanned
                          (text layer ≥   │        │  (image / no text layer)
                           threshold)     │        │
                                          │        ▼
                                          │  ┌───────────────┐
                                          │  │  D1 preprocess│  quality score:
                                          │  │  routing      │  blur + ink + contrast
                                          │  └──┬────┬────┬──┘
                                          │  ok │ rep│ deg│
                                          │     │  ▼ │    │  (degraded → lower D3 bar)
                                          │     │ enhance  │
                                          │     ▼    ▼     ▼
                                          │  ┌────────────────┐
                                          │  │  tool_layout   │  OCR (D3 gate) +
                                          │  │  OCR + regions │  block detect +
                                          │  └───────┬────────┘  classify
                                          │          │
                          PyMuPDF blocks  ▼          ▼  OCR words → blocks
                          ┌──────────────────────────────────────┐
                          │  READING ORDER  (XY-cut, 2-col aware) │
                          │  classify: heading/paragraph/list/    │
                          │  header_footer/blank                  │
                          └───────────────────┬──────────────────┘
                                              │  blocks in reading order
                                       ┌──────▼───────┐
                                       │  tool_correct│  per region:
                                       │  D4 accept?  │  ByT5 "correct: " ...
                                       │ edit ≤ 0.35  │  bounded-edit guard
                                       └──────┬───────┘  (+ optional LLM brain)
                                              │  accepted corrections / raw fallback
                                       ┌──────▼───────┐
                                       │ tool_assemble│  Markdown (## / -) +
                                       │              │  JSON blocks + plain text
                                       └──────┬───────┘
                                              ▼
                          {full_text, markdown, blocks[], decisions, metrics, manifest}
```

The pipeline is fixed and deterministic: **ingest → layout (preprocess + OCR + reading-order + classify) → correct → assemble**. The "intelligence" lives entirely in the **4 decision points D1–D4**, each of which has explicit inputs, thresholds, branches, and a safe fallback.

---

## 3. JobState — the context object

A single `JobState` (`agent/state.py`) is threaded through every tool and decision. It is the shared blackboard that makes the run auditable and resumable.

| Field | Meaning |
|---|---|
| `source` | original upload (image or PDF) + media type |
| `pages[]` | per-page record: `route` (born-digital / scanned), raster DPI, quality score |
| `regions[]` | detected blocks: `bbox`, `block_type`, `reading_index`, raw OCR text, OCR confidence, corrected text, region status |
| `decisions[]` | ordered list of `Decision` records (which of D1–D4, inputs, threshold, branch taken) |
| `traces[]` | ordered list of `ToolTrace` records (tool name, wall-clock duration, status) |
| `metrics` | aggregate CER/WER reduction, % regions auto-accepted vs flagged, ExactMatch |
| `manifest` | the final `manifest.json` snapshot of the whole run |

Nothing in the pipeline mutates global state — every tool reads from and writes back to `JobState`, which is what makes the **audit trace** (Section 7) complete and reproducible.

---

## 4. The four decision points (in detail)

### D1 — Page-quality / preprocess routing
*Applies to scanned pages only.* Computes a **quality score** from three cheap signals:

| Signal | Source | Reads on |
|---|---|---|
| Blur | Laplacian variance | focus / scan sharpness |
| Ink ratio | fraction of dark pixels | blank vs text-dense |
| Contrast | intensity spread | faded / washed-out scans |

**Branches:**
- `ok` → OCR as-is.
- `reprocess` → apply enhancement (deskew / denoise / binarize) then OCR.
- `degraded` → page is too poor to trust; OCR proceeds but **lowers the D3 confidence bar** so its regions are flagged more aggressively, and the page is marked for the human-review path.

**Fallback:** if `cv2` is unavailable the score is computed with a NumPy/PIL approximation; graceful degradation everywhere.

### D2 — Born-digital vs scanned routing
*Per page.* PDFs are opened with **PyMuPDF (fitz)**. If a page already carries a **text layer ≥ a character threshold**, it is **read directly** and **OCR is skipped entirely** — a huge speed win (~200 ms/page vs OCR-dominated). Images (png/jpg) and text-layer-less PDF pages take the **scanned** path. When *every* page is born-digital, the OCR engine is never loaded.

**Fallback:** any page where the text layer is below threshold is treated as scanned rather than risk an empty extraction.

### D3 — OCR-confidence gate
*Per region.* Tesseract's `image_to_data(output_type=DICT)` gives per-word confidence (0–100), aggregated to a per-region confidence in [0,1]. Regions below **threshold = 0.55** are **flagged for human review** (region status = review) rather than silently trusted. If D1 returned `degraded` for the page, the effective bar is raised so weak regions are caught.

**Fallback:** the offline **STUB engine** returns an empty result with confidence 0, so the whole agent + test suite runs with no OCR binary installed.

### D4 — Correction acceptance (the safety-critical gate)
This is the gate that stops the trainable corrector from **rewriting a region away or hallucinating**. For each region the ByT5 corrector produces a candidate; the candidate is **accepted only if BOTH** hold:

1. **Bounded edit:** normalized character edit ratio between raw OCR and candidate **≤ 0.35** (an *edit-budget guard* — a correction that changes more than ~a third of the characters is rejected as untrustworthy).
2. **Confident enough:** the region/correction confidence clears the acceptance bar.

If either check fails, the agent **keeps the raw OCR text** for that region. This is the single most important safeguard in the system: a corrector that fixes 40% of regions but breaks 30% is useless, so D4 plus the `% degraded` metric (Section 6 of the project metrics) jointly enforce *degraded ≪ improved*.

**Optional LLM brain:** for *flagged* regions only, an optional **Anthropic** LLM (`llm_orchestrator.py`) can be consulted; its output is **validated and falls back** to the bounded-edit rule if it fails the same guard. The LLM is **OFF by default** → **zero paid API calls** and a CPU-only default run.

---

## 5. Tool contracts

The four tools are pure functions over `JobState` (`agent/tools.py`); each emits a `ToolTrace` and zero or more `Decision` records.

| Tool | Input | Output (written to JobState) | Decisions raised |
|---|---|---|---|
| `tool_ingest` | upload bytes + media type | `pages[]` with route, raster (scanned) or text (born-digital) | **D2** |
| `tool_layout` | `pages[]` | `regions[]` with bbox, block_type, reading_index, raw OCR text + confidence | **D1**, **D3** |
| `tool_correct` | `regions[]` | corrected text per region (or raw kept) | **D4** (+ optional LLM) |
| `tool_assemble` | corrected `regions[]` | `full_text`, `markdown`, `blocks[]`, `metrics`, `manifest` | — |

`tool_layout` is the composite step that runs **preprocess (D1) + OCR (D3) + reading-order (XY-cut) + block classification** in sequence, which is what makes the system "document-level" rather than line-level.

---

## 6. Reading order & assembly

- **Reading order (XY-cut, multi-column aware):** detect a vertical whitespace gap near the page centre. If found → treat as **2 columns**, sort **column-then-top**; otherwise sort **top-down**. (Eval layout corpora: `pierreguillou/DocLayNet-base`, FUNSD, PubLayNet.)
- **Block classification (heuristic):** `heading` / `paragraph` / `list` / `header_footer` / `blank`.
- **Assembly:** blocks emitted in reading order →
  - **Markdown** (`##` for headings, `-` for list items),
  - **JSON blocks** (`type`, `bbox`, `text`, `reading_index`),
  - **plain text**.

---

## 7. Audit trace & manifest

Every step is **timed and traced**; every routing choice is **recorded**:

- **`ToolTrace`** — one per tool call: tool name, wall-clock duration, status. Used for the latency report.
- **`Decision`** — one per D1–D4 firing: which decision, the input values, the threshold, and the branch taken.
- **`manifest.json`** — the full end-of-run snapshot: source, per-page routes, per-region status, all decisions and traces, and the aggregate metrics.

This makes runs reproducible and explains *why* any region was flagged, corrected, or left raw — essential for the privacy/PII posture (minimize retention, on-prem/no-retention option, TTL cleanup).

**Validated offline:** the agent runs end-to-end (ingest → layout → correct → assemble), **all 4 decisions fire**, and a born-digital document produces **7 typed blocks** with correct Markdown (`##` headings, `-` lists).

---

## 8. Worked example — a 2-column scanned page

A scanned PDF page with a title spanning the full width and two text columns below it.

**1. Ingest (D2).** PyMuPDF opens the PDF. The page has **no text layer** → `route = scanned`. The page is rasterized at the configured DPI.
> `Decision(D2, text_layer_chars=0, threshold=…, branch=scanned)`

**2. Preprocess (D1).** Quality score: blur (Laplacian variance) acceptable, ink ratio normal, contrast slightly low → `branch = reprocess`; a light deskew/binarize is applied before OCR.
> `Decision(D1, blur=…, ink=…, contrast=…, branch=reprocess)`

**3. OCR + regions (D3).** Tesseract `image_to_data` returns per-word boxes + confidences. Words are grouped by block/par/line into regions. The full-width title region scores 0.91 (accepted); a column-2 paragraph scores 0.48 (< 0.55) → **flagged for review**.
> `Decision(D3, region=col2_para3, conf=0.48, threshold=0.55, branch=review)`

**4. Reading order.** XY-cut detects a vertical gap near the centre → **2 columns**. Blocks are sorted **column-then-top**: title → col-1 paragraphs (top→bottom) → col-2 paragraphs (top→bottom). Each block is classified (`heading` for the title, `paragraph`/`list` for the bodies).

**5. Correct (D4).** ByT5 runs `"correct: <raw region text>"` per region:
- Title `"lntroductlon"` → candidate `"Introduction"`; edit ratio ≈ 0.17 ≤ 0.35 **and** confident → **accepted**.
- A col-1 paragraph with `rn→m` slips → small bounded edits → **accepted**.
- The flagged col-2 paragraph: candidate rewrites > 50% of characters → edit ratio 0.52 > 0.35 → **rejected, raw OCR kept** (and the region stays in the review queue; the optional LLM brain, if enabled, would be consulted here and re-validated against the same guard).
> `Decision(D4, region=title, edit_ratio=0.17, branch=accept)`
> `Decision(D4, region=col2_para3, edit_ratio=0.52, threshold=0.35, branch=keep_raw)`

**6. Assemble.** Blocks emit in reading order:

```markdown
## Introduction

<col-1 paragraph 1, corrected>

<col-1 paragraph 2, corrected>

<col-2 paragraph 1, corrected>

<col-2 paragraph 3, RAW — flagged for review>
```

Plus JSON blocks (each with `type`, `bbox`, `text`, `reading_index`) and plain text. The `manifest.json` records every trace and all five decisions; `metrics` reports CER/WER **reduction** vs the identity baseline and the **% improved vs degraded** safety gate.

---

## 9. Design properties

- **Deterministic core, optional LLM edge.** The FSM is fully deterministic and CPU-runnable; the Anthropic LLM is an *optional* assistant for flagged regions only, off by default → 0 paid API.
- **Fail-safe by construction.** Stub OCR, identity corrector, and NumPy/PIL fallbacks (when `cv2`/`torch` are absent) mean every stage degrades gracefully rather than crashing.
- **Bounded, never destructive.** D4's edit-budget guard guarantees the corrector can only *improve within bounds* — when in doubt, the raw OCR text survives.
- **Fully auditable.** ToolTrace + Decision + manifest.json give a complete, reproducible record of what happened and why.
