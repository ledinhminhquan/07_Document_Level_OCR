# Ethics & Responsible AI Statement — P07 Document-Level OCR

**Project:** Document-Level OCR System (full-page image/PDF → clean, structured, reading-order-correct text)
**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — final assignment
**Package:** `src/dococr/`

---

## 1. Scope and what this system actually does

P07 takes a full-page document image or PDF and produces clean, structured, reading-order-correct text as Markdown, JSON blocks, and plain text. It combines three parts:

1. A **pretrained / algorithmic OCR front-end** (default Tesseract via `pytesseract`; optional doctr / PaddleOCR / Surya / EasyOCR / TrOCR) — not trained by us.
2. **Layout + reading-order** logic (PyMuPDF ingestion, block grouping, heuristic block typing, XY-cut multi-column reading order) — algorithmic.
3. The **trainable differentiator: a post-OCR error corrector** — a seq2seq text→text model (`google/byt5-small`, Apache-2.0; T4 fallback `google-t5/t5-small`) that lowers Character/Word Error Rate (CER/WER) on top of any OCR engine.

Because the system both **transcribes** documents and **rewrites** the transcribed text, it raises two distinct ethical surfaces: privacy/consent around what gets transcribed, and faithfulness around what the corrector changes. This statement addresses both.

---

## 2. Who benefits

| Stakeholder | Benefit |
|---|---|
| **Archivists & libraries** | Digitize scanned collections into searchable, structured text with correct reading order on multi-column pages and typed blocks (heading/paragraph/list). |
| **Businesses digitizing records** | Convert back-catalogues of invoices, forms, and reports to machine-readable data; born-digital pages skip OCR entirely (D2 routing) to cut cost per page. |
| **Accessibility users** | Reading-order-correct, structured text is far more usable with screen readers than a raw, unordered OCR dump. |
| **Researchers** | Clean corpora for downstream NLP; reproducible synthetic-noise generator and reported real-corpus CER make results auditable. |

The headline metric is deliberately **error reduction** (% reduction in CER/WER versus the raw-OCR identity baseline), so the benefit claimed is the benefit measured.

---

## 3. Who could be harmed

- **People whose private documents are transcribed and made searchable without consent.** Documents routinely contain PII — names, IDs, addresses, financial data. Turning a scanned pile of paper into indexed, searchable text changes the privacy calculus entirely: data that was practically obscure becomes instantly queryable.
- **Anyone relying on a high-stakes document the corrector silently altered.** The post-OCR corrector is a generative seq2seq model. In legal, medical, or financial documents, a corrector that "fixes" a real but unusual string (a case number, a dosage, an account figure, a name) into something more "plausible" can erase or change meaning. A hallucinated correction in such a document is the single most dangerous failure mode of this system.
- **Subjects of mis-OCR'd records** where downstream decisions (eligibility, identity matching) are made on text that is wrong but looks confident.

---

## 4. Bias & fairness

This system is **English-first by design and by data**, and that is a real limitation:

- The corrector's training data is the synthetic OCR-noise generator (`src/dococr/data/ocr_noise.py`) over **clean English text**, mixed with one real corpus, **PleIAs/Post-OCR-Correction** config `english` (CC0-1.0, 31.3K rows). Default synthetic sizes: train 60,000 / val 4,000 / test 4,000.
- A model trained mostly on **synthetic noise plus a single real English corpus** will likely **underperform on other languages, scripts, domains, and document genres** whose error distributions differ. The synthetic confusion model (`rn↔m`, `cl↔d`, `O↔0`, `l↔1`, `e↔c`, `S↔5`, `g↔q`, merges/splits, case flips at a tunable `char_error_rate`, default 0.08) encodes assumptions about Latin-script printed text.
- **Sim-to-real gap:** synthetic noise may not match a specific OCR engine's true error distribution. We mitigate by **mixing real PleIAs data** and by **reporting real-corpus CER alongside synthetic CER** rather than only synthetic numbers, so readers can see where the model is and is not validated.

**Honest disclosure:** do not deploy the corrector on non-English, non-Latin, or specialized-domain text without re-measuring CER reduction and the safety gate (Section 6) on that distribution. Equal headline accuracy across all inputs is **not** claimed.

---

## 5. Explainability for non-technical stakeholders

The system is built so a non-engineer (an archivist, a compliance reviewer, an auditor) can understand and challenge any output. Every page carries:

- **Per-region OCR confidence** — Tesseract reports per-word confidence (0–100); regions below the D3 threshold (**0.55**) are **flagged for human review** instead of being silently trusted.
- **A decision log** — the agent is a deterministic finite-state machine with four named decision points (D1 quality/preprocess routing, D2 born-digital vs scanned, D3 confidence gate, D4 correction acceptance). Each decision is recorded as a `Decision` and each step traced/timed (`ToolTrace`).
- **An edit-budget guard** — at D4 the corrector's output is accepted **only if** the edit is bounded (**edit ratio ≤ 0.35**) and confident enough; otherwise the **raw OCR text is kept**. A reviewer can see exactly when the model was allowed to change text and when it was overruled.
- **A full `manifest.json` audit trail** plus a per-block table (type, bbox, text, reading_index) in the Gradio UI and `/ocr` API response.

This means a stakeholder can answer "why does this say what it says?" without reading model internals: which regions were trusted, which were flagged, and where edits were and were not applied.

---

## 6. The corrector-altering-meaning risk and how it is mitigated

This is the core responsible-AI concern for P07, so it is called out explicitly.

**Risk:** a generative corrector can hallucinate — rewriting a region into fluent text that no longer reflects the source. In a high-stakes document this is worse than leaving an OCR error in place.

**Mitigations, in layers:**

1. **Bounded edit budget (D4).** Corrections are accepted only when `edit_ratio ≤ 0.35`. A correction that would rewrite more than ~a third of the region is rejected and the raw OCR is retained — a corrector structurally **cannot rewrite a region away**.
2. **The %-degraded safety gate (metric).** Headline error reduction is not enough. We report, per evaluation set, the share of examples **improved vs. degraded vs. unchanged**. A corrector that fixes 40% but breaks 30% is treated as a failure, not a success: we **require degraded ≪ improved**. ExactMatch is reported alongside CER/WER reduction.
3. **Human-review gate.** Low-confidence regions (D3 < 0.55) and degraded-quality pages (D1) are routed to human review rather than auto-accepted.
4. **Identity baseline kept honest.** The baseline to beat is **identity** (raw OCR, no correction). On the synthetic test distribution identity sits at CER ≈ 0.088 / WER ≈ 0.49 / exact-match ≈ 0.0005, so the corrector has a clear, measurable job — and any version that does not beat identity is rejected.

```
D4 acceptance rule (per region):
    accept corrected_text  IFF  edit_ratio(raw, corrected) <= 0.35
                                 AND confidence is sufficient
    else                          keep raw OCR text
```

The optional LLM brain (Anthropic) is **off by default** (zero paid API, CPU-runnable) and, when enabled, only assists flagged regions and is validated with fallback — it never gets an unbounded license to rewrite.

---

## 7. Misuse and safeguards

**Foreseeable misuse:**

- **Mass surveillance / bulk transcription of private documents** — turning seized, leaked, or scraped document piles into a searchable index of people's private records.
- **Scraping copyrighted books / paywalled material** at scale into clean machine-readable text.

**Safeguards built around the pipeline:**

| Safeguard | What it does |
|---|---|
| **Rights / consent checks** | OCR of private or copyrighted documents requires confirmed rights/consent before processing; this is stated as a precondition, not an afterthought. |
| **Audit manifest** | Every run emits `manifest.json` with timed traces and recorded decisions — processing is accountable and reviewable, not silent. |
| **Human-review gate** | D3 confidence gate (0.55) and D1 degraded path route uncertain content to people rather than auto-publishing it. |
| **Edit budget** | D4 bounds how much the corrector may change, limiting silent content alteration. |
| **Privacy-minimizing operation** | Minimize retention/logging of raw images and extracted text; on-prem / no-retention option; TTL cleanup. PII (names, IDs, addresses, financial data) is treated as sensitive by default. |

These are technical guardrails, not a substitute for policy: the operator remains responsible for lawful basis, consent, and copyright compliance. The system is designed to make those obligations **visible and auditable**, not to enforce them autonomously.

---

## 8. Robustness and graceful degradation

Responsible behavior under stress is part of the design:

- **Robustness reporting** across increasing `char_error_rate` levels so degradation is measured, not assumed.
- **OOD scans** (skew, blur) handled by preprocessing; the D1 quality score (Laplacian-variance blur + ink ratio + contrast) lowers the acceptance bar and flags degraded pages rather than over-trusting them.
- **Graceful degradation everywhere:** a stub OCR engine (empty result), an identity corrector, and numpy/PIL fallbacks when cv2/torch are absent — so the system fails safe and predictably rather than producing confident garbage.

---

## 9. Summary

P07 is built so that its two ethical risk surfaces — **what gets transcribed** (privacy/consent/copyright) and **what the corrector changes** (faithfulness in high-stakes text) — are constrained and observable. The bounded edit budget plus the %-improved-vs-degraded safety gate guard against meaning-altering hallucinations; per-region confidence, the decision log, the flagged-for-review path, and the audit manifest make every output explainable to non-technical stakeholders; and the English-first scope is disclosed honestly rather than overstated. The system is responsible **by construction**, but it does not replace human and legal judgment over rights, consent, and high-stakes review.
