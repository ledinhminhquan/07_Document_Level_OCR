# P07 — Document-Level OCR System: Problem Definition

**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — Final Assignment
**Package:** `src/dococr/`

---

## 1. Business Context & Motivation

Organizations sit on mountains of **scanned documents and PDFs** — contracts, invoices, forms, reports, archived correspondence, statements — that are effectively *invisible* to software. A page image is just pixels; you cannot search it, feed it to a retrieval-augmented (RAG) assistant, run analytics over it, audit it for compliance, or auto-fill a data-entry form from it. The first step in unlocking that value is converting the page into **clean, structured, reading-order-correct text**.

The P07 Document-Level OCR System takes a full-page document image or PDF and produces a faithful machine-readable rendering in three coordinated formats:

| Output | Purpose |
| --- | --- |
| **Markdown** | Human-readable structure (`##` headings, `-` lists) for review and rendering |
| **JSON blocks** | Machine-consumable blocks: `type`, `bbox`, `text`, `reading_index` — ideal for downstream pipelines, RAG chunking, and analytics |
| **Plain text** | A flat searchable/indexable stream |

These outputs feed the concrete business workflows that motivate the project:

- **Search & RAG** — structured, ordered text is what makes a document corpus retrievable and answerable.
- **Analytics** — typed blocks let you pull headings, tables, and fields out of unstructured scans.
- **Compliance & audit** — every step is timed and traced, and every routing decision is recorded in a `manifest.json` audit trail.
- **Data entry automation** — clean fields with confidence scores reduce manual keying.

### Target Users & Stakeholders

- **Operations / back-office teams** doing document intake who today re-key or proofread scans by hand.
- **Knowledge-management & RAG builders** who need a corpus of clean, chunkable document text.
- **Data & analytics teams** consuming the typed JSON blocks.
- **Compliance / audit functions** that require traceability of how each page was processed.
- **Platform / ML engineers** who deploy and operate the service (FastAPI + Gradio + CLI).

---

## 2. The Problem: Flat OCR Is Not Enough

Running a single, flat OCR pass over a page produces an **unordered, error-riddled text dump**. This is the core insight P07 is built around. The dump suffers from two distinct, compounding failures:

**(a) No structure.**
- **Wrong reading order** on multi-column pages — text from column 1 and column 2 gets interleaved into nonsense.
- **No block typing** — there is no distinction between a title, a paragraph, a list, a table, or a header/footer. Everything is one undifferentiated stream.

**(b) Character-level garble.**
OCR engines systematically confuse visually similar characters and mishandle word boundaries:

```
rn -> m        cl -> d       O <-> 0       l <-> 1
e <-> c        S <-> 5       g <-> q       merged/split words
```

These errors corrupt search hits, break downstream parsing, and force expensive manual review.

P07 solves **both** halves of the problem:

1. **Structure** comes from a layout + reading-order stage (region detection, block classification, XY-cut ordering) that turns pixels into typed, ordered blocks — this is what makes the system *document-level* rather than line-level.
2. **Error reduction** comes from a **trainable post-OCR corrector** that measurably lowers character and word error rates *on top of any OCR engine*.

The OCR front-end and layout analysis are **pretrained / algorithmic** components. The **trained differentiator** — the part this assignment builds and evaluates as its ML core — is the post-OCR corrector.

---

## 3. Why This Is an NLP Problem

The intelligence of the system is squarely in NLP, in three places:

### 3.1 The post-OCR corrector is a seq2seq language model
The headline trainable model frames **Post-OCR Error Correction as a text-to-text sequence-to-sequence task**: given noisy OCR text, generate the corrected text. The chosen model is **`google/byt5-small`** (Apache-2.0, ~300M params), a **char/byte-level** model that operates directly on bytes with **no SentencePiece tokenizer**. Byte-level modeling is deliberately chosen because it is robust to exactly the char-level noise OCR produces — the ByT5 paper (arXiv:2105.13626) shows byte models are markedly more noise-robust than subword models. Inputs are given the prefix `"correct: "`. A T4-class fallback swaps in `google-t5/t5-small` (Apache-2.0, 60.5M).

This is a language-modeling problem: the corrector must learn the statistics of well-formed English to decide that `rn` should have been `m` *in context*, while leaving a correct word untouched.

### 3.2 Layout & reading order are document-language structure
Recovering reading order on multi-column pages and classifying blocks (heading / paragraph / list / header_footer / blank) is the structural-NLP layer that turns a 2-D page into a linear, correctly-ordered document — the prerequisite for any search, RAG, or analytics consumer.

### 3.3 Agentic orchestration
A deterministic finite-state-machine agent (with an optional LLM brain) makes four explicit routing decisions per document — page-quality routing, born-digital vs. scanned routing, an OCR-confidence human-review gate, and a **bounded correction-acceptance gate** — and records every decision for audit. The corrector is only *trusted* when its edit is bounded, which is itself an NLP-quality judgment.

---

## 4. The Trainable Job, Quantified

The corrector has a clear, measurable target. On the synthetic OCR-noise test distribution, the **identity baseline** (raw OCR, no correction) — the canonical post-OCR baseline — performs as follows:

| Baseline (identity / raw OCR) | Value |
| --- | --- |
| Character Error Rate (CER) | ~0.088 |
| Word Error Rate (WER) | ~0.49 |
| Exact-match rate | ~0.0005 |

In other words, nearly half of all words and ~9% of characters are wrong, and almost no page comes out perfect. The trained ByT5 corrector must **reduce CER below this identity floor** — and it reduces CER substantially. Optional comparators are a **SymSpell** dictionary corrector (`symspellpy`, MIT) and `oliverguhr/spelling-correction-english-base` (MIT BART).

Training data is primarily a **reproducible synthetic OCR-noise generator** (`src/dococr/data/ocr_noise.py`) that corrupts clean English with a realistic confusion model (substitutions, insertions, deletions, word merge/split, case flips at a tunable `char_error_rate`, default 0.08), mixed with **real** data from **`PleIAs/Post-OCR-Correction`** config `english` (CC0-1.0, 31.3K rows; `text` = noisy OCR, `corrected_text` = gold). This real mix lets the project report real-world CER alongside synthetic results and guards against the sim-to-real gap.

---

## 5. Success Metrics

Metrics are split into business outcomes and technical measures. The **headline technical metric is error reduction**, paired with a **safety gate** that prevents a corrector from doing more harm than good.

### 5.1 Business Metrics

| Metric | Definition |
| --- | --- |
| **Manual-review reduction** | % of pages/regions auto-accepted at the OCR-confidence gate (fewer flagged for humans) |
| **Structure fidelity** | % of blocks correctly **typed and ordered** (right block type + right reading order) |
| **Cost per page** | Kept low: born-digital pages skip OCR entirely; the default pipeline runs CPU-only |

### 5.2 Technical Metrics

| Metric | Definition |
| --- | --- |
| **CER / WER reduction (headline)** | `% reduction = (CER_before − CER_after) / CER_before`, where `before` = CER(raw OCR, gold) and `after` = CER(corrected, gold); same for WER |
| **% improved vs. degraded (safety gate)** | Share of examples the corrector **improved** vs. **degraded** vs. left **unchanged** — `degraded` must be `<<` `improved`. A corrector that fixes 40% but breaks 30% is useless. |
| **ExactMatch** | % of outputs that exactly match the gold text |
| **Layout / reading-order correctness** | Correctness of block typing and reading order (eval against layout corpora such as DocLayNet / PubLayNet / FUNSD) |
| **Latency per page** | Born-digital ~200ms (OCR skipped); scanned is OCR-dominated; correction ~80ms/region |

Error-rate measurement uses jiwer-style char/word Levenshtein distance. The **safety gate is non-negotiable**: because the main risk is a corrector that *hallucinates* or rewrites a region, the pipeline only accepts a correction when it is a **bounded edit (edit ratio ≤ 0.35)** and confident enough, and the `% degraded` metric is reported explicitly so regressions cannot hide behind an improved average.

---

## 6. Summary

P07 reframes "OCR" from a single lossy pass into a **document-level NLP pipeline**: layout + reading order supply *structure*, and a trainable byte-level seq2seq corrector supplies *measurable error reduction* on top of any OCR engine — all orchestrated by an auditable agent with explicit, bounded decision gates. Success is judged not just on raw accuracy but on **how much manual review it removes**, **how faithfully it preserves document structure**, and a **safety gate** proving the corrector helps far more often than it hurts.
