# Data Privacy & Model Robustness

**Project:** Document-Level OCR System (P07)
**Author:** Le Dinh Minh Quan (23127460) — NLP in Industry, final assignment
**Package:** `src/dococr/`

This document covers two production concerns that sit on top of the core ML pipeline (layout + reading-order for structure, a trainable **post-OCR corrector** for accuracy): how the system handles **sensitive document data** and how it behaves **under adverse, out-of-distribution input**. Both are first-class design constraints, not afterthoughts — the OCR pipeline routinely ingests documents containing personally identifiable and financial information, and real-world scans are noisy, skewed, and rarely clean.

Primary code references: `analysis/robustness.py` (robustness report), `agent/policy.py` (the decision FSM that enforces the safety and quality gates).

---

## 1. Data Privacy

### 1.1 The threat: documents are PII-dense by nature

A document-level OCR system is, by definition, pointed at whole pages of human documents. Those pages routinely contain:

- **Names** (people, organizations)
- **Identifiers** (national IDs, account numbers, case numbers)
- **Addresses** (postal, email)
- **Financial data** (amounts, balances, transaction lines, invoice totals)

Both the raw input (the scanned image / PDF) and every downstream artifact (OCR text, layout blocks, corrected text, JSON output) can therefore carry sensitive content. The privacy posture is built around **data minimization** plus a **local-by-default** processing model.

### 1.2 Data minimization

| Principle | Implementation in P07 |
|---|---|
| Don't retain raw input beyond need | Raw images / PDFs are processed transiently; they are not persisted as a system of record. |
| Don't retain derived text beyond need | OCR text, blocks, and corrected output exist only as long as the request/job requires; **TTL cleanup** removes transient artifacts. |
| No raw PII in logs | Logs record decisions, timings, and metrics (via `ToolTrace` / `Decision`) — **not** raw page content. Audit trails capture *what the agent decided*, not *what the document said*. |
| On-prem / no-retention option | The default stack is CPU-able and self-hostable; a **no-retention mode** processes-and-discards so no document data is stored at all. |

The audit design intentionally separates **operational metadata** (which is retained for traceability — the per-step `ToolTrace`, every `Decision`, the `manifest.json`) from **document payload** (which is minimized and TTL-expired). This lets the system prove *how* a page was handled without keeping *the page itself*.

### 1.3 Local-by-default, no data egress

The most important privacy property is that the **trainable corrector runs locally**:

- The post-OCR corrector is **`google/byt5-small`** (Apache-2.0, ~300M, byte-level), loaded once at startup and run on-device. The T4 fallback **`google-t5/t5-small`** (Apache-2.0, 60.5M) is also local. **No document text leaves the machine** to be corrected.
- The OCR front-end default, **Tesseract** via `pytesseract`, is **CPU-only with no download** — OCR also happens entirely locally.
- The optional LLM "brain" (Anthropic) used by decision point **D4** for flagged regions is **OFF by default** → **zero paid API calls** and **no document data sent off-box** in the default configuration. When explicitly enabled by an operator, only flagged regions are sent, the output is validated, and the system falls back to the local result on failure.

This means a deployment can run the full pipeline — ingest → layout → correct → assemble — **air-gapped**, which is exactly what a privacy-sensitive on-prem customer requires.

### 1.4 Rights and consent

OCR of private or copyrighted documents requires the appropriate **rights / consent**. The training data itself avoids this problem: the primary signal is a **reproducible synthetic OCR-noise generator** (`src/dococr/data/ocr_noise.py`) over clean English text, with an optional real mix from **`PleIAs/Post-OCR-Correction`** (config `english`, **CC0-1.0**, 31.3K rows). The corrector is therefore trained without scraping sensitive real documents.

---

## 2. Model Robustness

OCR input is adversarial by default: scans are skewed, blurred, low-contrast, noisy; pages are multi-column, rotated, or packed with tables and figures. P07 treats robustness as an explicit, **measured** property rather than a hope.

### 2.1 Out-of-distribution (OOD) scans → preprocessing + degraded path

The agent's **D1 page-quality / preprocess routing** decision (see `agent/policy.py`) scores each scanned page before OCR:

- **Quality score** = Laplacian-variance blur + ink ratio + contrast.
- Routing outcome → **ok** / **reprocess** / **degraded**.

A page that scores poorly (skew, noise, blur, low contrast) is sent down the **degraded path**, which *lowers the downstream D3 bar* so the system does not over-reject a page that is simply a bad scan — it flags rather than silently dropping. Preprocessing (`src/dococr/ocr/preprocess.py`) is the first line of defense against OOD scan quality; D1 decides when the input is too far gone for the normal-confidence gate to apply.

### 2.2 The robustness report: CER reduction across OCR-noise severity

The headline robustness artifact is produced by **`analysis/robustness.py`**. Because the corrector's noise generator exposes a tunable `char_error_rate` (default **0.08**), we can sweep noise severity and measure whether the corrector *keeps helping* as input degrades.

The report measures, at each increasing noise level:

- **`CER_before`** = `CER(raw_OCR, gold)` — the identity baseline
- **`CER_after`** = `CER(corrected, gold)` — after the ByT5 corrector
- **% reduction** = `(before − after) / before`

The byte-level model is chosen specifically for this regime: ByT5 operates at the char/byte level (no SentencePiece), and the ByT5 paper (arXiv:2105.13626) shows byte models are markedly more noise-robust — exactly the property needed when noise severity climbs. On the synthetic test distribution the **identity baseline** sits at **CER ≈ 0.088, WER ≈ 0.49, exact-match ≈ 0.0005**, giving the corrector a clear, measurable job; the robustness report demonstrates that CER reduction holds across a *range* of severities, not just at one operating point.

### 2.3 Known failure cases

The system is **English-first** and layout-heuristic-based, so the following inputs are expected to be harder and are flagged rather than trusted:

| Failure class | Why it's hard | Mitigation |
|---|---|---|
| Dense tables | Block grouping by line/par hierarchy under-segments cells | D1 degraded path + region flag for review |
| Figures / mixed graphics | Not text; layout heuristics misfire | D3 confidence gate flags low-confidence regions |
| Rotated scans | Skew breaks reading order and OCR | D1 quality score → degraded path |
| Multi-column pages | XY-cut must detect the column gap correctly | XY-cut reading order (vertical-gap detection near page centre) |
| Non-English text | Trained/tuned on English (PleIAs english + English synthetic) | Out of declared scope; documented language limit |

The **sim-to-real gap** is also a known limitation: synthetic noise may not match a *specific* OCR engine's error distribution. This is mitigated by (a) a diverse confusion model (`rn↔m`, `cl↔d`, `O↔0`, `l↔1`, `e↔c`, `S↔5`, `g↔q`, …, plus insert/delete/merge/split/case-flip) and (b) **mixing real PleIAs data** and **reporting real PleIAs CER** alongside synthetic numbers.

### 2.4 The "corrector makes it worse" risk → D4 edit-budget guard + %-degraded metric

The single biggest model-robustness risk for a *trainable* post-OCR corrector is that **it rewrites or hallucinates**, turning a correct region into a wrong one. A corrector that fixes 40% but breaks 30% is worse than useless. P07 guards against this on two fronts:

**1. The D4 correction-acceptance gate (`agent/policy.py`).** The agent accepts the corrector's output **only if**:

- the edit is **bounded** — edit ratio **≤ 0.35** (it cannot rewrite a region away), **and**
- it is **confident enough**.

Otherwise the **raw OCR text is kept**. This is a hard, deterministic guard: a hallucinated, large rewrite is structurally rejected before it can reach the output.

**2. The safety-gate metric (the `%-degraded` measure).** Evaluation does not stop at average CER. It reports, per example, the split of **improved vs. degraded vs. unchanged**:

> **Requirement:** `degraded << improved`.

This makes the failure mode visible and gradeable — a corrector that lowers mean CER while quietly breaking a large fraction of examples does not pass, because the degraded share would be too high. `ExactMatch` is reported as well.

### 2.5 Graceful degradation everywhere

The pipeline is designed to **never hard-fail** on a missing dependency or missing input — a key robustness property for real deployments:

- **Stub OCR engine** — an empty-result engine so the agent and the test suite run **with no OCR binary** present (mirrors P05/P06). When every page is born-digital (text layer present), OCR is skipped entirely by **D2** anyway.
- **Identity corrector** — if the neural model is unavailable, the system falls back to identity (raw OCR passthrough); the corrector is *additive*, never a single point of failure.
- **numpy / PIL fallbacks** — preprocessing and image handling degrade to numpy/PIL paths when `cv2` / `torch` are absent.
- **Dataset loaders degrade gracefully** — e.g. the optional `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` benchmark has its viewer disabled, and the loader handles that rather than crashing.

This layered fallback means the worst case is *reduced functionality* (raw OCR, no correction), not an outage.

---

## 3. Summary

| Concern | Mechanism | Where |
|---|---|---|
| PII exposure | Minimization, TTL cleanup, no raw PII in logs, no-retention/on-prem mode | pipeline + audit design |
| Data egress | Local corrector (ByT5/T5) + local Tesseract; LLM brain OFF by default | corrector, `agent/policy.py` (D4) |
| OOD scans | Quality-score routing → degraded path; preprocessing | `agent/policy.py` (D1), `ocr/preprocess.py` |
| Noise severity | CER-reduction sweep across `char_error_rate` | `analysis/robustness.py` |
| Corrector hallucination | Bounded edit ratio ≤ 0.35 + confidence | `agent/policy.py` (D4) |
| Silent quality loss | `%-improved vs. degraded` safety gate (`degraded << improved`) | training/evaluate metrics |
| Dependency / input gaps | Stub OCR, identity corrector, numpy/PIL fallbacks | pipeline-wide |

Privacy and robustness are enforced by the **same deterministic agent FSM** that drives accuracy: D1 protects against bad scans, D2 skips OCR (privacy + speed), D3 flags low-confidence regions for human review, and D4 stops the corrector from doing harm — all while keeping document data local and minimally retained.
