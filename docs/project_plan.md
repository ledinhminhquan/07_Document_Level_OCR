# P07 — Document-Level OCR System: Project Plan & Teamwork

**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — Final Assignment
**Package:** `src/dococr/` (mirrors the P02–P06 template)

---

## 1. Project Overview

**Goal.** Turn a full-page document image or PDF into clean, structured, reading-order-correct text emitted as **Markdown + JSON blocks + plain text**.

**The key insight.** A flat OCR pass produces an unordered, error-riddled text dump: wrong reading order on multi-column pages, no block typing (title / paragraph / list / table), and char-level OCR garble (`rn`→`m`, `0`↔`O`, `1`↔`l`, merged/split words). P07 attacks both problems:

- **Structure** comes from layout detection + reading-order reconstruction (pretrained / algorithmic).
- **Accuracy** comes from a **trainable post-OCR corrector** (an NLP seq2seq model) that measurably lowers CER/WER on top of *any* OCR engine.

The **trained differentiator** is the post-OCR corrector. The OCR front-end and layout stack are pretrained or algorithmic and are not trained in this project.

### Component map

| Layer | What it does | Status |
|---|---|---|
| OCR front-end | Tesseract (default) → words + boxes + confidence | Pretrained / algorithmic |
| Layout + reading order | PyMuPDF ingest, block detection, XY-cut ordering, block classification | Algorithmic |
| **Post-OCR corrector** | **`google/byt5-small`** seq2seq, prefix `"correct: "` | **Trained (ML core)** |
| Agent | Deterministic FSM with 4 decision points (D1–D4) | Algorithmic + optional LLM |
| Deployment | FastAPI + Gradio + CLI + Docker | Engineering |

---

## 2. Project Plan & Timeline

The project is decomposed into eight phases. Each phase has a concrete, verifiable exit criterion (a "definition of done") so that progress is measurable rather than vibes-based.

### Milestone table

| # | Phase | Key deliverables | Exit criterion (DoD) |
|---|---|---|---|
| P1 | **Research & scoping** | Problem framing, model/dataset survey, success metrics, risk register | Verified model & dataset ids fixed; metric = **CER/WER % reduction** agreed |
| P2 | **Synthetic generator + baselines** | `data/ocr_noise.py` confusion-model corruptor; **identity** baseline; optional SymSpell | Synthetic train/val/test (60k/4k/4k) built; identity baseline measured (**CER ≈ 0.088, WER ≈ 0.49, exact ≈ 0.0005**) |
| P3 | **OCR / layout integration** | `ocr/preprocess.py`, `ocr/layout.py`, Tesseract engine, PyMuPDF ingest, XY-cut, block classifier | Born-digital doc → typed blocks in correct reading order; structured Markdown + JSON emitted |
| P4 | **ByT5 fine-tune** | `training/train_corrector.py` (Seq2SeqTrainer), tuning, checkpointing/resume | Trained ByT5 **reduces CER below identity** with degraded ≪ improved on the safety gate |
| P5 | **Agent** | `agent/` FSM with decisions D1–D4, ToolTrace, manifest, optional LLM brain | All 4 decisions fire end-to-end; full audit trace + `manifest.json` produced |
| P6 | **Deployment** | FastAPI (`/ocr`, `/correct`, health), Gradio UI, CLI, Docker, model registry | API serves uploads; Docker image (tesseract + poppler) builds; HF Space packaged |
| P7 | **Evaluation** | `training/evaluate.py`, error analysis, robustness sweep, latency profiling | Headline reduction reported; robustness across `char_error_rate` levels; real PleIAs CER reported |
| P8 | **Docs & report** | README, `project_plan.md`, auto-report (`report.pdf`) + slides (`slides.pptx`) | Report and slides generate; submission package complete |

### Critical path & dependencies

```
P1 ──> P2 ──> P4 ──> P5 ──> P6 ──> P7 ──> P8
         └──> P3 ──────┘ (P3 feeds P5 structure; P3 ⟂ P4 can overlap)
```

The two heaviest items — **P3 (OCR/layout)** and **P4 (ByT5 fine-tune)** — are independent: the corrector trains on synthetic + real text pairs and does not need the OCR front-end to exist. In solo execution they are sequenced; in a team they parallelize (see §4).

### Indicative schedule (solo, calendar weeks)

| Week | Focus |
|---|---|
| W1 | P1 research; P2 synthetic generator + identity/SymSpell baselines |
| W2 | P3 OCR + layout + reading order; born-digital fast path (D2) |
| W3 | P4 ByT5 fine-tune (≈ **3–6 h on one H100**), tuning, checkpoint/resume |
| W4 | P5 agent FSM (D1–D4) + audit trace |
| W5 | P6 deployment (FastAPI / Gradio / CLI / Docker / registry) |
| W6 | P7 evaluation, error analysis, robustness, latency; P8 docs, report, slides |

> The single longest compute task is the ByT5 fine-tune. GPU profiles are pre-defined (H100 bf16 bs32/accum8; A100-40 bf16 bs16/accum16 + ckpt; L4 bf16 bs8/accum32 + ckpt; T4 fp16 bs4/accum64 + ckpt with base switched to `google-t5/t5-small`), so the phase can run on whatever hardware is available without a redesign.

---

## 3. Task Breakdown by Role (simulated, solo execution)

This is a solo submission, but the work is organized as if six roles existed. Each phase was executed wearing the relevant "hat"; the table makes the responsibilities explicit and maps cleanly onto a real team.

| Role | Owns | Concrete tasks in this project |
|---|---|---|
| **PM** | Scope, schedule, risk, metrics | Phase plan & milestones; risk register (corrector-makes-it-worse, sim-to-real gap, PII); define headline metric = **% CER/WER reduction** + safety gate (% improved vs degraded) |
| **ML engineer** | The trainable core | Post-OCR corrector on `google/byt5-small` (`models/corrector.py`); `training/train_corrector.py`, `tune.py`, `metrics.py`; early stopping on CER, label smoothing, group-by-length; T4 fallback to `t5-small` |
| **Data engineer** | Data pipeline & quality | `data/ocr_noise.py` confusion-model generator (`rn`↔`m`, `O`↔`0`, `l`↔`1`, …, default `char_error_rate` 0.08); real mix from **PleIAs/Post-OCR-Correction** (english, CC0-1.0, 31.3k rows); leakage-free splits (dedup by clean text); optional ICDAR-2019 loader that degrades gracefully |
| **OCR / vision engineer** | Front-end & layout | Tesseract via `pytesseract` (`image_to_data` → boxes + conf + hierarchy); PyMuPDF ingest with born-digital text-layer detection; XY-cut multi-column reading order; heuristic block classifier; optional doctr/PaddleOCR/Surya/EasyOCR backends + stub engine |
| **Backend / deploy** | Serving & ops | FastAPI (`/ocr`, `/correct`, `/healthz`, `/readyz`, `/version`); Gradio UI; `dococr` CLI; Docker (python:3.11-slim + tesseract-ocr-eng + poppler-utils + libgl1) + compose; model registry (`model_meta.json` + latest pointer, `repo@revision`) |
| **QA** | Correctness & robustness | Offline tests with stub OCR; agent end-to-end check (all 4 decisions fire; born-digital doc → 7 typed blocks + correct Markdown); robustness sweep across `char_error_rate`; safety-gate enforcement (degraded ≪ improved); numpy/PIL fallbacks when cv2/torch absent |

**Solo coordination note.** Because one person plays all roles, the "interfaces" between roles are enforced through code contracts instead of meetings: the agent pipeline `ingest → layout → correct → assemble` defines the boundary between the OCR/vision work (produces blocks) and the ML work (corrects block text), and the `api/schemas.py` response contract (`full_text`, `markdown`, `blocks[]`, `decisions`, `metrics`) is the boundary between backend and everything upstream.

---

## 4. Reflection: Scaling to a Real Team

The codebase is deliberately modular so the simulated roles map to genuine parallel workstreams. Here is how P07 would scale with real headcount.

### 4.1 Parallel workstreams by module

The package directory structure is the org chart. With multiple engineers the phases stop being sequential:

- **OCR/vision squad** owns `ocr/` and `models/ocr_engine.py` — adds and benchmarks alternative backends (doctr, PaddleOCR/PP-StructureV3, Surya, EasyOCR) behind a stable engine interface. *(Licensing note: Surya weights are OpenRAIL-M non-commercial above $5M revenue and must be flagged before any commercial use.)*
- **ML squad** owns `models/corrector.py` + `training/` — iterates on the corrector independently because it trains on text pairs, not on the live OCR output.
- **Data squad** owns `data/` — runs the synthetic generator and real-data mixing as a standalone pipeline.
- **Platform squad** owns `api/`, `deploy/`, and the model registry.

These four streams only synchronize at two contracts: the **block schema** (vision → ML) and the **API response schema** (platform → consumers). Everything else proceeds in parallel.

### 4.2 CI

In a team, the offline guarantees that already exist (stub OCR, identity corrector, numpy/PIL fallbacks) become the foundation of CI. Every PR runs the agent end-to-end on the stub engine and the born-digital fixture, asserts that all four decisions fire and that the 7-block Markdown is produced, and runs the metrics suite. A **safety-gate regression check** is the most important gate: a PR is blocked if the corrector's `% degraded` rises relative to `% improved`, because "fixes 40% but breaks 30%" must never ship.

### 4.3 Model registry

The lightweight registry (`model_meta.json` + a latest pointer, addressed as `repo@revision`) generalizes into a real registry workflow: each training run publishes a versioned artifact, evaluation results are attached, and promotion to "latest" is a deliberate, reviewable step gated on CER reduction and the safety gate. This lets serving roll forward and back without code changes.

### 4.4 Data pipeline for harvesting production OCR pairs

The biggest single-quality lever at scale is closing the **sim-to-real gap**. The synthetic confusion model is a stand-in for a real engine's error distribution. In production we would harvest `(raw OCR, human-reviewed correction)` pairs from the **D3 low-confidence** and **human-review** flows: regions the confidence gate routes to a reviewer become labeled training data for the *exact* engine and document types in use. Over time the corrector is retrained on a real distribution rather than a synthetic one — the system gets better precisely where it currently fails. This must run under the project's privacy constraints (see §5): minimize retention of raw images/text, on-prem / no-retention option, TTL cleanup.

### 4.5 On-call & monitoring

The audit infrastructure (every step timed and traced via `ToolTrace`, every decision recorded as a `Decision`, full `manifest.json`) is the foundation for production observability. At scale this feeds dashboards and alerts on:

- **Auto-accept rate** at the D3 confidence gate (the business KPI — manual-review reduction).
- **Corrector safety** — live `% improved vs degraded`; a drift downward pages on-call.
- **Latency** — born-digital pages ≈ 200 ms (D2 skip) vs scanned pages OCR-dominated (≈ 0.6–1.2 s/region), correction ≈ 80 ms/region; regressions are caught per-percentile.
- **Layout failures** — spikes in the D1 *degraded* path (dense tables, figures, rotated scans) signal a content shift needing model attention.

Scalability itself is achieved through **page/region parallelism + GPU batching** of the corrector, which the stateless FastAPI service supports horizontally.

---

## 5. Risk, Privacy & Robustness (planning constraints)

These constraints shape the plan above and are owned jointly by PM and QA.

| Concern | Mitigation built into the plan |
|---|---|
| Corrector makes text **worse** / hallucinates | **D4 bounded-edit gate** (edit ratio ≤ 0.35, keep raw otherwise) + the `% degraded` safety metric as a CI gate |
| **PII** in documents (names, IDs, addresses, financial data) | Minimize retention/logging of raw images & text; on-prem / no-retention option; TTL cleanup; rights/consent for private or copyrighted documents |
| **Sim-to-real gap** | Mix real **PleIAs** data; report real-data CER alongside synthetic; harvest production pairs (§4.4) |
| **Layout failures** on complex pages | D1 *degraded* path + flag for review; robustness sweep across noise severities |
| **Language scope** | English-first (PleIAs english + English synthetic), stated explicitly |

---

## 6. Definition of Success

- **Business:** manual-review reduction (% pages auto-accepted at the confidence gate); structure fidelity (% blocks correctly typed + ordered); low cost per page (born-digital skips OCR; CPU-able default).
- **Technical (headline):** **CER/WER reduction** vs the identity baseline; **% improved vs degraded** safety gate; ExactMatch; layout/reading-order correctness; latency per page.

The plan is considered delivered when the trained ByT5 corrector demonstrably reduces CER below the identity baseline with degraded ≪ improved, the agent runs end-to-end with all four decisions firing, and the deployment + report artifacts generate cleanly.
