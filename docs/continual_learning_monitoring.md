# Continual Learning & Monitoring — P07 Document-Level OCR

This document describes how the Document-Level OCR System keeps improving after
deployment: how it harvests real correction data from production, how it
retrains the post-OCR corrector safely, how it detects degradation, what it
monitors, and which drift risks threaten the system. It is written for the
trainable ML core — the **post-OCR error corrector** (`google/byt5-small`,
prefix `"correct: "`) — which is the only learned component; the OCR front-end
(Tesseract by default) and the layout / reading-order stages are
pretrained/algorithmic and are not retrained here.

> Reference implementation for the drift/health report:
> `src/dococr/monitoring/drift_report.py`.

---

## 1. Why continual learning matters here

The corrector is trained primarily on a **synthetic OCR-noise generator**
(`src/dococr/data/ocr_noise.py`) that corrupts clean English text with a
confusion model (e.g. `rn<->m`, `cl<->d`, `O<->0`, `l<->1`, `e<->c`, `S<->5`,
`g<->q`, plus insert/delete, word merge/split, case flips) at a tunable
`char_error_rate` (default `0.08`). It mixes in real data from
**PleIAs/Post-OCR-Correction** (config `english`, CC0-1.0, 31.3K rows;
columns `text` = noisy OCR, `corrected_text` = gold).

The known weakness is the **sim-to-real gap**: synthetic noise may not match the
error distribution of the *specific* OCR engine, document domain, or scan
conditions seen in production. The cure is to close the loop — collect real
`(raw OCR, corrected)` pairs from live traffic and feed them back into training.

---

## 2. New-data collection (the feedback loop)

The agent already produces two natural sources of supervision, driven by its
deterministic decision points:

| Source | Where it comes from | Decision point | What it yields |
|---|---|---|---|
| **Flagged low-confidence regions** | OCR confidence below threshold `0.55` per region | **D3** OCR-confidence gate | Regions routed to human review (the "needs gold" queue) |
| **Rejected corrections** | Corrector output rejected because edit ratio > `0.35` or low confidence | **D4** correction acceptance | Regions where the model wanted a large/uncertain edit — high-value hard cases |
| **Human-corrected gold** | Reviewer fixes the flagged/rejected region in the UI | review tool | The clean target text |

### Feedback store

Every reviewed region becomes one training-ready record in a **feedback store**.
The minimum schema (derived from the API `/ocr` response — `blocks[]`,
`decisions`, `metrics` — plus the human edit) is:

```json
{
  "region_id": "doc123#p2#b7",
  "raw_ocr": "the q5uick brovvn f0x",
  "corrected_gold": "the quick brown fox",
  "ocr_confidence": 0.41,
  "block_type": "paragraph",
  "flag_reason": "d3_low_confidence",
  "model_revision": "byt5-small@<rev>",
  "engine": "tesseract",
  "doc_domain": "invoice",
  "language": "en",
  "scan_path": "scanned",
  "ts": "2026-06-26T10:04:11Z"
}
```

Key collection rules:

- **Harvest real `(raw OCR, corrected)` pairs from production.** The `raw_ocr`
  field is exactly the input the corrector saw; `corrected_gold` is the human
  target. These pairs mirror the PleIAs `text` / `corrected_text` columns, so
  they drop directly into the existing dataset pipeline (`data/dataset.py`).
- **Prioritise hard cases.** D3-flagged and D4-rejected regions are
  over-represented relative to easy auto-accepted regions; these are precisely
  the cases the current model gets wrong, so they carry the most learning signal.
- **Privacy first.** Documents may contain PII (names, IDs, addresses, financial
  data). Honour the on-prem / no-retention option and TTL cleanup: store the
  minimal text spans needed for training, avoid retaining raw page images, and
  redact or hash document identifiers. Collection must respect the same
  rights/consent constraints as OCR itself.

---

## 3. Retraining

Retraining is **continue-fine-tune**, not from scratch — start from the current
best checkpoint and continue with the existing recipe (`predict_with_generate`,
lr `5e-4` for byt5, cosine schedule, warmup `0.05`, `weight_decay 0.01`,
`label_smoothing 0.1`, early stopping patience `4` on CER,
`load_best_model_at_end`, `group_by_length`, resume via `get_last_checkpoint`).

### Training mix

Combine three streams so the model adapts to real errors without forgetting
general robustness:

1. **Real production corrections** — the harvested feedback pairs (highest
   weight; this is the whole point).
2. **Real public data** — PleIAs `english` slice, to anchor general post-OCR
   behaviour.
3. **Fresh synthetic noise** — newly regenerated from `ocr_noise.py` with a
   diverse confusion model, to keep broad coverage and prevent overfitting to a
   narrow recent distribution.

Splits stay **leakage-free** (dedup by clean text) and a real eval slice is held
out, exactly as in the base build.

### Registry versioning

Each retrain produces a new model version recorded in the model registry
(`models/model_registry.py`): `model_meta.json` + a `latest` pointer, addressed
as `repo@revision`. The `model_revision` stamped on every feedback record (see
schema above) ties each correction back to the model that produced it, so a bad
release can be traced and rolled back to the previous `repo@revision`.

### Canary / A/B rollout

Never promote a retrain blindly:

- **Canary:** route a small fraction of production regions to the new revision;
  serve the rest from the incumbent.
- **A/B:** compare canary vs. incumbent on the live monitoring metrics
  (Section 5) and on a frozen **golden set**.
- **Promotion gate:** promote only if CER-reduction on the golden set is at least
  as good as the incumbent **and** the safety gate holds (`% degraded` stays
  well below `% improved`). A retrain that fixes more but also breaks more is
  rejected — consistent with the build-time rule that a corrector that fixes 40%
  while breaking 30% is useless.

---

## 4. Degradation detection

Degradation is caught at two layers:

- **Offline, on the golden set.** A fixed, version-controlled set of
  `(raw OCR, gold)` regions (seeded from synthetic test + real PleIAs eval +
  curated production examples). Each candidate model is scored for
  **CER-reduction** = `(CER_before - CER_after) / CER_before`, WER-reduction,
  ExactMatch, and the `% improved / degraded / unchanged` split. A drop in
  CER-reduction or a rise in `% degraded` versus the previous revision is a
  regression and blocks promotion. (Build-time reference: identity baseline on
  the synthetic test distribution is CER `~0.088`, WER `~0.49`, exact-match
  `~0.0005` — the corrector must stay well below identity CER.)
- **Online, on live traffic.** `drift_report.py` aggregates the monitoring
  metrics over rolling windows and compares them against a baseline window. A
  sustained rise in **flag-rate**, drop in **mean region confidence**, or rise in
  **% degraded** signals that production has drifted away from the training
  distribution — even before the next labelled golden-set evaluation.

The edit-budget D4 gate (edit ratio `<= 0.35`) is also a runtime safety net: a
degrading model that starts proposing large rewrites gets its corrections
rejected automatically and those regions become new feedback records.

---

## 5. Proposed monitoring metrics

`src/dococr/monitoring/drift_report.py` computes and trends the following.
The first four are available continuously from production traffic; the last two
require a labelled golden set.

| Metric | Definition | Watch for | Action threshold (proposed) |
|---|---|---|---|
| **Flag-rate** | Fraction of regions sent to human review at D3 (conf < `0.55`) | Sustained increase | Investigate domain/engine/scan drift |
| **Mean region confidence** | Average OCR confidence (0–1) across regions | Sustained decrease | Same as above; possible input-quality shift |
| **Latency** | Per-page / per-region wall time (born-digital ~200ms via D2 skip; scanned OCR-dominated ~0.6–1.2s/region; correction ~80ms/region) | Tail growth, regressions | Profile front-end vs. corrector; check batching/GPU |
| **Status mix** | Share of regions `auto-accepted` / `flagged` / `degraded` (D1/D3/D4 outcomes) | Shift toward flagged/degraded | Triage which decision point is firing more |
| **CER-reduction (golden set)** | `(CER_before - CER_after)/CER_before` on the frozen golden set | Decline vs. previous revision | Block promotion / trigger retrain |
| **% degraded** | Share of regions made *worse* by the corrector (degraded vs. improved vs. unchanged) | Rising, or approaching `% improved` | Hard fail — safety gate; roll back |

Business-facing roll-ups derived from these: **manual-review reduction**
(% pages auto-accepted at the confidence gate) and **structure fidelity**
(% blocks correctly typed and ordered).

---

## 6. Drift risks and mitigation

The system is **English-first** (PleIAs `english` + English synthetic), built
around **Tesseract** + a synthetic noise model. The main drifts that break those
assumptions:

| Drift risk | Symptom in metrics | Mitigation |
|---|---|---|
| **New document domains** (e.g. invoices, legal, dense tables) the corrector and layout heuristics never saw | Flag-rate up, status mix shifts to `flagged`/`degraded`, structure fidelity down | Harvest domain pairs via the feedback loop; continue-fine-tune with domain mix; lean on D1 `degraded` path for complex pages |
| **New OCR engine** (swap Tesseract → doctr / PaddleOCR / Surya / EasyOCR / TrOCR) with a *different* error distribution | CER-reduction on golden set drops; `% degraded` rises (model tuned to old noise) | Regenerate feedback/golden sets with the new engine; retrain on its real error pairs; re-baseline `drift_report.py` |
| **New languages** beyond English | Confidence falls, flag-rate spikes; corrector out of scope | Detect and route out-of-scope languages to review; expand training data per language before claiming support |
| **Scan-condition shift** (skew, blur, low contrast, lower DPI) | Mean region confidence down, latency up, more D1 `degraded` | D1 quality routing (Laplacian-variance blur + ink ratio + contrast) reprocesses/flags bad scans; preprocessing handles skew/blur; persistent shift → retrain on real noisy pairs |
| **Sim-to-real gap** (synthetic noise diverges from production) | Golden-set gains don't translate to live flag-rate/`% degraded` | Always report real PleIAs CER alongside synthetic; raise the real-data weight in the retrain mix |

### Mitigation summary

1. **Watch** `drift_report.py` for flag-rate up / confidence down / `% degraded`
   up against a baseline window.
2. **Collect** the regions driving the drift (they are already being flagged at
   D3/D4) into the feedback store.
3. **Retrain** by continue-fine-tuning on real corrections + fresh synthetic,
   version the result in the registry, and roll out via canary/A-B.
4. **Gate** promotion on golden-set CER-reduction and the `% degraded` safety
   gate; roll back to the previous `repo@revision` if either regresses.

This loop turns every flagged region and every human correction into training
signal, so the post-OCR corrector tracks the real production error distribution
instead of drifting away from it.
