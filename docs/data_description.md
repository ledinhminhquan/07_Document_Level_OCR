# Data Description — P07 Document-Level OCR

**Project:** Document-Level OCR System (full-page document image/PDF → clean, reading-order-correct structured text)
**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — Final Assignment
**Package:** `src/dococr/`

This document describes the data used to train and evaluate the **ML core** of P07: the **Post-OCR Error Correction** model (a `text → text` seq2seq corrector). The OCR front-end (Tesseract/doctr/PaddleOCR/Surya), layout analysis, and reading-order logic are pretrained or algorithmic and are **not trained**, so they are out of scope for this data description except where they generate inputs for evaluation.

---

## 1. What the corrector learns from

The trainable model is **`google/byt5-small`** (Apache-2.0, char/byte-level, no SentencePiece), with `google-t5/t5-small` as a T4 fallback. Because the model operates at the **byte level**, the data it consumes is plain text pairs:

| Field | Meaning |
|-------|---------|
| `input` | Noisy OCR text (with the prefix `"correct: "`) |
| `target` | Gold (clean / corrected) text |

The learning objective is to map garbled OCR output back to clean text, fixing the characteristic error classes that any OCR engine produces.

---

## 2. Sourcing

Data comes from three sources, in priority order.

### 2.1 PRIMARY — Synthetic OCR-noise generator (`src/dococr/data/ocr_noise.py`)

The primary training signal is a **reproducible synthetic noise generator** that corrupts clean English text with a realistic OCR confusion model. This is the main data source because it is unlimited, fully reproducible, and lets us control the error distribution directly.

The corruption model applies, at a tunable **character error rate** (`char_error_rate`, default **0.08**):

- **Character substitutions** drawn from OCR confusion sets, e.g.
  `rn ↔ m`, `cl ↔ d`, `O ↔ 0`, `l ↔ 1`, `e ↔ c`, `S ↔ 5`, `g ↔ q`, …
- **Insertions** and **deletions** of characters.
- **Word merge / split** errors (spacing — two words run together, or one word broken in two).
- **Case flips** (upper/lower swaps).

These mirror the real failure modes of OCR: confusable glyph pairs (`rn`→`m`, `0`↔`O`, `1`↔`l`), spurious/dropped characters, and broken word boundaries.

**Default synthetic sizes:**

| Split | Rows |
|-------|------|
| train | 60,000 |
| val   | 4,000 |
| test  | 4,000 |

The clean text fed to the generator is the corpus source used as the `target`; the generator's corrupted output becomes the `input`.

### 2.2 Real mix — `PleIAs/Post-OCR-Correction` (config `english`)

To close the simulation-to-reality gap, the synthetic data is mixed with a **real** post-OCR correction corpus:

- **Dataset:** `PleIAs/Post-OCR-Correction`, config `"english"`
- **License:** CC0-1.0 (public domain)
- **Size:** 31.3K rows
- **Columns:** `text` (noisy OCR) + `corrected_text` (gold)

This provides genuine OCR error patterns from real documents, which the synthetic confusion model cannot perfectly reproduce. It is used both as a training mix and to provide a **real evaluation slice** (see §3).

### 2.3 Optional benchmark — ICDAR-2019 Post-OCR Text Correction

- **Dataset:** `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction`
- **Type:** manually curated competition corpus

The dataset viewer is disabled, so the loader **degrades gracefully** (it is treated as optional and skipped if unavailable rather than failing the pipeline).

### 2.4 Clean text source

The clean English text that seeds the synthetic generator is provided through the corpus layer (`src/dococr/data/corpus.py`). Each clean passage serves as the gold `target`; its corrupted counterpart is generated on the fly by the noise model.

---

## 3. Preprocessing

The data pipeline (`src/dococr/data/`) applies the following steps:

1. **Windowing.** Long passages are segmented into model-sized text windows so that inputs fit the corrector's sequence budget and training examples are length-comparable (`group_by_length` is also used at train time for efficient batching).
2. **Deduplication.** Examples are deduplicated **by clean text**, so that the same gold passage cannot appear in more than one split.
3. **Leakage-free splits.** Because dedup is keyed on the clean text, train / val / test are guaranteed not to share underlying source passages — a passage's noisy variants all land in the same split. This prevents the corrector from "memorizing" a target it will later be tested on.
4. **Real evaluation slice.** In addition to the synthetic test set, a slice of the **real** `PleIAs` data is held out for evaluation, so reported metrics reflect both the controlled synthetic distribution **and** real OCR errors.

---

## 4. Train / Validation / Test split & justification

| Split | Source | Default size | Purpose |
|-------|--------|--------------|---------|
| **train** | Synthetic (+ real `PleIAs` mix) | 60,000 | Fit the corrector across a diverse confusion model and real errors |
| **val** | Synthetic | 4,000 | Early stopping (monitor **CER**, lower is better) and checkpoint selection (`load_best_model_at_end`) |
| **test** | Synthetic + real `PleIAs` eval slice | 4,000 (synthetic) | Final, leakage-free evaluation of error reduction |

**Justification:**

- The split sizes give a large, varied training set while keeping val/test small enough for fast, frequent evaluation (early stopping patience 4 on CER).
- **Leakage-free by construction** (dedup on clean text) — essential for a corrector, where the same gold string in train and test would inflate exact-match and understate error.
- The **real eval slice** guards against over-fitting to the synthetic noise distribution: a corrector that only beats its own simulator is not useful, so we also measure on real `PleIAs` OCR.
- Mixing **diverse synthetic noise + real data** is part of the anti-overfitting strategy, alongside early stopping, weight decay (0.01), and label smoothing (0.1).

---

## 5. Baseline reference — the identity baseline

The canonical post-OCR baseline is **identity** (raw OCR, no correction): emit the noisy text unchanged. This is the bar the neural corrector must beat.

**Validated identity-baseline numbers on the synthetic test distribution:**

| Metric | Identity (raw OCR vs. gold) |
|--------|-----------------------------|
| **CER** | ~**0.088** |
| **WER** | ~**0.49** |
| **Exact match** | ~**0.0005** |

These numbers confirm the corrector has a clear, measurable job: nearly half the words and ~9% of characters are wrong before correction, and almost no passage is exactly correct. The trained ByT5 model reduces CER substantially below this baseline.

An optional **SymSpell dictionary corrector** (`symspellpy`, MIT) and an optional comparator (`oliverguhr/spelling-correction-english-base`, MIT BART) provide additional reference points.

### Headline metric: error reduction

The headline result is **error reduction**, not absolute error:

```
CER_before = CER(raw_OCR, gold)
CER_after  = CER(corrected, gold)
% reduction = (CER_before - CER_after) / CER_before
```

The same is computed for WER. A **safety gate** tracks the share of examples **improved vs. degraded vs. unchanged** — a corrector that fixes 40% of cases but breaks 30% is not acceptable, so we require *degraded ≪ improved*. ExactMatch is also reported. (All distances are jiwer-style char/word Levenshtein.)

---

## 6. Known limitations & biases

- **Simulation-to-reality gap.** The primary data is *synthetic*. A real OCR engine's error distribution may not match the confusion model in `ocr_noise.py`. Mitigation: mix real `PleIAs` data into training and **report real `PleIAs` CER** alongside synthetic CER, so the gap is visible rather than hidden.
- **Synthetic noise distribution may differ from a specific engine.** The confusion sets and error rate are tuned to be *representative* of OCR errors in general, not calibrated to any one engine (Tesseract vs. doctr vs. PaddleOCR vs. Surya). A deployment on a particular engine may see error patterns the generator under- or over-represents. Robustness across `char_error_rate` levels is measured in the robustness report.
- **English-first.** Both the synthetic generator (English clean text) and the real mix (`PleIAs` config `english`) are English. The corrector is not validated for other languages.
- **Benchmark availability.** The optional ICDAR-2019 corpus has its dataset viewer disabled; the loader degrades gracefully, so this benchmark may be unavailable in a given environment.
- **Corrector risk (data-driven).** Because correction is generative, the model can in principle rewrite or hallucinate a region. This is a property of the model rather than the data, but it shapes evaluation: the **% improved vs. degraded** safety gate and the agent's bounded edit-budget acceptance (edit ratio ≤ 0.35) exist specifically to catch corrections that make text worse.

---

## 7. Summary

| Aspect | Choice |
|--------|--------|
| Primary data | Reproducible **synthetic OCR-noise generator** (`ocr_noise.py`, default `char_error_rate` 0.08) |
| Real mix / eval | `PleIAs/Post-OCR-Correction` `english` (CC0-1.0, 31.3K, `text` + `corrected_text`) |
| Optional benchmark | `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` (graceful skip) |
| Default split sizes | train 60,000 / val 4,000 / test 4,000 (+ real eval slice) |
| Leakage control | Dedup by clean text → leakage-free splits |
| Baseline | **Identity** (raw OCR): CER ~0.088, WER ~0.49, ExactMatch ~0.0005 |
| Headline metric | **% CER/WER reduction** + improved/degraded safety gate |
