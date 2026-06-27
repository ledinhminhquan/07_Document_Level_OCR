# Correction Quality Evaluation — P07 Document-Level OCR

**Author:** Le Dinh Minh Quan (23127460) · **Course:** NLP in Industry (final assignment)
**Scope of this document:** how we measure whether the **trainable post-OCR corrector** (the ML core of P07) actually improves text quality — and, just as importantly, whether it ever makes things *worse*.

The OCR front-end (Tesseract / docTR / PaddleOCR) and the layout + reading-order stage are pretrained / algorithmic and are evaluated separately (structure fidelity, reading-order correctness, latency). This document is the **quality deep-dive for the corrector**: a `google/byt5-small` seq2seq model that maps noisy OCR text to clean gold text under the `"correct: "` prefix, on top of *any* OCR engine.

---

## 1. Why a dedicated correction evaluation?

A post-OCR corrector is a **risky** component: it is a generative model rewriting text, so it can hallucinate or paraphrase a region into something that no longer matches the source. A naïve "average CER went down" headline can hide the fact that the model fixed many easy cases while corrupting a meaningful minority. Our evaluation therefore reports **two things that must both hold**:

1. **Aggregate error reduction** — CER and WER drop materially versus the raw-OCR floor.
2. **A safety gate** — far more examples are *improved* than *degraded*; the corrector must not be a net liability per-example even when the average looks good.

All metric code lives in **`src/dococr/training/metrics.py`**; the evaluation driver is **`src/dococr/training/evaluate.py`**; per-example error breakdowns are in **`src/dococr/analysis/error_analysis.py`**; noise-severity sweeps are in **`src/dococr/analysis/robustness.py`**.

---

## 2. Metric definitions

All edit-distance metrics are Levenshtein-based (jiwer-style char/word distance). Let `ref` be the gold text and `hyp` the candidate (raw OCR for the baseline, model output for the corrector).

### 2.1 Character Error Rate (CER)

$$
\text{CER} = \frac{S_c + D_c + I_c}{N_c}
$$

where `S_c`, `D_c`, `I_c` are character-level substitutions, deletions, and insertions in the optimal alignment of `hyp` to `ref`, and `N_c` is the number of characters in `ref`. Lower is better. CER is the **primary** corrector metric because OCR noise is fundamentally character-level (e.g. `rn`→`m`, `0`↔`O`), and ByT5's byte-level vocabulary is matched to exactly this granularity.

### 2.2 Word Error Rate (WER)

$$
\text{WER} = \frac{S_w + D_w + I_w}{N_w}
$$

the same formula at word granularity (`N_w` = number of words in `ref`). WER is the secondary metric: it captures word merges/splits and spacing errors that the noise model injects.

### 2.3 Exact Match (EM)

$$
\text{ExactMatch} = \frac{1}{M}\sum_{i=1}^{M} \mathbb{1}\big[\,\text{hyp}_i = \text{ref}_i\,\big]
$$

fraction of test examples reproduced character-for-character. On the synthetic distribution raw OCR almost never matches gold (EM ≈ 0.0005), so any non-trivial EM from the corrector is signal.

### 2.4 Headline metric — Error Reduction (before vs. after)

The thing we actually advertise is **relative reduction** of error versus the raw OCR input:

$$
\Delta\text{CER}\% = \frac{\text{CER}_{\text{before}} - \text{CER}_{\text{after}}}{\text{CER}_{\text{before}}}\times 100
\qquad
\Delta\text{WER}\% = \frac{\text{WER}_{\text{before}} - \text{WER}_{\text{after}}}{\text{WER}_{\text{before}}}\times 100
$$

- `CER_before = CER(raw_OCR, gold)` — error of the text *entering* the corrector.
- `CER_after = CER(corrected, gold)` — error of the text *leaving* the corrector.

A positive `ΔCER%` means the corrector removed that fraction of the character errors; the same applies to `ΔWER%`. This pairing is the headline result of the whole P07 ML effort.

### 2.5 Safety gate — improved vs. degraded vs. unchanged

For each test example we compare its CER before and after correction and bucket it:

| Bucket | Condition | Meaning |
|---|---|---|
| **Improved** | `CER_after < CER_before` | corrector helped |
| **Degraded** | `CER_after > CER_before` | corrector *hurt* this example |
| **Unchanged** | `CER_after = CER_before` | no net effect |

We report the **percentage in each bucket**. The acceptance rule is plain:

> A corrector that fixes 40% of examples but breaks 30% is useless. We require **degraded ≪ improved**.

This guards against the central risk of the project (the corrector making things worse / hallucinating). It is reported *alongside* the headline reduction — the average can look great while the degraded share is unacceptable, and only the safety gate exposes that.

> **Note on alignment with the agent.** The runtime agent enforces a complementary guard at decision point **D4**: the corrector's output is accepted only if it is a **bounded edit** (edit ratio ≤ 0.35) and confident enough; otherwise the raw region is kept. The `degraded` bucket measures what *would* happen without that gate, so the two numbers together tell us both the model's raw quality and the residual risk the gate has to absorb.

---

## 3. Evaluation protocol

### 3.1 Two evaluation slices (sim-to-real)

| Slice | Source | Why it exists |
|---|---|---|
| **Synthetic test slice** | `src/dococr/data/ocr_noise.py` generator over clean English text (default test size **4 000**) | controlled, reproducible noise with a known confusion model; lets us sweep noise severity |
| **Real slice** | `PleIAs/Post-OCR-Correction`, config `"english"` (CC0-1.0, 31.3K rows; `text` = noisy OCR, `corrected_text` = gold) | real OCR error distribution from real documents — closes the **sim-to-real gap** |

We always report the real PleIAs CER too, not just synthetic, because synthetic noise may not match a specific engine's error distribution. The two slices together are the sim-to-real check. Optionally the manually-corrected `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` corpus can be added as an extra benchmark (its viewer is disabled, so the loader degrades gracefully).

**Leakage control.** Splits are deduplicated by clean text so the same gold passage cannot appear in both train and test. Synthetic defaults: train 60 000 / val 4 000 / test 4 000.

### 3.2 What gets compared

For every example in a slice we record `(gold, raw_OCR, candidate)` triples for each system, then compute CER / WER / EM and the before-vs-after reduction against the **raw OCR** as the `before` reference. The driver is `evaluate.py`; per-error-type diagnostics (which confusion classes survive correction) come from `error_analysis.py`.

---

## 4. Baselines

The corrector is judged against two reference points, in increasing strength:

1. **Identity (raw OCR, no correction)** — the canonical post-OCR baseline and the **floor** the neural model must beat. By definition its `CER_after = CER_before`, so its error reduction is 0% and its safety-gate split is 100% unchanged.
2. **SymSpell dictionary corrector** (`symspellpy`, MIT) — a non-neural dictionary/edit-distance speller. It fixes some isolated misspellings but has no document/byte-level context. An optional neural comparator, `oliverguhr/spelling-correction-english-base` (MIT, BART), may also be reported.

The bar is unambiguous: **the trained ByT5 corrector must reduce CER below identity**, and ideally below the dictionary baseline, while keeping the degraded share far below the improved share.

### 4.1 Validated floor (identity)

These identity numbers are **measured** on the synthetic test distribution and are the fixed floor every other column is read against:

| Metric | Identity (raw OCR) |
|---|---|
| CER | **0.088** |
| WER | **0.49** |
| ExactMatch | **0.0005** |

So roughly 8.8% of characters and 49% of words are wrong before correction, and essentially nothing matches gold exactly — i.e. the corrector has a clear, measurable job to do. The trained ByT5 reduces CER substantially below this floor.

---

## 5. Results table template (fill after training)

Reported on the **synthetic test slice** (default 4 000 examples) and, separately, on the **real PleIAs `english` slice**. Identity column is pre-filled with the validated floor.

| Metric | Identity (raw OCR) | Dictionary (SymSpell) | ByT5 (`google/byt5-small`) |
|---|---|---|---|
| CER ↓ | **0.088** | _fill_ | _fill_ |
| WER ↓ | **0.49** | _fill_ | _fill_ |
| ExactMatch ↑ | **0.0005** | _fill_ | _fill_ |
| **ΔCER% (reduction) ↑** | 0% (floor) | _fill_ | _fill_ |
| **ΔWER% (reduction) ↑** | 0% (floor) | _fill_ | _fill_ |
| % Improved ↑ | 0% | _fill_ | _fill_ |
| % Degraded ↓ | 0% | _fill_ | _fill_ |
| % Unchanged | 100% | _fill_ | _fill_ |

Repeat the same table for the **PleIAs real slice** to expose any sim-to-real gap (identity CER on the real slice is measured, not assumed to be 0.088).

**Reading the table.**
- The ByT5 `CER` cell must be **below 0.088** to clear the floor.
- The headline lives in the `ΔCER%` / `ΔWER%` rows.
- The submission is only valid if **% Degraded ≪ % Improved** for the ByT5 column.

---

## 6. Robustness across noise levels

A corrector trained at one noise level can collapse at another. Because the synthetic generator exposes a tunable `char_error_rate` (default **0.08**, which produces the validated CER ≈ 0.088 floor), we sweep it and re-run the full metric suite at each level. This is implemented in **`src/dococr/analysis/robustness.py`** and produces the robustness report.

| `char_error_rate` (input noise) | Identity CER (before) | ByT5 CER (after) | ΔCER% | % Improved | % Degraded |
|---|---|---|---|---|---|
| 0.04 (light) | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| 0.08 (default, validated floor 0.088) | ≈ 0.088 | _fill_ | _fill_ | _fill_ | _fill_ |
| 0.12 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| 0.16 (heavy) | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

What we look for:

- **Monotonic input difficulty** — identity (before) CER should climb with `char_error_rate`, confirming the sweep is real.
- **Graceful degradation** — `ΔCER%` may shrink at high noise, but it should stay **positive**; the corrector should not flip to a net-negative reduction.
- **Safety holds under stress** — the degraded share must not explode as noise rises. If it does, the agent's D4 edit-budget gate is doing more of the work and the model needs more diverse confusion training.

The diverse confusion model (substitutions `rn↔m`, `cl↔d`, `O↔0`, `l↔1`, `e↔c`, `S↔5`, `g↔q`, …, plus insertions, deletions, word merge/split, case flips) is precisely what gives the byte-level ByT5 its noise robustness, consistent with the ByT5 finding (arXiv:2105.13626) that byte models are markedly more robust to character-level noise than subword models.

---

## 7. Reproducing the evaluation

```bash
# Identity + dictionary + ByT5 on the synthetic test slice and the real PleIAs slice
dococr evaluate

# Per-error-type breakdown (which confusion classes survive correction)
dococr error-analysis

# Noise-severity sweep (CER/WER reduction + safety gate per char_error_rate)
dococr robustness
```

| Concern | Where it lives |
|---|---|
| CER / WER / ExactMatch / reduction / safety-gate buckets | `src/dococr/training/metrics.py` |
| Evaluation driver (baselines + ByT5, both slices) | `src/dococr/training/evaluate.py` |
| Per-example & per-error-type diagnostics | `src/dococr/analysis/error_analysis.py` |
| Noise-level robustness sweep | `src/dococr/analysis/robustness.py` |

---

## 8. Pass criteria (summary)

The corrector evaluation is considered a success when, on the ByT5 column:

1. **CER < 0.088** and **WER < 0.49** on the synthetic test slice (beats the validated identity floor), and ideally beats the SymSpell dictionary baseline.
2. **ΔCER% and ΔWER% are clearly positive** — a material headline reduction.
3. **% Degraded ≪ % Improved** — the safety gate holds; the corrector is a net help per-example, not just on average.
4. **The real PleIAs slice corroborates the synthetic result**, with the sim-to-real gap reported rather than hidden.
5. **Robustness sweep stays positive** across rising `char_error_rate`, with no blow-up in the degraded share.
