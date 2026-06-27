# Model Card: `dococr-postocr-byt5-small`

Post-OCR error-correction model for English document text. Engine-agnostic: it sits on top of **any** OCR front-end (Tesseract, docTR, PaddleOCR, Surya, EasyOCR, TrOCR, ...) and measurably lowers Character Error Rate (CER) and Word Error Rate (WER) relative to the raw OCR output.

This model is the trainable ML core of the **P07 Document-Level OCR System**. The OCR front-end, layout analysis, and reading-order recovery in that system are pretrained/algorithmic; this corrector is the one component that is *trained* on the task and is the system's measured differentiator.

| | |
|---|---|
| **Model name** | `dococr-postocr-byt5-small` |
| **Base model** | [`google/byt5-small`](https://huggingface.co/google/byt5-small) (Apache-2.0, ~300M params) |
| **Architecture** | Encoder–decoder (seq2seq), text → text |
| **Tokenization** | Byte/character-level — **no SentencePiece**, robust to character-level OCR noise (ByT5, arXiv:2105.13626) |
| **Task** | Post-OCR error correction |
| **Task prefix** | `correct: ` (prepend to every input) |
| **Language** | English |
| **License** | Apache-2.0 (inherited from base) |
| **T4 fallback base** | [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small) (Apache-2.0, 60.5M params) |

---

## Model details

`dococr-postocr-byt5-small` fine-tunes `google/byt5-small` to map noisy OCR text to its clean, corrected form. It is a standard sequence-to-sequence model and is loaded with `AutoModelForSeq2SeqLM`.

The base model is **ByT5**, a byte-level / character-level encoder–decoder. This is a deliberate choice: OCR errors are overwhelmingly *character-level* (`rn`→`m`, `0`↔`O`, `1`↔`l`, `e`↔`c`, `S`↔`5`, merged/split words). A byte-level model has no fixed subword vocabulary to break on garbled tokens, and the ByT5 paper (arXiv:2105.13626) shows byte models are markedly more noise-robust than subword models — exactly the property needed for post-OCR cleanup.

Every input must be prefixed with `correct: `. The model outputs the corrected text directly.

---

## Intended use

- **Primary use:** Post-OCR error correction of **English document text**. Given a string of raw OCR output, produce a cleaner string with OCR-induced character and word errors fixed.
- **Engine-agnostic:** The model does not depend on which OCR engine produced its input. It improves text from Tesseract, docTR, PaddleOCR, Surya, EasyOCR, TrOCR, or any other source.
- **Where it fits:** In the P07 pipeline it runs per text region after OCR and reading-order recovery (`ingest → layout → correct → assemble`), and is also exposed directly via the `POST /correct` API endpoint (raw text in, corrected text out, no image required).

### Out-of-scope use

This model **must not change the meaning** of the text. It is a corrector, not a rewriter. It is **not** intended for:

- **Non-English text** — trained English-first; other languages are out of scope.
- **Handwriting** — trained on printed-document OCR noise, not handwritten recognition.
- **Content rewriting, paraphrasing, or summarization** — the model should fix OCR garble and nothing else. It must not summarize, rephrase, translate, or otherwise alter content meaning.

Because over-correction (hallucinated rewrites) is the principal failure mode, the production pipeline wraps the model in an **edit-budget guard** at inference (see *Limitations*).

---

## Training data

Training mixes a reproducible synthetic OCR-noise generator (primary) with a real post-OCR correction corpus.

### Synthetic OCR-noise generator (primary)

A reproducible generator (`src/dococr/data/ocr_noise.py`) corrupts clean English text with a realistic OCR confusion model:

- **Character substitutions** from confusion sets: `rn`↔`m`, `cl`↔`d`, `O`↔`0`, `l`↔`1`, `e`↔`c`, `S`↔`5`, `g`↔`q`, ...
- **Insertions** and **deletions**
- **Word merge / split** (spacing errors)
- **Case flips**

Corruption severity is controlled by a tunable `char_error_rate` (default `0.08`). Default synthetic split sizes:

| Split | Size |
|---|---|
| train | 60,000 |
| validation | 4,000 |
| test | 4,000 |

### Real corpus (mix-in + real eval slice)

- **[`PleIAs/Post-OCR-Correction`](https://huggingface.co/datasets/PleIAs/Post-OCR-Correction)**, config `english` (CC0-1.0, 31.3K rows). Columns: `text` (noisy OCR) and `corrected_text` (gold). Mixed into training and used as a real evaluation slice.
- **Optional benchmark:** `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` (manually corrected corpus; its dataset viewer is disabled, so the loader degrades gracefully).

Splits are kept **leakage-free** by deduplicating on the clean text, with a dedicated real-data evaluation slice so synthetic and real performance can be reported separately.

---

## Training procedure

Training uses the Hugging Face `Seq2SeqTrainer` with `predict_with_generate`.

### Hyperparameters

| Setting | Value |
|---|---|
| Optimizer schedule | Cosine, warmup ratio `0.05` |
| Learning rate | `5e-4` (byt5) / `3e-4` (t5-small fallback) |
| Effective batch size | 256 (per-device 32 × grad-accum 8 on H100) |
| Weight decay | `0.01` |
| Label smoothing | `0.1` |
| Precision | bf16 + tf32 (H100/A100), fp16 (T4) |
| Early stopping | Patience 4 on CER (lower is better) |
| Checkpointing | `load_best_model_at_end`, `group_by_length`, resume via `get_last_checkpoint` |

### Hardware / GPU profiles

| GPU | Precision | Per-device BS | Grad-accum | Notes |
|---|---|---|---|---|
| H100 | bf16 | 32 | 8 | reference profile |
| A100-40 | bf16 | 16 | 16 | gradient checkpointing |
| L4 | bf16 | 8 | 32 | gradient checkpointing |
| T4 | fp16 | 4 | 64 | gradient checkpointing + switch base to `t5-small` |

Rough training time for `byt5-small` is **~3–6 h on a single H100**.

### Anti-overfitting

A diverse confusion model, mixing in real PleIAs data, early stopping on CER, and weight decay together guard against overfitting to any single synthetic noise distribution.

---

## Evaluation

The headline metric is **error reduction**, not absolute error.

- **CER reduction** = `(CER_before − CER_after) / CER_before`, where `CER_before = CER(raw_OCR, gold)` and `CER_after = CER(corrected, gold)`.
- **WER reduction** — same formula on words.
- **Safety gate** — per-example breakdown of **% improved vs % degraded vs % unchanged**. A corrector that fixes 40% of examples but breaks 30% is useless; the requirement is `degraded ≪ improved`.
- **ExactMatch** — fraction of outputs exactly equal to gold.

Distances are jiwer-style character/word Levenshtein.

### Baselines

| Baseline | Description |
|---|---|
| **Identity** (raw OCR, no correction) | Canonical post-OCR baseline. The neural corrector must reduce CER **below** identity. |
| **SymSpell dictionary** (optional) | `symspellpy` dictionary corrector (MIT). |
| **BART comparator** (optional) | `oliverguhr/spelling-correction-english-base` (MIT). |

### Validated identity baseline (synthetic test distribution)

| Metric | Identity (raw OCR) |
|---|---|
| CER | ~0.088 |
| WER | ~0.49 |
| ExactMatch | ~0.0005 |

These numbers establish that the corrector has a clear, measurable job. The trained ByT5 reduces CER substantially relative to this baseline.

### Real vs synthetic

Performance is reported separately on the synthetic test slice and on the real **PleIAs `english`** slice, so the synthetic-to-real gap is visible rather than hidden.

---

## Limitations and biases

- **Simulation-to-real gap.** The synthetic noise model may not match a specific OCR engine's true error distribution. Mitigations: mix in real PleIAs data during training, and always report the **real PleIAs CER** alongside the synthetic CER.
- **Over-correction / hallucination is the main risk.** A seq2seq corrector can rewrite or hallucinate text instead of merely fixing it. The P07 pipeline mitigates this with an **edit-budget guard (decision point D4)**: the corrector's output is accepted **only if it is a bounded edit** (edit ratio ≤ `0.35`) and confident enough; otherwise the raw OCR text is kept. The **% degraded** metric directly tracks this failure mode.
- **English-only.** Trained on English (PleIAs `english` + English synthetic); other languages are out of scope.
- **Printed text, not handwriting.** Trained on printed-document OCR noise.
- **Robustness.** Behavior under increasing noise severity is measured (a robustness report sweeps `char_error_rate` levels); out-of-distribution scans (skew/blur) are handled upstream by preprocessing rather than by this model.

---

## Ethical considerations

- **Privacy / PII.** Documents may contain personally identifiable or sensitive information (names, IDs, addresses, financial data). Deployments should minimize retention and logging of raw images and text, offer an on-prem / no-retention option, and apply TTL cleanup.
- **Rights and consent.** OCR and correction of private or copyrighted documents requires appropriate rights and consent.
- **Meaning preservation.** Because the corrector edits text that may be relied upon downstream, the system enforces bounded edits and routes low-confidence regions to human review (OCR-confidence gate D3, threshold `0.55`) rather than silently accepting risky rewrites.

---

## How to use

Load with `AutoModelForSeq2SeqLM` and **prepend the `correct: ` prefix** to every input.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "dococr-postocr-byt5-small"  # local path or hub id
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

raw_ocr = "Tlie qulck br0wn f0x junlps over tl1e lazy d0g."
inputs = tokenizer("correct: " + raw_ocr, return_tensors="pt")

output_ids = model.generate(**inputs, max_new_tokens=256)
corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(corrected)
```

> **Note:** In production, wrap the model output in the edit-budget guard (accept only if the edit ratio vs. the raw input is ≤ `0.35`) so the corrector cannot rewrite a region away. In the P07 system this is handled automatically by the agent at decision point D4, and the model is also served directly via `POST /correct`.

---

*Model card for the P07 Document-Level OCR final-assignment submission. Author: Le Dinh Minh Quan (student 23127460), NLP in Industry. Package: `src/dococr/`.*
