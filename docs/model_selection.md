# Model Selection & Optimization — P07 Document-Level OCR

**Project:** Document-Level OCR System
**Author:** Le Dinh Minh Quan (23127460)
**Course:** NLP in Industry — Final Assignment

This document explains the model choice, architecture, training procedure, baselines, error analysis, and trade-offs for the **trainable ML core** of P07: the **post-OCR error corrector**. The OCR front-end (Tesseract / docTR / PaddleOCR / Surya) and the layout/reading-order logic are pretrained or algorithmic — they are *not* trained here. The single component we train, tune, and measure is a sequence-to-sequence text→text model that fixes OCR errors on top of **any** OCR engine.

---

## 1. The problem this model solves

A flat OCR pass produces an error-riddled text dump. Beyond the structural problems (wrong reading order, no block typing) that layout analysis handles, the raw text contains **character-level OCR garble**:

- character confusions: `rn → m`, `0 ↔ O`, `1 ↔ l`, `cl ↔ d`, `e ↔ c`, `S ↔ 5`, `g ↔ q`
- spurious insertions and deletions
- word **merge / split** errors (lost or inserted spaces)
- random case flips

On our synthetic test distribution, the **identity baseline** (raw OCR, no correction) measures:

| Metric | Identity (raw OCR) |
| --- | --- |
| CER | ~0.088 |
| WER | ~0.49 |
| ExactMatch | ~0.0005 |

That ~0.49 WER and near-zero exact-match rate mean the corrector has a **clear, measurable job**: reduce the error rate of whatever the OCR engine produced.

The task is therefore framed as **Post-OCR Error Correction**: given noisy OCR text, emit the corrected text. We prepend the prefix `correct: ` to every input, in the T5 text-to-text convention.

---

## 2. Why ByT5-small

**Selected model: `google/byt5-small`** (Apache-2.0, ~300M parameters, byte/character-level, **no SentencePiece**).

### 2.1 The decisive property: byte-level robustness to character noise

OCR errors are fundamentally **character-level** events: a single glyph flips (`0→O`), one character is inserted, two characters merge into one. Subword tokenizers (SentencePiece / BPE, as used by `t5-small`) are exactly the wrong granularity for this:

- A one-character corruption can **shatter the subword segmentation** of an entire word, turning one in-vocabulary token into several rare or `<unk>`-adjacent pieces. The model never sees the corruption as a local, learnable edit — it sees a global tokenization change.
- The errors we care about (`rn↔m`, `cl↔d`) live *inside* tokens, where a subword model has no handle on them.

ByT5 operates directly on **UTF-8 bytes**. There is no learned subword vocabulary at all — the input alphabet is the 256 byte values plus a handful of special IDs. A character flip is a *byte* flip: a small, local perturbation the model can learn to undo. This is precisely why the **ByT5 paper (Xue et al., arXiv:2105.13626)** reports that byte-level models are **markedly more robust to noise** (including synthetic spelling/character corruption) than their subword counterparts. Post-OCR correction is a near-ideal application of that finding.

### 2.2 What we rejected and why

| Candidate | Why not chosen |
| --- | --- |
| `t5-small` (subword) | SentencePiece segmentation is brittle under character-level noise; a single OCR flip re-tokenizes the word. Kept only as the **T4 fallback** (see §3.2). |
| Word-level / token-classification corrector | Cannot model merge/split (space) errors or sub-word edits; assumes word boundaries are already correct, which OCR breaks. |
| Dictionary-only (e.g. SymSpell alone) | No context: cannot disambiguate real-word errors, cannot fix merges/splits cleanly, no learning from the actual OCR error distribution. Kept only as a **baseline** (§4). |

ByT5's only real cost is sequence length (bytes are more numerous than subword tokens — see §6), which we accept in exchange for the robustness that the task demands.

### 2.3 Architecture

ByT5 is a **T5 encoder–decoder Transformer**:

- **Encoder** reads the byte sequence of the noisy input (`correct: <noisy text>`).
- **Decoder** autoregressively generates the byte sequence of the corrected text.
- **Vocabulary:** raw UTF-8 bytes (no SentencePiece model, no subword merges). ByT5 re-balances parameters toward a heavier encoder relative to standard T5 to compensate for longer byte sequences.
- Training objective: standard seq2seq cross-entropy (with label smoothing), decoding via `predict_with_generate`.

---

## 3. Training procedure

Training uses the Hugging Face **`Seq2SeqTrainer`** with `predict_with_generate=True` so that CER/WER are computed on actually-generated text, not teacher-forced logits.

### 3.1 Hyperparameter configuration

```python
TRAIN_CONFIG = {
    "model_name": "google/byt5-small",   # T4 fallback: "google-t5/t5-small"
    "prefix": "correct: ",

    # optimization
    "learning_rate": 5e-4,               # 3e-4 for t5-small
    "effective_batch_size": 256,         # per-device 32 x grad-accum 8 (H100)
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "label_smoothing_factor": 0.1,

    # precision
    "bf16": True,                        # H100 / A100
    "tf32": True,                        # H100 / A100
    "fp16": False,                       # True on T4 instead

    # generation / eval
    "predict_with_generate": True,
    "metric_for_best_model": "cer",      # lower is better
    "greater_is_better": False,
    "load_best_model_at_end": True,

    # efficiency / robustness
    "group_by_length": True,
    "early_stopping_patience": 4,        # on CER
    "resume_from_checkpoint": "auto",    # via get_last_checkpoint
}
```

Key choices:

- **`metric_for_best_model = "cer"`, lower is better** — we optimize directly for the headline metric, not validation loss.
- **Early stopping (patience 4) on CER** + `load_best_model_at_end` — checkpoint selection on the metric that matters, with `get_last_checkpoint` allowing crash-safe resume.
- **`group_by_length`** — batches similar-length byte sequences together, important because byte sequences vary widely and ByT5 sequences are long.
- **Cosine schedule, warmup 0.05** — stable convergence for a from-pretrained fine-tune.

### 3.2 GPU profile table

The trainer adapts precision, per-device batch size, gradient accumulation, and (on the smallest GPU) the base model to the available hardware, while keeping the **effective batch size near 256**:

| GPU | Precision | Per-device BS | Grad-accum | Grad checkpointing | Base model |
| --- | --- | --- | --- | --- | --- |
| **H100** | bf16 + tf32 | 32 | 8 | off | byt5-small |
| **A100-40** | bf16 + tf32 | 16 | 16 | on | byt5-small |
| **L4** | bf16 | 8 | 32 | on | byt5-small |
| **T4** | fp16 | 4 | 64 | on | **t5-small** (fallback) |

On **T4** the base model switches to `t5-small` because byte sequences are too long/memory-heavy for the 16 GB card; fp16 replaces bf16 (no bf16 support). Rough wall-clock for byt5-small is **~3–6 h on a single H100**.

### 3.3 Anti-overfitting

The corrector must generalize across OCR engines and noise levels, not memorize one synthetic distribution. Defenses:

- **Diverse confusion model** — the synthetic generator (`src/dococr/data/ocr_noise.py`) samples a wide range of substitution/insertion/deletion/merge/split/case errors at a tunable `char_error_rate` (default **0.08**), so the model sees varied corruption rather than a fixed pattern.
- **Real-data mixing** — training mixes in the real **`PleIAs/Post-OCR-Correction`** (config `english`) corpus so the model is exposed to *actual* OCR error distributions, not only synthetic ones.
- **Early stopping** on CER + **weight decay (0.01)** + **label smoothing (0.1)**.
- **Leakage-free splits** — splits are deduplicated by clean text, and a real eval slice is held out, so validation/test cannot leak through shared source text.

### 3.4 Data summary

| Source | Role | License / size |
| --- | --- | --- |
| Synthetic OCR-noise generator (`ocr_noise.py`) | **Primary** training data (reproducible) | default `char_error_rate=0.08`; train **60000**, val **4000**, test **4000** |
| `PleIAs/Post-OCR-Correction` (config `english`) | Real-data mix + real eval slice | CC0-1.0, 31.3K rows; `text` (noisy) + `corrected_text` (gold) |
| `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` | Optional benchmark | manual corpus; viewer disabled → loader **degrades gracefully** |

---

## 4. Baselines and the bar to beat

The corrector is only worth shipping if it beats the trivial alternative. We compare against three baselines:

1. **Identity** (raw OCR, no correction) — the canonical post-OCR baseline. CER ~0.088, WER ~0.49 on the synthetic test set. **The neural corrector must reduce CER below this.**
2. **SymSpell dictionary corrector** (`symspellpy`, MIT) — optional classical baseline; dictionary lookup with edit-distance, no context.
3. **`oliverguhr/spelling-correction-english-base`** (MIT, BART) — optional neural comparator from a different model family.

### 4.1 Headline metric: error **reduction**

We report **reduction**, not just absolute error, so improvement over raw OCR is explicit:

```
CER_before = CER(raw_OCR, gold)
CER_after  = CER(corrected, gold)
% reduction = (CER_before - CER_after) / CER_before     # same for WER
```

Metrics are jiwer-style character/word Levenshtein. The trained **ByT5 reduces CER substantially** below the identity baseline.

### 4.2 Safety gate: improved vs degraded

A corrector that fixes 40% of examples but **breaks** 30% is useless — and dangerous, because it silently corrupts text that was already correct. We therefore track, per example, whether the correction made it **improved / degraded / unchanged**, and require:

> **degraded ≪ improved**

ExactMatch is reported alongside. This safety gate is enforced again at inference time by the agent's **D4 correction-acceptance** decision: a correction is accepted only when it is a **bounded edit** (edit ratio ≤ 0.35) and confident enough; otherwise the raw OCR text is kept. This prevents the corrector from hallucinating or rewriting a region away.

---

## 5. Error analysis approach

Evaluation goes beyond aggregate CER/WER to *characterize* the corrector's behaviour (`src/dococr/analysis/error_analysis.py`):

- **Improved vs degraded vs unchanged** buckets — the core safety view: how often the model helps vs hurts, with concrete degraded examples surfaced for inspection.
- **Error-type breakdown** — which corruption classes (substitution, insertion, deletion, merge/split, case) the model fixes well vs poorly, mapped back to the confusion sets in the synthetic generator.
- **Robustness sweep** (`analysis/robustness.py`) — CER reduction measured across increasing `char_error_rate` levels, to see where the corrector saturates or starts to degrade.
- **Sim-to-real check** — report CER on the **real PleIAs** slice as well as synthetic, to quantify the sim-to-real gap (synthetic noise may not match a specific engine's error distribution).

---

## 6. Trade-offs

### 6.1 Accuracy vs speed

- **ByT5 byte sequences are longer** than subword sequences for the same text — more encoder/decoder steps per document, so **slower per token** than a subword model. We accept this because byte-level granularity is what makes the model robust to character-level OCR noise (§2.1). Correction runs at **~80 ms/region** (small model), which is small next to OCR itself (~0.6–1.2 s/region scanned).
- **small vs base** — we use the *small* variant. A larger ByT5 would likely correct more, but at higher latency and memory cost; for a CPU-deployable, per-region corrector that runs behind a bounded-edit gate, small is the right accuracy/cost point. The **T4 fallback to `t5-small`** is a further speed/memory concession when byte sequences won't fit.
- Throughput is recovered via **page/region parallelism and GPU batching**, and by the agent's **D2** decision, which skips OCR *and* correction entirely on born-digital pages (~200 ms/page).

### 6.2 Complexity vs maintainability

- A trainable seq2seq corrector is more complex than a dictionary, but it is the **measurable differentiator** of P07: it learns the actual OCR error distribution and lowers CER/WER on top of any engine.
- Complexity is contained by keeping the corrector **modular and swappable** (`models/corrector.py` behind a model registry), by the **bounded-edit D4 gate** that caps the blast radius of any single bad correction, and by **graceful degradation everywhere** — if `torch` is unavailable the pipeline falls back to the identity corrector, so the document system still runs end-to-end.

---

## 7. Summary

| Decision | Choice | Rationale |
| --- | --- | --- |
| Model | `google/byt5-small` | Byte-level → robust to char-level OCR noise (arXiv:2105.13626); no SentencePiece |
| Fallback | `google-t5/t5-small` | Fits T4 16 GB where byte sequences don't |
| Architecture | T5 encoder–decoder, byte vocab | Local byte edits = learnable corrections |
| Trainer | HF `Seq2SeqTrainer`, `predict_with_generate` | Optimize/select on generated-text CER |
| Headline metric | % CER/WER **reduction** vs identity | Must beat raw OCR |
| Safety gate | improved ≫ degraded; bounded edit (≤0.35) | A corrector that breaks text is useless |

The trained ByT5-small corrector reduces CER substantially below the identity baseline while keeping the degraded rate low — exactly the two conditions a post-OCR corrector must satisfy to be worth deploying.
