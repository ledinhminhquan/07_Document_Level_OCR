# 📄 Document-Level OCR System

> Turn document **images / PDFs** into clean, **structured, reading-order-correct** text
> (Markdown + JSON blocks + plain text) — with an OCR front-end, layout & reading-order
> analysis, a **trainable post-OCR correction model**, and an **agentic** pipeline.

**NLP in Industry — Final Assignment.** Author: **Le Dinh Minh Quan** (Student `23127460`).

A flat OCR pass gives an unordered, error-riddled dump (`rn→m`, `0↔O`, `1↔l`, merged words;
wrong reading order on multi-column pages). P07 fixes both: **layout + reading order** give
structure, and a trained **ByT5 post-OCR corrector** measurably lowers CER/WER on top of *any*
OCR engine. An **agent** orchestrates the whole document → structured-text pipeline.

---

## ✅ How this repo meets every assignment requirement

| Requirement | Where it is delivered |
|---|---|
| **Business problem** | [`docs/problem_definition.md`](docs/problem_definition.md) |
| **Dev infra & tooling** | `src/` package, `pyproject.toml`, `requirements*.txt`, `Makefile`, Docker, CI |
| **Data management** | synthetic OCR-noise generator ([`data/ocr_noise.py`](src/dococr/data/ocr_noise.py)) + real PleIAs; [`docs/data_description.md`](docs/data_description.md), [`docs/data_card.md`](docs/data_card.md) |
| **Model selection & optimization** | ByT5 post-OCR corrector + **identity/dictionary baselines**; CER/WER-reduction; [`docs/model_selection.md`](docs/model_selection.md) |
| **Deployment** | FastAPI `/ocr` + `/correct` + Gradio + CLI + Docker + HF Space; [`docs/deployment.md`](docs/deployment.md) |
| **Agentic AI** | deterministic FSM with **4 decision points** + optional LLM brain; [`docs/agent_architecture.md`](docs/agent_architecture.md) |
| **Continual learning & monitoring** | [`docs/continual_learning_monitoring.md`](docs/continual_learning_monitoring.md) + [`monitoring/drift_report.py`](src/dococr/monitoring/drift_report.py) |
| **Privacy & robustness** | [`docs/privacy_robustness.md`](docs/privacy_robustness.md) + [`analysis/robustness.py`](src/dococr/analysis/robustness.py) |
| **Project management** | [`docs/project_plan.md`](docs/project_plan.md) |
| **Ethics** | [`docs/ethics_statement.md`](docs/ethics_statement.md) |
| **Report + slides** | auto-generated `report.pdf` + `slides.pptx` (`autopilot`) |

---

## 🏗️ Pipeline

```
document image / PDF
   │  preprocess (deskew · denoise · binarize)              ── D1 page-quality routing
   ▼
born-digital (text layer) vs scanned  ───────────────────── ── D2 routing (skip OCR if digital)
   │  layout: OCR regions → reading order (XY-cut) → classify
   ▼
recognize regions (Tesseract / docTR / stub)               ── D3 OCR-confidence gate
   │  POST-OCR CORRECTION (trained ByT5)                    ── D4 acceptance (edit budget)
   ▼
assemble → Markdown (## headings, - lists) + JSON blocks + plain text + manifest
```

## 📦 Models & data (ids VERIFIED on the HF Hub)

| Role | Id | License |
|---|---|---|
| **Post-OCR corrector (trained)** | `google/byt5-small` (T4: `google-t5/t5-small`) | Apache-2.0 |
| **Baselines** | identity (raw OCR) · SymSpell dictionary (`symspellpy`) | — |
| OCR engine (default) | Tesseract (`pytesseract`) | Apache-2.0 |
| OCR upgrades | `python-doctr`, PaddleOCR / PP-Structure, EasyOCR | Apache-2.0 |
| Real data | `PleIAs/Post-OCR-Correction` (`english`, `text`/`corrected_text`) | CC0-1.0 |
| Train data (primary) | synthetic OCR-noise generator | code MIT |
| Benchmark (optional) | `FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction` | — |

## 🗂️ Repository layout

```
src/dococr/
├── config.py  cli.py  logging_utils.py
├── data/         corpus.py · ocr_noise.py · dataset.py · samples.py · download_dataset.py
├── models/       text_utils.py · corrector.py · baseline.py · ocr_engine.py · model_registry.py
├── ocr/          preprocess.py · layout.py
├── training/     train_corrector.py · evaluate.py · tune.py · metrics.py
├── agent/        state.py · policy.py · tools.py · llm_orchestrator.py · doc_agent.py
├── api/          schemas.py · dependencies.py · main.py · ui.py · app_combined.py
├── analysis/ autoreport/ monitoring/ automation/ grading/
configs/ · data/ · models/ · tests/ · docs/ · notebooks/ · app/ · deploy/ · sample_data/
```

---

## 🚀 Quickstart

```bash
pip install -e ".[ml,ocr,api,report]"

dococr data --task corpus               # synthetic corpus + samples
dococr synth --n 6                      # preview (noisy -> clean) pairs
dococr demo-agent                       # agent on a simulated-OCR document
dococr correct --text "The cornpany rep0rted gr0wth in 2O21."
dococr ocr --file sample_data/sample_page.png   # OCR + structure (needs tesseract)
```

### Train
```bash
dococr --config configs/train.yaml train        # fine-tune ByT5 (auto-resumes)
dococr evaluate --which test                    # corrector vs baselines, CER/WER reduction
dococr evaluate --which real                    # on the real PleIAs slice
```
On Colab/GPU use the notebook (below) — it auto-profiles H100/A100/L4/T4.

### Serve
```bash
dococr serve --ui --port 7860        # FastAPI /ocr + /correct + Gradio UI at /ui
```

### One-button report + slides + self-grade
```bash
dococr autopilot --no-train          # eval → analysis → report.pdf + slides.pptx + bundle
dococr grade
```

---

## 🤖 The agent (mandatory agentic component)

A **deterministic FSM** with **four decision points** acting on intermediate outputs, plus an
*optional* LLM brain (`anthropic`) that validates its output and **falls back to rules**:

- **D1** page-quality / preprocess routing
- **D2** born-digital vs scanned routing (skip OCR when a text layer exists)
- **D3** OCR-confidence gate (flag low-confidence regions for review)
- **D4** correction acceptance — accept a correction only within an **edit budget** so it can't hallucinate

Every step is timed + traced and a full `manifest.json` is written. See
[`docs/agent_architecture.md`](docs/agent_architecture.md).

## ☁️ Colab / H100 training

Open [`notebooks/DocOCR_Colab_Training_H100_AUTOPILOT.ipynb`](notebooks/DocOCR_Colab_Training_H100_AUTOPILOT.ipynb)
— mounts Drive, installs Colab-safe deps + Tesseract, auto-profiles the GPU, trains resume-safely,
evaluates vs baselines, runs the agent, and generates the report/slides. Step-by-step:
[`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md).

## 🧪 Tests

```bash
pytest -q        # CPU-only, no model/network downloads (synthetic + stub OCR + identity corrector)
```

## 📚 Docs index

`docs/`: problem_definition · data_description · data_card · model_selection · evaluation ·
agent_architecture · deployment · continual_learning_monitoring · privacy_robustness ·
project_plan · ethics_statement · architecture · model_card · slide_deck_outline · DESIGN_BRIEF.

## 📝 License

MIT — see [`LICENSE`](LICENSE). Pretrained models/engines keep their own licenses (table above).
OCR of private or copyrighted documents requires appropriate rights/consent.
