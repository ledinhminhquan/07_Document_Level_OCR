.PHONY: help install install-all data synth train evaluate demo serve ui test grade report slides autopilot clean

help:
	@echo "Document-Level OCR System — common tasks"
	@echo "  make install       core install (pip install -e .)"
	@echo "  make install-all   full install (.[all])"
	@echo "  make data          build the post-OCR corpus + samples"
	@echo "  make synth         show a few synthetic (noisy, clean) pairs"
	@echo "  make train         fine-tune the ByT5 post-OCR corrector"
	@echo "  make evaluate      corrector vs baselines (CER/WER reduction)"
	@echo "  make demo          run the agent on a simulated-OCR sample doc"
	@echo "  make serve / ui    FastAPI server / + Gradio UI at /ui"
	@echo "  make test report slides autopilot grade"

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

data:
	dococr data --task corpus

synth:
	dococr synth --n 6

train:
	dococr --config configs/train.yaml train

evaluate:
	dococr evaluate --which test

demo:
	dococr demo-agent

serve:
	dococr --config configs/infer.yaml serve --host 0.0.0.0 --port 8000

ui:
	dococr serve --ui --host 0.0.0.0 --port 7860

test:
	pytest -q

grade:
	dococr grade

report:
	dococr generate-report

slides:
	dococr generate-slides

autopilot:
	dococr autopilot --no-train

clean:
	rm -rf artifacts __pycache__ .pytest_cache src/*.egg-info build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
