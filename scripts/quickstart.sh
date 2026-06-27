#!/usr/bin/env bash
# Quickstart: install, prepare data, run the offline agent demo, and self-grade.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> install (core + extras)"
pip install -e ".[ml,ocr,api,report]"

echo "==> build the post-OCR corpus + samples"
dococr data --task corpus

echo "==> see the baseline you must beat (identity = raw OCR)"
dococr evaluate --which test || true

echo "==> run the agent on a simulated-OCR sample document (offline-friendly)"
dococr demo-agent

echo "==> generate report.pdf + slides.pptx and self-grade"
dococr generate-report
dococr generate-slides
dococr grade
