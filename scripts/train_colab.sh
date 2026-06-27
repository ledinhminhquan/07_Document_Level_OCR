#!/usr/bin/env bash
# Train the ByT5 post-OCR corrector on Colab/GPU. The notebook does this with an
# auto GPU profile; this is the plain CLI equivalent.
set -euo pipefail
cd "$(dirname "$0")/.."

export DOCOCR_ARTIFACTS_DIR="${DOCOCR_ARTIFACTS_DIR:-/content/drive/MyDrive/dococr}"

dococr data --task corpus
dococr --config configs/train.yaml train
dococr evaluate --which test
dococr evaluate --which real
dococr error-analysis
dococr robustness
dococr autopilot --no-train
