#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${RSNA_PYTHON:-/opt/homebrew/opt/python@3.12/bin/python3.12}"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3.12 || command -v python3)"; fi
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,kaggle,dicom]"
# torch is optional local extra; skip CUDA wheels on macOS.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || python -m pip install torch || true
rsna doctor
echo "OK. Next: rsna smoke --synthetic"
