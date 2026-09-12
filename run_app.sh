#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
if [ ! -f artifacts/geb_ai_model_bundle.joblib ]; then
  python train_models.py
fi
python -m streamlit run app.py

