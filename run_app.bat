@echo off
if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate
python -m pip install -r requirements.txt
if not exist artifacts\geb_ai_model_bundle.joblib python train_models.py
python -m streamlit run app.py

