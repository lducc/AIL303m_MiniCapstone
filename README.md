# Heart Disease Risk Prediction

A machine learning project predicting heart disease risk and generating counterfactual action plans.

## Project Structure
project_root/
  app/streamlit_app.py
  data/
  models/
  evaluation/model_evaluation.ipynb
  utils/data_loader.py
  xai/
  requirements.txt

## Setup
conda create -n heart_disease_env python=3.10
conda activate heart_disease_env
pip install -r requirements.txt

## Run
1. Train: python -m models.scripts.train_all
2. Evaluate: Jupyter notebook evaluation/model_evaluation.ipynb
3. xAI test: python -m xai.test_counterfactual
4. App: streamlit run app/streamlit_app.py
