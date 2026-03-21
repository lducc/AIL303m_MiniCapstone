# Heart Disease Risk Prediction

A machine learning pipeline that shifts from *explanatory* to *actionable* AI. This project predicts heart disease risk using binary classification models and provides domain-constrained counterfactual explanations to generate personalized, realistic health action plans.

## Project Details

Traditional Explainable AI (like SHAP or LIME) tells a patient *why* they are at high risk. This project uses **Counterfactual Explanations** to tell the patient *what to do* to reverse that risk. 

To ensure clinical realism, the counterfactual optimization engine strictly enforces:
- **Immutable Features**: Characteristics like Age, Gender, Height, and Family History cannot be altered.
- **Directional Constraints**: Metrics like blood pressure and cholesterol can only decrease, while physical activity and sleep can only increase.
- **Human Effort Costs**: Difficult metabolic changes requiring medication are penalized more heavily during optimization than simple behavioral adjustments.

## Setup

conda create -n heart_disease python=3.10
conda activate heart_disease
pip install -r requirements.txt

## Usage

1. Train models:
`python -m models.scripts.train_all`

2. Evaluate models:
`jupyter notebook evaluation/model_evaluation.ipynb`

3. Test xAI:
`python -m xai.test_counterfactual`

4. Run app:
`streamlit run app/streamlit_app.py`
