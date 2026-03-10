# Heart Disease Risk Prediction & Action Plan

This is a machine learning project with a Streamlit frontend designed to predict the risk of heart disease based on metrics. 

## Features
- **Predictive Modeling**: Classifies patients as "At Risk" or "No Risk" using various optimized algorithms (Logistic Regression, Random Forest, XGBoost, SVM, KNN).
- **Explainable AI (xAI)**: Uses SHAP for feature importance and custom Counterfactual Explanations to generate actionable health changes.
- **Interactive UI**: A Streamlit app that accepts user input, predicts risk, and automatically calculates required lifestyle/health changes to reach a safe threshold.

## Project Structure
```text
project_root/
│
├── app/
│   └── streamlit_app.py              # Main interactive Streamlit application
│
├── data/
│   ├── healthcare_synthetic_data.csv # Raw dataset
│   ├── processed_data.csv            # Cleaned & feature-engineered dataset
│   └── eda_feature_engineering.ipynb # Notebook for cleaning data (EDA)
│
├── models/
│   ├── outputs/                      # Saved .joblib model weights and scaler
│   └── scripts/                      # Training scripts for each ML algorithm
│       ├── train_logistic_regression.py
│       ├── train_random_forest.py
│       ├── train_xgboost.py
│       ├── train_svm.py
│       └── train_knn.py
│
├── evaluation/
│   └── model_evaluation.ipynb        # Comprehensive model metrics & SHAP charts
│
├── utils/
│   └── data_loader.py                # Shared utilities for splitting and scaling
│
├── xai/
│   ├── counterfactual.py             # Custom explainer class 
│
└── requirements.txt                  # Python dependencies
```

## Installation & Setup

1. **Clone the repository** (or navigate to the project directory).
2. **Create a virtual environment** (optional but recommended):
   ```bash
   conda create -n heart_disease_env python=3.10
   conda activate heart_disease_env
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Train the Models (Optional)
The pre-trained models are already saved in the `models/outputs/` directory. However, if you wish to retrain them (which includes a comprehensive GridSearchCV hyperparameter tuning for F2 score optimization), you can run any of the scripts individually from the project root:

```bash
python -m models.scripts.train_logistic_regression
python -m models.scripts.train_random_forest
python -m models.scripts.train_xgboost
```

### 2. View the Evaluation
Open the Jupyter notebook located at `evaluation/model_evaluation.ipynb`. 
This notebook allows you to:
- Compare Accuracy, F1, F2, and ROC-AUC across all models.
- View Confusion Matrices and ROC Curves.
- Visualize global and local feature importance via SHAP summary plots.

### 3. Test the Explainer (xAI)
To see how the counterfactual action plan generator works under the hood for random test-set patients, run:
```bash
python -m xai.test_counterfactual
```

### 4. Run the Streamlit Application
The highest level interactive experience is through the Streamlit App. This allows a user to input their custom vitals and receive an instant prediction and custom lifestyle adjustment plan.

Run the app from the root directory:
```bash
streamlit run app/streamlit_app.py
```
This will open a local web server (typically at `http://localhost:8501/`) in your default browser.

Once in the app:
1. Simply adjust the sliders for Age, BP, Cholesterol, Sleep, etc.
2. Click **Predict Risk & Generate Action Plan**.
3. If the risk is high, read the **Primary Goals** section to see exactly what numerical changes you should make to lower your risk back down.
