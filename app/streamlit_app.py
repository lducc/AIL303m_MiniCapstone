import sys
import os
# Must add the project root to sys.path so streamlit can find utils and xai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import streamlit as st


from utils.data_loader import load_model, load_scaler, load_feature_names
from xai.counterfactual import CounterfactualExplainer, recompute_derived, DERIVED

# --- Config ---
st.set_page_config(
    page_title="Heart Disease Risk Prediction & Action Plan",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_app_assets():
    """Load models, scaler, features only once."""
    model = load_model("logistic_regression") # LR is our best for xAI
    
    # Patch for Scikit-learn to avoid GridSearchCV model attribute errors 
    if not hasattr(model, 'multi_class'):
        model.multi_class = 'auto'
    if not hasattr(model, 'classes_'):
        model.classes_ = np.array([0, 1])
    if hasattr(model, 'best_estimator_'):
        if not hasattr(model.best_estimator_, 'multi_class'):
            model.best_estimator_.multi_class = 'auto'
        if not hasattr(model.best_estimator_, 'classes_'):
            model.best_estimator_.classes_ = np.array([0, 1])
    
    scaler = load_scaler()
    features = load_feature_names()
    
    # We need realistic min/max bounds for the explainer
    # Let's load the processed data just for bounding
    from utils.data_loader import load_processed_data
    X_full, _ = load_processed_data()
    
    explainer = CounterfactualExplainer(model, scaler, features, X_full)
    return model, scaler, features, explainer, X_full

try:
    model, scaler, features, explainer, X_full = load_app_assets()
except Exception as e:
    st.error(f"Error loading models or data. Did you run train_all.py first?\n{e}")
    st.stop()


# --- UI ---
st.title("❤️ Heart Disease Risk Prediction & Action Plan")
st.markdown("""
Enter your health information below to predict your risk of heart disease. 
If you are at high risk, the AI will provide a **personalized, minimum-action plan** to lower your risk back to a healthy range.
""")

st.divider()

col1, col2, col3 = st.columns(3)

# Dictionary to hold raw user inputs
inputs = {}

with col1:
    st.subheader("Demographics")
    inputs["Age"] = st.slider("Age", 20, 100, 50)
    gender_map = {"Male": 1, "Female": 0}
    inputs["Gender"] = gender_map[st.selectbox("Gender", ["Male", "Female"])]
    inputs["Height_cm"] = st.number_input("Height (cm)", 140.0, 220.0, 170.0)
    inputs["Weight_kg"] = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
    inputs["Family_History"] = 1 if st.checkbox("Family History of Heart Disease") else 0

with col2:
    st.subheader("Vitals & Labs")
    inputs["Systolic_BP"] = st.slider("Systolic BP", 90, 200, 120)
    inputs["Diastolic_BP"] = st.slider("Diastolic BP", 60, 130, 80)
    inputs["Cholesterol_Total"] = st.slider("Total Cholesterol", 100, 400, 200)
    inputs["Cholesterol_HDL"] = st.slider("HDL Cholesterol", 20, 100, 50)
    inputs["Cholesterol_LDL"] = st.slider("LDL Cholesterol", 50, 300, 100)
    inputs["Fasting_Blood_Sugar"] = st.slider("Fasting Blood Sugar", 70, 200, 90)

with col3:
    st.subheader("Lifestyle")
    inputs["Smoking_Status"] = 1 if st.checkbox("Current Smoker") else 0
    inputs["Alcohol_Consumption"] = st.selectbox("Alcohol Consumption", [0, 1, 2, 3])
    inputs["Physical_Activity_Level"] = st.selectbox("Physical Activity Level", [0, 1, 2, 3, 4], index=2)
    inputs["Stress_Level"] = st.slider("Stress Level (0-10)", 0, 10, 5)
    inputs["Sleep_Hours"] = st.slider("Sleep Hours", 3, 12, 7)

# We must initialize the derived features with zeros, they will be recomputed
for df_name in DERIVED:
    inputs[df_name] = 0.0

st.divider()

if st.button("Predict Risk & Generate Action Plan", type="primary"):
    
    # 1. Structure as array matching exactly the feature_names order
    fi = {f: i for i, f in enumerate(features)}
    x_raw = np.zeros(len(features))
    for f in features:
        if f in inputs:
            x_raw[fi[f]] = inputs[f]
            
    # 2. Recompute the derived features (BMI, ratios, etc)
    x_raw = recompute_derived(x_raw, fi)
    
    # 3. Predict using the explainer's helper
    pred, proba = explainer._predict_from_raw(x_raw)
    risk_prob = proba[1]
    
    if pred == 0:
        st.success(f"### 🎉 Low Risk (Probability: {risk_prob:.1%})")
        st.markdown("Great job! Based on your metrics, your risk of heart disease is currently low. Keep up the good work.")
        
        with st.expander("View calculated metrics"):
            st.write(f"- BMI: {x_raw[fi['BMI']]:.1f}")
            st.write(f"- Cholesterol Ratio: {x_raw[fi['Cholesterol_Ratio']]:.1f}")
            st.write(f"- LDL/HDL Ratio: {x_raw[fi['LDL_HDL_Ratio']]:.1f}")
            st.write(f"- Pulse Pressure: {x_raw[fi['Pulse_Pressure']]:.1f}")
            st.write(f"- Mean Arterial Pressure: {x_raw[fi['MAP']]:.1f}")
            
    else:
        st.error(f"### ⚠️ High Risk (Probability: {risk_prob:.1%})")
        st.markdown("Based on your metrics, you are at an elevated risk of heart disease.")
        
        with st.spinner("Generating personalized action plan..."):
            res = explainer.generate(x_raw, desired_class=0, n_restarts=10)
            
        if not res['success']:
            st.warning("We couldn't find a simple action plan that brings your risk down to a safe level. Please consult a doctor immediately.")
        else:
            st.success("### 📝 Your Personalized Action Plan")
            st.markdown(f"By making the following changes, your predicted risk will drop to **{res['cf_proba'][1]:.1%}**.")
            
            # Display actionable changes
            if res["changes"]:
                st.markdown("#### Primary Goals")
                for feat, info in sorted(res["changes"].items(), key=lambda x: x[1]["cost"], reverse=True):
                    orig_val = info['original']
                    new_val = info['counterfactual']
                    delta = info['delta']
                    arrow = "⬇️ Reduce" if delta < 0 else "⬆️ Increase"
                    
                    st.info(f"**{arrow} {feat.replace('_', ' ')}**\n"
                            f"Target: **{new_val:.1f}** (Currently: {orig_val:.1f})")
            else:
                st.write("No actionable changes found.")

            # Display derived changes
            if res.get("derived_changes"):
                st.markdown("#### Secondary Effects (Automatic)")
                st.markdown("Achieving the goals above will automatically improve these metrics:")
                for feat, info in res["derived_changes"].items():
                    orig_val = info['original']
                    new_val = info['counterfactual']
                    st.write(f"- **{feat.replace('_', ' ')}**: {orig_val:.1f} → **{new_val:.1f}**")
