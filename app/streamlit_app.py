import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_model, load_scaler, load_feature_names
from xai.counterfactual import CounterfactualExplainer, recompute_derived, DERIVED

st.set_page_config(
    page_title="Heart Disease Risk Prediction & Action Plan",
    layout="wide"
)

@st.cache_resource
def load_app_assets():
    model = load_model("logistic_regression")
    if not hasattr(model, 'multi_class'): model.multi_class = 'auto'
    if not hasattr(model, 'classes_'): model.classes_ = np.array([0, 1])
    if hasattr(model, 'best_estimator_'):
        if not hasattr(model.best_estimator_, 'multi_class'): model.best_estimator_.multi_class = 'auto'
        if not hasattr(model.best_estimator_, 'classes_'): model.best_estimator_.classes_ = np.array([0, 1])

    scaler = load_scaler()
    features = load_feature_names()
    from utils.data_loader import load_processed_data
    X_full, _ = load_processed_data()
    explainer = CounterfactualExplainer(model, scaler, features, X_full)
    return model, scaler, features, explainer, X_full

try:
    model, scaler, features, explainer, X_full = load_app_assets()
except Exception as e:
    st.error(f"Error loading models or data. Did you run train_all.py first?\n{e}")
    st.stop()


def make_gauge(value, title="Risk Level"):
    pct = value * 100
    color = "#FF4B4B" if pct >= 50 else "#FFA500" if pct >= 30 else "#00CC66"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0,204,102,0.15)"},
                {"range": [30, 50], "color": "rgba(255,165,0,0.15)"},
                {"range": [50, 100], "color": "rgba(255,75,75,0.15)"},
            ],
            "threshold": {"line": {"color": "#333", "width": 3}, "thickness": 0.8, "value": 50},
        }
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font_color="#FAFAFA")
    return fig
BINARY_FEAT = {"Smoking_Status", "Family_History"}
LABELS = {
    "Smoking_Status": {0: "Non-Smoker", 1: "Smoker"},
    "Alcohol_Consumption": {0: "None", 1: "Light", 2: "Moderate", 3: "Heavy"},
    "Physical_Activity_Level": {0: "Sedentary", 1: "Low", 2: "Moderate", 3: "Active", 4: "Very Active"},
}

def make_comparison_chart(changes):
    feats, originals, targets = [], [], []

    for feat, info in sorted(changes.items(), key=lambda x: x[1]["cost"], reverse=True):
        if feat in BINARY_FEAT:
            continue  # Binary features shown separately as text
        feats.append(feat.replace("_", " "))
        originals.append(info["original"])
        targets.append(info["counterfactual"])

    if not feats:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feats, x=originals, name="Current",
        orientation="h", marker_color="#FF6B6B",
        text=[f"{v:.1f}" for v in originals], textposition="auto"
    ))
    fig.add_trace(go.Bar(
        y=feats, x=targets, name="Target",
        orientation="h", marker_color="#51CF66",
        text=[f"{v:.1f}" for v in targets], textposition="auto"
    ))
    fig.update_layout(
        barmode="group", title="Current vs Target Values",
        height=max(250, len(feats) * 80 + 100),
        margin=dict(t=60, b=30, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# --- UI ---
st.title("Heart Disease Risk Prediction & Action Plan")
st.markdown("""
Enter your health information below to predict your risk of heart disease. 
If you are at high risk, the AI will provide a **personalized, minimum-action plan** to lower your risk.
""")

st.divider()

col1, col2, col3 = st.columns(3)
inputs = {}

with col1:
    st.subheader("Demographics")
    inputs["Age"] = st.slider("Age", 20, 100, 50)
    inputs["Gender"] = {"Male": 1, "Female": 0}[st.selectbox("Gender", ["Male", "Female"])]
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

for df_name in DERIVED:
    inputs[df_name] = 0.0

st.divider()

if st.button("Predict Risk & Generate Action Plan", type="primary"):

    fi = {f: i for i, f in enumerate(features)}
    x_raw = np.zeros(len(features))
    for f in features:
        if f in inputs: x_raw[fi[f]] = inputs[f]
    x_raw = recompute_derived(x_raw, fi)

    pred, proba = explainer._predict_from_raw(x_raw)
    risk_prob = proba[1]

    if pred == 0:
        # ---- LOW RISK ----
        st.plotly_chart(make_gauge(risk_prob, "Your Heart Disease Risk"), use_container_width=True)
        st.success("### Low Risk - You're in great shape!")
        st.markdown("Based on your metrics, your risk of heart disease is currently low. Keep up the healthy habits!")

        with st.expander("View calculated metrics"):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("BMI", f"{x_raw[fi['BMI']]:.1f}")
            mc2.metric("Cholesterol Ratio", f"{x_raw[fi['Cholesterol_Ratio']]:.1f}")
            mc3.metric("LDL/HDL Ratio", f"{x_raw[fi['LDL_HDL_Ratio']]:.1f}")
            mc4, mc5, _ = st.columns(3)
            mc4.metric("Pulse Pressure", f"{x_raw[fi['Pulse_Pressure']]:.0f}")
            mc5.metric("Mean Arterial Pressure", f"{x_raw[fi['MAP']]:.0f}")

    else:
        # ---- HIGH RISK ----
        st.plotly_chart(make_gauge(risk_prob, "Your Current Risk"), use_container_width=True)
        st.error("### High Risk - Action plan recommended")
        st.markdown("Based on your metrics, you are at an elevated risk of heart disease. The AI is generating the easiest possible plan to reduce your risk.")

        with st.spinner("Generating personalized action plan..."):
            res = explainer.generate(x_raw, desired_class=0, n_restarts=4)

        if not res['success']:
            st.warning("We couldn't find a simple action plan. Please consult a doctor immediately.")
        else:
            st.divider()

            # --- Risk Comparison Gauges ---
            st.markdown("## Risk Before vs After")
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(make_gauge(res['orig_proba'][1], "Before (Current)"), use_container_width=True)
            with g2:
                st.plotly_chart(make_gauge(res['cf_proba'][1], "After (Target)"), use_container_width=True)

            st.divider()

            # --- Before vs After Bar Chart ---
            st.markdown("## Your Personalized Action Plan")
            comparison_fig = make_comparison_chart(res["changes"])
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)

            # --- Action Cards with st.metric ---
            if res["changes"]:
                # Separate binary/categorical from continuous
                binary_changes = {k: v for k, v in res["changes"].items() if k in BINARY_FEAT}
                labeled_changes = {k: v for k, v in res["changes"].items() if k in LABELS and k not in BINARY_FEAT}
                numeric_changes = {k: v for k, v in res["changes"].items() if k not in BINARY_FEAT and k not in LABELS}

                # Show binary changes as prominent text cards
                if binary_changes:
                    for feat, info in binary_changes.items():
                        label_map = LABELS.get(feat, {})
                        old_label = label_map.get(int(info["original"]), str(int(info["original"])))
                        new_label = label_map.get(int(info["counterfactual"]), str(int(info["counterfactual"])))
                        st.success(f"**{feat.replace('_', ' ')}**: {old_label} → {new_label}")

                # Show labeled categorical changes with readable names
                if labeled_changes:
                    st.markdown("### Lifestyle Changes")
                    lc_cols = st.columns(min(len(labeled_changes), 3))
                    for i, (feat, info) in enumerate(labeled_changes.items()):
                        with lc_cols[i % 3]:
                            label_map = LABELS.get(feat, {})
                            old_label = label_map.get(int(info["original"]), str(int(info["original"])))
                            new_label = label_map.get(int(info["counterfactual"]), str(int(info["counterfactual"])))
                            st.metric(label=feat.replace("_", " "), value=new_label, delta=f"was {old_label}")

                # Show numeric continuous changes with delta arrows
                if numeric_changes:
                    st.markdown("### Medical Targets")
                    nc_cols = st.columns(min(len(numeric_changes), 3))
                    for i, (feat, info) in enumerate(sorted(numeric_changes.items(), key=lambda x: x[1]["cost"], reverse=True)):
                        with nc_cols[i % 3]:
                            st.metric(
                                label=feat.replace("_", " "),
                                value=f"{info['counterfactual']:.1f}",
                                delta=f"{info['delta']:+.1f}",
                                delta_color="inverse"
                            )

            # --- Derived Changes ---
            if res.get("derived_changes"):
                st.markdown("### Automatic Secondary Effects")
                st.caption("These metrics will improve automatically when you achieve the goals above.")
                dcols = st.columns(min(len(res["derived_changes"]), 3))
                for i, (feat, info) in enumerate(res["derived_changes"].items()):
                    with dcols[i % 3]:
                        st.metric(
                            label=feat.replace("_", " "),
                            value=f"{info['counterfactual']:.1f}",
                            delta=f"{info['delta']:+.1f}",
                            delta_color="inverse"
                        )
