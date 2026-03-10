import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_processed_data, load_model, load_scaler, load_feature_names
from xai.counterfactual import CounterfactualExplainer

def main():
    print("Loading data and model...")
    X, y = load_processed_data()
    model = load_model("logistic_regression")
    scaler = load_scaler()
    features = load_feature_names()
    
    print("Initializing Explainer...")
    explainer = CounterfactualExplainer(model, scaler, features, X)
    
    print("Scanning for high risk patients...")
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    preds = model.predict(X_scaled)
    
    high_risk_idx = np.where(preds == 1)[0]
    if len(high_risk_idx) == 0:
        print("No high risk patients found in dataset.")
        return
        
    np.random.seed(42)  # For reproducibility during testing
    test_patients = np.random.choice(high_risk_idx, 5, replace=False)
    
    for i, patient_idx in enumerate(test_patients):
        patient_raw = X.iloc[patient_idx].values
        
        print("\n" + "="*60)
        print(f"TESTING PATIENT #{i+1} (Dataset Index: {patient_idx})")
        print("="*60)
        
        res = explainer.generate(patient_raw, desired_class=0, n_restarts=5)
        
        orig_pred, orig_prob = res['orig_pred'], res['orig_proba'][1]
        cf_pred, cf_prob = res['cf_pred'], res['cf_proba'][1]
        
        print(f"Original Risk : {orig_prob:.1%} (High Risk)")
        print(f"New Target Risk : {cf_prob:.1%} (Safe)")
        print(f"Total Effort Cost : {res['total_cost']:.2f}\n")
        
        print("--- Actionable Targets ---")
        if res['changes']:
            for feat, info in sorted(res['changes'].items(), key=lambda x: x[1]['cost'], reverse=True):
                print(f"{feat:25s}: {info['original']:6.1f} -> {info['counterfactual']:6.1f} (Delta: {info['delta']:+6.1f})")
        else:
            print("(No direct actions found or algorithm failed to find safe plan)")
            
        print("\n--- Automatic Secondary Effects ---")
        if res['derived_changes']:
            for feat, info in res['derived_changes'].items():
                print(f"{feat:25s}: {info['original']:6.1f} -> {info['counterfactual']:6.1f}")
        else:
            print("(No secondary effects)")

if __name__ == "__main__":
    main()
