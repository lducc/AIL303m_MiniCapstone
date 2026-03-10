import numpy as np
import pandas as pd
from scipy.optimize import minimize

IMMUTABLE = {"Age", "Gender", "Height_cm", "Family_History"}
DERIVED = {"BMI", "Cholesterol_Ratio", "LDL_HDL_Ratio", "Pulse_Pressure", "MAP"}
INT_FEAT = {"Smoking_Status", "Alcohol_Consumption", "Physical_Activity_Level", "Stress_Level", "Sleep_Hours"}
DIR = {"Weight_kg": "d", "Systolic_BP": "d", "Diastolic_BP": "d", "Cholesterol_Total": "d", "Cholesterol_LDL": "d", "Cholesterol_HDL": "u", "Fasting_Blood_Sugar": "d", "Smoking_Status": "d", "Alcohol_Consumption": "d", "Physical_Activity_Level": "u", "Stress_Level": "d", "Sleep_Hours": "u"}
COSTS = {"Weight_kg": 3.0, "Systolic_BP": 4.0, "Diastolic_BP": 4.0, "Cholesterol_Total": 5.0, "Cholesterol_LDL": 5.0, "Cholesterol_HDL": 5.0, "Fasting_Blood_Sugar": 4.0, "Smoking_Status": 2.0, "Alcohol_Consumption": 1.0, "Physical_Activity_Level": 1.5, "Stress_Level": 2.0, "Sleep_Hours": 1.5}

def recompute_derived(x, fi):
    x[fi["BMI"]] = x[fi["Weight_kg"]] / ((x[fi["Height_cm"]] / 100.0) ** 2)
    hdl = max(x[fi["Cholesterol_HDL"]], 1e-6)
    x[fi["Cholesterol_Ratio"]], x[fi["LDL_HDL_Ratio"]]  = x[fi["Cholesterol_Total"]] / hdl, x[fi["Cholesterol_LDL"]] / hdl
    x[fi["Pulse_Pressure"]] = x[fi["Systolic_BP"]] - x[fi["Diastolic_BP"]]
    x[fi["MAP"]] = x[fi["Diastolic_BP"]] + x[fi["Pulse_Pressure"]] / 3.0
    return x

class CounterfactualExplainer:
    def __init__(self, model, scaler, feature_names, X_full):
        self.model, self.scaler, self.features = model, scaler, list(feature_names)
        self.fi = {f: i for i, f in enumerate(self.features)}
        self.src = [f for f in self.features if f not in IMMUTABLE | DERIVED]
        self.idx = [self.fi[f] for f in self.src]
        
        # Bounding box limits for optimization (min and max possible values observed in real life)
        self.lb = np.array([X_full[f].min() for f in self.features], dtype=float)
        self.ub = np.array([X_full[f].max() for f in self.features], dtype=float)
        self.ranges = np.maximum([self.ub[i] - self.lb[i] for i in self.idx], 1e-6)
        self.cost_weights = np.array([COSTS[f] for f in self.src])

    def _predict_from_raw(self, x):
        xs = pd.DataFrame(self.scaler.transform(pd.DataFrame([x], columns=self.features)), columns=self.features)
        return int(self.model.predict(xs)[0]), self.model.predict_proba(xs)[0]

    def generate(self, original, desired_class=0, n_restarts=8, max_iter=800):
        orig, fi = np.array(original, dtype=float), self.fi
        
        # Determine maximum allowed change based on direction rules (d=down, u=up)
        bounds = [
            (self.lb[i]-orig[i], 0.0) if DIR.get(f) == "d" else 
            (0.0, self.ub[i]-orig[i]) if DIR.get(f) == "u" else 
            (self.lb[i]-orig[i], self.ub[i]-orig[i]) 
            for f, i in zip(self.src, self.idx)
        ]

        def objective(delta):
            cf = orig.copy()
            for j, i in enumerate(self.idx): cf[i] += delta[j]
            cf = np.clip(recompute_derived(cf, fi), self.lb, self.ub)
            if np.isnan(cf).any(): return 1e10
            
            # Score = Human Effort + AI Prediction Penalty
            df = pd.DataFrame(cf.reshape(1, -1), columns=self.features)
            xs = pd.DataFrame(self.scaler.transform(df), columns=self.features)
            proba = self.model.predict_proba(xs)[0]
            penalty = max(0.0, proba[1 if desired_class == 0 else 0] - 0.40) * 100.0
            cost = float(np.sum(self.cost_weights * np.abs(cf[self.idx] - orig[self.idx]) / self.ranges))
            return cost + penalty

        # Try optimizing from multiple random starting points to find the absolute best plan
        best, best_fun, best_cf = None, np.inf, None
        for r in range(n_restarts):
            # Pick a random starting tweak within bounds (or 0 for the first run)
            x0 = np.zeros(len(self.src)) if r == 0 else np.array([np.random.uniform(b[0]*0.2, b[1]*0.2) for b in bounds])
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-9})
            
            # Apply and enforce realism (integers and rounding)
            cf = orig.copy()
            for j, i in enumerate(self.idx): cf[i] += res.x[j]
            for f in INT_FEAT: cf[fi[f]] = np.round(cf[fi[f]])
            for f in [f for f in self.features if f not in INT_FEAT | DERIVED]: cf[fi[f]] = np.round(cf[fi[f]], 1)
            cf = np.clip(recompute_derived(cf, fi), self.lb, self.ub)
            
            # Score this realistic rounded version to make sure it functions
            df = pd.DataFrame(cf.reshape(1, -1), columns=self.features)
            proba = self.model.predict_proba(pd.DataFrame(self.scaler.transform(df), columns=self.features))[0]
            penalty = max(0.0, proba[1 if desired_class == 0 else 0] - 0.40) * 100.0
            cost = float(np.sum(self.cost_weights * np.abs(cf[self.idx] - orig[self.idx]) / self.ranges))
            real_fun = cost + penalty
            
            if real_fun < best_fun: best_fun, best, best_cf = real_fun, res, cf

        cf = best_cf

        # Calculate final changes tracking dictionaries
        cf_pred, cf_proba = self._predict_from_raw(cf)
        
        chg = {f: {"original": orig[self.idx[j]], "counterfactual": cf[self.idx[j]], "delta": cf[self.idx[j]] - orig[self.idx[j]], "cost": self.cost_weights[j] * abs(cf[self.idx[j]] - orig[self.idx[j]]) / self.ranges[j]} 
               for j, f in enumerate(self.src) if abs(cf[self.idx[j]] - orig[self.idx[j]]) > 1e-6}
               
        d_chg = {f: {"original": orig[fi[f]], "counterfactual": cf[fi[f]], "delta": cf[fi[f]] - orig[fi[f]]} 
                 for f in DERIVED if abs(cf[fi[f]] - orig[fi[f]]) > 1e-4}

        orig_pred, orig_proba = self._predict_from_raw(orig)
        return {"success": (cf_pred == desired_class), "changes": chg, "derived_changes": d_chg, "total_cost": best.fun, "original": orig, "counterfactual": cf, "orig_pred": orig_pred, "orig_proba": orig_proba, "cf_pred": cf_pred, "cf_proba": cf_proba}
