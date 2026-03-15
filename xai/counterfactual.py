import numpy as np
import pandas as pd
from scipy.optimize import minimize

IMMUTABLE = {"Age", "Gender", "Height_cm", "Family_History"}
DERIVED = {"BMI", "Cholesterol_Ratio", "LDL_HDL_Ratio", "Pulse_Pressure", "MAP"}
INT_FEAT = {"Smoking_Status", "Alcohol_Consumption", "Physical_Activity_Level", "Stress_Level", "Sleep_Hours"}
DIR = {"Weight_kg": "d", "Systolic_BP": "d", "Diastolic_BP": "d", "Cholesterol_Total": "d", "Cholesterol_LDL": "d", "Cholesterol_HDL": "u", "Fasting_Blood_Sugar": "d", "Smoking_Status": "d", "Alcohol_Consumption": "d", "Physical_Activity_Level": "u", "Stress_Level": "d", "Sleep_Hours": "u"}
COSTS = {"Weight_kg": 3.0, "Systolic_BP": 4.0, "Diastolic_BP": 4.0, "Cholesterol_Total": 5.0, "Cholesterol_LDL": 5.0, "Cholesterol_HDL": 5.0, "Fasting_Blood_Sugar": 4.0, "Smoking_Status": 2.0, "Alcohol_Consumption": 1.0, "Physical_Activity_Level": 1.5, "Stress_Level": 2.0, "Sleep_Hours": 1.5}
MIN_STEP = {"Weight_kg": 2.0, "Systolic_BP": 5.0, "Diastolic_BP": 5.0, "Cholesterol_Total": 10.0, "Cholesterol_LDL": 10.0, "Cholesterol_HDL": 5.0, "Fasting_Blood_Sugar": 5.0, "Smoking_Status": 1, "Alcohol_Consumption": 1, "Physical_Activity_Level": 1, "Stress_Level": 1, "Sleep_Hours": 1}

def recompute_derived(x, fi):
    x[fi["BMI"]] = x[fi["Weight_kg"]] / ((x[fi["Height_cm"]] / 100.0) ** 2)
    hdl = max(x[fi["Cholesterol_HDL"]], 1e-6)
    x[fi["Cholesterol_Ratio"]], x[fi["LDL_HDL_Ratio"]] = x[fi["Cholesterol_Total"]] / hdl, x[fi["Cholesterol_LDL"]] / hdl
    x[fi["Pulse_Pressure"]] = x[fi["Systolic_BP"]] - x[fi["Diastolic_BP"]]
    x[fi["MAP"]] = x[fi["Diastolic_BP"]] + x[fi["Pulse_Pressure"]] / 3.0
    return x

class CounterfactualExplainer:
    def __init__(self, model, scaler, feature_names, X_full):
        self.model, self.scaler, self.features = model, scaler, list(feature_names)
        self.fi = {f: i for i, f in enumerate(self.features)}
        self.src = [f for f in self.features if f not in IMMUTABLE | DERIVED]
        self.idx = [self.fi[f] for f in self.src]

        self.lb = np.array([X_full[f].min() for f in self.features], dtype=float)
        self.ub = np.array([X_full[f].max() for f in self.features], dtype=float)
        self.ranges = np.maximum([self.ub[i] - self.lb[i] for i in self.idx], 1e-6)
        self.cost_weights = np.array([COSTS[f] for f in self.src])

        self.sc_mean = scaler.mean_.copy()
        self.sc_scale = scaler.scale_.copy()

        estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        self.coef = estimator.coef_.ravel()
        self.intercept = float(estimator.intercept_[0])

    def _fast_proba(self, x_raw):
        x_scaled = (x_raw - self.sc_mean) / self.sc_scale
        logit = np.dot(self.coef, x_scaled) + self.intercept
        p1 = 1.0 / (1.0 + np.exp(-logit))
        return np.array([1.0 - p1, p1])

    def _predict_from_raw(self, x):
        proba = self._fast_proba(x)
        pred = 1 if proba[1] >= 0.5 else 0
        return pred, proba

    def generate(self, original, desired_class=0, n_restarts=4, max_iter=200):
        orig, fi = np.array(original, dtype=float), self.fi
        idx_arr = np.array(self.idx)

        raw_bounds = [
            (self.lb[i]-orig[i], 0.0) if DIR.get(f) == "d" else
            (0.0, self.ub[i]-orig[i]) if DIR.get(f) == "u" else
            (self.lb[i]-orig[i], self.ub[i]-orig[i])
            for f, i in zip(self.src, self.idx)
        ]
        bounds = [(min(lo, hi), max(lo, hi)) for lo, hi in raw_bounds]

        def objective(delta):
            cf = orig.copy()
            cf[idx_arr] += delta
            cf = np.clip(recompute_derived(cf, fi), self.lb, self.ub)
            if np.isnan(cf).any(): return 1e10

            proba = self._fast_proba(cf)
            penalty = max(0.0, proba[1 if desired_class == 0 else 0] - 0.30) * 100.0
            cost = float(np.sum(self.cost_weights * np.abs(cf[idx_arr] - orig[idx_arr]) / self.ranges))
            return cost + penalty

        best, best_fun, best_cf = None, np.inf, None
        for r in range(n_restarts):
            x0 = np.zeros(len(self.src)) if r == 0 else np.array([np.random.uniform(b[0]*0.2, b[1]*0.2) for b in bounds])
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-9})

            cf = orig.copy()
            cf[idx_arr] += res.x
            for f in INT_FEAT: cf[fi[f]] = np.round(cf[fi[f]])
            for f in [f for f in self.features if f not in INT_FEAT | DERIVED]: cf[fi[f]] = np.round(cf[fi[f]], 1)
            cf = np.clip(recompute_derived(cf, fi), self.lb, self.ub)

            proba = self._fast_proba(cf)
            penalty = max(0.0, proba[1 if desired_class == 0 else 0] - 0.30) * 100.0
            cost = float(np.sum(self.cost_weights * np.abs(cf[idx_arr] - orig[idx_arr]) / self.ranges))
            real_fun = cost + penalty

            if real_fun < best_fun:
                best_fun, best, best_cf = real_fun, res, cf

        cf = best_cf
        cf_pred, cf_proba = self._predict_from_raw(cf)

        chg = {f: {"original": orig[self.idx[j]], "counterfactual": cf[self.idx[j]], "delta": cf[self.idx[j]] - orig[self.idx[j]], "cost": self.cost_weights[j] * abs(cf[self.idx[j]] - orig[self.idx[j]]) / self.ranges[j]}
               for j, f in enumerate(self.src) if abs(cf[self.idx[j]] - orig[self.idx[j]]) >= MIN_STEP.get(f, 1e-6)}

        d_chg = {f: {"original": orig[fi[f]], "counterfactual": cf[fi[f]], "delta": cf[fi[f]] - orig[fi[f]]}
                 for f in DERIVED if abs(cf[fi[f]] - orig[fi[f]]) > 1e-4}

        orig_pred, orig_proba = self._predict_from_raw(orig)
        return {"success": (cf_pred == desired_class), "changes": chg, "derived_changes": d_chg, "total_cost": best_fun, "original": orig, "counterfactual": cf, "orig_pred": orig_pred, "orig_proba": orig_proba, "cf_pred": cf_pred, "cf_proba": cf_proba}
