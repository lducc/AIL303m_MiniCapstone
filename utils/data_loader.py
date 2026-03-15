import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "healthcare_synthetic_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

TARGET_COL = "Heart_Disease_Risk"


def load_raw_data():
    return pd.read_csv(RAW_DATA_PATH)


def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42, save=True):
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    if save:
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, "scaler.joblib"))
        joblib.dump(feature_names, os.path.join(MODEL_OUTPUT_DIR, "feature_names.joblib"))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def load_model(name):
    path = os.path.join(MODEL_OUTPUT_DIR, f"{name}.joblib")
    return joblib.load(path)

def load_scaler():
    return joblib.load(os.path.join(MODEL_OUTPUT_DIR, "scaler.joblib"))

def load_feature_names():
    return joblib.load(os.path.join(MODEL_OUTPUT_DIR, "feature_names.joblib"))
