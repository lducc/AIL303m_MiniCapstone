import os, joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from utils.data_loader import load_processed_data, split_and_scale, MODEL_OUTPUT_DIR

def train():
    X, y = load_processed_data()
    X_train_sc, X_test_sc, y_train, y_test, _, _ = split_and_scale(X, y)

    f2_scorer = make_scorer(fbeta_score, beta=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'p': [1, 2]}

    base_model = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring=f2_scorer, cv=cv, n_jobs=-1)
    grid_search.fit(X_train_sc, y_train)
    model = grid_search.best_estimator_

    y_pred = model.predict(X_test_sc)
    print(f"KNN F2: {fbeta_score(y_test, y_pred, beta=2):.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_OUTPUT_DIR, "knn.joblib")
    joblib.dump(model, out_path)
    print(f"Saved KNN to {out_path}")

if __name__ == "__main__":
    train()
