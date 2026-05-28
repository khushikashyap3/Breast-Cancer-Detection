# ============================================================
#  Flask Web App — Breast Cancer Detection
#  Render-ready: auto-trains model on first startup
# ============================================================

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

MODEL_PATH    = "models/logistic_model.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/selected_features.pkl"

model    = None
scaler   = None
features = None


def train_and_save():
    """Train the model from scratch and save artifacts."""
    print("[INFO] Training model from scratch...")

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, f_classif
    from imblearn.over_sampling import SMOTE
    import pandas as pd

    raw = load_breast_cancer()
    df  = pd.DataFrame(raw.data, columns=raw.feature_names)
    df['target'] = raw.target

    X = df.drop(columns=['target'])
    y = df['target']

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=15)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()].tolist()
    X_sel = X[selected]

    # SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_sel, y)

    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)

    # Tune & train
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    gs = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1
    )
    gs.fit(X_train_sc, y_train)
    best = gs.best_estimator_

    os.makedirs("models", exist_ok=True)
    joblib.dump(best,     MODEL_PATH)
    joblib.dump(sc,       SCALER_PATH)
    joblib.dump(selected, FEATURES_PATH)
    print(f"[INFO] Model trained & saved. Best params: {gs.best_params_}")
    return best, sc, selected


def load_artifacts():
    global model, scaler, features
    if os.path.exists(MODEL_PATH):
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        print(f"[INFO] Loaded saved model ({len(features)} features).")
    else:
        # No saved model — train now (happens on first Render deploy)
        model, scaler, features = train_and_save()


load_artifacts()


# ── Routes ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           features=features if features else [])


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available."}), 500
    try:
        data   = request.get_json()
        values = [float(data.get(f, 0)) for f in features]
        X      = np.array(values).reshape(1, -1)
        X_sc   = scaler.transform(X)

        pred  = model.predict(X_sc)[0]
        prob  = model.predict_proba(X_sc)[0]

        label      = "Benign" if pred == 1 else "Malignant"
        confidence = float(prob[pred]) * 100

        return jsonify({
            "prediction":     label,
            "confidence":     round(confidence, 2),
            "prob_benign":    round(float(prob[1]) * 100, 2),
            "prob_malignant": round(float(prob[0]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "features":     features if features else []
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
