# 🩺 Breast Cancer Detection — End-to-End ML Project

**Dataset**: Breast Cancer Wisconsin (Diagnostic) — available on Kaggle  
**Task**: Binary Classification (Benign / Malignant)  
**Model**: Logistic Regression  
**Deployment**: Flask Web Application

---

## 📁 Project Structure

```
breast_cancer_project/
│
├── train_model.py          ← Full ML pipeline (run this first)
├── app.py                  ← Flask web application
├── requirements.txt        ← Python dependencies
│
├── templates/
│   └── index.html          ← Web UI (auto-served by Flask)
│
├── models/                 ← Auto-created after training
│   ├── logistic_model.pkl
│   ├── scaler.pkl
│   └── selected_features.pkl
│
└── plots/                  ← Auto-created after training
    ├── eda.png
    ├── boxplot_mean_radius.png
    ├── evaluation.png
    └── feature_importance.png
```

---

## ⚙️ ML Lifecycle Phases Covered

| Phase | Description |
|-------|-------------|
| 1 | **Data Collection** — Load Breast Cancer Wisconsin dataset |
| 2 | **Cleaning & Preprocessing** — Missing value check, deduplication |
| 3 | **EDA** — Class distribution, correlation heatmap, boxplots |
| 4 | **Feature Engineering & Selection** — ANOVA F-test (SelectKBest, top 15) |
| 5 | **Imbalanced Data Handling** — SMOTE oversampling |
| 6 | **Train/Test Split + Scaling** — 80/20 split, StandardScaler |
| 7 | **Model Training** — Logistic Regression + 5-Fold CV |
| 8 | **Hyperparameter Tuning** — GridSearchCV (C, solver) |
| 9 | **Evaluation** — Accuracy, AUC, Confusion Matrix, ROC Curve |
| 10 | **Deployment** — Flask REST API + Web UI |
| 11 | **Maintenance** — Modular artifacts (model/scaler saved via joblib) |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This will:
- Run the full ML pipeline
- Print metrics to the terminal
- Save model artifacts to `models/`
- Save evaluation plots to `plots/`

### 3. Start the Flask app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 📊 Expected Results

| Metric | Expected Value |
|--------|---------------|
| Accuracy | ~97–99% |
| ROC-AUC | ~0.99 |
| Cross-Val Accuracy | ~96–98% |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/predict` | POST | JSON prediction |
| `/health` | GET | Health check |

### Example `/predict` Request
```json
POST /predict
Content-Type: application/json

{
  "mean radius": 12.0,
  "mean texture": 17.0,
  "mean perimeter": 78.0,
  ...
}
```

### Example Response
```json
{
  "prediction": "Benign",
  "confidence": 97.42,
  "prob_benign": 97.42,
  "prob_malignant": 2.58
}
```

---

## 📝 Notes

- The model uses the **sklearn built-in** version of the Wisconsin dataset (identical to the Kaggle version). To use the Kaggle CSV directly, replace the `load_breast_cancer()` call in `train_model.py` with `pd.read_csv('data.csv')` and adjust the target column accordingly.
- SMOTE is applied only on the training set to prevent data leakage.
- The scaler is fit only on training data and applied to test/inference data.
