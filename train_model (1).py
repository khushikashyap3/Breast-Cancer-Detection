# ============================================================
#  Breast Cancer Detection - End-to-End ML Pipeline
#  Dataset: Breast Cancer Wisconsin (Kaggle)
#  Model: Logistic Regression (Binary Classification)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("  BREAST CANCER DETECTION — ML LIFECYCLE PIPELINE")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# PHASE 1: DATA COLLECTION
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 1] Data Collection")
print("-" * 40)

# Using sklearn's built-in Wisconsin dataset (same as Kaggle version)
raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df['target'] = raw.target          # 0 = malignant, 1 = benign
df['diagnosis'] = df['target'].map({0: 'M', 1: 'B'})

print(f"Dataset shape : {df.shape}")
print(f"Features      : {df.shape[1] - 2}")
print(f"Samples       : {df.shape[0]}")
print(f"\nClass distribution:\n{df['diagnosis'].value_counts()}")

# ──────────────────────────────────────────────────────────────
# PHASE 2: DATA CLEANING & PREPROCESSING
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 2] Data Cleaning & Preprocessing")
print("-" * 40)

print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Drop label helper column for modeling
X = df.drop(columns=['target', 'diagnosis'])
y = df['target']   # 1 = benign, 0 = malignant

print(f"\nFeature stats (mean ± std):")
print(X.describe().loc[['mean','std']].T.head(5).to_string())

# ──────────────────────────────────────────────────────────────
# PHASE 3: EDA (Exploratory Data Analysis)
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 3] Exploratory Data Analysis")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("EDA — Breast Cancer Wisconsin Dataset", fontsize=14)

# Class balance
counts = df['diagnosis'].value_counts()
axes[0].bar(['Benign (B)', 'Malignant (M)'], counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

# Correlation heatmap (top 10 features)
top_features = X.corrwith(y).abs().nlargest(10).index
corr = X[top_features].corr()
sns.heatmap(corr, ax=axes[1], cmap='coolwarm', annot=False, linewidths=0.3)
axes[1].set_title("Correlation Heatmap (Top 10 Features)")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("plots/eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → EDA plot saved to plots/eda.png")

# Benign vs Malignant — mean radius boxplot
fig, ax = plt.subplots(figsize=(8, 4))
df.boxplot(column='mean radius', by='diagnosis', ax=ax,
           patch_artist=True)
ax.set_title("Mean Radius by Diagnosis")
plt.suptitle("")
plt.savefig("plots/boxplot_mean_radius.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Boxplot saved to plots/boxplot_mean_radius.png")

# ──────────────────────────────────────────────────────────────
# PHASE 4: FEATURE ENGINEERING & SELECTION
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 4] Feature Engineering & Selection")
print("-" * 40)

# SelectKBest — pick top 15 features
selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(X, y)
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask].tolist()
print(f"  Top 15 features selected via ANOVA F-test:")
for f in selected_features:
    print(f"    • {f}")

X_selected = X[selected_features]

# ──────────────────────────────────────────────────────────────
# PHASE 5: HANDLE IMBALANCED DATA
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 5] Handling Class Imbalance")
print("-" * 40)

print(f"  Before SMOTE → Benign: {(y==1).sum()}, Malignant: {(y==0).sum()}")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_selected, y)
print(f"  After  SMOTE → Benign: {(y_bal==1).sum()}, Malignant: {(y_bal==0).sum()}")

# ──────────────────────────────────────────────────────────────
# PHASE 6: TRAIN / TEST SPLIT + SCALING
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 6] Train/Test Split & Feature Scaling")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────────
# PHASE 7: MODEL SELECTION — LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 7] Model Training — Logistic Regression")
print("-" * 40)

base_model = LogisticRegression(random_state=42, max_iter=1000)
cv_scores = cross_val_score(base_model, X_train_sc, y_train, cv=5, scoring='accuracy')
print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ──────────────────────────────────────────────────────────────
# PHASE 8: HYPERPARAMETER TUNING (GridSearchCV)
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 8] Hyperparameter Tuning (GridSearchCV)")
print("-" * 40)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2']
}
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train_sc, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"  Best params : {best_params}")
print(f"  Best CV AUC : {grid_search.best_score_:.4f}")

# ──────────────────────────────────────────────────────────────
# PHASE 9: MODEL EVALUATION
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 9] Model Evaluation on Test Set")
print("-" * 40)

y_pred      = best_model.predict(X_test_sc)
y_prob      = best_model.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])

print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"\n{report}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Model Evaluation", fontsize=14)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Malignant', 'Benign'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f"AUC = {auc:.4f}")
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Evaluation plots saved to plots/evaluation.png")

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'feature': selected_features,
    'coefficient': best_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='black')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Logistic Regression — Feature Coefficients")
ax.set_xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Feature importance plot saved to plots/feature_importance.png")

# ──────────────────────────────────────────────────────────────
# PHASE 10: SAVE MODEL & ARTIFACTS
# ──────────────────────────────────────────────────────────────
print("\n[PHASE 10] Saving Model & Artifacts")
print("-" * 40)

joblib.dump(best_model, "models/logistic_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(selected_features, "models/selected_features.pkl")

print("  → models/logistic_model.pkl")
print("  → models/scaler.pkl")
print("  → models/selected_features.pkl")

print("\n" + "=" * 60)
print(f"  PIPELINE COMPLETE  |  Accuracy: {acc*100:.2f}%  |  AUC: {auc:.4f}")
print("=" * 60)
