import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
print("Loading Breast Cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Train/Test Split
# Fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Define Base Models
# Note: Feature scaling is applied where appropriate using pipelines.

# Model 1: Calibrated GaussianNB
# Scaling is often helpful for GaussianNB if variance differs significantly, though not strictly required.
# We include it as requested "where needed".
gnb = GaussianNB()
calibrated_gnb = CalibratedClassifierCV(gnb, method='sigmoid', cv=5)
# GNB generally doesn't require scaling, but we'll keep the raw features or scale?
# The prompt asks for scaling "especially for Logistic Regression and GaussianNB".
# We will wrap it in a pipeline with scaler for safety and adherence to prompt.
model_gnb = make_pipeline(StandardScaler(), calibrated_gnb)

# Model 2: Logistic Regression
# Scaling is crucial for convergence and regularization.
model_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
)

# Model 3: Random Forest
# Tree-based models are scale-invariant, so no scaler needed.
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

base_estimators = [
    ('gnb', model_gnb),
    ('lr', model_lr),
    ('rf', model_rf)
]

# 4. Define Stacking Classifier
# Uses LogisticRegression as the meta-learner (final_estimator)
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(solver='liblinear', random_state=42),
    cv=5  # 5-fold cross-validation for training meta-learner
)

# 5. Training & Evaluation

model_accuracies = {}

print("\n--- Training Individual Base Models ---")
for name, model in base_estimators:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_accuracies[name] = acc
    print(f"Model: {name}, Test Accuracy: {acc:.4f}")

print("\n--- Training Stacking Ensemble ---")
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)
stack_acc = accuracy_score(y_test, y_pred_stack)
print(f"Stacked Ensemble Test Accuracy: {stack_acc:.4f}")
print("\nClassification Report (Stacked Ensemble):")
print(classification_report(y_test, y_pred_stack))

# 6. Comparison Section

federated_ensemble_accuracy = 0.00  # replace with your FL ensemble accuracy

print("\n" + "="*40)
print("       MODEL PERFORMANCE COMPARISON       ")
print("="*40)
print(f"{'Model':<25} | {'Accuracy':<10}")
print("-" * 40)
for name, acc in model_accuracies.items():
    print(f"{name:<25} | {acc:.4f}")
print("-" * 40)
print(f"{'Centralized Stacked':<25} | {stack_acc:.4f}")
print(f"{'Federated Ensemble (FL)':<25} | {federated_ensemble_accuracy:.4f}")
print("="*40)
