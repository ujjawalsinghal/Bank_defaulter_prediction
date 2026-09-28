import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import FinancialFeatureEngineer


DATA_PATH = "data/bank_defaulter_clean.csv"
MODEL_PATH = "models/bank_defaulter_model.pkl"
METRICS_PATH = "models/model_metrics.json"

DERIVED_FEATURES = {
    "loan_to_income_ratio",
    "debt_to_income_ratio",
    "log_income",
    "log_loan_amount",
    "log_savings",
}


df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customer_id", "default"]).copy()
y = df["default"].copy()

# The transformer will recreate these columns. Do not feed the old
# precomputed versions into the model, otherwise feature engineering
# would be duplicated.
X = X.drop(columns=[c for c in DERIVED_FEATURES if c in X.columns])

categorical_features = ["employment_type"]
numeric_features = [c for c in X.columns if c not in categorical_features]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features + sorted(DERIVED_FEATURES)),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("feature_engineer", FinancialFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=0.1,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.50).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "f1": f1_score(y_test, predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_test, probabilities),
    "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
}

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Model saved:", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)
print("\nHeld-out test metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")
