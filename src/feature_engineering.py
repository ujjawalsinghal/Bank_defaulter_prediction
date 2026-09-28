import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the same financial features during training and inference."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()

        safe_income = np.where(
            pd.to_numeric(X_new["annual_income"], errors="coerce") <= 0,
            1.0,
            pd.to_numeric(X_new["annual_income"], errors="coerce"),
        )
        safe_tenure = np.where(
            pd.to_numeric(X_new["loan_tenure_months"], errors="coerce") <= 0,
            1.0,
            pd.to_numeric(X_new["loan_tenure_months"], errors="coerce"),
        )

        X_new["loan_to_income_ratio"] = (
            X_new["loan_amount"] / safe_income
        ).astype(float)

        X_new["debt_to_income_ratio"] = (
            (X_new["monthly_debt"] + X_new["loan_amount"] / safe_tenure)
            / (safe_income / 12)
        ).astype(float)

        X_new["log_income"] = np.log1p(
            np.clip(X_new["annual_income"], a_min=0, a_max=None)
        )
        X_new["log_loan_amount"] = np.log1p(
            np.clip(X_new["loan_amount"], a_min=0, a_max=None)
        )
        X_new["log_savings"] = np.log1p(
            np.clip(X_new["savings_balance"], a_min=0, a_max=None)
        )

        return X_new

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([
                "loan_to_income_ratio",
                "debt_to_income_ratio",
                "log_income",
                "log_loan_amount",
                "log_savings",
            ], dtype=object)

        return np.asarray(input_features, dtype=object)
