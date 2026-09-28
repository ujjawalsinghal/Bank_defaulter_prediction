# 🏦 Bank Default Risk Predictor

An end-to-end machine-learning project that predicts the probability of bank-customer default from financial, credit, employment, and behavioral attributes.

The project intentionally starts with a messy synthetic banking dataset so the workflow demonstrates **realistic data cleaning, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, threshold analysis, explainability, and deployment with Streamlit**.

> **Important:** This is an educational ML project. The model is not a standalone lending decision system and has not undergone the validation, calibration, fairness, governance, security, or regulatory review required for real-world lending.

## ✨ Features

- Data-quality inspection and cleaning
- Missing-value treatment and invalid-value handling
- Duplicate-record and duplicate-customer detection
- Categorical-label standardization
- Financial ratio feature engineering
- Log transformations for highly skewed variables
- Exploratory data analysis
- Logistic Regression, Decision Tree, and Random Forest baselines
- 5-fold ROC-AUC hyperparameter tuning
- Probability-threshold analysis
- Logistic Regression feature-contribution explainability
- Streamlit interactive prediction dashboard
- Training-serving consistency through a custom feature-engineering transformer

## 🧠 Machine-Learning Workflow

```text
Raw anomalous data
        ↓
Data inspection
        ↓
Cleaning & validation
        ↓
EDA
        ↓
Feature engineering
        ↓
Train / test split
        ↓
Preprocessing pipeline
        ↓
Baseline model comparison
        ↓
Hyperparameter tuning
        ↓
Threshold analysis
        ↓
Final Logistic Regression pipeline
        ↓
Streamlit dashboard
```

## 📊 Dataset

The cleaned dataset contains **10,000 customer records** and a binary target:

- `0` → Non-defaulter
- `1` → Defaulter

The cleaned project data contains financial, credit, employment, and transaction-related variables such as:

- Age
- Employment type
- Annual income
- Credit score
- Loan amount
- Loan tenure
- Existing loans
- Monthly debt and expenses
- Missed payments
- Previous defaults
- Credit utilization
- Savings balance
- Transaction frequency
- Loan-to-income ratio
- Debt-to-income ratio

The original raw dataset intentionally included missing values, duplicate records, inconsistent categorical labels, invalid values, and synthetic anomalies.

## 🧹 Data Cleaning

The cleaning workflow included:

1. Removing exact duplicate records.
2. Converting intended numeric fields to numeric values.
3. Median imputation for selected missing numeric values.
4. Validating age and credit-score ranges.
5. Handling non-positive loan amounts.
6. Validating employment duration against age.
7. Standardizing employment categories.
8. Removing duplicate customer IDs.
9. Recalculating financial ratios after cleaning.
10. Detecting and handling deliberately injected transaction-frequency anomalies.

## 🔧 Feature Engineering

The model uses several derived features:

```text
loan_to_income_ratio
debt_to_income_ratio
log_income
log_loan_amount
log_savings
```

Feature engineering is implemented inside the trained scikit-learn pipeline using a custom transformer. This keeps training and inference transformations consistent.

## 🤖 Models Evaluated

Baseline models:

- Logistic Regression
- Decision Tree
- Random Forest

Hyperparameter tuning was performed for:

- Logistic Regression
- Random Forest

Grid search used **5-fold cross-validation** with **ROC-AUC** as the optimization metric.

### Held-out test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7100 | 0.5065 | 0.6593 | 0.5729 | 0.7632 |
| Decision Tree | 0.7250 | 0.5329 | 0.5492 | 0.5409 | 0.7243 |
| Random Forest | 0.7315 | 0.5509 | 0.4864 | 0.5167 | 0.7449 |
| Tuned Random Forest | 0.7270 | 0.5362 | 0.5525 | 0.5442 | 0.7545 |
| Tuned Logistic Regression | 0.7090 | 0.5052 | 0.6627 | 0.5733 | 0.7626 |

The deployed model is the tuned Logistic Regression pipeline because it provides a useful combination of discrimination, recall, and interpretability for this educational project.

## 🔍 Explainability

The dashboard displays feature-level Logistic Regression contributions.

A positive contribution means that, for this model and this particular input, the feature moves the model's score toward default. A negative contribution moves it away from default.

These are **model contributions, not causal explanations**.

## 🖥️ Streamlit Dashboard

The dashboard contains:

### Prediction
Enter a customer's financial and credit information and receive:

- Default probability
- Risk category
- Model classification
- Debt-to-income ratio
- Loan-to-income ratio
- Customer risk signals
- Feature-level model contributions

### Model Performance
Displays:

- Model comparison
- ROC-AUC comparison
- Confusion matrix for the deployed model

### Methodology
Documents the complete ML workflow and project limitations.

## 📁 Project Structure

```text
bank-default-risk/
│
├── app.py
├── retrain_model.py
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── bank_defaulter_raw_anomalous.csv
│   └── bank_defaulter_clean.csv
│
├── models/
│   └── bank_defaulter_model.pkl
│
├── notebooks/
│   └── bank_defaulter_analysis.ipynb
│
└── src/
    ├── __init__.py
    └── feature_engineering.py
```

## 🚀 Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

From the project root:

```bash
python retrain_model.py
```

Then launch the dashboard:

```bash
streamlit run app.py
```

The browser should open the Streamlit application automatically.

## 🔁 Reproducibility

The training script rebuilds the model from the cleaned dataset and saves the resulting pipeline to:

```text
models/bank_defaulter_model.pkl
```

The saved pipeline contains the feature-engineering, preprocessing, and classification stages so the same transformations can be applied during inference.

## ⚠️ Limitations

This project uses a synthetic dataset and therefore does not establish real-world lending performance.

A production credit-risk system would additionally require, among other things:

- Probability calibration
- Out-of-time validation
- Robust missing-data strategy
- Population stability and drift monitoring
- Fairness and bias evaluation
- Model governance
- Security controls
- Auditability
- Regulatory compliance
- Human oversight

## 👨‍💻 Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib / data visualization tools
- Jupyter Notebook

## 📌 Learning Outcomes

This project demonstrates an end-to-end understanding of:

- Data preprocessing
- EDA
- Feature engineering
- Classification
- Class imbalance
- Model evaluation
- Cross-validation
- Hyperparameter tuning
- Threshold selection
- Explainability
- ML pipelines
- Model serialization
- Streamlit deployment
- Training-serving consistency
