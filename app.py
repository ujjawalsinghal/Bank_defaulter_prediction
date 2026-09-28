import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Required when loading a model trained with the feature-engineering pipeline.
try:
    from src.feature_engineering import FinancialFeatureEngineer
except ImportError:
    FinancialFeatureEngineer = None

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .hero {
        padding: 1.7rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #374151 100%);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2.25rem;
    }
    .hero p {
        color: #d1d5db;
        margin: .45rem 0 0;
        font-size: 1rem;
    }
    .risk-card {
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid #374151;
        background: #111827;
        text-align: center;
        margin: .5rem 0 1rem;
    }
    .risk-label {
        font-size: 1.55rem;
        font-weight: 800;
    }
    .risk-description {
        color: #9ca3af;
        margin-top: .35rem;
    }
    .signal-positive {
        padding: .7rem .9rem;
        border-radius: 10px;
        background: rgba(220, 38, 38, .10);
        border-left: 4px solid #dc2626;
        margin-bottom: .55rem;
    }
    .signal-negative {
        padding: .7rem .9rem;
        border-radius: 10px;
        background: rgba(22, 163, 74, .10);
        border-left: 4px solid #16a34a;
        margin-bottom: .55rem;
    }
    .disclaimer {
        color: #6b7280;
        font-size: .82rem;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/bank_defaulter_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Could not find models/bank_defaulter_model.pkl")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🏦 Bank Default Risk Predictor</h1>
        <p>Predictive modeling, risk signals, and model performance in one dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Model Overview")
    has_feature_engineering = "feature_engineer" in getattr(model, "named_steps", {})
    st.write(
        "The deployed model is a Logistic Regression pipeline with "
        "numeric scaling and categorical one-hot encoding."
        + (" Feature engineering is embedded in the pipeline." if has_feature_engineering else "")
    )

    st.divider()

    st.subheader("Held-out Test Results")
    a, b = st.columns(2)
    with a:
        st.metric("Accuracy", "70.9%")
        st.metric("Recall", "66.3%")
    with b:
        st.metric("ROC-AUC", "0.7626")
        st.metric("F1", "57.3%")

    st.divider()
    st.caption(
        "Metrics are from the held-out test set. The model is a demonstration "
        "of predictive modeling and is not a standalone lending decision system."
    )

# ============================================================
# TABS
# ============================================================

predict_tab, performance_tab, methodology_tab = st.tabs(
    ["🔍 Prediction", "📊 Model Performance", "📘 Methodology"]
)

# ============================================================
# PREDICTION TAB
# ============================================================

with predict_tab:

    st.subheader("👤 Customer Financial Profile")

    with st.form("customer_form"):

        st.markdown("#### Personal & Employment")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            age = st.number_input("Age", 18, 100, 30, 1)

        with c2:
            employment_type = st.selectbox(
                "Employment Type",
                ["Salaried", "Self-Employed", "Business", "Unemployed"],
            )

        with c3:
            employment_years = st.number_input(
                "Employment Years", 0.0, 82.0, 5.0, .5
            )

        with c4:
            bank_account_age_years = st.number_input(
                "Bank Account Age (Years)", 0.0, 100.0, 5.0, .5
            )

        st.markdown("#### Income & Loan")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            annual_income = st.number_input(
                "Annual Income (₹)", 1.0, 1e12, 500000.0, 10000.0
            )

        with c2:
            loan_amount = st.number_input(
                "Loan Amount (₹)", 0.0, 1e12, 500000.0, 10000.0
            )

        with c3:
            loan_tenure_months = st.number_input(
                "Loan Tenure (Months)", 1, 600, 36, 1
            )

        with c4:
            existing_loans = st.number_input(
                "Existing Loans", 0, 50, 1, 1
            )

        st.markdown("#### Monthly Finances")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            monthly_debt = st.number_input(
                "Monthly Debt (₹)", 0.0, 1e9, 10000.0, 1000.0
            )

        with c2:
            monthly_expenses = st.number_input(
                "Monthly Expenses (₹)", 0.0, 1e9, 20000.0, 1000.0
            )

        with c3:
            savings_balance = st.number_input(
                "Savings Balance (₹)", 0.0, 1e12, 100000.0, 5000.0
            )

        with c4:
            transaction_frequency = st.number_input(
                "Transaction Frequency", 0.0, 100.0, 30.0, 1.0
            )

        st.markdown("#### Credit Behaviour")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            credit_score = st.number_input(
                "Credit Score", 300.0, 850.0, 700.0, 1.0
            )

        with c2:
            num_credit_cards = st.number_input(
                "Credit Cards", 0, 50, 2, 1
            )

        with c3:
            credit_utilization = st.number_input(
                "Credit Utilization",
                0.0, 1.0, .30, .01,
                help="0.30 means 30%."
            )

        with c4:
            missed_payments_last_year = st.number_input(
                "Missed Payments (Last Year)", 0, 50, 0, 1
            )

        previous_defaults = st.number_input(
            "Previous Defaults", 0, 20, 0, 1
        )

        submitted = st.form_submit_button(
            "🔍 Predict Default Risk",
            use_container_width=True
        )

    if submitted:

        if employment_years > max(age - 18, 0):
            st.warning(
                "Employment years cannot exceed the years available since age 18."
            )
            st.stop()

        loan_to_income_ratio = loan_amount / annual_income

        debt_to_income_ratio = (
            (monthly_debt + loan_amount / loan_tenure_months)
            / (annual_income / 12)
        )

        log_income = np.log1p(annual_income)
        log_loan_amount = np.log1p(loan_amount)
        log_savings = np.log1p(savings_balance)

        input_data = pd.DataFrame([{
            "age": age,
            "employment_type": employment_type,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "loan_tenure_months": loan_tenure_months,
            "existing_loans": existing_loans,
            "monthly_debt": monthly_debt,
            "monthly_expenses": monthly_expenses,
            "employment_years": employment_years,
            "bank_account_age_years": bank_account_age_years,
            "num_credit_cards": num_credit_cards,
            "missed_payments_last_year": missed_payments_last_year,
            "previous_defaults": previous_defaults,
            "credit_utilization": credit_utilization,
            "savings_balance": savings_balance,
            "transaction_frequency": transaction_frequency,
        }])

        # Backward compatibility:
        # - New model: feature_engineer creates ratios/logs internally.
        # - Old model: create the engineered columns here exactly as before.
        if "feature_engineer" not in getattr(model, "named_steps", {}):
            input_data["loan_to_income_ratio"] = loan_amount / annual_income
            input_data["debt_to_income_ratio"] = (
                (monthly_debt + loan_amount / loan_tenure_months)
                / (annual_income / 12)
            )
            input_data["log_income"] = np.log1p(annual_income)
            input_data["log_loan_amount"] = np.log1p(loan_amount)
            input_data["log_savings"] = np.log1p(savings_balance)

        # Calculate these for display regardless of where the model computes them.
        loan_to_income_ratio = loan_amount / annual_income
        debt_to_income_ratio = (
            (monthly_debt + loan_amount / loan_tenure_months)
            / (annual_income / 12)
        )

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        if probability < .30:
            risk_label = "🟢 Low Risk"
            risk_message = "The model estimates a relatively low probability of default."
        elif probability < .60:
            risk_label = "🟡 Moderate Risk"
            risk_message = "The model estimates a moderate probability of default."
        else:
            risk_label = "🔴 High Risk"
            risk_message = "The model estimates a relatively high probability of default."

        st.divider()
        st.subheader("📊 Prediction Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Default Probability", f"{probability * 100:.2f}%")
        with c2:
            st.metric("Debt-to-Income", f"{debt_to_income_ratio:.3f}")
        with c3:
            st.metric("Loan-to-Income", f"{loan_to_income_ratio:.3f}")

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="risk-label">{risk_label}</div>
                <div class="risk-description">{risk_message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            min(max(probability, 0), 1),
            text=f"Model probability: {probability * 100:.2f}%"
        )

        if prediction == 1:
            st.error(
                "⚠️ Model classification: Potential Defaulter "
                "(probability ≥ 50%)"
            )
        else:
            st.success(
                "✅ Model classification: Predicted Non-Defaulter "
                "(probability < 50%)"
            )

        # ----------------------------------------------------
        # MODEL EXPLAINABILITY
        # ----------------------------------------------------

        st.subheader("🧠 Why did the model produce this prediction?")

        st.caption(
            "These are model contributions, not causal explanations. "
            "Positive contributions increase the model's predicted default score; "
            "negative contributions decrease it."
        )
        st.caption(
            "A contribution describes how this trained model used the customer's "
            "inputs. It does not mean that changing a factor would necessarily "
            "change real-world default risk."
        )
        st.info(
            "Positive contributions increase the model's predicted default score; "
            "negative contributions decrease it. These values are model-level "
            "signals, not causal effects."
        )

        try:
            classifier = model.named_steps["model"]

            if "feature_engineer" in getattr(model, "named_steps", {}):
                # Run the raw input through every preprocessing step exactly
                # as the model does during prediction.
                preprocessing_pipeline = model[:-1]
                transformed = preprocessing_pipeline.transform(input_data)
                preprocessor = model.named_steps["preprocessor"]
            else:
                # Legacy model: engineered columns are already present in input_data.
                preprocessor = model.named_steps["preprocessor"]
                transformed = preprocessor.transform(input_data)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            feature_names = preprocessor.get_feature_names_out()
            coefficients = classifier.coef_[0]

            contributions = transformed[0] * coefficients

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "Contribution": contributions
            })

            explanation_df["Absolute"] = explanation_df["Contribution"].abs()

            top_positive = (
                explanation_df[explanation_df["Contribution"] > 0]
                .sort_values("Contribution", ascending=False)
                .head(5)
            )

            top_negative = (
                explanation_df[explanation_df["Contribution"] < 0]
                .sort_values("Contribution")
                .head(5)
            )

            def display_feature_name(name):
                name = name.replace("numeric__", "").replace("categorical__", "")
                if name.startswith("employment_type_"):
                    value = name.replace("employment_type_", "", 1)
                    return f"Employment Type: {value}"
                labels = {
                    "loan_to_income_ratio": "Loan-to-Income Ratio",
                    "debt_to_income_ratio": "Debt-to-Income Ratio",
                    "log_income": "Log Income",
                    "log_loan_amount": "Log Loan Amount",
                    "log_savings": "Log Savings",
                    "monthly_expenses": "Monthly Expenses",
                    "monthly_debt": "Monthly Debt",
                    "loan_tenure_months": "Loan Tenure (Months)",
                    "missed_payments_last_year": "Missed Payments (Last Year)",
                    "previous_defaults": "Previous Defaults",
                    "credit_utilization": "Credit Utilization",
                    "credit_score": "Credit Score",
                    "annual_income": "Annual Income",
                    "loan_amount": "Loan Amount",
                    "savings_balance": "Savings Balance",
                    "employment_years": "Employment Years",
                    "age": "Age",
                    "existing_loans": "Existing Loans",
                    "num_credit_cards": "Credit Cards",
                    "bank_account_age_years": "Bank Account Age (Years)",
                    "transaction_frequency": "Transaction Frequency",
                }
                return labels.get(name, name.replace("_", " ").title())

            ec1, ec2 = st.columns(2)

            with ec1:
                st.markdown("#### 🔴 Positive model contributions")
                if len(top_positive) == 0:
                    st.info("No positive model contributions detected.")
                else:
                    for _, row in top_positive.iterrows():
                        name = display_feature_name(row["Feature"])
                        st.markdown(
                            f"""
                            <div class="signal-positive">
                                <strong>{name}</strong><br>
                                Contribution: +{row["Contribution"]:.3f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with ec2:
                st.markdown("#### 🟢 Negative model contributions")
                if len(top_negative) == 0:
                    st.info("No negative model contributions detected.")
                else:
                    for _, row in top_negative.iterrows():
                        name = display_feature_name(row["Feature"])
                        st.markdown(
                            f"""
                            <div class="signal-negative">
                                <strong>{name}</strong><br>
                                Contribution: {row["Contribution"]:.3f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with st.expander("View all model contributions"):
                all_explanations = explanation_df.sort_values(
                    "Absolute", ascending=False
                ).drop(columns="Absolute")

                all_explanations["Feature"] = all_explanations["Feature"].apply(
                    display_feature_name
                )

                st.dataframe(
                    all_explanations,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.warning(
                "Prediction was successful, but model contribution details "
                f"could not be displayed: {exc}"
            )

        # ----------------------------------------------------
        # SIGNAL CHECKS
        # ----------------------------------------------------

        st.subheader("🔎 Customer Risk Signals")

        c1, c2 = st.columns(2)

        with c1:
            if credit_score < 650:
                st.warning(f"Credit score: {credit_score:.0f}")
            else:
                st.info(f"Credit score: {credit_score:.0f}")

            if previous_defaults > 0:
                st.warning(f"Previous defaults: {previous_defaults}")
            else:
                st.success("No previous defaults reported.")

            if missed_payments_last_year > 0:
                st.warning(
                    f"Missed payments: {missed_payments_last_year}"
                )
            else:
                st.success("No missed payments reported.")

        with c2:
            if debt_to_income_ratio > 1:
                st.warning(
                    f"Debt-to-income ratio: {debt_to_income_ratio:.3f}"
                )
            else:
                st.info(
                    f"Debt-to-income ratio: {debt_to_income_ratio:.3f}"
                )

            if credit_utilization > .70:
                st.warning(
                    f"Credit utilization: {credit_utilization * 100:.1f}%"
                )
            else:
                st.info(
                    f"Credit utilization: {credit_utilization * 100:.1f}%"
                )

            st.info(
                f"Loan-to-income ratio: {loan_to_income_ratio:.3f}"
            )

        with st.expander("📋 View Customer Input Summary"):
            summary = pd.DataFrame({
                "Feature": [
                    "Age", "Employment Type", "Annual Income", "Credit Score",
                    "Loan Amount", "Loan Tenure", "Existing Loans",
                    "Monthly Debt", "Monthly Expenses", "Employment Years",
                    "Bank Account Age", "Credit Cards", "Missed Payments",
                    "Previous Defaults", "Credit Utilization", "Savings Balance",
                    "Transaction Frequency", "Debt-to-Income Ratio",
                    "Loan-to-Income Ratio"
                ],
                "Value": [
                    age, employment_type, f"₹{annual_income:,.0f}",
                    f"{credit_score:.0f}", f"₹{loan_amount:,.0f}",
                    f"{loan_tenure_months} months", existing_loans,
                    f"₹{monthly_debt:,.0f}", f"₹{monthly_expenses:,.0f}",
                    f"{employment_years:.1f} years",
                    f"{bank_account_age_years:.1f} years",
                    num_credit_cards, missed_payments_last_year,
                    previous_defaults, f"{credit_utilization * 100:.1f}%",
                    f"₹{savings_balance:,.0f}",
                    f"{transaction_frequency:.0f}",
                    f"{debt_to_income_ratio:.3f}",
                    f"{loan_to_income_ratio:.3f}"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.subheader("📊 Model Performance")

    st.write(
        "The following results were obtained during the notebook-based "
        "evaluation on the 2,000-row held-out test set."
    )

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Tuned Random Forest",
            "Tuned Logistic Regression",
        ],
        "Accuracy": [.7100, .7250, .7315, .7270, .7090],
        "Precision": [.5065, .5329, .5509, .5362, .5052],
        "Recall": [.6593, .5492, .4864, .5525, .6627],
        "F1 Score": [.5729, .5409, .5167, .5442, .5733],
        "ROC-AUC": [.7632, .7243, .7449, .7545, .7626],
    })

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### ROC-AUC Comparison")

    auc_chart = results[["Model", "ROC-AUC"]].set_index("Model")
    st.bar_chart(auc_chart)

    st.markdown("#### Confusion Matrix — Tuned Logistic Regression")

    cm = pd.DataFrame(
        [[1027, 383], [199, 391]],
        index=["Actual Non-Default", "Actual Default"],
        columns=["Predicted Non-Default", "Predicted Default"],
    )

    st.dataframe(cm, use_container_width=True)

    st.info(
        "The ROC-AUC values summarize ranking/discrimination ability. "
        "For this project, recall is also important because it measures "
        "the proportion of actual defaulters detected by the model."
    )

# ============================================================
# METHODOLOGY TAB
# ============================================================

with methodology_tab:

    st.subheader("📘 Project Methodology")

    st.markdown("""
    **1. Data preparation**

    The raw banking dataset was intentionally constructed with data-quality
    problems such as missing values, duplicate records, inconsistent
    categorical labels, invalid values, and synthetic anomalies.

    **2. Data cleaning**

    The workflow handled missing values, numeric conversion, impossible
    ranges, categorical standardization, duplicate customer IDs, derived
    financial ratios, and deliberately injected transaction-frequency
    anomalies.

    **3. Exploratory Data Analysis**

    The target distribution was approximately:

    - Non-default: **70.5%**
    - Default: **29.5%**

    Important observed associations included debt-to-income ratio,
    previous defaults, missed payments, and credit score.

    **4. Modeling**

    Three baseline algorithms were evaluated:

    - Logistic Regression
    - Decision Tree
    - Random Forest

    Hyperparameter tuning was subsequently performed for Logistic Regression
    and Random Forest using 5-fold cross-validation and ROC-AUC scoring.

    **5. Final deployed model**

    The deployed pipeline is **Logistic Regression** because it provided
    strong discrimination and recall while remaining relatively interpretable.

    **6. Threshold analysis**

    Classification was evaluated around the standard 0.50 probability
    threshold. Lower thresholds generally increase recall at the cost of
    precision, while higher thresholds do the opposite.

    **7. Explainability**

    The prediction page displays feature-level Logistic Regression
    contributions. These indicate how the model's inputs move the prediction
    in log-odds space; they should not be interpreted as causal effects.
    """)

    st.warning(
        "This application is an educational machine-learning project. "
        "A real lending system would require extensive validation, "
        "calibration, fairness testing, governance, security, and regulatory "
        "review before deployment."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        Bank Default Risk Predictor · Machine Learning Project<br>
        Model estimates should not be treated as standalone financial decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
