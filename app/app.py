import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ============================================================
# REUSABLE PROJECT MODULES
# ============================================================

from src.feature_engineering import engineer_features
from src.feature_engineering import ORIGINAL_FEATURE_COLUMNS
from src.feature_engineering import FEATURE_COLUMNS

from src.risk_scoring import assign_risk_segment
from src.risk_scoring import assign_priority

from src.model_utils import predict_fraud_probability


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "final_fraud_model.joblib"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "fraud_modeling_data.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection & Transaction Risk Intelligence",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💳 Fraud Detection & Transaction Risk Intelligence Platform")

st.write(
    """
    This application predicts the probability that a transaction is fraudulent
    and provides a risk segment, investigation priority, and SHAP-based
    explanation of the model prediction.
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_shap_explainer():

    model = joblib.load(MODEL_PATH)

    # Extract the Gradient Boosting estimator
    # from the calibrated model.

    gb_model = model.calibrated_classifiers_[0].estimator

    explainer = shap.TreeExplainer(gb_model)

    return explainer


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    return data


# ============================================================
# SHAP VALUE CALCULATION
# ============================================================

def calculate_shap_values(
    explainer,
    features
):

    shap_values = explainer.shap_values(features)

    # Handle possible list output
    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    # Handle possible 3-dimensional output
    if shap_values.ndim == 3:

        shap_values = shap_values[:, :, 1]

    return shap_values


# ============================================================
# DISPLAY SHAP EXPLANATION
# ============================================================

def display_shap_explanation(
    explainer,
    features
):

    try:

        shap_values = calculate_shap_values(
            explainer,
            features
        )

        values = shap_values[0]

        feature_names = features.columns.tolist()

        shap_table = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": values,
            "Absolute SHAP": np.abs(values)
        })

        shap_table = shap_table.sort_values(
            "Absolute SHAP",
            ascending=False
        )

        st.subheader("🔎 SHAP Explanation")

        st.write(
            """
            SHAP values show how each feature contributed to this
            individual prediction.

            Positive values push the prediction toward fraud.
            Negative values push the prediction away from fraud.
            """
        )

        # Top 10 features
        top_shap = shap_table.head(10).copy()

        st.dataframe(
            top_shap[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ],
            use_container_width=True
        )

        # Plot
        plot_data = top_shap.sort_values(
            "SHAP Value"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.barh(
            plot_data["Feature"],
            plot_data["SHAP Value"]
        )

        ax.set_xlabel("SHAP Value")
        ax.set_ylabel("Feature")
        ax.set_title(
            "Top Features Contributing to Prediction"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    except Exception as error:

        st.warning(
            f"SHAP explanation could not be generated: {error}"
        )


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:

    model = load_model()

    explainer = load_shap_explainer()

    data = load_data()

except Exception as error:

    st.error(
        f"Unable to load project files: {error}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Transaction Analysis")

analysis_mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    [
        "Example Fraud Transaction",
        "Example Legitimate Transaction",
        "Manual Input"
    ]
)


# ============================================================
# SELECT TRANSACTION
# ============================================================

actual_class = None


# ------------------------------------------------------------
# EXAMPLE FRAUD TRANSACTION
# ------------------------------------------------------------

if analysis_mode == "Example Fraud Transaction":

    fraud_transactions = data[
        data["Class"] == 1
    ]

    if len(fraud_transactions) == 0:

        st.error(
            "No fraud transactions were found in the dataset."
        )

        st.stop()

    transaction = fraud_transactions.iloc[
        [0]
    ].copy()

    actual_class = 1

    st.info(
        "Using an actual fraud transaction from the dataset."
    )


# ------------------------------------------------------------
# EXAMPLE LEGITIMATE TRANSACTION
# ------------------------------------------------------------

elif analysis_mode == "Example Legitimate Transaction":

    legitimate_transactions = data[
        data["Class"] == 0
    ]

    if len(legitimate_transactions) == 0:

        st.error(
            "No legitimate transactions were found in the dataset."
        )

        st.stop()

    transaction = legitimate_transactions.iloc[
        [0]
    ].copy()

    actual_class = 0

    st.info(
        "Using an actual legitimate transaction from the dataset."
    )


# ------------------------------------------------------------
# MANUAL INPUT
# ------------------------------------------------------------

else:

    st.sidebar.subheader("Transaction Inputs")

    time_value = st.sidebar.number_input(
        "Time",
        min_value=0.0,
        value=1000.0
    )

    amount_value = st.sidebar.number_input(
        "Amount",
        min_value=0.0,
        value=100.0
    )

    manual_values = {}

    manual_values["Time"] = time_value

    for i in range(1, 29):

        manual_values[f"V{i}"] = st.sidebar.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )

    manual_values["Amount"] = amount_value

    transaction = pd.DataFrame(
        [manual_values]
    )


# ============================================================
# PREPARE TRANSACTION
# ============================================================

original_transaction = transaction[
    ORIGINAL_FEATURE_COLUMNS
].copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

try:

    engineered_features = engineer_features(
        original_transaction
    )

except Exception as error:

    st.error(
        f"Feature engineering failed: {error}"
    )

    st.stop()


# ============================================================
# MODEL PREDICTION
# ============================================================

try:

    probability = predict_fraud_probability(
        model,
        engineered_features
    )[0]

except Exception as error:

    st.error(
        f"Prediction failed: {error}"
    )

    st.stop()


# ============================================================
# RISK SEGMENT
# ============================================================

risk_segment = assign_risk_segment(
    probability
)

priority = assign_priority(
    risk_segment
)


# ============================================================
# MAIN RESULTS
# ============================================================

st.header("Transaction Risk Assessment")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )


with col2:

    st.metric(
        "Risk Segment",
        risk_segment
    )


with col3:

    st.metric(
        "Investigation Priority",
        priority
    )


with col4:

    if probability >= 0.50:

        status = "Critical"

    elif probability >= 0.10:

        status = "Review"

    else:

        status = "Low Concern"

    st.metric(
        "Recommended Status",
        status
    )


# ============================================================
# ACTUAL LABEL
# ============================================================

if actual_class is not None:

    st.subheader("Dataset Ground Truth")

    if actual_class == 1:

        st.error(
            "Actual dataset label: FRAUD"
        )

    else:

        st.success(
            "Actual dataset label: LEGITIMATE"
        )


# ============================================================
# RISK INTERPRETATION
# ============================================================

st.subheader("Risk Interpretation")

if risk_segment == "Very Low":

    st.success(
        "Very Low Risk — the model assigns a very low estimated "
        "probability of fraud."
    )

elif risk_segment == "Low":

    st.success(
        "Low Risk — the estimated fraud probability is relatively low."
    )

elif risk_segment == "Medium":

    st.warning(
        "Medium Risk — the transaction may warrant additional attention."
    )

elif risk_segment == "High":

    st.warning(
        "High Risk — the transaction should receive increased "
        "fraud-investigation attention."
    )

else:

    st.error(
        "Critical Risk — the model assigns a high estimated "
        "probability of fraud."
    )


# ============================================================
# ORIGINAL TRANSACTION FEATURES
# ============================================================

with st.expander("📋 Original Transaction Features"):

    st.dataframe(
        original_transaction.T.rename(
            columns={original_transaction.index[0]: "Value"}
        ),
        use_container_width=True
    )


# ============================================================
# ENGINEERED FEATURES
# ============================================================

with st.expander("⚙️ Engineered Model Features"):

    st.dataframe(
        engineered_features.T.rename(
            columns={engineered_features.index[0]: "Value"}
        ),
        use_container_width=True
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

display_shap_explanation(
    explainer,
    engineered_features
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fraud Detection & Transaction Risk Intelligence Platform | "
    "Machine Learning Portfolio Project"
)