import numpy as np
import pandas as pd


# ============================================================
# EXPECTED MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Amount_Log",
    "Time_Hours",
    "Hour_Sin",
    "Hour_Cos"
]


# ============================================================
# ORIGINAL FEATURES
# ============================================================

ORIGINAL_FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def engineer_features(df):
    """
    Create the same features used during model development.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the original transaction features.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the model-ready features.
    """

    data = df.copy()

    # Check that all original features exist
    missing_columns = [
        column
        for column in ORIGINAL_FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Amount transformation
    # --------------------------------------------------------

    data["Amount_Log"] = np.log1p(
        data["Amount"]
    )

    # --------------------------------------------------------
    # Time transformation
    # --------------------------------------------------------

    data["Time_Hours"] = (
        data["Time"] / 3600
    )

    # --------------------------------------------------------
    # Cyclical time features
    # --------------------------------------------------------

    hour_of_day = (
        data["Time_Hours"] % 24
    )

    data["Hour_Sin"] = np.sin(
        2 * np.pi * hour_of_day / 24
    )

    data["Hour_Cos"] = np.cos(
        2 * np.pi * hour_of_day / 24
    )

    # --------------------------------------------------------
    # Select exact model features
    # --------------------------------------------------------

    data = data[FEATURE_COLUMNS]

    return data