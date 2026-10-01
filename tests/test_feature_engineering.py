import pandas as pd

from src.feature_engineering import engineer_features
from src.feature_engineering import FEATURE_COLUMNS


def test_engineer_features_shape():
    data = pd.DataFrame({
        "Time": [1000, 2000],
        "V1": [0.1, 0.2],
        "V2": [0.3, 0.4],
        "V3": [0.5, 0.6],
        "V4": [0.7, 0.8],
        "V5": [0.9, 1.0],
        "V6": [1.1, 1.2],
        "V7": [1.3, 1.4],
        "V8": [1.5, 1.6],
        "V9": [1.7, 1.8],
        "V10": [1.9, 2.0],
        "V11": [2.1, 2.2],
        "V12": [2.3, 2.4],
        "V13": [2.5, 2.6],
        "V14": [2.7, 2.8],
        "V15": [2.9, 3.0],
        "V16": [3.1, 3.2],
        "V17": [3.3, 3.4],
        "V18": [3.5, 3.6],
        "V19": [3.7, 3.8],
        "V20": [3.9, 4.0],
        "V21": [4.1, 4.2],
        "V22": [4.3, 4.4],
        "V23": [4.5, 4.6],
        "V24": [4.7, 4.8],
        "V25": [4.9, 5.0],
        "V26": [5.1, 5.2],
        "V27": [5.3, 5.4],
        "V28": [5.5, 5.6],
        "Amount": [100, 200]
    })

    result = engineer_features(data)

    assert result.shape == (2, 34)
    assert result.columns.tolist() == FEATURE_COLUMNS


def test_engineer_features_creates_log_amount():
    data = pd.DataFrame({
        "Time": [3600],
        **{f"V{i}": [0.0] for i in range(1, 29)},
        "Amount": [100]
    })

    result = engineer_features(data)

    assert "Amount_Log" in result.columns
    assert result["Amount_Log"].iloc[0] > 0