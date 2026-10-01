import numpy as np

from src.model_utils import predict_fraud_probability


class DummyModel:

    def predict_proba(self, features):
        return np.array([
            [0.90, 0.10],
            [0.20, 0.80],
            [0.70, 0.30]
        ])


def test_predict_fraud_probability():

    model = DummyModel()

    features = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    probabilities = predict_fraud_probability(
        model,
        features
    )

    assert len(probabilities) == 3
    assert probabilities[0] == 0.10
    assert probabilities[1] == 0.80
    assert probabilities[2] == 0.30