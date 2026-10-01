import joblib


def load_model(model_path):
    """
    Load a saved machine-learning model.
    """

    return joblib.load(model_path)


def predict_fraud_probability(model, features):
    """
    Generate fraud probabilities for transactions.
    """

    probabilities = model.predict_proba(
        features
    )[:, 1]

    return probabilities