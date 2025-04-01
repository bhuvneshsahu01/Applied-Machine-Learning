import joblib
import numpy as np
from sklearn.base import BaseEstimator

# Load the previously trained TF-IDF vectorizer
vectorizer = joblib.load("vectorizer.pkl")

def score(text: str, model: BaseEstimator, threshold: float) -> tuple[bool, float]:
    """
    Evaluates the given text using a trained classification model.

    Parameters:
    - text (str): The input text to analyze.
    - model (sklearn.base.BaseEstimator): Pre-trained classification model.
    - threshold (float): Cutoff value to classify the text.

    Returns:
    - prediction (bool): True if classified as spam, False otherwise.
    - propensity (float): Probability score for spam classification.
    """
    if not isinstance(text, str) or not isinstance(threshold, (int, float)):
        raise ValueError("Invalid data types provided.")

    # Restrict threshold to a valid probability range
    threshold = max(0, min(1, threshold))

    # Convert text to its TF-IDF representation
    text_tfidf = vectorizer.transform([text])  # Apply vectorization to input text
    print(text_tfidf)

    # Extract probability for the spam class (class 1)
    prediction_proba = model.predict_proba(text_tfidf)[0, 1]
    print(prediction_proba)

    # Make final classification decision based on threshold
    prediction = prediction_proba >= threshold

    return bool(prediction), float(prediction_proba)
