import joblib
from sklearn.base import BaseEstimator

# Load and cache the TF-IDF vectorizer
VECTORIZER = joblib.load("tfidf_vectorizer.pkl")

def score(message: str, estimator: BaseEstimator, threshold: float):
    """
    Evaluate a message for spam classification using a machine learning model.

    Parameters:
    - message (str): Text to be analyzed.
    - estimator (BaseEstimator): A pre-trained classification model.
    - threshold (float): Classification cutoff threshold.

    Returns:
    - (bool): True if spam, False otherwise.
    - (float): Model's predicted probability for spam.
    """

    # Validate inputs
    if type(message) is not str:
        raise TypeError("Expected a string for the message input.")
    if not isinstance(threshold, (int, float)):
        raise TypeError("Threshold must be a numeric type.")

    # Normalize threshold
    threshold = float(min(max(threshold, 0), 1))

    # Preprocess input message
    vectorized_input = VECTORIZER.transform([message])

    # Get spam probability (assumes binary classification)
    probability = estimator.predict_proba(vectorized_input)[0,1]
    print("Spam probability:", probability)

    # Determine label
    is_spam_flag = probability >= threshold
    return is_spam_flag, probability
