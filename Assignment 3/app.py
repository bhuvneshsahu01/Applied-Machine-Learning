from flask import Flask, request, jsonify
import joblib


app = Flask(__name__)

# Load pre-trained model and vectorizer from files
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")



@app.route("/score", methods=["POST"])
def score_text():
    # Get the JSON data from the POST request
    data = request.get_json()
    text = data.get("text", "")

    # Check if the input is a valid non-empty string
    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "Invalid input. Please provide a valid text."}), 400

    # Vectorize the input text using the pre-loaded vectorizer
    text_vectorized = vectorizer.transform([text])

    # Calculate the probability score for the positive class
    prediction_proba = model.predict_proba(text_vectorized)[0, 1]

    # Apply the threshold for classification (default is 0.5)
    threshold = 0.5
    prediction = int(prediction_proba >= threshold)

    # Return the prediction and probability score in the response
    return jsonify({"prediction": prediction, "propensity": prediction_proba})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
