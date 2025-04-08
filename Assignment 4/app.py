from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load serialized model and vectorizer
MODEL = joblib.load("best_model.pkl")
TFIDF = joblib.load("vectorizer.pkl")

@app.route("/score", methods=["POST"])
def score_text():
    payload = request.get_json(force=True)
    input_text = payload.get("text", "").strip()

    # Validate the input
    if not input_text:
        return jsonify({"error": "Invalid input. Text must be a non-empty string."}), 400

    # Vectorize input
    vector_input = TFIDF.transform([input_text])

    # Get spam probability (class 1)
    spam_prob = MODEL.predict_proba(vector_input)[0,1]

    # Binary prediction using fixed threshold
    cutoff = 0.5
    is_spam = int(spam_prob >= cutoff)

    return jsonify({
        "prediction": is_spam,
        "propensity": spam_prob
    })

if __name__ == "__main__":
    # Run on all network interfaces for Docker use
    app.run(host="0.0.0.0", port=5000, debug=True)
