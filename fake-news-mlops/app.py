from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from src.preprocess import clean_text

app = Flask(__name__)
CORS(app)   # 🔥 THIS LINE FIXES THE ISSUE

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news_text = data.get("text", "")

    cleaned_text = clean_text(news_text)
    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()

    THRESHOLD = 0.65  # 65%

    if confidence >= THRESHOLD:
        label = "REAL" if prediction == 1 else "FAKE"
    else:
        label = "UNCERTAIN"

    return jsonify({
    "prediction": label,
    "confidence": round(confidence * 100, 2),
    "probabilities": {
        "fake": round(model.predict_proba(vector)[0][0] * 100, 2),
        "real": round(model.predict_proba(vector)[0][1] * 100, 2)
    }
})


if __name__ == "__main__":
    app.run(debug=True)
