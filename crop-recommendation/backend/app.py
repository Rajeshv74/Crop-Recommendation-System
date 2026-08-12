"""
app.py
------
Flask REST API for the Crop Recommendation System.

Endpoints:
    GET  /api/health         -> health check
    POST /api/predict        -> predict crop from 7 input features
    GET  /api/crops          -> list all crops the model knows about
    GET  /api/feature-info   -> min/max/mean stats per feature (for frontend hints)
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from the React dev server

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ---- Load trained artifacts at startup ----
model = None
label_encoder = None

def load_artifacts():
    global model, label_encoder
    model_path = os.path.join(MODEL_DIR, "crop_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python train_model.py --data <csv>` first."
        )

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("Model and label encoder loaded successfully.")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Validate all 7 fields are present and numeric
    missing = [f for f in FEATURE_COLUMNS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = [float(data[f]) for f in FEATURE_COLUMNS]
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric"}), 400

    X = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    pred_idx = model.predict(X)[0]
    pred_crop = label_encoder.inverse_transform([pred_idx])[0]

    # Top-3 predictions with probabilities, for a richer UI
    probs = model.predict_proba(X)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"crop": label_encoder.inverse_transform([i])[0], "confidence": round(float(probs[i]) * 100, 2)}
        for i in top3_idx
    ]

    return jsonify({
        "recommended_crop": pred_crop,
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "top_3": top3,
        "input": dict(zip(FEATURE_COLUMNS, values)),
    })


@app.route("/api/crops", methods=["GET"])
def list_crops():
    if label_encoder is None:
        return jsonify({"error": "Model not loaded on server"}), 500
    return jsonify({"crops": sorted(label_encoder.classes_.tolist())})


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True, host="0.0.0.0", port=5000)
