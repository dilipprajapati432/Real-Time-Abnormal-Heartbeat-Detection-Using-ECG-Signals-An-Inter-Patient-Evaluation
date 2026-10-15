"""
Part 4: FastAPI backend - real-time heartbeat scoring service.

Run with:
    uvicorn api:app --reload --port 8000

POST one heartbeat's features to /predict and get a prediction back.
The saved model is a full sklearn Pipeline, so any scaling the model
needs is applied inside it - the API never has to guess.
"""

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model

from config import BEST_MODEL_PATH, MODEL_META_PATH

# Load the trained pipeline and its metadata once, when the server starts
model = joblib.load(BEST_MODEL_PATH)
with open(MODEL_META_PATH) as f:
    META = json.load(f)

FEATURES = META["feature_columns"]
THRESHOLD = META.get("decision_threshold", 0.5)

# Request schema is generated from the exact features the model was trained on
HeartbeatFeatures = create_model("HeartbeatFeatures", **{name: (float, ...) for name in FEATURES})

app = FastAPI(title="Real-Time Abnormal Heartbeat Detection API")


@app.get("/")
def root():
    return {"message": "Real-Time Abnormal Heartbeat Detection API is running.",
            "model": META["model_name"], "features": FEATURES}


@app.get("/model-info")
def model_info():
    return META


@app.post("/predict")
def predict(features: HeartbeatFeatures):
    """Return Normal/Abnormal plus the abnormal-beat probability for one heartbeat."""
    X = pd.DataFrame([features.model_dump()], columns=FEATURES)
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "prediction": "Abnormal" if prob >= THRESHOLD else "Normal",
        "abnormal_probability": round(prob, 4),
    }
