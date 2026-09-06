
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


# -----------------------------
# Load Model
# -----------------------------

artifact = joblib.load(
    "fraud_detection_model.pkl"
)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]


# -----------------------------
# Create FastAPI App
# -----------------------------

app = FastAPI(
    title="Bank Fraud Detection API",
    description="Machine Learning API for detecting fraudulent transactions",
    version="1.0"
)


# -----------------------------
# Request Model
# -----------------------------

class TransactionRequest(BaseModel):

    data: dict[str, float]


# -----------------------------
# Home Endpoint
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "Bank Fraud Detection API is running",
        "status": "success"
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(request: TransactionRequest):

    received_features = set(
        request.data.keys()
    )

    expected_features = set(
        features
    )

    missing_features = expected_features - received_features

    if missing_features:

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing features",
                "missing_features": list(
                    missing_features
                )
            }
        )

    # Arrange features in correct order

    input_data = pd.DataFrame(
        [[request.data[feature] for feature in features]],
        columns=features
    )

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= threshold
    )

    if prediction == 1:
        result = "FRAUD"
    else:
        result = "NORMAL"

    return {
        "prediction": result,
        "fraud_probability": float(probability),
        "threshold": float(threshold)
    }
