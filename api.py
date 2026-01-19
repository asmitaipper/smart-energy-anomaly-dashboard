from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np

app = FastAPI(
    title="Smart Energy Anomaly Detection API",
    description="API to detect anomalies in energy usage data using IsolationForest.",
    version="1.0.0",
)

MODEL_PATH = Path("models/energy_iforest.joblib")
SCALER_PATH = Path("models/scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class Reading(BaseModel):
    power_kw: float
    voltage: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Energy anomaly detection API running"}


@app.post("/predict")
def predict(reading: Reading):
    x = np.array([[reading.power_kw, reading.voltage]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]       # 1 = normal, -1 = anomaly
    score = model.decision_function(x_scaled)[0]

    is_anomaly = pred == -1

    return {
        "is_anomaly": bool(is_anomaly),
        "raw_prediction": int(pred),
        "anomaly_score": float(score),
        "details": "Anomalous usage" if is_anomaly else "Normal usage",
    }
