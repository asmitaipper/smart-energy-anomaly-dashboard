# Smart Energy Anomaly Dashboard

End-to-end project for monitoring energy usage and detecting anomalies using Python, IsolationForest, FastAPI, and a simple web dashboard.

## Features

- Energy usage data ingestion from CSV
- IsolationForest-based anomaly detection on power and voltage
- FastAPI REST API for predictions
- Simple HTML/JS dashboard to test readings visually

## How to run

```bash
pip install -r requirements.txt
python src/train_model.py
uvicorn src.api:app --reload
