"""
ML Inference API — Replaces the rule-based /predict-risk endpoint.
Exposes the same REST contract so existing callers need zero changes.

Endpoints:
  POST /predict-risk   → ML model prediction (Low / Medium / High)
  POST /predict-proba  → Raw churn probability
  GET  /model/info     → Current loaded model metadata
  GET  /health         → Liveness check
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.feature_engineering import build_feature_matrix, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction ML API",
    description="ML-based customer churn risk scoring — replaces rule engine",
    version="2.0.0",
)
Instrumentator().instrument(app).expose(app)
# ──────────────────────────────────────────────
# Model loader (singleton)
# ──────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.pkl")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "models/model_schema.json")
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

_model = None
_schema = {}


def get_model():
    global _model, _schema
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run ml/train.py first."
            )
        logger.info("Loading model from %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)

        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH) as f:
                _schema = json.load(f)
        logger.info("Model loaded. Schema: %s", _schema.get("model_type", "unknown"))
    return _model


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class TicketLog(BaseModel):
    ticket_id: str
    created_at: str                           # ISO 8601
    category: str = "other"                   # complaint|billing|technical|other
    sentiment_score: float = Field(0.0, ge=-1.0, le=1.0)


class CustomerInput(BaseModel):
    customer_id: str
    monthly_charges: float
    prior_monthly_charges: Optional[float] = None
    tenure: int = 12
    contract: str = "Month-to-Month"          # Month-to-Month|One year|Two year
    internet_service: str = "DSL"             # No|DSL|Fiber optic
    tickets: List[TicketLog] = []


class RiskPrediction(BaseModel):
    customer_id: str
    risk_category: str                        # Low | Medium | High
    churn_probability: float
    model_version: str
    predicted_at: str


class BatchInput(BaseModel):
    customers: List[CustomerInput]


# ──────────────────────────────────────────────
# Helper: build feature matrix from request
# ──────────────────────────────────────────────

def _request_to_features(customers: List[CustomerInput]) -> np.ndarray:
    cust_rows = []
    ticket_rows = []

    for c in customers:
        cust_rows.append({
            "customer_id": c.customer_id,
            "monthly_charges": c.monthly_charges,
            "prior_monthly_charges": c.prior_monthly_charges or c.monthly_charges,
            "tenure": c.tenure,
            "contract": c.contract,
            "internet_service": c.internet_service,
        })
        for t in c.tickets:
            ticket_rows.append({
                "customer_id": c.customer_id,
                "created_at": t.created_at,
                "category": t.category,
                "sentiment_score": t.sentiment_score,
            })

    customers_df = pd.DataFrame(cust_rows)
    tickets_df = pd.DataFrame(ticket_rows) if ticket_rows else pd.DataFrame(
        columns=["customer_id", "created_at", "category", "sentiment_score"]
    )

    matrix = build_feature_matrix(customers_df, tickets_df)

    # Ensure columns present and in correct order
    for col in FEATURE_COLUMNS:
        if col not in matrix.columns:
            matrix[col] = 0.0

    return matrix[FEATURE_COLUMNS].values.astype(float), matrix["customer_id"].tolist()


def _prob_to_risk(prob: float) -> str:
    """
    Map churn probability to human-readable risk tier.
    Thresholds are configurable via env vars.
    """
    high_threshold = float(os.getenv("HIGH_RISK_THRESHOLD", "0.65"))
    medium_threshold = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.35"))

    if prob >= high_threshold:
        return "High"
    elif prob >= medium_threshold:
        return "Medium"
    else:
        return "Low"


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup to avoid cold-start latency on first request."""
    try:
        get_model()
    except RuntimeError as e:
        logger.warning("Model not pre-loaded: %s", e)


@app.get("/health")
def health():
    model_loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model/info")
def model_info():
    """Return metadata about the currently loaded model."""
    try:
        get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "model_path": MODEL_PATH,
        "schema": _schema,
        "feature_columns": FEATURE_COLUMNS,
        "thresholds": {
            "high_risk": float(os.getenv("HIGH_RISK_THRESHOLD", "0.65")),
            "medium_risk": float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.35")),
        },
    }


@app.post("/predict-risk", response_model=RiskPrediction)
def predict_risk(customer: CustomerInput):
    """
    Single-customer churn risk prediction.
    Replaces the rule-based engine — same API contract.
    """
    model = get_model()
    X, customer_ids = _request_to_features([customer])

    try:
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return RiskPrediction(
        customer_id=customer.customer_id,
        risk_category=_prob_to_risk(prob),
        churn_probability=round(prob, 4),
        model_version=_schema.get("trained_at", "unknown"),
        predicted_at=datetime.utcnow().isoformat(),
    )


@app.post("/predict-proba")
def predict_proba(customer: CustomerInput):
    """Returns raw probability scores — useful for downstream systems."""
    model = get_model()
    X, _ = _request_to_features([customer])
    probs = model.predict_proba(X)[0]
    return {
        "customer_id": customer.customer_id,
        "prob_not_churn": round(float(probs[0]), 4),
        "prob_churn": round(float(probs[1]), 4),
    }


@app.post("/batch-predict")
def batch_predict(batch: BatchInput):
    """
    Batch prediction for multiple customers.
    Returns list of RiskPrediction objects.
    """
    model = get_model()
    X, customer_ids = _request_to_features(batch.customers)

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {e}")

    results = []
    for cid, prob in zip(customer_ids, probs):
        results.append({
            "customer_id": cid,
            "risk_category": _prob_to_risk(float(prob)),
            "churn_probability": round(float(prob), 4),
            "model_version": _schema.get("trained_at", "unknown"),
            "predicted_at": datetime.utcnow().isoformat(),
        })

    return {"predictions": results, "count": len(results)}


# ──────────────────────────────────────────────
# Dev entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.ml_inference:app", host="0.0.0.0", port=8000, reload=True)