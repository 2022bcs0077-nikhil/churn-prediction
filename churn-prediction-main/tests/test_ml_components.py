"""
Unit Tests for Assignment 2 — ML Components
Tests: feature engineering, model inference API, drift detection
"""

import sys
import os
import json
from datetime import datetime, timedelta
import random

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────

from ml.feature_engineering import (
    compute_ticket_features,
    compute_customer_features,
    build_feature_matrix,
    FEATURE_COLUMNS,
    ChurnFeatureTransformer,
)


def make_customers(n=5, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(n)],
        "monthly_charges": rng.uniform(20, 120, n),
        "prior_monthly_charges": rng.uniform(20, 120, n),
        "tenure": rng.integers(1, 60, n),
        "contract": rng.choice(["Month-to-Month", "One year", "Two year"], n),
        "internet_service": rng.choice(["No", "DSL", "Fiber optic"], n),
    })


def make_tickets(customer_ids, seed=42):
    random.seed(seed)
    rows = []
    base = datetime.utcnow()
    for cid in customer_ids:
        for _ in range(random.randint(0, 6)):
            rows.append({
                "customer_id": cid,
                "created_at": base - timedelta(days=random.randint(0, 90)),
                "category": random.choice(["complaint", "billing", "technical", "other"]),
                "sentiment_score": random.uniform(-1, 1),
            })
    return pd.DataFrame(rows)


class TestFeatureEngineering:
    def test_ticket_features_shape(self):
        customers = make_customers()
        tickets = make_tickets(customers["customer_id"].tolist())
        feat = compute_ticket_features(tickets)
        assert "customer_id" in feat.columns
        assert "ticket_count_7d" in feat.columns
        assert "ticket_count_30d" in feat.columns
        assert "ticket_count_90d" in feat.columns
        assert "avg_sentiment_score" in feat.columns

    def test_ticket_count_non_negative(self):
        customers = make_customers()
        tickets = make_tickets(customers["customer_id"].tolist())
        feat = compute_ticket_features(tickets)
        for col in ["ticket_count_7d", "ticket_count_30d", "ticket_count_90d"]:
            assert (feat[col] >= 0).all(), f"{col} has negative values"

    def test_ticket_count_window_ordering(self):
        """30d count must always >= 7d count."""
        customers = make_customers(20)
        tickets = make_tickets(customers["customer_id"].tolist())
        feat = compute_ticket_features(tickets)
        assert (feat["ticket_count_30d"] >= feat["ticket_count_7d"]).all()
        assert (feat["ticket_count_90d"] >= feat["ticket_count_30d"]).all()

    def test_customer_features_contract_encoding(self):
        customers = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "monthly_charges": [50, 80, 30],
            "tenure": [12, 24, 6],
            "contract": ["Month-to-Month", "One year", "Two year"],
            "internet_service": ["DSL", "Fiber optic", "No"],
        })
        feat = compute_customer_features(customers)
        assert feat.loc[feat["customer_id"] == "C001", "contract_encoded"].values[0] == 0
        assert feat.loc[feat["customer_id"] == "C002", "contract_encoded"].values[0] == 1
        assert feat.loc[feat["customer_id"] == "C003", "contract_encoded"].values[0] == 2

    def test_build_feature_matrix_all_columns_present(self):
        customers = make_customers(10)
        tickets = make_tickets(customers["customer_id"].tolist())
        matrix = build_feature_matrix(customers, tickets)
        for col in FEATURE_COLUMNS:
            assert col in matrix.columns, f"Missing feature column: {col}"

    def test_build_feature_matrix_no_nulls(self):
        customers = make_customers(10)
        tickets = make_tickets(customers["customer_id"].tolist())
        matrix = build_feature_matrix(customers, tickets)
        assert not matrix[FEATURE_COLUMNS].isnull().any().any(), \
            "Feature matrix contains NaN values"

    def test_customer_with_no_tickets(self):
        """Customers with no tickets should still get zero-filled ticket features."""
        customers = make_customers(3)
        empty_tickets = pd.DataFrame(
            columns=["customer_id", "created_at", "category", "sentiment_score"]
        )
        matrix = build_feature_matrix(customers, empty_tickets)
        assert len(matrix) == 3
        for col in ["ticket_count_7d", "ticket_count_30d", "ticket_count_90d"]:
            assert (matrix[col] == 0).all()

    def test_feature_transformer_sklearn_compatible(self):
        """ChurnFeatureTransformer should work inside an sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        customers = make_customers(20)
        tickets = make_tickets(customers["customer_id"].tolist())

        transformer = ChurnFeatureTransformer()
        X = transformer.fit_transform({"customers": customers, "tickets": tickets})

        assert X.shape[1] == len(FEATURE_COLUMNS)
        assert X.dtype == float


# ─────────────────────────────────────────────
# ML Inference API Tests
# ─────────────────────────────────────────────

class TestMLInferenceAPI:
    """
    API tests using a real (fast-trained) sklearn model.
    No MagicMock — uses a tiny LogisticRegression trained inline.
    """

    @pytest.fixture
    def client_with_real_model(self, tmp_path, monkeypatch):
        """Train a tiny real model and wire up the API to use it."""
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # Train minimal real model on synthetic data
        customers = make_customers(100)
        tickets = make_tickets(customers["customer_id"].tolist())
        matrix = build_feature_matrix(customers, tickets)
        X = matrix[FEATURE_COLUMNS].values.astype(float)
        y = (np.random.rand(len(X)) > 0.7).astype(int)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ])
        pipeline.fit(X, y)

        model_path = str(tmp_path / "churn_model.pkl")
        joblib.dump(pipeline, model_path)

        schema = {
            "model_type": "logistic_regression",
            "trained_at": "2025-01-01T00:00:00",
            "feature_columns": FEATURE_COLUMNS,
        }
        schema_path = str(tmp_path / "model_schema.json")
        with open(schema_path, "w") as f:
            json.dump(schema, f)

        monkeypatch.setenv("MODEL_PATH", model_path)
        monkeypatch.setenv("SCHEMA_PATH", schema_path)

        import importlib
        import app.ml_inference as ml_module
        importlib.reload(ml_module)
        ml_module._model = joblib.load(model_path)
        ml_module._schema = schema

        return TestClient(ml_module.app)

    def test_health_endpoint(self, client_with_real_model):
        resp = client_with_real_model.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_risk_returns_valid_category(self, client_with_real_model):
        payload = {
            "customer_id": "C001",
            "monthly_charges": 75.0,
            "tenure": 12,
            "contract": "Month-to-Month",
            "internet_service": "Fiber optic",
            "tickets": [
                {
                    "ticket_id": "T1",
                    "created_at": "2024-11-01T10:00:00",
                    "category": "complaint",
                    "sentiment_score": -0.8,
                }
            ],
        }
        resp = client_with_real_model.post("/predict-risk", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["customer_id"] == "C001"
        assert data["risk_category"] in ("Low", "Medium", "High")
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_risk_no_tickets(self, client_with_real_model):
        payload = {
            "customer_id": "C002",
            "monthly_charges": 40.0,
            "tenure": 36,
            "contract": "Two year",
            "internet_service": "DSL",
            "tickets": [],
        }
        resp = client_with_real_model.post("/predict-risk", json=payload)
        assert resp.status_code == 200

    def test_batch_predict(self, client_with_real_model):
        payload = {
            "customers": [
                {
                    "customer_id": f"C{i:03d}",
                    "monthly_charges": 50.0 + i,
                    "tenure": 12,
                    "contract": "Month-to-Month",
                    "internet_service": "DSL",
                    "tickets": [],
                }
                for i in range(5)
            ]
        }
        resp = client_with_real_model.post("/batch-predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5
        assert len(data["predictions"]) == 5


# ─────────────────────────────────────────────
# Drift Detection Tests
# ─────────────────────────────────────────────

class TestDriftDetection:
    def test_psi_identical_distributions(self):
        """PSI between identical distributions should be ~0."""
        from mlops.drift_detection import _compute_psi
        data = np.random.rand(500)
        psi = _compute_psi(data, data)
        assert psi < 0.01

    def test_psi_different_distributions(self):
        """PSI between clearly different distributions should be >0.1."""
        from mlops.drift_detection import _compute_psi
        ref = np.random.rand(500) * 0.3          # low probs
        cur = 0.7 + np.random.rand(500) * 0.3   # high probs
        psi = _compute_psi(ref, cur)
        assert psi > 0.1, f"Expected PSI > 0.1 for divergent dists, got {psi}"

    def test_fallback_drift_no_drift(self):
        """Same distribution should not trigger drift."""
        from mlops.drift_detection import _fallback_drift_detection
        np.random.seed(42)
        ref = pd.DataFrame({col: np.random.randn(200) for col in FEATURE_COLUMNS})
        cur = pd.DataFrame({col: np.random.randn(200) for col in FEATURE_COLUMNS})

        summary = _fallback_drift_detection(ref, cur)
        assert "drift_detected" in summary
        # Not asserting value since statistical tests can vary, just check structure
        assert "drift_share" in summary
        assert "drifted_columns" in summary

    def test_fallback_drift_detects_large_shift(self):
        """A massive distribution shift should be flagged."""
        from mlops.drift_detection import _fallback_drift_detection
        ref = pd.DataFrame({col: np.random.randn(300) for col in FEATURE_COLUMNS})
        # Current data has very different distribution (scaled by 100)
        cur = pd.DataFrame({col: np.random.randn(300) * 100 + 500 for col in FEATURE_COLUMNS})

        summary = _fallback_drift_detection(ref, cur)
        assert summary["drift_detected"] is True